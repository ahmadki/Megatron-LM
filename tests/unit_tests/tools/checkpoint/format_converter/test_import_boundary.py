# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The package must import and run without megatron, Transformer Engine or nvrx.

Each check runs in a fresh interpreter whose ``sys.meta_path`` refuses those
packages, so an accidental top-level import anywhere in the package fails here
even in environments (CI containers) where they are installed. An AST scan
also keeps each subpackage to the layers below it.
"""

from __future__ import annotations

import ast
import enum
import importlib
import inspect
import json
import pkgutil
import subprocess
import sys
import textwrap

import pytest
import torch

from ._harness import REPO_ROOT
from ._synthetic import default_args, torch_save_bytes, write_torch_dist_store

ENUMS = "megatron.core.transformer.enums"


class FakeAttnBackend(enum.Enum):
    """Pickled under Megatron's name ``AttnBackend``."""

    auto = 5


def megatron_common(**common):
    """``torch.save([common])`` whose args reference Megatron's ``AttnBackend`` enum."""
    args = default_args(attention_backend=FakeAttnBackend.auto)
    return torch_save_bytes(
        [{"args": args, **common}], aliases={FakeAttnBackend: (ENUMS, "AttnBackend")}
    )


_BLOCKER = textwrap.dedent("""
    import importlib
    import pkgutil
    import sys

    BLOCKED = ("megatron", "transformer_engine", "nvidia_resiliency_ext")

    class _BlockedImportFinder:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError(f"import of {name} is not allowed in format_converter")
            return None

    sys.meta_path.insert(0, _BlockedImportFinder())
    """)


def _run(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + textwrap.dedent(script)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


@pytest.mark.usefixtures("local_rank_zero")
def test_package_import_does_not_import_torch(scrubbed_env):
    result = _run(
        """
        import tools.checkpoint.format_converter as fc
        assert "torch" not in sys.modules, "package __init__ imported torch"
        assert fc.__version__ == "0.1.0"
        assert {
            "convert_checkpoint", "inspect_checkpoint", "plan_conversion", "verify_checkpoint"
        } <= set(dir(fc))
        print("ok")
        """,
        scrubbed_env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


@pytest.mark.usefixtures("local_rank_zero")
def test_every_module_imports_without_megatron(scrubbed_env):
    result = _run(
        """
        import tools.checkpoint.format_converter as fc
        names = [
            info.name
            for info in pkgutil.walk_packages(fc.__path__, fc.__name__ + ".")
        ]
        for name in names:
            importlib.import_module(name)
        print(len(names))
        """,
        scrubbed_env,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) >= 4


def test_api_exports_are_functions_after_the_subpackages_load():
    """Importing ``verify/`` sets the package attribute ``verify``; no export may clash."""
    importlib.import_module("tools.checkpoint.format_converter.verify.compare")
    fc = importlib.import_module("tools.checkpoint.format_converter")
    from tools.checkpoint.format_converter import (  # pylint: disable=import-outside-toplevel
        inspect_checkpoint,
        verify_checkpoint,
    )

    for name in fc.__all__:
        if name != "__version__":
            assert inspect.isfunction(getattr(fc, name)), name
    assert inspect.isfunction(inspect_checkpoint) and inspect.isfunction(verify_checkpoint)
    assert {"plan_conversion", "convert_checkpoint"} <= set(fc.__all__)
    assert set(fc.__all__).isdisjoint(info.name for info in pkgutil.iter_modules(fc.__path__))


@pytest.mark.usefixtures("local_rank_zero")
def test_usage_errors_do_not_import_torch(scrubbed_env):
    result = _run(
        """
        from tools.checkpoint.format_converter import cli
        for argv in (
            ["verify", "A", "--mode", "bogus"],
            ["verify", "A", "--seed", "x"],
            ["inspect", "A", "--max-item-bytes", "lots"],
            ["inspect", "A", "--from", "zarr"],
        ):
            try:
                cli.main(argv)
            except SystemExit as exit_:
                assert exit_.code == 2, (argv, exit_.code)
            else:
                raise AssertionError(f"{argv} was accepted")
        assert "torch" not in sys.modules, "a usage error imported torch"
        print("ok")
        """,
        scrubbed_env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


@pytest.mark.usefixtures("local_rank_zero")
def test_blocker_really_blocks(scrubbed_env):
    result = _run("import megatron.core\n", scrubbed_env)
    assert result.returncode != 0
    assert "not allowed in format_converter" in result.stderr


@pytest.mark.usefixtures("local_rank_zero")
def test_plain_codec_refuses_megatron_classes_with_fc_e01(tmp_path, scrubbed_env):
    """Runs where Megatron is blocked, so CI containers that have it exercise it too."""
    payload = tmp_path / "common"
    payload.write_bytes(megatron_common())
    result = _run(
        f"""
        from pathlib import Path
        from tools.checkpoint.format_converter.errors import RefusedError
        from tools.checkpoint.format_converter.formats.base import WHOLE, CommonLayout, RawCommon
        from tools.checkpoint.format_converter.mcore_state import common_state

        assert common_state.auto_mode() is common_state.CodecMode.PLAIN
        payload = Path({str(payload)!r}).read_bytes()
        raw = RawCommon("test payload", CommonLayout.WRAPPED, {{WHOLE: payload}})
        try:
            common_state.decode(raw, common_state.CodecMode.PLAIN)
        except RefusedError as err:
            print(int(err.exit_code), err.format())
        """,
        scrubbed_env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        f"3 FC-E01: test payload references classes only the Megatron codec can load "
        f"({ENUMS}.AttnBackend) hint: run inside the Megatron container"
    )


@pytest.mark.usefixtures("local_rank_zero")
def test_commands_run_without_megatron(tmp_path, scrubbed_env):
    store = write_torch_dist_store(
        tmp_path / "store",
        {"decoder.final_layernorm.weight": torch.ones(4)},
        common={"args": default_args(), "iteration": 7, "checkpoint_version": 3.0},
    )
    digests = tmp_path / "digests.json"
    dst = tmp_path / "dst" / "iter_0000007"
    result = _run(
        f"""
        from tools.checkpoint.format_converter import cli
        inspected = cli.main(["inspect", {str(store)!r}, "--digests", {str(digests)!r}])
        verified = cli.main(["verify", {str(store)!r}, {str(store)!r}])
        planned = cli.main(["plan", {str(store)!r}, "--to", "torch_dist"])
        converted = cli.main(["convert", {str(store)!r}, {str(dst)!r}, "--to", "torch_dist",
                              "--workers", "0"])
        import torch
        assert not torch.cuda.is_initialized()
        assert not [m for m in sys.modules if m.split(".")[0] in BLOCKED]
        sys.exit(inspected or verified or planned or converted)
        """,
        scrubbed_env,
    )
    assert result.returncode == 0, result.stderr
    assert "format:     torch_dist (mcore)" in result.stdout
    assert "verify:     OK" in result.stdout
    assert "(plain codec)" in result.stdout
    assert f"published:  {dst}" in result.stdout
    assert digests.is_file()


@pytest.mark.usefixtures("local_rank_zero")
def test_plan_without_megatron_stubs_megatron_classes_and_convert_refuses(tmp_path, scrubbed_env):
    """``plan`` stays complete on a host without Megatron; ``convert`` needs the container."""
    store = write_torch_dist_store(
        tmp_path / "store",
        {"decoder.final_layernorm.weight": torch.ones(4)},
        common={
            "args": default_args(attention_backend=FakeAttnBackend.auto, no_save_optim=False),
            "iteration": 7,
            "checkpoint_version": 3.0,
            "optimizer": {"param_groups": []},
        },
        common_aliases={FakeAttnBackend: (ENUMS, "AttnBackend")},
    )
    dst = tmp_path / "dst"
    result = _run(
        f"""
        from tools.checkpoint.format_converter import cli
        planned = cli.main(["plan", {str(store)!r}, "--to", "torch_dist", "--json"])
        converted = cli.main(["convert", {str(store)!r}, {str(dst)!r}, "--to", "torch_dist",
                              "--workers", "0"])
        print("exit", planned, converted)
        """,
        scrubbed_env,
    )
    assert result.returncode == 0, result.stderr
    report, _, codes = result.stdout.rpartition("exit ")
    assert codes.split() == ["0", "3"]
    common = json.loads(report)["common"]
    assert common["codec"] == "stub"
    assert common["dropped"] == ["optimizer"]
    assert common["args_changes"] == {"no_save_optim": [False, True]}
    assert (
        f"FC-E01: common_state/shard_0_1 references classes only the Megatron codec can load "
        f"({ENUMS}.AttnBackend)" in result.stderr
    )
    assert not dst.exists()


PACKAGE = REPO_ROOT / "tools" / "checkpoint" / "format_converter"
LAYERS = {
    "engine": {"errors"},
    "core": {"errors", "engine"},
    "formats": {"errors", "engine", "core"},
    "rules": {"errors", "engine", "core"},
    "verify": {"errors", "engine", "core", "formats", "options"},
    "mcore_state": {"errors", "engine", "core", "formats"},
}
"""What each subpackage may import from the rest of the package (besides itself)."""


def _package_imports(path) -> set[str]:
    """The top-level package modules (``engine``, ``errors``, ...) that ``path`` imports."""
    parts = path.relative_to(REPO_ROOT).with_suffix("").parts
    package = parts[:-1]
    root = PACKAGE.relative_to(REPO_ROOT).parts
    found = set()
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else ()
            module = (*base, *(node.module or "").split(".")) if node.module else base
            names = [module] if node.module else [(*module, a.name) for a in node.names]
        elif isinstance(node, ast.Import):
            names = [tuple(a.name.split(".")) for a in node.names]
        else:
            continue
        found.update(n[len(root)] for n in names if n[: len(root)] == root and len(n) > len(root))
    return found


def test_each_layer_imports_only_the_layers_below_it():
    violations = []
    for layer, allowed in LAYERS.items():
        for path in sorted((PACKAGE / layer).glob("**/*.py")):
            extra = _package_imports(path) - allowed - {layer}
            if extra:
                violations.append(f"{path.relative_to(PACKAGE)} imports {sorted(extra)}")
    assert violations == []
    assert _package_imports(PACKAGE / "planning.py") >= {"engine", "formats", "mcore_state"}
