# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The package must import and run without megatron, Transformer Engine or nvrx.

Each check runs in a fresh interpreter whose ``sys.meta_path`` refuses those
packages, so an accidental top-level import anywhere in the package fails here
even in environments (CI containers) where they are installed.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import subprocess
import sys
import textwrap

import pytest
import torch

from ._harness import REPO_ROOT
from ._synthetic import default_args, write_torch_dist_store

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
        assert {"inspect_checkpoint", "verify_checkpoint"} <= set(dir(fc))
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
def test_commands_run_without_megatron(tmp_path, scrubbed_env):
    store = write_torch_dist_store(
        tmp_path / "store",
        {"decoder.final_layernorm.weight": torch.ones(4)},
        common={"args": default_args(), "iteration": 7, "checkpoint_version": 3.0},
    )
    digests = tmp_path / "digests.json"
    result = _run(
        f"""
        from tools.checkpoint.format_converter import cli
        inspected = cli.main(["inspect", {str(store)!r}, "--digests", {str(digests)!r}])
        verified = cli.main(["verify", {str(store)!r}, {str(store)!r}])
        import torch
        assert not torch.cuda.is_initialized()
        assert not [m for m in sys.modules if m.split(".")[0] in BLOCKED]
        sys.exit(inspected or verified)
        """,
        scrubbed_env,
    )
    assert result.returncode == 0, result.stderr
    assert "format:     torch_dist (mcore)" in result.stdout
    assert "verify:     OK" in result.stdout
    assert digests.is_file()
