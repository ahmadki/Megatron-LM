# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The package must import and run without megatron, Transformer Engine or nvrx.

Each check runs in a fresh interpreter whose ``sys.meta_path`` refuses those
packages, so an accidental top-level import anywhere in the package fails here
even in environments (CI containers) where they are installed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from ._harness import REPO_ROOT

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


@pytest.mark.usefixtures("local_rank_zero")
def test_blocker_really_blocks(scrubbed_env):
    result = _run("import megatron.core\n", scrubbed_env)
    assert result.returncode != 0
    assert "not allowed in format_converter" in result.stderr
