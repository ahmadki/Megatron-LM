# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CLI plumbing: help, usage errors and the ``python -m`` entry point."""

from __future__ import annotations

import subprocess
import sys

import pytest

from tools.checkpoint.format_converter import __version__, cli

from ._harness import REPO_ROOT


def test_help_exits_zero_and_lists_exit_codes(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    out = " ".join(capsys.readouterr().out.split())  # argparse wraps at the terminal width
    assert "usage: python -m tools.checkpoint.format_converter" in out
    assert "8 memory budget exceeded" in out


def test_missing_command_is_usage_error(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2
    assert "required" in capsys.readouterr().err


def test_unknown_command_is_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["no-such-command"])
    assert excinfo.value.code == 2


def test_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


@pytest.mark.usefixtures("local_rank_zero")
def test_python_dash_m_entry_point(scrubbed_env):
    ok = subprocess.run(
        [sys.executable, "-m", "tools.checkpoint.format_converter", "--help"],
        cwd=REPO_ROOT,
        env=scrubbed_env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert ok.returncode == 0, ok.stderr
    assert "usage:" in ok.stdout
    bad = subprocess.run(
        [sys.executable, "-m", "tools.checkpoint.format_converter"],
        cwd=REPO_ROOT,
        env=scrubbed_env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert bad.returncode == 2
