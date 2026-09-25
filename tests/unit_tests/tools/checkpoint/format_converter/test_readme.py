# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The package README documents every flag, exit code and refusal code, and its example runs."""

from __future__ import annotations

import re

import pytest

from tools.checkpoint.format_converter import cli, errors
from tools.checkpoint.format_converter.errors import ExitCode

from ._harness import REPO_ROOT

README = REPO_ROOT / "tools" / "checkpoint" / "format_converter" / "README.md"


@pytest.fixture(name="readme", scope="module")
def fixture_readme() -> str:
    return README.read_text()


@pytest.mark.parametrize("command", [None, "inspect", "verify"])
def test_every_flag_is_documented(readme, command, capsys):
    argv = ["--help"] if command is None else [command, "--help"]
    with pytest.raises(SystemExit):
        cli.main(argv)
    flags = set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", capsys.readouterr().out)) - {"--help"}
    assert flags, "no flags found in the help text"
    missing = sorted(flag for flag in flags if f"`{flag}" not in readme)
    assert missing == []


def test_every_exit_code_is_documented(readme):
    for code in ExitCode:
        assert re.search(rf"^\| {int(code)} \| ", readme, re.MULTILINE), code


def test_every_refusal_code_is_documented(readme):
    codes = {value for name, value in vars(errors).items() if re.fullmatch(r"FC_[RPAE]\d{2}", name)}
    assert codes
    for code in codes:
        assert f"| `{code}` |" in readme, code


def test_key_map_example_runs(readme):
    blocks = re.findall(r"```python\n(.*?)```", readme, re.DOTALL)
    (example,) = [block for block in blocks if "def to_store_key" in block]
    namespace: dict = {}
    exec(compile(example, str(README), "exec"), namespace)  # pylint: disable=exec-used
    to_store_key = namespace["to_store_key"]
    assert to_store_key("decoder.layers.0.mlp.experts.linear_fc2.weight3", 2) == (
        "model.module.decoder.layers.0.mlp.experts.linear_fc2.weight19"
    )
    assert to_store_key("decoder.final_layernorm.weight", 1) == (
        "model.module.decoder.final_layernorm.weight"
    )


def test_readme_has_no_local_paths(readme):
    assert re.search(r"/(home|datasets|Users)/", readme) is None
