# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Mamba-2 layouts pinned to ``MambaMixer.sharded_state_dict``, read with ``ast`` (host-safe).

``MambaMixer.sharded_state_dict`` decides how torch_dist stores the mixer: which
tensors its split factory cuts into which parts (in order), which it leaves
whole, and the checkpoint names of the convolution the mixer owns. The
converter's Mamba-2 layouts, its tensors kept whole and its convolution names
must be those. The factory's sizes are pinned on a real module in the training
container (``real/test_mamba.py``).
"""

from __future__ import annotations

import ast

import pytest

from tools.checkpoint.format_converter.core.keys import MixerTensor, join_mixer_tensor
from tools.checkpoint.format_converter.core.mixers import MAMBA2, TORCH_DIST_WHOLE, mamba2_parts

from .._harness import REPO_ROOT

pytestmark = pytest.mark.covers("mamba2", "parity")
MAMBA_MIXER = REPO_ROOT / "megatron" / "core" / "ssm" / "mamba_mixer.py"


@pytest.fixture(name="sharded_state_dict", scope="module")
def fixture_sharded_state_dict() -> ast.FunctionDef:
    """``MambaMixer.sharded_state_dict``."""
    if not MAMBA_MIXER.is_file():
        pytest.skip(f"{MAMBA_MIXER} is not in this checkout")
    tree = ast.parse(MAMBA_MIXER.read_text(), str(MAMBA_MIXER))
    (mixer,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MambaMixer"]
    (method,) = [
        n for n in mixer.body if isinstance(n, ast.FunctionDef) and n.name == "sharded_state_dict"
    ]
    return method


def _tensor_names(target: ast.expr, loops: dict[str, list[str]]) -> list[str]:
    """The tensors ``sharded_state_dict[f"{prefix}<name>"]`` names, expanding loop variables."""
    assert isinstance(target, ast.Subscript) and isinstance(target.slice, ast.JoinedStr)
    names = [""]
    for value in target.slice.values:
        if isinstance(value, ast.Constant):
            names = [name + value.value for name in names]
        elif isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name):
            if value.value.id != "prefix":
                names = [name + item for name in names for item in loops[value.value.id]]
    return names


def _splits(method: ast.FunctionDef) -> dict[str, list[str]]:
    """Every tensor ``_split_tensor_factory`` splits, and its part names in order."""
    loops = {
        node.target.id: ast.literal_eval(node.iter)
        for node in ast.walk(method)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
        if isinstance(node.iter, ast.List)
    }
    splits = {}
    for node in ast.walk(method):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_split_tensor_factory":
            target, _, names, axis = node.args
            assert ast.literal_eval(axis) == 0
            for tensor in _tensor_names(target, loops):
                splits[tensor] = ast.literal_eval(names)
    return splits


def test_the_split_factory_cuts_the_converters_parts_in_order(sharded_state_dict):
    in_proj, conv = mamba2_parts(8, 6, 4)
    assert _splits(sharded_state_dict) == {
        "in_proj.weight": list(in_proj.parts),
        "conv1d_weight": list(conv.parts),
        "conv1d_bias": list(conv.parts),
    }


def test_only_the_in_proj_bias_is_left_whole(sharded_state_dict):
    """``MambaMixer`` asserts ``bias=False``; an ``in_proj.bias`` of a mixer built otherwise would
    not be split."""
    strings = {
        value.value
        for node in ast.walk(sharded_state_dict)
        if isinstance(node, ast.JoinedStr)
        for value in node.values
        if isinstance(value, ast.Constant)
    }
    assert "in_proj.bias" not in strings and "in_proj.weight" in strings
    assert TORCH_DIST_WHOLE[MAMBA2] == {("in_proj", "bias")}


def test_the_convolution_keeps_the_conv1d_modules_checkpoint_names(sharded_state_dict):
    (key_map,) = [
        ast.literal_eval(node.value)
        for node in ast.walk(sharded_state_dict)
        if isinstance(node, ast.Assign)
        and [getattr(t, "id", None) for t in node.targets] == ["conv_checkpoint_key_map"]
    ]
    expected = {}
    for leaf in ("weight", "bias"):
        _, parameter = join_mixer_tensor(MixerTensor("mixer", "conv1d", leaf, conv_parameter=True))
        path, module_leaf = join_mixer_tensor(MixerTensor("mixer", "conv1d", leaf))
        expected[parameter] = f"{path.removeprefix('mixer.')}.{module_leaf}"
    assert key_map == expected == {"conv1d_weight": "conv1d.weight", "conv1d_bias": "conv1d.bias"}
