# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Logical ids, slices, copies and outputs validate their invariants."""

from __future__ import annotations

import pytest
import torch

from tools.checkpoint.format_converter.core.logical import (
    LogicalCopy,
    LogicalId,
    LogicalOutput,
    LogicalSlice,
)
from tools.checkpoint.format_converter.engine.regions import Region
from tools.checkpoint.format_converter.errors import InternalError

QKV = LogicalId("decoder", 3, "self_attention.linear_qkv", "weight")
NORM = LogicalId(None, None, "decoder.final_norm", "weight")


def test_ids_render_block_layer_and_name():
    assert str(QKV) == "decoder[3].self_attention.linear_qkv.weight"
    assert QKV.name == "self_attention.linear_qkv.weight"
    assert str(NORM) == "decoder.final_norm.weight"
    assert str(LogicalId(None, None, "", "scale")) == "scale"
    assert {QKV, LogicalId("decoder", 3, "self_attention.linear_qkv", "weight")} == {QKV}


@pytest.mark.parametrize(
    ("block", "layer", "leaf"),
    [("decoder", None, "weight"), (None, 0, "weight"), ("decoder", -1, "weight"), (None, None, "")],
)
def test_invalid_ids_are_internal_errors(block, layer, leaf):
    with pytest.raises(InternalError):
        LogicalId(block, layer, "mlp.linear_fc1", leaf)


def test_slice_shape_is_its_region():
    piece = LogicalSlice(
        QKV,
        "decoder.layers.3.self_attention.linear_qkv.weight",
        Region((0, 0), (24, 8)),
        torch.bfloat16,
    )
    assert piece.shape == (24, 8)


def test_copies_need_boxes_of_one_shape():
    LogicalCopy(QKV, Region((0, 0), (2, 8)), Region((4, 0), (2, 8)))
    with pytest.raises(InternalError, match="differ in shape"):
        LogicalCopy(QKV, Region((0, 0), (2, 8)), Region((0, 0), (8, 2)))


def test_outputs_must_be_tiled_by_their_copies():
    halves = (
        LogicalCopy(QKV, Region((0, 0), (12, 8)), Region((0, 0), (12, 8))),
        LogicalCopy(NORM, Region((0, 0), (12, 8)), Region((12, 0), (12, 8))),
    )
    LogicalOutput(QKV, (24, 8), torch.float32, halves)
    with pytest.raises(InternalError, match="do not tile"):
        LogicalOutput(QKV, (24, 8), torch.float32, halves[:1])
    with pytest.raises(InternalError, match="do not tile"):
        LogicalOutput(QKV, (24, 8), torch.float32, (halves[0], halves[0], halves[1]))
    scalar = LogicalCopy(NORM, Region((), ()), Region((), ()))
    LogicalOutput(NORM, (), torch.float32, (scalar,))
