# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""HybridStack block layouts and their cross-check against the stored layer indices."""

from __future__ import annotations

import argparse

import pytest

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.blocks import BlockLayout, classify_blocks
from tools.checkpoint.format_converter.core.keys import KeyGrammarError
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, TORCH_DIST
from tools.checkpoint.format_converter.errors import InternalError, RefusedError


def _args(**fields) -> ArgsView:
    return ArgsView(argparse.Namespace(**fields))


def _keys(*indices: int) -> dict[str, None]:
    keys = {"embedding.word_embeddings.weight": None, "decoder.final_norm.weight": None}
    keys.update({f"decoder.layers.{i}.mlp.linear_fc1.weight": None for i in indices})
    return keys


@pytest.mark.parametrize("fmt", [TORCH_DIST, FSDP_DTENSOR])
def test_a_hybrid_decoder_is_one_per_layer_block(fmt):
    (block,) = classify_blocks(
        _args(hybrid_layer_pattern="*-|*-/*-", num_layers=4), _keys(0, 1, 2, 3), fmt
    )
    assert block.to_json() == {
        "prefix": "decoder",
        "kind": "hybrid_stack",
        "num_modules": 4,
        "td_layout": "per_layer",
        "dense_to_td": [0, 1, 2, 3],
        "symbols": ["*", "-", "*", "-"],
        "evidence": "hybrid layer pattern '*-|*-/*-' (4 layers)",
    }
    assert [block.dense_index(i, fmt) for i in range(4)] == [0, 1, 2, 3]
    assert block.stored_index(3, fmt) == 3


def test_models_without_a_pattern_have_no_hybrid_block():
    assert classify_blocks(_args(hybrid_layer_pattern=None), _keys(0), TORCH_DIST) == ()


@pytest.mark.parametrize(
    ("stored", "message"),
    [
        ((0, 1, 2), "means torch_dist layer indices 0-3, but the store has 0-2 (missing [3]"),
        ((0, 1, 2, 3, 5), "but the store has 0-3, 5 (missing none, unexpected [5])"),
        ((0, 2, 3, 4), "the store has 0, 2-4 (missing [1], unexpected [4])"),
    ],
)
def test_stored_indices_must_match_the_pattern(stored, message):
    with pytest.raises(RefusedError) as excinfo:
        classify_blocks(_args(hybrid_layer_pattern="*-*-"), _keys(*stored), TORCH_DIST)
    assert excinfo.value.refusal.code == errors.FC_R11
    assert message in excinfo.value.refusal.message


def test_num_layers_must_match_the_pattern():
    with pytest.raises(RefusedError) as excinfo:
        classify_blocks(
            _args(hybrid_layer_pattern="*-*-", num_layers=6), _keys(0, 1, 2, 3), TORCH_DIST
        )
    assert excinfo.value.refusal.code == errors.FC_R11
    assert excinfo.value.refusal.message.startswith("args.num_layers=6 but hybrid layer pattern")


def test_a_layout_needs_one_distinct_stored_index_per_module():
    with pytest.raises(InternalError):
        BlockLayout("decoder", "hybrid_stack", 2, "per_layer", (0, 0), None, "")
    block = BlockLayout("decoder", "hybrid_stack", 2, "per_layer", (0, 2), None, "")
    assert block.dense_index(2, TORCH_DIST) == 1 and block.dense_index(1, FSDP_DTENSOR) == 1
    with pytest.raises(KeyGrammarError):
        block.dense_index(1, TORCH_DIST)
