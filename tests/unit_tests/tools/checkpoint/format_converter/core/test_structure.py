# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Model keys decompose into logical ids and compose back, per format."""

from __future__ import annotations

import pytest

from tools.checkpoint.format_converter.core.blocks import BlockLayout
from tools.checkpoint.format_converter.core.keys import KeyGrammarError
from tools.checkpoint.format_converter.core.logical import LogicalId
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, TORCH_DIST
from tools.checkpoint.format_converter.core.structure import compose, decompose
from tools.checkpoint.format_converter.errors import InternalError

HYBRID = BlockLayout("decoder", "hybrid_stack", 3, "per_layer", (0, 1, 2), ("*", "-", "*"), "")
# A block whose formats number layers differently (as shortcut-MoE grouping will).
SPARSE = BlockLayout("decoder", "hybrid_stack", 3, "per_layer", (0, 2, 3), None, "")


@pytest.mark.parametrize(
    ("key", "lid"),
    [
        (
            "decoder.layers.2.self_attention.linear_qkv.layer_norm_weight",
            LogicalId("decoder", 2, "self_attention.linear_qkv", "layer_norm_weight"),
        ),
        ("decoder.final_norm.weight", LogicalId(None, None, "decoder.final_norm", "weight")),
        (
            "embedding.word_embeddings.weight",
            LogicalId(None, None, "embedding.word_embeddings", "weight"),
        ),
    ],
)
@pytest.mark.parametrize("fmt", [TORCH_DIST, FSDP_DTENSOR])
def test_round_trip(key, lid, fmt):
    assert decompose(key, [HYBRID], fmt) == lid
    assert compose(lid, [HYBRID], fmt) == key


def test_formats_number_the_layers_of_a_block_independently():
    lid = decompose("decoder.layers.3.mlp.linear_fc1.weight", [SPARSE], TORCH_DIST)
    assert lid == LogicalId("decoder", 2, "mlp.linear_fc1", "weight")
    assert compose(lid, [SPARSE], FSDP_DTENSOR) == "decoder.layers.2.mlp.linear_fc1.weight"
    assert compose(lid, [SPARSE], TORCH_DIST) == "decoder.layers.3.mlp.linear_fc1.weight"


def test_the_longest_block_prefix_wins():
    inner = BlockLayout("decoder.layers.0.inner", "hybrid_stack", 1, "per_layer", (0,), None, "")
    lid = decompose("decoder.layers.0.inner.layers.0.weight", [HYBRID, inner], TORCH_DIST)
    assert (lid.block, lid.layer, lid.path, lid.leaf) == ("decoder.layers.0.inner", 0, "", "weight")


@pytest.mark.parametrize(
    "key", ["decoder.layers.7.mlp.linear_fc1.weight", "decoder.layers.x.weight", "decoder.layers.1"]
)
def test_keys_the_layout_cannot_place_are_grammar_errors(key):
    with pytest.raises(KeyGrammarError):
        decompose(key, [HYBRID], TORCH_DIST)


def test_composing_an_unknown_block_is_a_bug():
    with pytest.raises(InternalError):
        compose(LogicalId("encoder", 0, "x", "weight"), [HYBRID], TORCH_DIST)
