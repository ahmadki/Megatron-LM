# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Model keys decompose into logical slices and compose back, per format."""

from __future__ import annotations

import pytest
import torch

from tools.checkpoint.format_converter.core.blocks import BlockLayout
from tools.checkpoint.format_converter.core.experts import ExpertLayout
from tools.checkpoint.format_converter.core.keys import KeyGrammarError
from tools.checkpoint.format_converter.core.logical import LogicalId, LogicalSlice, OutputSlice
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, TORCH_DIST
from tools.checkpoint.format_converter.core.structure import compose, decompose
from tools.checkpoint.format_converter.engine.metadata import TensorEntry
from tools.checkpoint.format_converter.engine.regions import Region
from tools.checkpoint.format_converter.errors import InternalError

HYBRID = BlockLayout("decoder", "hybrid_stack", 3, "per_layer", (0, 1, 2), ("*", "-", "*"), "")
# A block whose formats number layers differently (as shortcut-MoE grouping will).
SPARSE = BlockLayout("decoder", "hybrid_stack", 3, "per_layer", (0, 2, 3), None, "")
STACKED = BlockLayout("decoder", "transformer_block", 3, "stacked", (0, 1, 2), None, "")


def entry(key: str, shape: tuple[int, ...] = (4,)) -> TensorEntry:
    return TensorEntry(key, shape, torch.bfloat16, ())


def decompose_one(key: str, blocks, fmt: str) -> LogicalId:
    """The logical id of a key that decomposes into one slice holding the whole tensor."""
    (piece,) = decompose(key, entry(key), blocks, fmt)
    assert piece.region == Region.full((4,)) and piece.shape == (4,)
    return piece.lid


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
    assert decompose_one(key, [HYBRID], fmt) == lid
    assert compose(lid, (4,), [HYBRID], fmt) == OutputSlice(key, (4,), Region.full((4,)))


def test_formats_number_the_layers_of_a_block_independently():
    lid = decompose_one("decoder.layers.3.mlp.linear_fc1.weight", [SPARSE], TORCH_DIST)
    assert lid == LogicalId("decoder", 2, "mlp.linear_fc1", "weight")
    assert (
        compose(lid, (4,), [SPARSE], FSDP_DTENSOR).key == "decoder.layers.2.mlp.linear_fc1.weight"
    )
    assert compose(lid, (4,), [SPARSE], TORCH_DIST).key == "decoder.layers.3.mlp.linear_fc1.weight"


def test_the_longest_block_prefix_wins():
    inner = BlockLayout("decoder.layers.0.inner", "hybrid_stack", 1, "per_layer", (0,), None, "")
    lid = decompose_one("decoder.layers.0.inner.layers.0.weight", [HYBRID, inner], TORCH_DIST)
    assert (lid.block, lid.layer, lid.path, lid.leaf) == ("decoder.layers.0.inner", 0, "", "weight")


def test_blocks_of_other_namespaces_keep_their_own_numbering():
    language = BlockLayout(
        "language_model.decoder", "hybrid_stack", 3, "per_layer", (0, 2, 3), None, ""
    )
    vision = BlockLayout("vision_model.decoder", "hybrid_stack", 2, "per_layer", (0, 1), None, "")
    mtp = BlockLayout("mtp", "hybrid_stack", 2, "per_layer", (0, 1), None, "")
    blocks = [language, vision, mtp, SPARSE]
    for key, lid in [
        (
            "language_model.decoder.layers.3.mlp.linear_fc1.weight",
            LogicalId("language_model.decoder", 2, "mlp.linear_fc1", "weight"),
        ),
        (
            "vision_model.decoder.layers.1.mlp.linear_fc1.weight",
            LogicalId("vision_model.decoder", 1, "mlp.linear_fc1", "weight"),
        ),
        (
            "mtp.layers.1.transformer_layer.mlp.linear_fc1.weight",
            LogicalId("mtp", 1, "transformer_layer.mlp.linear_fc1", "weight"),
        ),
        (
            "decoder.layers.2.mlp.linear_fc1.weight",
            LogicalId("decoder", 1, "mlp.linear_fc1", "weight"),
        ),
    ]:
        assert decompose_one(key, blocks, TORCH_DIST) == lid
        assert compose(lid, (4,), blocks, TORCH_DIST).key == key
    dense = LogicalId("language_model.decoder", 2, "mlp.linear_fc1", "weight")
    expected = "language_model.decoder.layers.2.mlp.linear_fc1.weight"
    assert compose(dense, (4,), blocks, FSDP_DTENSOR).key == expected


@pytest.mark.parametrize(
    "key",
    [
        "decoder.layers.7.mlp.linear_fc1.weight",  # not a stored index
        "decoder.layers.x.weight",  # stacked over a per-layer block
        "decoder.layers.1",
        "decoder.layers.01.x.weight",
    ],
)
def test_keys_the_layout_cannot_place_are_grammar_errors(key):
    with pytest.raises(KeyGrammarError):
        decompose(key, entry(key), [HYBRID], TORCH_DIST)


def test_a_stacked_tensor_decomposes_into_one_slice_per_layer():
    key = "decoder.layers.self_attention.linear_qkv.weight"
    slices = decompose(key, TensorEntry(key, (3, 6, 4), torch.float32, ()), [STACKED], TORCH_DIST)
    assert slices == [
        LogicalSlice(
            LogicalId("decoder", layer, "self_attention.linear_qkv", "weight"),
            key,
            Region((layer, 0, 0), (1, 6, 4)),
            torch.float32,
            squeezed=1,
        )
        for layer in range(3)
    ]
    assert slices[2].shape == (6, 4)
    assert slices[2].stored_box(Region((2, 1), (3, 2))) == Region((2, 2, 1), (1, 3, 2))
    norms = decompose("decoder.layers.norm.bias", entry("n", (3,)), [STACKED], TORCH_DIST)
    assert [(s.region, s.shape) for s in norms] == [(Region((i,), (1,)), ()) for i in range(3)]


@pytest.mark.parametrize(
    ("key", "shape", "fmt"),
    [
        ("decoder.layers.mlp.linear_fc1.weight", (2, 8, 4), TORCH_DIST),  # 2 of 3 layers
        ("decoder.layers.mlp.linear_fc1.weight", (), TORCH_DIST),
        ("decoder.layers.1.mlp.linear_fc1.weight", (8, 4), TORCH_DIST),  # per layer
        ("decoder.layers.mlp.linear_fc1.weight", (3, 8, 4), FSDP_DTENSOR),  # never stacked
    ],
)
def test_a_stacked_block_takes_only_complete_stacked_tensors(key, shape, fmt):
    with pytest.raises(KeyGrammarError):
        decompose(key, TensorEntry(key, shape, torch.float32, ()), [STACKED], fmt)


def test_a_stacked_block_is_one_layer_per_key_in_fsdp_dtensor():
    key = "decoder.layers.2.mlp.linear_fc1.weight"
    lid = decompose_one(key, [STACKED], FSDP_DTENSOR)
    assert lid == LogicalId("decoder", 2, "mlp.linear_fc1", "weight")
    assert compose(lid, (8, 4), [STACKED], FSDP_DTENSOR) == OutputSlice(
        key, (8, 4), Region.full((8, 4))
    )


def test_a_layer_of_a_stacked_block_is_placed_in_its_stacked_tensor():
    lid = LogicalId("decoder", 2, "mlp.linear_fc1", "weight")
    placed = compose(lid, (8, 4), [STACKED], TORCH_DIST)
    assert placed == OutputSlice(
        "decoder.layers.mlp.linear_fc1.weight", (3, 8, 4), Region((2, 0, 0), (1, 8, 4)), 1
    )
    assert placed.shape == (8, 4)
    assert placed.logical_box(Region((2, 4, 0), (1, 4, 4))) == Region((4, 0), (4, 4))
    scalar = compose(LogicalId("decoder", 0, "", "scale"), (), [STACKED], TORCH_DIST)
    assert (scalar.key, scalar.global_shape, scalar.region) == (
        "decoder.layers.scale",
        (3,),
        Region((0,), (1,)),
    )
    (piece,) = [
        s
        for s in decompose(
            placed.key, TensorEntry(placed.key, (3, 8, 4), torch.float32, ()), [STACKED], TORCH_DIST
        )
        if s.lid == lid
    ]
    assert piece.region == placed.region  # decompose is the inverse


def test_composing_an_unknown_block_is_a_bug():
    with pytest.raises(InternalError):
        compose(LogicalId("encoder", 0, "x", "weight"), (4,), [HYBRID], TORCH_DIST)


FOUR_EXPERTS = ExpertLayout(4)
STACKED_FC1 = "decoder.layers.mlp.experts.experts.linear_fc1.weight"


def test_a_stacked_expert_tensor_decomposes_into_one_slice_per_layer_and_expert():
    """torch_dist's ``[L, E, ...]``: layer ``l``, expert ``e`` is ``[l:l+1, e:e+1, ...]``."""
    entry_ = TensorEntry(STACKED_FC1, (3, 4, 6, 5), torch.bfloat16, ())
    slices = decompose(STACKED_FC1, entry_, [STACKED], TORCH_DIST)
    assert slices == [
        LogicalSlice(
            LogicalId("decoder", layer, "mlp.experts.linear_fc1", "weight", expert),
            STACKED_FC1,
            Region((layer, expert, 0, 0), (1, 1, 6, 5)),
            torch.bfloat16,
            squeezed=2,
        )
        for layer in range(3)
        for expert in range(4)
    ]
    assert slices[6].shape == (6, 5)
    assert slices[6].stored_box(Region((2, 1), (4, 3))) == Region((1, 2, 2, 1), (1, 1, 4, 3))


def test_a_per_layer_expert_tensor_decomposes_into_one_slice_per_expert():
    key = "decoder.layers.2.mlp.experts.experts.linear_fc2.bias"
    slices = decompose(key, TensorEntry(key, (4, 6), torch.float32, ()), [HYBRID], TORCH_DIST)
    assert [(s.lid, s.region, s.squeezed) for s in slices] == [
        (
            LogicalId("decoder", 2, "mlp.experts.linear_fc2", "bias", expert),
            Region((expert, 0), (1, 6)),
            1,
        )
        for expert in range(4)
    ]


@pytest.mark.parametrize(
    ("key", "shape", "blocks"),
    [
        ("decoder.layers.1.mlp.experts.experts.linear_fc1.weight", (), [HYBRID]),
        ("decoder.layers.1.mlp.experts.experts.linear_fc1.weight", (0, 4), [HYBRID]),
        (STACKED_FC1, (3,), [STACKED]),
        ("decoder.layers.1.mlp.experts.experts.3.linear_fc1.weight", (6, 5), [HYBRID]),
        ("decoder.layers.1.mlp.experts.linear_fc1.weight", (6, 5), [HYBRID]),
    ],
)
def test_malformed_torch_dist_expert_tensors_are_grammar_errors(key, shape, blocks):
    with pytest.raises(KeyGrammarError):
        decompose(key, TensorEntry(key, shape, torch.float32, ()), blocks, TORCH_DIST)


def test_an_expert_is_placed_at_its_index_of_the_expert_axis():
    lid = LogicalId("decoder", 1, "mlp.experts.linear_fc1", "weight", 2)
    placed = compose(lid, (6, 5), [STACKED], TORCH_DIST, FOUR_EXPERTS)
    assert placed == OutputSlice(STACKED_FC1, (3, 4, 6, 5), Region((1, 2, 0, 0), (1, 1, 6, 5)), 2)
    assert placed.logical_box(Region((1, 2, 3, 0), (1, 1, 3, 5))) == Region((3, 0), (3, 5))
    per_layer = compose(lid, (6, 5), [HYBRID], TORCH_DIST, FOUR_EXPERTS)
    assert per_layer == OutputSlice(
        "decoder.layers.1.mlp.experts.experts.linear_fc1.weight",
        (4, 6, 5),
        Region((2, 0, 0), (1, 6, 5)),
        1,
    )
    entry_ = TensorEntry(STACKED_FC1, (3, 4, 6, 5), torch.float32, ())
    (piece,) = [s for s in decompose(STACKED_FC1, entry_, [STACKED], TORCH_DIST) if s.lid == lid]
    assert piece.region == placed.region  # decompose is the inverse


@pytest.mark.parametrize("experts", [None, ExpertLayout(2)])
def test_placing_an_expert_outside_the_expert_layout_is_a_bug(experts):
    lid = LogicalId("decoder", 1, "mlp.experts.linear_fc1", "weight", 2)
    with pytest.raises(InternalError):
        compose(lid, (6, 5), [STACKED], TORCH_DIST, experts)


def test_expert_layouts_have_experts():
    assert FOUR_EXPERTS.to_json() == {"num_experts": 4}
    with pytest.raises(InternalError):
        ExpertLayout(0)
