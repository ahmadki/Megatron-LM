# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The model-key grammar: layer segments, leaves and roots."""

from __future__ import annotations

import pytest

from tools.checkpoint.format_converter.core.keys import (
    ExpertName,
    KeyGrammarError,
    LayerKey,
    attribute_layer_key,
    innermost_layer_key,
    join_layer_key,
    join_leaf,
    join_torch_dist_expert,
    root_of,
    split_layer_key,
    split_leaf,
    split_torch_dist_expert,
    torch_dist_chunk_axes,
)


@pytest.mark.parametrize(
    ("key", "block", "expected"),
    [
        (
            "decoder.layers.0.self_attention.linear_qkv.weight",
            "decoder",
            LayerKey("decoder", 0, "self_attention.linear_qkv.weight"),
        ),
        (
            "decoder.layers.17.final_norm.weight",
            "decoder",
            LayerKey("decoder", 17, "final_norm.weight"),
        ),
        (
            "language_model.decoder.layers.3.mlp.linear_fc1.bias",
            "language_model.decoder",
            LayerKey("language_model.decoder", 3, "mlp.linear_fc1.bias"),
        ),
        (
            "vision_model.decoder.layers.self_attention.linear_qkv.weight",
            "vision_model.decoder",
            LayerKey("vision_model.decoder", None, "self_attention.linear_qkv.weight"),
        ),
        ("decoder.layers._private.weight", "decoder", LayerKey("decoder", None, "_private.weight")),
        ("decoder.final_norm.weight", "decoder", None),
        ("embedding.word_embeddings.weight", "decoder", None),
        ("vision_model.decoder.layers.1.x", "decoder", None),
    ],
)
def test_split_layer_key(key, block, expected):
    assert split_layer_key(key, block) == expected
    if expected is not None:
        assert expected.render() == key
        assert expected.stacked == (expected.index is None)


@pytest.mark.parametrize(
    "key",
    [
        "decoder.layers.01.mlp.linear_fc1.weight",  # non-canonical index
        "decoder.layers.-1.mlp.linear_fc1.weight",
        "decoder.layers.3",  # index without a name
        "decoder.layers.3.",
        "decoder.layers.3x.weight",
        "decoder.layers.",
    ],
)
def test_malformed_layer_segments_are_refused(key):
    with pytest.raises(KeyGrammarError):
        split_layer_key(key, "decoder")


def test_join_layer_key_rejects_negative_indices():
    assert join_layer_key("decoder", 12, "mlp.linear_fc2.weight") == (
        "decoder.layers.12.mlp.linear_fc2.weight"
    )
    assert join_layer_key("decoder", None, "mlp.linear_fc2.weight") == (
        "decoder.layers.mlp.linear_fc2.weight"
    )
    with pytest.raises(KeyGrammarError):
        join_layer_key("decoder", -1, "x.weight")


VISION_STACKED = "vision_model.decoder.layers.mlp.linear_fc1.weight"
LANGUAGE_LAYER = "language_model.decoder.layers.3.self_attention.linear_proj.weight"
MTP_LAYER = "mtp.layers.1.transformer_layer.mlp.linear_fc2.weight"
MTP_INNER = "mtp.layers.0.mtp_model_layer.layers.2.mixer.in_proj.weight"
NAMESPACED_BLOCKS = (
    "decoder",
    "language_model.decoder",
    "vision_model.decoder",
    "mtp",
    "mtp.layers.0.mtp_model_layer",
)
"""Blocks of several namespaces, one of them nested in a layer of another."""


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (VISION_STACKED, LayerKey("vision_model.decoder", None, "mlp.linear_fc1.weight")),
        (
            LANGUAGE_LAYER,
            LayerKey("language_model.decoder", 3, "self_attention.linear_proj.weight"),
        ),
        (MTP_LAYER, LayerKey("mtp", 1, "transformer_layer.mlp.linear_fc2.weight")),
        (MTP_INNER, LayerKey("mtp.layers.0.mtp_model_layer", 2, "mixer.in_proj.weight")),
        ("vision_model.decoder.final_layernorm.weight", None),
        ("language_model.embedding.word_embeddings.weight", None),
    ],
)
def test_keys_belong_to_the_innermost_block_of_their_namespace(key, expected):
    assert attribute_layer_key(key, NAMESPACED_BLOCKS) == expected
    assert innermost_layer_key(key) == expected
    if expected is not None:
        assert expected.render() == key


def test_attribution_only_considers_known_blocks():
    assert attribute_layer_key(MTP_INNER, ("mtp",)) == LayerKey(
        "mtp", 0, "mtp_model_layer.layers.2.mixer.in_proj.weight"
    )
    assert attribute_layer_key(VISION_STACKED, ("decoder", "language_model.decoder")) is None
    assert innermost_layer_key("layers.0.weight") is None  # no block prefix
    with pytest.raises(KeyGrammarError):
        innermost_layer_key("decoder.layers.1.inner.layers.07.weight")


@pytest.mark.parametrize(
    ("name", "path", "leaf"),
    [
        (
            "self_attention.linear_qkv.layer_norm_weight",
            "self_attention.linear_qkv",
            "layer_norm_weight",
        ),
        ("weight", "", "weight"),
        ("router.expert_bias", "router", "expert_bias"),
    ],
)
def test_split_and_join_leaf(name, path, leaf):
    assert split_leaf(name) == (path, leaf)
    assert join_leaf(path, leaf) == name


@pytest.mark.parametrize("name", ["", "mlp."])
def test_names_without_a_leaf_are_refused(name):
    with pytest.raises(KeyGrammarError):
        split_leaf(name)


def test_root_of():
    assert root_of("decoder.layers.0.x.weight") == "decoder"
    assert root_of("output_layer.weight") == "output_layer"
    assert root_of("bare") == "bare"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mlp.experts.experts.linear_fc1.weight", ExpertName("mlp.experts.linear_fc1", "weight")),
        (
            "decoder.layers.3.mlp.experts.experts.linear_fc2.bias",
            ExpertName("decoder.layers.3.mlp.experts.linear_fc2", "bias"),
        ),
        (
            "transformer_layer.mlp.experts.experts.linear_fc1.weight",
            ExpertName("transformer_layer.mlp.experts.linear_fc1", "weight"),
        ),
        ("mlp.shared_experts.linear_fc1.weight", None),
        ("mlp.router.expert_bias", None),
        ("mlp.linear_fc1.weight", None),
    ],
)
def test_torch_dist_names_every_expert_in_one_tensor(name, expected):
    assert split_torch_dist_expert(name) == expected
    if expected is not None:
        assert join_torch_dist_expert(expected.path, expected.leaf) == name


@pytest.mark.parametrize(
    "name",
    [
        "mlp.experts.experts.3.linear_fc1.weight",  # singleton local shards
        "mlp.experts.linear_fc1.weight3",  # an fsdp_dtensor name
        "mlp.experts.local_experts.3.linear_fc1.weight",
        "mlp.experts.experts.linear_fc1.layer_norm_weight",
        "mlp.experts.experts.linear_fc1.weight_w",
        "mlp.experts.weight1",
    ],
)
def test_other_names_under_an_experts_module_are_not_torch_dist(name):
    with pytest.raises(KeyGrammarError, match="torch_dist stores routed-expert tensors"):
        split_torch_dist_expert(name)


def test_joining_needs_an_experts_linear_layer():
    with pytest.raises(KeyGrammarError):
        join_torch_dist_expert("mlp.linear_fc1", "weight")


@pytest.mark.parametrize(
    ("key", "axes"),
    [
        ("decoder.layers.mlp.experts.experts.linear_fc1.weight", 2),
        ("decoder.layers.2.mlp.experts.experts.linear_fc1.weight", 1),
        ("decoder.layers.mlp.linear_fc1.weight", 1),
        ("decoder.layers.2.mlp.linear_fc1.weight", 0),
        ("decoder.layers.mlp.router.expert_bias", 1),
        ("output_layer.weight", 0),
    ],
)
def test_torch_dist_saves_a_chunk_per_layer_and_expert(key, axes):
    assert torch_dist_chunk_axes(key) == axes
