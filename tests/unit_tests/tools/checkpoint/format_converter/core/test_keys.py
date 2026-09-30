# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The model-key grammar: layer segments, leaves and roots."""

from __future__ import annotations

import pytest

from tools.checkpoint.format_converter.core.keys import (
    MIXER_TENSOR,
    ExpertName,
    KeyGrammarError,
    LayerKey,
    MixerTensor,
    attribute_layer_key,
    fc2_path_of,
    innermost_layer_key,
    is_fc1_path,
    is_glu_half,
    is_swiglu_key,
    join_factory_part,
    join_glu_half,
    join_layer_key,
    join_leaf,
    join_torch_dist_expert,
    root_of,
    split_factory_part,
    split_glu_half,
    split_layer_key,
    split_leaf,
    split_mixer_tensor,
    split_torch_dist_expert,
    torch_dist_chunk_axes,
)
from tools.checkpoint.format_converter.errors import InternalError


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


@pytest.mark.parametrize(
    ("name", "tensor", "part"),
    [
        ("decoder.layers.1.mlp.linear_fc1.weight_w", "decoder.layers.1.mlp.linear_fc1.weight", "w"),
        ("mlp.linear_fc1.bias_v", "mlp.linear_fc1.bias", "v"),
        ("mlp.experts.linear_fc1.weight17_v", "mlp.experts.linear_fc1.weight17", "v"),
        (
            "mlp.experts.local_experts.3.linear_fc1.weight_w",
            "mlp.experts.local_experts.3.linear_fc1.weight",
            "w",
        ),
        ("mlp.linear_fc1.weight", "mlp.linear_fc1.weight", None),
        ("mlp.linear_fc2.weight_w", "mlp.linear_fc2.weight_w", None),
        ("mlp.linear_fc1.layer_norm_weight_w", "mlp.linear_fc1.layer_norm_weight_w", None),
        ("mlp.linear_fc1.weight_x", "mlp.linear_fc1.weight_x", None),
    ],
)
def test_swiglu_halves_name_their_fused_tensor(name, tensor, part):
    assert split_glu_half(name) == (tensor, part)
    if part is not None:
        assert join_glu_half(tensor, part) == name


def test_only_the_gate_and_value_are_halves():
    with pytest.raises(InternalError):
        join_glu_half("mlp.linear_fc1.weight", "z")


@pytest.mark.parametrize(
    ("name", "split"),
    [
        ("decoder.layers.1.mlp.linear_fc1.weight", True),
        ("decoder.layers.1.mlp.linear_fc1.bias", True),
        ("decoder.layers.1.mlp.experts.linear_fc1.weight17", True),
        ("decoder.layers.1.mlp.experts.linear_fc1.bias3", True),
        ("decoder.layers.1.mlp.experts.local_experts.3.linear_fc1.weight", True),
        ("decoder.layers.1.mlp.experts.local_experts.3.linear_fc1.bias", True),
        ("decoder.layers.1.mlp.shared_experts.linear_fc1.weight", True),
        ("vision_model.decoder.layers.0.mlp.linear_fc1.bias", True),
        ("vision_projection.encoder.linear_fc1.weight", False),
        ("mlp.linear_fc1.weight", False),  # Megatron's patterns need a module before mlp
        ("decoder.layers.1.mlp.linear_fc1.layer_norm_weight", False),
        ("decoder.layers.1.mlp.linear_fc2.weight", False),
        ("decoder.layers.1.mlp.experts.linear_fc1.weight", False),
    ],
)
def test_megatron_fsdp_splits_the_swiglu_names(name, split):
    assert is_swiglu_key(name) is split


def test_the_second_projection_beside_a_first():
    assert is_fc1_path("linear_fc1") and is_fc1_path("mlp.experts.linear_fc1")
    assert not is_fc1_path("mlp.linear_fc10") and not is_fc1_path("mlp.xlinear_fc1")
    assert fc2_path_of("mlp.experts.linear_fc1") == "mlp.experts.linear_fc2"
    with pytest.raises(KeyGrammarError):
        fc2_path_of("mlp.linear_fc2")


@pytest.mark.parametrize(
    ("name", "tensor", "part"),
    [
        ("mixer.in_proj.weight.V0", "mixer.in_proj.weight", "V0"),
        ("mixer.in_proj.bias.a", "mixer.in_proj.bias", "a"),
        ("mixer.conv1d.weight.K12", "mixer.conv1d.weight", "K12"),
        ("attn_layer.mixer.in_proj.weight.z", "attn_layer.mixer.in_proj.weight", "z"),
        ("self_attention.in_proj.weight.query", "self_attention.in_proj.weight", "query"),
        ("self_attention.conv1d.bias.value", "self_attention.conv1d.bias", "value"),
        # whole tensors, and names that are no fused projection's part
        ("mixer.in_proj.weight", "mixer.in_proj.weight", None),
        ("mixer.in_proj.layer_norm_weight", "mixer.in_proj.layer_norm_weight", None),
        ("mixer.out_proj.weight.z", "mixer.out_proj.weight.z", None),
        ("mlp.linear_fc1.weight.w", "mlp.linear_fc1.weight.w", None),
        ("mixer.in_proj.weight.0", "mixer.in_proj.weight.0", None),
        ("in_proj.weight.z", "in_proj.weight.z", None),
    ],
)
def test_torch_dist_factory_parts_name_their_fused_tensor(name, tensor, part):
    assert split_factory_part(name) == (tensor, part)
    if part is not None:
        assert join_factory_part(tensor, part) == name


def test_only_fused_projections_are_joined_with_parts():
    with pytest.raises(InternalError):
        join_factory_part("mlp.linear_fc1.weight", "z")


def test_swiglu_halves_are_parts_of_first_projections_only():
    assert is_glu_half("mlp.linear_fc1", "w") and is_glu_half("mlp.experts.linear_fc1", "v")
    assert not is_glu_half("mlp.linear_fc2", "w")
    assert not is_glu_half("self_attention.in_proj", "w")  # a Gated DeltaNet 2 part


@pytest.mark.parametrize(
    ("key", "path", "leaf", "tensor"),
    [
        (
            "decoder.layers.0.mixer.in_proj.weight.V0",
            "mixer.in_proj",
            "weight",
            MixerTensor("mixer", "in_proj", "weight"),
        ),
        (
            "decoder.layers.1.attn_layer.mixer.conv1d.bias",
            "attn_layer.mixer.conv1d",
            "bias",
            MixerTensor("attn_layer.mixer", "conv1d", "bias"),
        ),
        (
            "decoder.layers.2.mixer.conv1d_weight",
            "mixer",
            "conv1d_weight",
            MixerTensor("mixer", "conv1d", "weight", conv_parameter=True),
        ),
        (
            "decoder.layers.2.mixer.conv1d_bias",
            "mixer",
            "conv1d_bias",
            MixerTensor("mixer", "conv1d", "bias", conv_parameter=True),
        ),
    ],
)
def test_mixer_tensors_are_named_by_mixer_projection_and_leaf(key, path, leaf, tensor):
    assert MIXER_TENSOR.search(key)
    assert split_mixer_tensor(path, leaf) == tensor


@pytest.mark.parametrize(
    ("key", "path", "leaf"),
    [
        ("decoder.layers.0.mixer.in_proj.layer_norm_weight", "mixer.in_proj", "layer_norm_weight"),
        ("decoder.layers.0.mixer.out_proj.weight", "mixer.out_proj", "weight"),
        ("decoder.layers.0.mixer.A_log", "mixer", "A_log"),
        ("decoder.layers.0.self_attention.in_proj.weight", "self_attention.in_proj", "weight"),
        ("decoder.layers.0.mixer.norm.weight", "mixer.norm", "weight"),
    ],
)
def test_other_mixer_tensors_are_no_fused_projections(key, path, leaf):
    assert MIXER_TENSOR.search(key) is None
    assert split_mixer_tensor(path, leaf) is None
