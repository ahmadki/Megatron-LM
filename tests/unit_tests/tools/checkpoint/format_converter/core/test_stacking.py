# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TransformerBlock layouts: observed in torch_dist stores and decided for fsdp_dtensor ones.

Every case is a hand-built key set; the expected layouts follow Megatron's
``TransformerBlock.sharded_state_dict`` (the parity pin in
``parity/test_parity_megatron.py`` checks the predicate against it).
"""

from __future__ import annotations

import argparse

import pytest
import torch

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.blocks import BlockFacts, BlockLayout, classify_blocks
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, TORCH_DIST
from tools.checkpoint.format_converter.core.stacking import (
    per_layer_reasons,
    requested_layout_warnings,
    transformer_block,
)
from tools.checkpoint.format_converter.engine.metadata import TensorEntry
from tools.checkpoint.format_converter.errors import AmbiguousError, RefusedError, UsageError

from .test_refusals import codes, facts

H = 4


def _model(shapes: dict[str, tuple[int, ...]]) -> dict[str, TensorEntry]:
    return {key: TensorEntry(key, shape, torch.bfloat16, ()) for key, shape in shapes.items()}


def stacked_gpt(layers: int = 3, prefix: str = "decoder") -> dict[str, tuple[int, ...]]:
    """Shapes of a GPT decoder stored stacked (``[L, ...]`` per tensor)."""
    return {
        f"{prefix}.layers.self_attention.linear_qkv.weight": (layers, 3 * H, H),
        f"{prefix}.layers.self_attention.linear_qkv.layer_norm_weight": (layers, H),
        f"{prefix}.layers.mlp.linear_fc2.weight": (layers, H, 2 * H),
        f"{prefix}.final_layernorm.weight": (H,),
    }


def per_layer_gpt(indices, prefix: str = "decoder") -> dict[str, tuple[int, ...]]:
    """Shapes of a GPT decoder stored one layer per key."""
    shapes = {f"{prefix}.final_layernorm.weight": (H,)}
    for index in indices:
        shapes[f"{prefix}.layers.{index}.self_attention.linear_qkv.weight"] = (3 * H, H)
        shapes[f"{prefix}.layers.{index}.mlp.linear_fc2.weight"] = (H, 2 * H)
    return shapes


def classify(
    shapes, *, source=TORCH_DIST, content_metadata=None, layouts=None, per_layer=False, **args
) -> tuple[BlockLayout, ...]:
    facts = BlockFacts(
        ArgsView(argparse.Namespace(**args)),
        source,
        _model(shapes),
        content_metadata,
        dict(layouts or {}),
        per_layer,
    )
    return classify_blocks(facts, default=transformer_block)


def refusal(shapes, **kwargs):
    with pytest.raises(RefusedError) as excinfo:
        classify(shapes, **kwargs)
    assert excinfo.value.refusal.code == errors.FC_R12
    return excinfo.value.refusal


def decided(shapes=None, **kwargs) -> BlockLayout:
    """The one block of an fsdp_dtensor store of the GPT decoder (3 layers by default)."""
    (block,) = classify(
        per_layer_gpt(range(3)) if shapes is None else shapes, source=FSDP_DTENSOR, **kwargs
    )
    return block


def test_a_stacked_decoder_is_observed_with_its_layer_count():
    (block,) = classify(stacked_gpt(), num_layers=3)
    assert block.to_json() == {
        "prefix": "decoder",
        "kind": "transformer_block",
        "num_modules": 3,
        "td_layout": "stacked",
        "dense_to_td": [0, 1, 2],
        "symbols": None,
        "evidence": "torch_dist stores 3 tensors stacked over 3 layers",
    }


def test_a_per_layer_decoder_is_observed_with_global_indices():
    (block,) = classify(per_layer_gpt(range(4)), num_layers=4)
    assert (block.td_layout, block.num_modules, block.dense_to_td) == ("per_layer", 4, (0, 1, 2, 3))
    assert block.evidence == "torch_dist stores 4 layers one per key"


def test_blocks_other_than_the_decoder_have_their_own_length():
    shapes = {**stacked_gpt(3), **stacked_gpt(2, prefix="encoder")}
    decoder, encoder = classify(shapes, num_layers=3)
    assert (decoder.prefix, decoder.num_modules) == ("decoder", 3)
    assert (encoder.prefix, encoder.num_modules, encoder.stacked) == ("encoder", 2, True)


def test_stacked_tensors_must_agree_on_the_number_of_layers():
    shapes = {**stacked_gpt(3), "decoder.layers.mlp.linear_fc1.weight": (2, 2 * H, H)}
    found = refusal(shapes)
    assert "disagree on the number of layers" in found.message
    assert "mlp.linear_fc1.weight 2" in found.message and "mlp.linear_fc2.weight 3" in found.message
    assert found.keys[0] == "decoder.layers.mlp.linear_fc1.weight"


def test_a_stacked_scalar_is_refused():
    assert "disagree" in refusal({"decoder.layers.scale": ()}).message


def test_a_block_holds_one_form_only():
    shapes = {**stacked_gpt(2), **per_layer_gpt([0, 1])}
    found = refusal(shapes)
    assert found.message.startswith(
        "decoder holds both tensors stacked over its layers and tensors of single layers (0-1)"
    )
    assert "decoder.layers.0.mlp.linear_fc2.weight" in found.keys


@pytest.mark.parametrize(
    ("indices", "message"),
    [((0, 1, 3), "stores layers 0-1, 3, not 0-2"), ((1, 2), "stores layers 1-2, not 0-1")],
)
def test_per_layer_indices_must_be_contiguous_from_zero(indices, message):
    assert message in refusal(per_layer_gpt(indices)).message


@pytest.mark.parametrize("shapes", [stacked_gpt(3), per_layer_gpt(range(3))])
def test_the_decoder_has_num_layers_layers(shapes):
    found = refusal(shapes, num_layers=4)
    assert found.message == "args.num_layers=4 but the store holds 3 layers of decoder"


def test_the_decoder_of_an_fsdp_store_is_stacked_when_megatron_stacks_it():
    block = decided(num_layers=3, moe_layer_freq=1, linear_attention_freq=None)
    assert (block.td_layout, block.num_modules, block.dense_to_td) == ("stacked", 3, (0, 1, 2))
    assert block.evidence == "Megatron stacks it (no per-layer setting) and its layers are alike"


@pytest.mark.parametrize(
    ("content_metadata", "args", "reason"),
    [
        ({"non_homogeneous_layers": True}, {}, "content_metadata.non_homogeneous_layers=True"),
        ({"singleton_local_shards": True}, {}, "content_metadata.singleton_local_shards=True"),
        (None, {"moe_layer_freq": 2}, "args.moe_layer_freq=2"),
        (None, {"moe_layer_freq": [1, 0, 1]}, "args.moe_layer_freq=[1, 0, 1]"),
        (None, {"linear_attention_freq": 4}, "args.linear_attention_freq=4"),
        (None, {"linear_attention_freq": [0, 1, 1]}, "args.linear_attention_freq=[0, 1, 1]"),
    ],
)
def test_the_decoder_is_per_layer_when_megatron_builds_it_per_layer(content_metadata, args, reason):
    block = decided(content_metadata=content_metadata, **args)
    assert block.td_layout == "per_layer"
    assert block.evidence == f"Megatron builds it per layer ({reason})"


@pytest.mark.parametrize(
    ("content_metadata", "args"),
    [
        ({"non_homogeneous_layers": False, "singleton_local_shards": False}, {}),
        (None, {"moe_layer_freq": 1, "linear_attention_freq": 1}),
    ],
)
def test_settings_that_keep_the_decoder_stacked(content_metadata, args):
    assert decided(content_metadata=content_metadata, **args).td_layout == "stacked"
    assert per_layer_reasons(ArgsView(argparse.Namespace(**args)), content_metadata) == []


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ({"heterogeneous_layers_config_path": "layers.json"}, errors.FC_R03),
        ({"heterogeneous_layers_config_encoded_json": "{}"}, errors.FC_R03),
        ({"experimental_attention_variant": "dsv4_hybrid"}, errors.FC_R20),
    ],
)
def test_the_arguments_behind_megatrons_configuration_only_flags_are_refused(args, code):
    """Megatron builds a block per layer for ``heterogeneous_block_specs`` (set by a
    heterogeneous layers configuration) and ``hetereogenous_dist_checkpoint`` (set for
    ``dsv4_hybrid`` attention), which :func:`per_layer_reasons` does not read: the refusal
    fence must keep these models out (lifting either code needs them in the predicate)."""
    assert per_layer_reasons(ArgsView(argparse.Namespace(**args)), None) == []
    assert code in codes(facts(args=args))


def test_layers_that_differ_are_not_stacked_silently():
    shapes = per_layer_gpt(range(3))
    shapes["decoder.layers.2.self_attention.linear_qkv.weight"] = (2 * H, H)
    found = refusal(shapes, source=FSDP_DTENSOR)
    assert found.message == (
        "the arguments make Megatron stack decoder, but its layers differ: "
        "self_attention.linear_qkv.weight is [12, 4] torch.bfloat16 in layer 0 and "
        "[8, 4] torch.bfloat16 in layer 2"
    )
    del shapes["decoder.layers.2.self_attention.linear_qkv.weight"]
    assert "layer 0 has self_attention.linear_qkv.weight, layer 2 does not" in (
        refusal(shapes, source=FSDP_DTENSOR).message
    )
    per_layer = {"non_homogeneous_layers": True}
    assert decided(shapes, content_metadata=per_layer).td_layout == "per_layer"


def test_block_layout_overrides_the_decision():
    block = decided(layouts={"decoder": "per_layer"})
    assert (block.td_layout, block.evidence) == ("per_layer", "--block-layout decoder=per_layer")
    stacked = decided(
        content_metadata={"non_homogeneous_layers": True}, layouts={"decoder": "stacked"}
    )
    assert stacked.td_layout == "stacked"
    shapes = per_layer_gpt(range(3))
    del shapes["decoder.layers.1.mlp.linear_fc2.weight"]
    found = refusal(shapes, source=FSDP_DTENSOR, layouts={"decoder": "stacked"})
    assert found.message.startswith(
        "--block-layout decoder=stacked cannot stack layers that differ"
    )
    assert found.hint == "pass --block-layout decoder=per_layer"


def warnings_of(shapes=None, *, content_metadata=None, layouts=None, **args) -> list[str]:
    facts_ = BlockFacts(
        ArgsView(argparse.Namespace(**args)),
        FSDP_DTENSOR,
        _model(per_layer_gpt(range(3)) if shapes is None else shapes),
        content_metadata,
        dict(layouts or {}),
    )
    return requested_layout_warnings(facts_, classify_blocks(facts_, default=transformer_block))


def test_a_block_layout_megatron_does_not_build_is_warned_about():
    assert warnings_of(layouts={"decoder": "per_layer"}) == [
        "--block-layout decoder=per_layer: a job built from these arguments and this "
        "content_metadata builds decoder stacked (no per-layer setting), so it cannot load "
        "this store strictly; --torch-dist-layout per-layer marks the per-layer blocks in "
        "content_metadata, so that Megatron loads them per layer"
    ]
    (warning,) = warnings_of(layouts={"decoder": "stacked"}, moe_layer_freq=2)
    assert warning.startswith("--block-layout decoder=stacked: ")
    assert "builds decoder per_layer (args.moe_layer_freq=2)" in warning
    assert warnings_of() == []
    assert warnings_of(layouts={"decoder": "stacked"}) == []
    per_layer = {"non_homogeneous_layers": True}
    assert warnings_of(layouts={"decoder": "per_layer"}, content_metadata=per_layer) == []
    shapes = {**per_layer_gpt(range(3)), **per_layer_gpt(range(2), prefix="encoder")}
    assert warnings_of(shapes, layouts={"encoder": "per_layer"}) == []  # its kind is unknown


def test_a_layer_list_of_unknown_kind_needs_a_layout():
    shapes = {**per_layer_gpt(range(3)), **per_layer_gpt(range(2), prefix="encoder")}
    with pytest.raises(AmbiguousError) as excinfo:
        classify(shapes, source=FSDP_DTENSOR)
    assert excinfo.value.refusal.code == errors.FC_A02
    assert "--block-layout encoder=stacked" in excinfo.value.refusal.hint
    blocks = classify(shapes, source=FSDP_DTENSOR, layouts={"encoder": "per_layer"})
    assert [(b.prefix, b.td_layout) for b in blocks] == [
        ("decoder", "stacked"),
        ("encoder", "per_layer"),
    ]


@pytest.mark.parametrize(
    ("layouts", "message"),
    [
        ({"decoder.layers.0": "stacked"}, "no layer list of this model"),
        ({"decoder": "stacked"}, "a hybrid_stack block"),
    ],
)
def test_block_layout_names_transformer_blocks_only(layouts, message):
    hybrid = {"hybrid_layer_pattern": "*-*"} if "hybrid" in message else {}
    with pytest.raises(UsageError, match=message):
        classify(per_layer_gpt(range(3)), source=FSDP_DTENSOR, layouts=layouts, **hybrid)


def test_an_fsdp_store_holds_one_layer_per_key():
    found = refusal(stacked_gpt(), source=FSDP_DTENSOR)
    assert found.message.startswith("decoder: fsdp_dtensor stores one key per layer")
    assert "not 0-1" in refusal(per_layer_gpt((0, 2)), source=FSDP_DTENSOR).message
    assert (
        "args.num_layers=2"
        in refusal(per_layer_gpt(range(3)), source=FSDP_DTENSOR, num_layers=2).message
    )


def test_torch_dist_layout_per_layer_makes_every_block_per_layer():
    shapes = {**per_layer_gpt(range(3)), **per_layer_gpt(range(2), prefix="encoder")}
    decoder, encoder = classify(shapes, source=FSDP_DTENSOR, per_layer=True)
    assert [(b.td_layout, b.evidence) for b in (decoder, encoder)] == [
        ("per_layer", "--torch-dist-layout per-layer")
    ] * 2
    shapes["decoder.layers.2.self_attention.linear_qkv.weight"] = (2 * H, H)  # layers differ
    decoder, _ = classify(shapes, source=FSDP_DTENSOR, per_layer=True)
    assert decoder.td_layout == "per_layer"
    blocks = classify(shapes, source=FSDP_DTENSOR, per_layer=True, layouts={"encoder": "stacked"})
    assert [b.td_layout for b in blocks] == ["per_layer", "stacked"]


def llava(vision=None, language=None) -> dict[str, tuple[int, ...]]:
    """A vision-language model: a GPT language model and a vision encoder (both per layer)."""
    return {
        **(language or per_layer_gpt(range(3), prefix="language_model.decoder")),
        **(vision or per_layer_gpt(range(2), prefix="vision_model.decoder")),
        "vision_projection.encoder.linear_fc1.weight": (H, H),
    }


def test_the_blocks_of_a_vision_language_model_are_observed():
    shapes = {
        **stacked_gpt(3, prefix="language_model.decoder"),
        **stacked_gpt(2, prefix="vision_model.decoder"),
    }
    language, vision = classify(shapes, num_layers=3)
    assert (language.prefix, language.num_modules, language.stacked) == (
        "language_model.decoder",
        3,
        True,
    )
    assert (vision.prefix, vision.num_modules, vision.stacked) == ("vision_model.decoder", 2, True)
    assert "args.num_layers=2" in refusal(shapes, num_layers=2).message  # the language model's


@pytest.mark.parametrize("source", [TORCH_DIST, FSDP_DTENSOR])
def test_the_language_model_of_a_vision_language_model_may_have_decoder_num_layers(source):
    """``pretrain_vlm.py`` gives its language model ``args.decoder_num_layers`` layers; with
    ``--encoder-num-layers 2 --decoder-num-layers 3``, ``validate_args`` sets num_layers=2."""
    shapes = (
        llava()
        if source == FSDP_DTENSOR
        else {
            **stacked_gpt(3, prefix="language_model.decoder"),
            **stacked_gpt(2, prefix="vision_model.decoder"),
        }
    )
    args = {"num_layers": 2, "encoder_num_layers": 2, "decoder_num_layers": 3}
    language, vision = classify(shapes, source=source, **args)
    assert (language.num_modules, vision.num_modules) == (3, 2)
    found = refusal(shapes, source=source, **{**args, "decoder_num_layers": 4})
    assert found.message == (
        "args.decoder_num_layers=4 but the store holds 3 layers of language_model.decoder"
    )
    single = stacked_gpt(3) if source == TORCH_DIST else per_layer_gpt(range(3))
    assert "args.num_layers=2" in refusal(single, source=source, **args).message  # not a VLM


VISION_COPY = "its configuration is a copy of the language model's: "


def test_the_vision_encoder_follows_the_language_models_configuration():
    """Megatron's builders deep-copy the language model's configuration for the encoder."""
    language, vision = classify(
        llava(), source=FSDP_DTENSOR, num_layers=3, moe_layer_freq=[1, 0, 1]
    )
    assert (language.td_layout, vision.td_layout) == ("per_layer", "per_layer")
    assert vision.evidence == (
        f"{VISION_COPY}Megatron builds it per layer (args.moe_layer_freq=[1, 0, 1])"
    )
    _, vision = classify(llava(), source=FSDP_DTENSOR, num_layers=3, moe_layer_freq=1)
    assert vision.td_layout == "stacked"
    assert vision.evidence == (
        f"{VISION_COPY}Megatron stacks it (no per-layer setting) and its layers are alike"
    )
    flagged = {"non_homogeneous_layers": True}
    blocks = classify(llava(), source=FSDP_DTENSOR, content_metadata=flagged)
    reason = "Megatron builds it per layer (content_metadata.non_homogeneous_layers=True)"
    assert [(b.td_layout, b.evidence) for b in blocks] == [
        ("per_layer", reason),
        ("per_layer", VISION_COPY + reason),
    ]


def test_a_vision_encoder_whose_layers_differ_is_not_stacked_silently():
    vision = per_layer_gpt(range(2), prefix="vision_model.decoder")
    del vision["vision_model.decoder.layers.1.mlp.linear_fc2.weight"]
    found = refusal(llava(vision=vision), source=FSDP_DTENSOR)
    assert found.message.startswith(
        "the arguments make Megatron stack vision_model.decoder, but its layers differ: "
    )
    assert found.hint.startswith("pass --torch-dist-layout per-layer")
    assert found.hint.endswith(
        "or --block-layout vision_model.decoder=per_layer if the job that loads the store "
        "builds vision_model.decoder per layer from a configuration of its own"
    )
    layouts = {"vision_model.decoder": "per_layer"}
    _, block = classify(llava(vision=vision), source=FSDP_DTENSOR, layouts=layouts)
    assert block.td_layout == "per_layer"
    (warning,) = warnings_of(llava(vision=vision), layouts=layouts)
    assert warning.startswith(
        "--block-layout vision_model.decoder=per_layer: a job built from these arguments and "
        "this content_metadata builds vision_model.decoder stacked"
    )
    per_layer = {"moe_layer_freq": 2}
    assert warnings_of(llava(vision=vision), layouts=layouts, **per_layer) == []
