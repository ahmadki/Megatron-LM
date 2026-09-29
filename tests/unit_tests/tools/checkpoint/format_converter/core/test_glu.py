# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Gatedness of every MLP first projection, from hand-built key and shape tables."""

from __future__ import annotations

import argparse

import pytest
import torch

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.blocks import BlockLayout
from tools.checkpoint.format_converter.core.experts import ExpertLayout
from tools.checkpoint.format_converter.core.glu import glu_layout, swiglu_split
from tools.checkpoint.format_converter.core.logical import LogicalId
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, GROUPED, TORCH_DIST
from tools.checkpoint.format_converter.engine.metadata import TensorEntry
from tools.checkpoint.format_converter.errors import RefusedError

H, F, S = 8, 6, 4
"""Hidden size, expert and dense FFN size, shared-expert FFN size."""
STACKED = BlockLayout("decoder", "transformer_block", 2, "stacked", (0, 1), None, "")
PER_LAYER = BlockLayout("decoder", "transformer_block", 2, "per_layer", (0, 1), None, "")
VISION = BlockLayout("vision_model.decoder", "transformer_block", 1, "per_layer", (0,), None, "")
LANGUAGE = BlockLayout(
    "language_model.decoder", "transformer_block", 1, "per_layer", (0,), None, ""
)


def entries(shapes: dict[str, tuple[int, ...]], prefix: str = "") -> dict[str, TensorEntry]:
    return {
        key: TensorEntry(prefix + key, shape, torch.bfloat16, ()) for key, shape in shapes.items()
    }


def derive(shapes, blocks, *, source=TORCH_DIST, lm="decoder", prefix="", **args):
    return glu_layout(
        ArgsView(argparse.Namespace(**args)), entries(shapes, prefix), blocks, source, lm
    )


def fc1(block, layer, path, expert=None) -> LogicalId:
    return LogicalId(block, layer, path, "weight", expert)


def test_a_stacked_swiglu_decoder_is_gated_layer_by_layer():
    shapes = {
        "decoder.layers.mlp.linear_fc1.weight": (2, 2 * F, H),
        "decoder.layers.mlp.linear_fc1.bias": (2, 2 * F),
        "decoder.layers.mlp.linear_fc2.weight": (2, H, F),
    }
    layout, evidence = derive(shapes, [STACKED], swiglu=True)
    assert {lid: (u.key, u.rows, u.gated) for lid, u in layout.fc1s.items()} == {
        fc1("decoder", layer, "mlp.linear_fc1"): ("decoder.layers.mlp.linear_fc1.weight", 12, True)
        for layer in (0, 1)
    }
    assert layout.gated_json() == {"decoder": {"gated": 2}}
    assert evidence == (
        "2 linear_fc1 weights against their linear_fc2 inputs (gated: twice as many rows); "
        "args.swiglu=True, args.quick_geglu=False make the language model's MLPs (in decoder) gated"
    )
    bias = LogicalId("decoder", 1, "mlp.linear_fc1", "bias")
    assert layout.of(bias) is layout.of(fc1("decoder", 1, "mlp.linear_fc1"))
    assert layout.of(LogicalId("decoder", 1, "mlp.linear_fc1", "layer_norm_weight")) is None
    assert layout.of(LogicalId("decoder", 1, "mlp.linear_fc2", "weight")) is None


def test_quick_geglu_gates_the_language_model_too():
    shapes = {
        "decoder.layers.0.mlp.linear_fc1.weight": (2 * F, H),
        "decoder.layers.0.mlp.linear_fc2.weight": (H, F),
        "decoder.layers.1.self_attention.linear_proj.weight": (H, H),
    }
    layout, _ = derive(shapes, [PER_LAYER], swiglu=False, quick_geglu=True)
    assert layout.gated_json() == {"decoder": {"gated": 1}}


def test_every_routed_expert_and_the_shared_experts_are_classified():
    """Grouped fsdp_dtensor experts (one tensor per global expert) and shared experts."""
    shapes = {
        **{f"decoder.layers.1.mlp.experts.linear_fc1.weight{e}": (2 * F, H) for e in range(3)},
        **{f"decoder.layers.1.mlp.experts.linear_fc2.weight{e}": (H, F) for e in range(3)},
        "decoder.layers.1.mlp.shared_experts.linear_fc1.weight": (2 * S, H),
        "decoder.layers.1.mlp.shared_experts.linear_fc2.weight": (H, S),
    }
    layout, _ = derive(
        shapes, [PER_LAYER], source=FSDP_DTENSOR, prefix="model.module.", swiglu=True
    )
    assert sorted(map(str, layout.fc1s)) == [
        "decoder[1].mlp.experts.linear_fc1.weight[expert 0]",
        "decoder[1].mlp.experts.linear_fc1.weight[expert 1]",
        "decoder[1].mlp.experts.linear_fc1.weight[expert 2]",
        "decoder[1].mlp.shared_experts.linear_fc1.weight",
    ]
    expert = layout.of(LogicalId("decoder", 1, "mlp.experts.linear_fc1", "bias", 2))
    assert expert.key == "model.module.decoder.layers.1.mlp.experts.linear_fc1.weight2"
    assert all(u.gated for u in layout.fc1s.values())


def test_a_torch_dist_expert_tensor_is_classified_per_expert():
    shapes = {
        "decoder.layers.mlp.experts.experts.linear_fc1.weight": (2, 3, 2 * F, H),
        "decoder.layers.mlp.experts.experts.linear_fc2.weight": (2, 3, H, F),
    }
    layout, _ = derive(shapes, [STACKED], swiglu=True)
    assert layout.gated_json() == {"decoder": {"gated": 6}}
    assert fc1("decoder", 1, "mlp.experts.linear_fc1", 2) in layout.fc1s


def test_a_vision_encoder_and_projection_have_configurations_of_their_own():
    """A gated vision encoder and projection under a plain language model (args.swiglu off)."""
    shapes = {
        "language_model.decoder.layers.0.mlp.linear_fc1.weight": (F, H),
        "language_model.decoder.layers.0.mlp.linear_fc2.weight": (H, F),
        "vision_model.decoder.layers.0.mlp.linear_fc1.weight": (2 * S, H),
        "vision_model.decoder.layers.0.mlp.linear_fc2.weight": (H, S),
        "vision_projection.encoder.linear_fc1.weight": (2 * H, S),
        "vision_projection.encoder.linear_fc2.weight": (H, H),
    }
    layout, _ = derive(shapes, [LANGUAGE, VISION], lm="language_model.decoder", swiglu=False)
    assert layout.gated_json() == {
        "language_model.decoder": {"plain": 1},
        "vision_model.decoder": {"gated": 1},
        "vision_projection.encoder": {"gated": 1},
    }
    assert layout.of(LogicalId(None, None, "vision_projection.encoder.linear_fc1", "bias")).gated


@pytest.mark.parametrize(
    ("rows", "args", "observed"),
    [
        (2 * F, {"swiglu": False}, "gated (twice the rows linear_fc2 has inputs)"),
        (F, {"swiglu": True}, "plain"),
        (F, {"quick_geglu": True}, "plain"),
    ],
)
def test_the_language_model_must_be_gated_as_its_arguments_say(rows, args, observed):
    shapes = {
        "decoder.layers.mlp.linear_fc1.weight": (2, rows, H),
        "decoder.layers.mlp.linear_fc2.weight": (2, H, F),
    }
    with pytest.raises(RefusedError) as caught:
        derive(shapes, [STACKED], **args)
    refusal = caught.value.refusal
    assert refusal.code == errors.FC_R12
    assert refusal.message.startswith(
        f"the language model's linear_fc1 weights (in decoder) are {observed}, but "
    )
    assert refusal.keys == ("decoder.layers.mlp.linear_fc1.weight",)


def test_the_mlps_of_a_layer_are_all_gated_or_all_plain():
    shapes = {
        "decoder.layers.1.mlp.experts.linear_fc1.weight0": (2 * F, H),
        "decoder.layers.1.mlp.experts.linear_fc1.weight1": (F, H),
        **{f"decoder.layers.1.mlp.experts.linear_fc2.weight{e}": (H, F) for e in range(2)},
    }
    with pytest.raises(RefusedError) as caught:
        derive(shapes, [PER_LAYER], source=FSDP_DTENSOR, swiglu=True)
    assert caught.value.refusal.code == errors.FC_R12
    assert caught.value.refusal.message.startswith("decoder[1] has gated and plain linear_fc1")
    assert caught.value.refusal.keys == tuple(sorted(shapes)[:2])


def test_a_first_projection_without_its_second_is_refused():
    shapes = {"decoder.layers.0.mlp.linear_fc1.weight": (2 * F, H)}
    with pytest.raises(RefusedError) as caught:
        derive(shapes, [PER_LAYER], swiglu=True)
    assert caught.value.refusal.code == errors.FC_R12
    assert caught.value.refusal.message == (
        "decoder[0].mlp.linear_fc1.weight has no mlp.linear_fc2 weight beside it, so whether it "
        "is a gated linear unit cannot be told"
    )


def test_a_linear_fc1_outside_every_layer_without_a_linear_fc2_is_a_plain_linear_layer():
    """A patch merger (``PixtralLargePatchMerger``) holds a plain ``nn.Linear`` named
    ``linear_fc1``, with no ``linear_fc2``: no MLP's first projection, so it is left out."""
    shapes = {
        "language_model.decoder.layers.0.mlp.linear_fc1.weight": (F, H),
        "language_model.decoder.layers.0.mlp.linear_fc2.weight": (H, F),
        "vision_model.merger.linear_fc1.weight": (S, 4 * S),
    }
    layout, evidence = derive(shapes, [LANGUAGE], lm="language_model.decoder")
    assert list(layout.fc1s) == [fc1("language_model.decoder", 0, "mlp.linear_fc1")]
    assert layout.of(LogicalId(None, None, "vision_model.merger.linear_fc1", "weight")) is None
    assert evidence.endswith(
        "(in language_model.decoder) plain; 1 linear_fc1 outside every layer with no linear_fc2 "
        "beside it (plain linear layers, converted unchanged): vision_model.merger.linear_fc1.weight"
    )


def test_a_lone_linear_fc1_under_a_swiglu_name_is_refused_outside_layers_too():
    """Megatron-FSDP splits a first projection outside every layer by its name alone: one under
    a SwiGLU name with no linear_fc2 cannot be told gated or plain."""
    shapes = {"encoder.mlp.linear_fc1.weight": (2 * F, H)}
    with pytest.raises(RefusedError) as caught:
        derive(shapes, [PER_LAYER], swiglu=True)
    assert caught.value.refusal.code == errors.FC_R12
    assert caught.value.refusal.keys == ("encoder.mlp.linear_fc1.weight",)


def test_a_model_without_mlps_has_no_first_projections():
    shapes = {"decoder.layers.0.self_attention.linear_qkv.weight": (3 * H, H)}
    layout, evidence = derive(shapes, [PER_LAYER])
    assert layout.fc1s == {} and evidence.startswith("0 linear_fc1 weights")


FOUR = ExpertLayout(4, GROUPED)
MOE = {
    **{f"decoder.layers.1.mlp.experts.linear_fc1.weight{e}": (2 * F, H) for e in range(4)},
    **{f"decoder.layers.1.mlp.experts.linear_fc2.weight{e}": (H, F) for e in range(4)},
    "decoder.layers.1.mlp.shared_experts.linear_fc1.weight": (2 * S, H),
    "decoder.layers.1.mlp.shared_experts.linear_fc2.weight": (H, S),
    "decoder.layers.0.mlp.linear_fc1.weight": (2 * F, H),
    "decoder.layers.0.mlp.linear_fc2.weight": (H, F),
}
"""A dense SwiGLU layer and a SwiGLU MoE layer with shared experts, one tensor per expert."""
MOE_TD = {
    "decoder.layers.1.mlp.experts.experts.linear_fc1.weight": (4, 2 * F, H),
    "decoder.layers.1.mlp.experts.experts.linear_fc2.weight": (4, H, F),
    **{k: v for k, v in MOE.items() if ".mlp.experts." not in k},
}
"""The same model as torch_dist stores it (every expert in one tensor)."""


def split(shapes, blocks, *, fsdp_swiglu, source=TORCH_DIST, lm="decoder", experts=None, **args):
    layout, _ = derive(shapes, blocks, source=source, lm=lm, **args)
    return swiglu_split(layout, fsdp_swiglu, blocks, source, experts)


def test_the_split_predicate_on_a_torch_dist_source():
    layout, evidence = split(MOE_TD, [PER_LAYER], fsdp_swiglu=True, experts=FOUR, swiglu=True)
    assert {str(lid): fc1.split for lid, fc1 in layout.fc1s.items()} == {
        "decoder[0].mlp.linear_fc1.weight": True,
        **{f"decoder[1].mlp.experts.linear_fc1.weight[expert {e}]": True for e in range(4)},
        "decoder[1].mlp.shared_experts.linear_fc1.weight": True,
    }
    assert layout.split_json() == {"decoder": {"split": 6}}
    assert evidence.endswith(": 6 split, 0 gated kept fused")
    kept, _ = split(MOE_TD, [PER_LAYER], fsdp_swiglu=False, experts=FOUR, quick_geglu=True)
    assert kept.split_json() == {"decoder": {"fused": 6}}


def test_only_the_swiglu_names_of_gated_layers_are_split():
    """A gated projection's ``encoder.linear_fc1`` is no SwiGLU name; a plain vision encoder's
    layers are not gated; an MLP outside every layer has no owner and is split."""
    shapes = {
        "language_model.decoder.layers.0.mlp.linear_fc1.weight": (2 * F, H),
        "language_model.decoder.layers.0.mlp.linear_fc2.weight": (H, F),
        "vision_model.decoder.layers.0.mlp.linear_fc1.weight": (S, H),
        "vision_model.decoder.layers.0.mlp.linear_fc2.weight": (H, S),
        "vision_projection.encoder.linear_fc1.weight": (2 * H, S),
        "vision_projection.encoder.linear_fc2.weight": (H, H),
        "vision_projection.head.mlp.linear_fc1.weight": (S, H),
        "vision_projection.head.mlp.linear_fc2.weight": (H, S),
    }
    layout, _ = split(
        shapes, [LANGUAGE, VISION], fsdp_swiglu=True, lm="language_model.decoder", swiglu=True
    )
    assert {str(lid): (fc1.gated, fc1.split) for lid, fc1 in layout.fc1s.items()} == {
        "language_model.decoder[0].mlp.linear_fc1.weight": (True, True),
        "vision_model.decoder[0].mlp.linear_fc1.weight": (False, False),
        "vision_projection.encoder.linear_fc1.weight": (True, False),
        "vision_projection.head.mlp.linear_fc1.weight": (False, True),
    }
    assert layout.split_json() == {
        "language_model.decoder": {"split": 1},
        "vision_projection.encoder": {"fused": 1},
        "vision_projection.head.mlp": {"split": 1},
    }


def test_an_odd_first_projection_to_split_is_refused():
    shapes = {"head.mlp.linear_fc1.weight": (5, H), "head.mlp.linear_fc2.weight": (H, 5)}
    with pytest.raises(RefusedError) as caught:
        split(shapes, [PER_LAYER], fsdp_swiglu=True)
    assert caught.value.refusal.code == errors.FC_R12
    assert "has 5 rows" in caught.value.refusal.message


def halves(shapes: dict[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    """``shapes`` with every gated linear_fc1 weight and bias stored as SwiGLU halves."""
    stored = {}
    for key, shape in shapes.items():
        if "linear_fc1" in key and key.rsplit(".", 1)[1].startswith(("weight", "bias")):
            half = (shape[0] // 2, *shape[1:])
            stored[f"{key}_w"], stored[f"{key}_v"] = half, half
        else:
            stored[key] = shape
    return stored


def test_an_fsdp_source_stores_what_the_predicate_says():
    stored = halves(MOE)
    layout, _ = split(
        stored, [PER_LAYER], fsdp_swiglu=True, source=FSDP_DTENSOR, experts=FOUR, swiglu=True
    )
    assert all(fc1.halves and fc1.split for fc1 in layout.fc1s.values())
    expert = layout.of(LogicalId("decoder", 1, "mlp.experts.linear_fc1", "weight", 3, "v"))
    assert (expert.key, expert.rows, expert.gated, expert.halves, expert.split) == (
        "decoder.layers.1.mlp.experts.linear_fc1.weight3_w",
        2 * F,
        True,
        True,
        True,
    )
    with pytest.raises(RefusedError) as caught:
        split(
            stored,
            [PER_LAYER],
            fsdp_swiglu=False,
            source=FSDP_DTENSOR,
            experts=FOUR,
            quick_geglu=True,
        )
    assert caught.value.refusal.code == errors.FC_R12
    assert caught.value.refusal.message.startswith(
        "the store holds SwiGLU halves that Megatron-FSDP does not write"
    )
    with pytest.raises(RefusedError) as caught:
        split(MOE, [PER_LAYER], fsdp_swiglu=True, source=FSDP_DTENSOR, experts=FOUR, swiglu=True)
    assert caught.value.refusal.message.startswith(
        "the store holds gated linear_fc1 whole that Megatron-FSDP splits into SwiGLU halves"
    )
    assert len(caught.value.refusal.keys) == 6


GATE = "decoder.layers.0.mlp.linear_fc1.weight"
DENSE = {GATE: (2 * F, H), "decoder.layers.0.mlp.linear_fc2.weight": (H, F)}


@pytest.mark.parametrize(
    ("stored", "code", "message"),
    [
        ({f"{GATE}_w": (F, H)}, errors.FC_R20, "half w (6, 8)"),
        ({f"{GATE}_w": (F, H), f"{GATE}_v": (F - 1, H)}, errors.FC_R20, "half v (5, 8)"),
        ({GATE: (2 * F, H), f"{GATE}_v": (F, H)}, errors.FC_R20, "the whole tensor (12, 8)"),
        (
            {
                f"{GATE}_w": (F, H),
                f"{GATE}_v": (F, H),
                "decoder.layers.0.mlp.linear_fc1.bias": (2 * F,),
            },
            errors.FC_R12,
            "and its bias are not both stored as SwiGLU halves",
        ),
    ],
)
def test_swiglu_halves_come_in_pairs_that_replace_the_whole_tensor(stored, code, message):
    shapes = {**{k: v for k, v in DENSE.items() if k != GATE}, **stored}
    with pytest.raises(RefusedError) as caught:
        derive(shapes, [PER_LAYER], source=FSDP_DTENSOR, swiglu=True)
    assert caught.value.refusal.code == code
    assert message in caught.value.refusal.message
