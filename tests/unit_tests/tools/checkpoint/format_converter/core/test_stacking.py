# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TransformerBlock layouts observed in torch_dist stores, checked against hand-built keys."""

from __future__ import annotations

import argparse

import pytest
import torch

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.blocks import BlockFacts, BlockLayout, classify_blocks
from tools.checkpoint.format_converter.core.names import TORCH_DIST
from tools.checkpoint.format_converter.core.stacking import transformer_block
from tools.checkpoint.format_converter.engine.metadata import TensorEntry
from tools.checkpoint.format_converter.errors import RefusedError

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


def classify(shapes, **args) -> tuple[BlockLayout, ...]:
    facts = BlockFacts(ArgsView(argparse.Namespace(**args)), TORCH_DIST, _model(shapes))
    return classify_blocks(facts, default=transformer_block)


def refusal(shapes, **args):
    with pytest.raises(RefusedError) as excinfo:
        classify(shapes, **args)
    assert excinfo.value.refusal.code == errors.FC_R12
    return excinfo.value.refusal


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
