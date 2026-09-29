# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Gated linear units (``linear_fc1`` of SwiGLU and GEGLU MLPs) across formats, on synthetic stores.

The gate ``W`` and value ``V`` of every gated ``linear_fc1`` are drawn
separately (``gated_units``): the fsdp_dtensor store Megatron-FSDP writes with
``--swiglu`` holds them as the halves ``<key>_w`` and ``<key>_v``, and the
torch_dist store holds ``torch.cat([W, V])``, stacked with ``torch.stack``
over experts and layers. The expected stores therefore never come from the
converter; keys, chunk grids and derivations are written out by hand.
"""

from __future__ import annotations

import pytest
import torch

from tools.checkpoint.format_converter.api import (
    convert_checkpoint,
    plan_conversion,
    verify_checkpoint,
)
from tools.checkpoint.format_converter.options import ConvertOptions, PlanOptions, VerifyOptions

from ._synthetic import (
    expert_grids,
    gated_units,
    gpt_args,
    gpt_moe_tensors,
    gpt_tensors,
    hybrid_args,
    hybrid_tensors,
    layer_grids,
    sequential_experts,
    stack_experts,
    stack_layers,
    swiglu_grids,
    vision_language_tensors,
    write_fsdp_store,
)
from .test_convert_blocks import ITER, expect_fsdp_store, grid_of, run, write_gpt_source
from .test_convert_cross import bitwise_equal, read_tensors

pytestmark = pytest.mark.covers("glu", "ground_truth")
LAYERS, EXPERTS, FFN, HIDDEN = 3, 4, 16, 8
HALVES, FUSED = gated_units(gpt_tensors(LAYERS, HIDDEN, FFN))
"""A SwiGLU GPT one layer per key: as fsdp_dtensor splits it, and fused as torch_dist stores it."""
STACKED = stack_layers(FUSED, "decoder")
FC1 = "decoder.layers.mlp.linear_fc1.weight"


def swiglu_args(layers: int = LAYERS, **overrides):
    return gpt_args(layers, **{"swiglu": True, **overrides})


def megatron_grids(tensors):
    """Megatron's grid: one chunk per layer and expert, and per half of every gated linear_fc1."""
    return {**layer_grids(tensors, "decoder"), **expert_grids(tensors), **swiglu_grids(tensors)}


def to_fsdp(tmp_path, src, name="fsdp", **options):
    report = convert_checkpoint(
        src, tmp_path / name, ConvertOptions(to_format="fsdp_dtensor", workers=0, **options)
    )
    return tmp_path / name / ITER, report.to_json()


def to_torch_dist(tmp_path, store, name="td", **options):
    report = convert_checkpoint(store, tmp_path / name, ConvertOptions(workers=0, **options))
    return tmp_path / name / ITER, report.to_json()


def assert_store(out, expected):
    tensors = read_tensors(out)
    assert set(tensors) == set(expected)
    for key, tensor in tensors.items():
        assert bitwise_equal(tensor, expected[key]), key


def per_half(layers: int, rows: int, width: int, experts: int | None = None) -> list:
    """The chunks of a stacked gated linear_fc1 weight saved one chunk per (layer, expert, half)."""
    if experts is None:
        return [((l, h * rows, 0), (1, rows, width)) for l in range(layers) for h in (0, 1)]
    return [
        ((l, e, h * rows, 0), (1, 1, rows, width))
        for l in range(layers)
        for e in range(experts)
        for h in (0, 1)
    ]


def test_the_plan_logs_which_first_projections_are_gated(tmp_path):
    src = write_gpt_source(tmp_path / "src", stack_layers(gpt_tensors(LAYERS), "decoder"))
    plan = plan_conversion(src, PlanOptions(to_format="fsdp_dtensor")).to_json()
    (gated,) = [d for d in plan["derivations"] if d["field"] == "gated_fc1"]
    assert gated["value"] == {"decoder": {"plain": LAYERS}}
    assert gated["evidence"].endswith("make the language model's MLPs (in decoder) plain")
    assert plan["common"]["args_consulted"]["swiglu"] is False


def test_a_language_model_gated_otherwise_than_its_arguments_is_refused(tmp_path, capsys):
    """``--quick-geglu`` builds gated MLPs, but the store's are plain."""
    model = stack_layers(gpt_tensors(LAYERS), "decoder")
    src = write_gpt_source(tmp_path / "src", model, args=gpt_args(LAYERS, quick_geglu=True))
    assert run("plan", src, "--to", "fsdp_dtensor") == 3
    err = capsys.readouterr().err
    assert (
        "error: FC-R12: the language model's linear_fc1 weights (in decoder) are plain, but "
        "args.swiglu=False and args.quick_geglu=True make its MLPs gated "
        "[decoder.layers.mlp.linear_fc1.weight]"
    ) in err


def test_a_swiglu_gpt_is_split_into_halves_for_fsdp_dtensor(tmp_path):
    src = write_gpt_source(
        tmp_path / "src", STACKED, grids=megatron_grids(STACKED), args=swiglu_args()
    )
    out, report = to_fsdp(tmp_path, src)
    expect_fsdp_store(out, HALVES)
    assert report["claims"] == {"glu": 2 * LAYERS, "identity": len(FUSED) - 2 * LAYERS}
    derived = {d["field"]: d["value"] for d in report["derivations"]}
    assert derived["gated_fc1"] == {"decoder": {"gated": LAYERS}}
    assert derived["fsdp_swiglu"] is True
    assert derived["swiglu_split"] == {"decoder": {"split": LAYERS}}
    run_stats = report["run"]
    assert run_stats["torch_load_calls"] == run_stats["items_touched"]  # each chunk read once
    assert run_stats["items_touched"] == sum(len(g) for g in megatron_grids(STACKED).values()) + 3


def test_swiglu_halves_merge_back_into_megatrons_chunk_per_half(tmp_path):
    src = write_gpt_source(
        tmp_path / "src", STACKED, grids=megatron_grids(STACKED), args=swiglu_args()
    )
    fsdp, _ = to_fsdp(tmp_path, src)
    out, report = to_torch_dist(tmp_path, fsdp)
    assert_store(out, STACKED)
    assert grid_of(out, FC1) == per_half(LAYERS, FFN, HIDDEN)
    assert grid_of(out, "decoder.layers.mlp.linear_fc1.bias") == [
        ((l, h * FFN), (1, FFN)) for l in range(LAYERS) for h in (0, 1)
    ]
    assert report["claims"]["glu"] == 4 * LAYERS  # both halves of the weight and the bias
    assert verify_checkpoint(out, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok


def test_an_fsdp_swiglu_training_save_converts_to_a_stacked_store(tmp_path):
    """Megatron-FSDP stores fp32 main weights; the halves are TP-canonical, so any TP works."""
    fp32 = {key: tensor.float() for key, tensor in HALVES.items()}
    args = swiglu_args(ckpt_format="fsdp_dtensor", use_megatron_fsdp=True, bf16=False)
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, fp32, args=args)
    out, _ = to_torch_dist(tmp_path, store)
    expected = stack_layers({key: tensor.float() for key, tensor in FUSED.items()}, "decoder")
    assert_store(out, expected)
    assert grid_of(out, FC1) == per_half(LAYERS, FFN, HIDDEN)


def moe_units() -> tuple[dict, dict]:
    """A SwiGLU GPT whose two layers are mixture-of-experts layers with shared experts, as
    fsdp_dtensor names grouped experts: halves and fused."""
    generator = torch.Generator().manual_seed(5)
    shared = {}
    for layer in range(2):
        head = f"decoder.layers.{layer}.mlp.shared_experts."
        shared[head + "linear_fc1.weight"] = torch.randn(4, HIDDEN, generator=generator).bfloat16()
        shared[head + "linear_fc2.weight"] = torch.randn(HIDDEN, 4, generator=generator).bfloat16()
    return gated_units({**gpt_moe_tensors(2, EXPERTS, HIDDEN), **shared})


MOE_HALVES, MOE_FUSED = moe_units()
MOE_STACKED = stack_layers(stack_experts(MOE_FUSED), "decoder")
EXPERT_FC1 = "decoder.layers.mlp.experts.experts.linear_fc1.weight"


@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "sequential"])
def test_swiglu_experts_are_split_under_their_global_index(tmp_path, grouped):
    """Grouped ``linear_fc1.weight<e>_w`` or sequential ``local_experts.<e>.linear_fc1.weight_w``,
    and the shared experts' halves; the round trip keeps one chunk per (layer, expert, half)."""
    args = swiglu_args(2, num_experts=EXPERTS, moe_grouped_gemm=grouped)
    src = write_gpt_source(
        tmp_path / "src", MOE_STACKED, grids=megatron_grids(MOE_STACKED), args=args
    )
    fsdp, report = to_fsdp(tmp_path, src)
    expected = MOE_HALVES if grouped else sequential_experts(MOE_HALVES)
    expect_fsdp_store(fsdp, expected)
    half = "decoder.layers.1.mlp.experts." + (
        "linear_fc1.weight3_v" if grouped else "local_experts.3.linear_fc1.weight_v"
    )
    assert half in expected
    derived = {d["field"]: d["value"] for d in report["derivations"]}
    assert derived["swiglu_split"] == {"decoder": {"split": 2 * (EXPERTS + 1)}}
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, MOE_STACKED)
    assert grid_of(out, EXPERT_FC1) == per_half(2, 6, HIDDEN, EXPERTS)
    assert verify_checkpoint(out, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok


def test_a_swiglu_hybrid_model_round_trips(tmp_path):
    """The MLP layers (``-``) of a hybrid model are gated MLPLayers (TransformerLayers)."""
    halves, fused = gated_units(hybrid_tensors("*-*-"))
    args = hybrid_args("*-|*-", swiglu=True)
    src = write_gpt_source(tmp_path / "src", fused, grids=swiglu_grids(fused), args=args)
    fsdp, _ = to_fsdp(tmp_path, src)
    expect_fsdp_store(fsdp, halves)
    assert "decoder.layers.3.mlp.linear_fc1.weight_v" in halves
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, fused)
    assert grid_of(out, "decoder.layers.1.mlp.linear_fc1.weight") == [
        ((0, 0), (16, 8)),
        ((16, 0), (16, 8)),
    ]


def test_only_the_language_model_of_a_vision_language_model_is_split(tmp_path):
    """A SwiGLU language model under a plain vision encoder and projection."""
    vlm_halves, vlm_fused = vision_language_tensors(HALVES), vision_language_tensors(FUSED)
    source = stack_layers(stack_layers(vlm_fused, "language_model.decoder"), "vision_model.decoder")
    src = write_gpt_source(tmp_path / "src", source, args=swiglu_args())
    fsdp, report = to_fsdp(tmp_path, src)
    expect_fsdp_store(fsdp, vlm_halves)
    derived = {d["field"]: d["value"] for d in report["derivations"]}
    assert derived["gated_fc1"] == {
        "language_model.decoder": {"gated": LAYERS},
        "vision_model.decoder": {"plain": 2},
        "vision_projection.encoder": {"plain": 1},
    }
    assert derived["swiglu_split"] == {"language_model.decoder": {"split": LAYERS}}
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, source)


def test_the_plan_names_the_split_rule(tmp_path, capsys):
    src = write_gpt_source(tmp_path / "src", STACKED, args=swiglu_args())
    assert run("plan", src, "--to", "fsdp_dtensor") == 0
    text = capsys.readouterr().out
    assert f"model tensors claimed: glu {2 * LAYERS}, identity {len(FUSED) - 2 * LAYERS}" in text


def fsdp_swiglu_store(root, model, **overrides):
    args = swiglu_args(ckpt_format="fsdp_dtensor", **overrides)
    return write_fsdp_store(root / ITER, model, args=args)


def test_an_unpaired_swiglu_half_is_refused(tmp_path, capsys):
    half = "decoder.layers.1.mlp.linear_fc1.weight_v"
    store = fsdp_swiglu_store(tmp_path / "fsdp", {k: t for k, t in HALVES.items() if k != half})
    assert run("plan", store, "--to", "torch_dist", "--torch-dist-layout", "per-layer") == 3
    err = capsys.readouterr().err
    assert (
        "error: FC-R20: decoder[1].mlp.linear_fc1.weight is stored as half w (16, 8): SwiGLU "
        "halves come in pairs (w and v) of one shape that replace the whole tensor "
        "[model.module.decoder.layers.1.mlp.linear_fc1.weight_w]"
    ) in err


def test_halves_that_megatron_fsdp_does_not_write_are_refused(tmp_path, capsys):
    """Without ``args.swiglu`` Megatron-FSDP keeps linear_fc1 whole (quick-GEGLU is gated)."""
    store = fsdp_swiglu_store(tmp_path / "fsdp", HALVES, swiglu=False, quick_geglu=True)
    assert run("plan", store, "--to", "torch_dist") == 3
    err = capsys.readouterr().err
    assert "error: FC-R12: the store holds SwiGLU halves that Megatron-FSDP does not write" in err
    assert "(here False) is set" in err


def test_pack_sizes_are_given_for_an_fsdp_dtensor_target_only(tmp_path, capsys, caplog):
    src = write_gpt_source(tmp_path / "src", STACKED, args=swiglu_args())
    fsdp, report = to_fsdp(tmp_path, src, fsdp_tp=2)
    assert "--fsdp-tp 2 has no effect: the fsdp_dtensor store packs no tensor per" in caplog.text
    assert "tensor_model_parallel_size" not in report["common"]["args_changes"]  # halves only
    for store in (fsdp, src):
        assert run("plan", store, "--to", "torch_dist", "--fsdp-etp", "2") == 2
        err = capsys.readouterr().err
        assert "error: only torch_dist to fsdp_dtensor conversions take --fsdp-etp" in err
