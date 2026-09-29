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
    packed_units,
    product_grid,
    sequential_experts,
    stack_experts,
    stack_layers,
    swiglu_grids,
    vision_language_tensors,
    write_fsdp_store,
)
from .test_convert_blocks import (
    ITER,
    expect_fsdp_store,
    grid_of,
    read_common,
    run,
    write_gpt_source,
)
from .test_convert_cross import bitwise_equal, read_items, read_tensors

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


def geglu_args(layers: int = LAYERS, **overrides):
    """A GEGLU model (``--quick-geglu``): gated MLPs that Megatron-FSDP keeps fused."""
    return gpt_args(layers, **{"quick_geglu": True, **overrides})


def tp2_grids(tensors):
    """Megatron's grid of a gated linear_fc1 saved at tensor-parallel size 2: per layer, the
    shards ``W_0, W_1, V_0, V_1`` (rows ``[0, F/2)``, ``[F/2, F)``, ...), and one chunk per layer
    of everything else."""
    grids = layer_grids(tensors, "decoder")
    for key in (FC1, "decoder.layers.mlp.linear_fc1.bias"):
        shape = tuple(tensors[key].shape)
        rows = shape[1]
        grids[key] = product_grid(
            shape,
            [list(range(1, LAYERS)), [rows // 4, rows // 2, 3 * rows // 4]]
            + [[]] * (len(shape) - 2),
        )
    return grids


def test_a_geglu_gpt_keeps_every_linear_fc1_fused(tmp_path):
    """Packed at tensor-parallel size 1, a fused tensor is the canonical ``[W; V]``."""
    src = write_gpt_source(
        tmp_path / "src", STACKED, grids=megatron_grids(STACKED), args=geglu_args()
    )
    fsdp, report = to_fsdp(tmp_path, src)
    expect_fsdp_store(fsdp, FUSED)
    assert report["claims"] == {"glu": 2 * LAYERS, "identity": len(FUSED) - 2 * LAYERS}
    derived = {d["field"]: d for d in report["derivations"]}
    assert derived["swiglu_split"]["value"] == {"decoder": {"fused": LAYERS}}
    assert derived["pack_tp"]["value"] == 1 and derived["pack_etp"]["value"] is None
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, STACKED)
    assert grid_of(out, FC1) == per_half(LAYERS, FFN, HIDDEN)
    assert verify_checkpoint(out, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok


def test_a_tensor_parallel_source_is_packed_at_the_loading_jobs_size(tmp_path, capsys):
    """Saved at tensor-parallel size 2: fsdp_dtensor holds ``[W_0; V_0; W_1; V_1]``."""
    args = geglu_args(tensor_model_parallel_size=2, tensor_parallel_num_weight_shards=2)
    src = write_gpt_source(tmp_path / "src", STACKED, grids=tp2_grids(STACKED), args=args)
    assert run("plan", src, "--to", "fsdp_dtensor") == 4
    err = capsys.readouterr().err
    assert "error: FC-A01: fsdp_dtensor packs 3 gated linear_fc1 per rank of the job" in err
    assert "hint: pass --fsdp-tp N with the tensor_model_parallel_size of that job" in err
    fsdp, report = to_fsdp(tmp_path, src, fsdp_tp=2)
    expect_fsdp_store(fsdp, packed_units(HALVES, lambda key: 2))
    assert not {k for k in report["common"]["args_changes"] if "tensor" in k}  # already 2
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, STACKED)
    assert grid_of(out, FC1) == per_half(LAYERS, FFN, HIDDEN)


def test_packing_at_another_size_rewrites_and_restores_the_arguments(tmp_path):
    args = geglu_args(tensor_model_parallel_size=2, tensor_parallel_num_weight_shards=2)
    src = write_gpt_source(tmp_path / "src", STACKED, grids=tp2_grids(STACKED), args=args)
    fsdp, report = to_fsdp(tmp_path, src, fsdp_tp=4)
    expect_fsdp_store(fsdp, packed_units(HALVES, lambda key: 4))
    assert report["common"]["args_changes"]["tensor_model_parallel_size"] == [2, 4]
    assert report["common"]["args_changes"]["tensor_parallel_num_weight_shards"] == [2, 4]
    stored = read_items(fsdp)["args"]
    assert (stored.tensor_model_parallel_size, stored.tensor_parallel_num_weight_shards) == (4, 4)
    out, report = to_torch_dist(tmp_path, fsdp)
    assert_store(out, STACKED)
    restored = read_common(out)["args"]
    assert (restored.tensor_model_parallel_size, restored.tensor_parallel_num_weight_shards) == (
        2,
        2,
    )
    (pack,) = [d for d in report["derivations"] if d["field"] == "pack_tp"]
    assert pack["value"] == 4 and pack["evidence"].startswith(
        "args.tensor_model_parallel_size=4 of the job that saved the source"
    )


def test_a_megatron_fsdp_save_packed_at_two_ranks_is_unpacked(tmp_path):
    """A GEGLU training save at tensor-parallel size 2: fp32 main weights, rank-major."""
    fp32 = {key: tensor.float() for key, tensor in HALVES.items()}
    args = geglu_args(ckpt_format="fsdp_dtensor", tensor_model_parallel_size=2, bf16=False)
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, packed_units(fp32, lambda key: 2), args=args)
    out, _ = to_torch_dist(tmp_path, store)
    assert_store(out, stack_layers({key: t.float() for key, t in FUSED.items()}, "decoder"))
    assert grid_of(out, FC1) == per_half(LAYERS, FFN, HIDDEN)


def test_routed_experts_are_packed_at_the_expert_size(tmp_path, capsys):
    """Expert tensor-parallel size 2 with tensor-parallel size 1: each expert's linear_fc1 is
    ``[W_0; V_0; W_1; V_1]``, the shared experts' ``[W; V]``."""
    args = geglu_args(2, num_experts=EXPERTS, moe_grouped_gemm=True, expert_tensor_parallel_size=2)
    src = write_gpt_source(
        tmp_path / "src", MOE_STACKED, grids=megatron_grids(MOE_STACKED), args=args
    )
    assert run("plan", src, "--to", "fsdp_dtensor") == 4
    err = capsys.readouterr().err
    assert "args.expert_tensor_parallel_size=2" in err and "--fsdp-etp N" in err
    fsdp, report = to_fsdp(tmp_path, src, fsdp_etp=2)
    expected = packed_units(MOE_HALVES, lambda key: 2 if ".mlp.experts." in key else 1)
    expect_fsdp_store(fsdp, expected)
    derived = {d["field"]: d["value"] for d in report["derivations"]}
    assert (derived["pack_tp"], derived["pack_etp"]) == (1, 2)
    assert derived["swiglu_split"] == {"decoder": {"fused": 2 * (EXPERTS + 1)}}
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, MOE_STACKED)
    assert grid_of(out, EXPERT_FC1) == per_half(2, 6, HIDDEN, EXPERTS)


def test_another_tensor_parallel_size_needs_the_loading_jobs_expert_size(tmp_path, capsys, caplog):
    """A job at --fsdp-tp 2 runs its experts at expert tensor-parallel size 2 unless it sets
    --expert-tensor-parallel-size: the source's expert size 1 is no default then."""
    args = geglu_args(2, num_experts=EXPERTS, moe_grouped_gemm=True, expert_tensor_parallel_size=1)
    src = write_gpt_source(
        tmp_path / "src", MOE_STACKED, grids=megatron_grids(MOE_STACKED), args=args
    )
    assert run("plan", src, "--to", "fsdp_dtensor", "--fsdp-tp", "2") == 4
    err = capsys.readouterr().err
    assert (
        "error: FC-A01: fsdp_dtensor packs 8 gated linear_fc1 of routed experts per expert "
        "tensor-parallel rank of the job that loads it, and --fsdp-tp 2 is not the size the "
        "source was saved with (args.tensor_model_parallel_size=1)"
    ) in err
    assert "hint: pass --fsdp-etp N with the expert_tensor_parallel_size of that job" in err
    fsdp, report = to_fsdp(tmp_path, src, fsdp_tp=2, fsdp_etp=2)
    expect_fsdp_store(fsdp, packed_units(MOE_HALVES, lambda key: 2))
    changes = report["common"]["args_changes"]
    assert changes["tensor_model_parallel_size"] == changes["expert_tensor_parallel_size"] == [1, 2]
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, MOE_STACKED)
    halves, fused = gated_units(gpt_moe_tensors(2, EXPERTS, HIDDEN))  # no shared experts
    experts = stack_layers(stack_experts(fused), "decoder")
    src = write_gpt_source(tmp_path / "experts", experts, grids=megatron_grids(experts), args=args)
    fsdp, report = to_fsdp(tmp_path, src, name="experts_fsdp", fsdp_tp=2, fsdp_etp=2)
    expect_fsdp_store(fsdp, packed_units(halves, lambda key: 2))
    assert "tensor_model_parallel_size" not in report["common"]["args_changes"]
    assert (
        "--fsdp-tp 2 has no effect: the fsdp_dtensor store packs no tensor per tensor-parallel "
        "rank, but it packs 8 gated linear_fc1 of routed experts per expert tensor-parallel rank"
    ) in caplog.text


HYBRID_VLM = vision_language_tensors(hybrid_tensors("*-*-"))
VIT_HALVES, VIT_FUSED = gated_units(HYBRID_VLM, prefix="vision_model.")
"""A hybrid language model (plain MLPs) under a GLU vision encoder: the encoder's layers gated."""
VIT_STACKED = stack_layers(VIT_FUSED, "vision_model.decoder")
VIT_FC1 = "vision_model.decoder.layers.mlp.linear_fc1.weight"


@pytest.mark.parametrize("tp", [1, 2])
def test_a_glu_vision_encoder_under_a_plain_hybrid_language_model(tmp_path, tp):
    """The encoder's configuration gates its MLPs without args.swiglu: Megatron-FSDP keeps them
    fused, packed at the tensor-parallel size."""
    args = hybrid_args("*-*-", swiglu=False, tensor_model_parallel_size=tp)
    grids = {**layer_grids(VIT_STACKED, "vision_model.decoder"), **swiglu_grids(VIT_STACKED)}
    src = write_gpt_source(tmp_path / "src", VIT_STACKED, grids=grids, args=args)
    fsdp, report = to_fsdp(tmp_path, src, fsdp_tp=tp)
    expect_fsdp_store(fsdp, packed_units(VIT_HALVES, lambda key: tp))
    derived = {d["field"]: d["value"] for d in report["derivations"]}
    assert derived["gated_fc1"] == {
        "language_model.decoder": {"plain": 2},
        "vision_model.decoder": {"gated": 2},
        "vision_projection.encoder": {"plain": 1},
    }
    assert derived["swiglu_split"] == {"vision_model.decoder": {"fused": 2}}
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, VIT_STACKED)
    assert grid_of(out, VIT_FC1) == per_half(2, 8, 4)


def test_a_glu_projection_stays_fused_under_a_swiglu_language_model(tmp_path):
    """``encoder.linear_fc1`` is no SwiGLU name: a split language model, a packed projection."""
    vlm = vision_language_tensors(gpt_tensors(LAYERS, HIDDEN, FFN))
    lm_halves, lm_fused = gated_units(vlm, prefix="language_model.")
    halves, _ = gated_units(lm_halves, seed=23, prefix="vision_projection.")
    _, fused = gated_units(lm_fused, seed=23, prefix="vision_projection.")
    source = stack_layers(stack_layers(fused, "language_model.decoder"), "vision_model.decoder")
    src = write_gpt_source(tmp_path / "src", source, args=swiglu_args())
    fsdp, report = to_fsdp(tmp_path, src)
    projection = {k: v for k, v in fused.items() if k.startswith("vision_projection.")}
    rest = {k: v for k, v in halves.items() if not k.startswith("vision_projection.")}
    expect_fsdp_store(fsdp, {**rest, **projection})
    assert projection["vision_projection.encoder.linear_fc1.weight"].shape == (2 * HIDDEN, 4)
    derived = {d["field"]: d["value"] for d in report["derivations"]}
    assert derived["swiglu_split"] == {
        "language_model.decoder": {"split": LAYERS},
        "vision_projection.encoder": {"fused": 1},
    }
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, source)


def test_target_swiglu_may_not_change_the_language_models_activation(tmp_path, capsys):
    """--swiglu makes a job's MLPs SiLU-gated, and a job without it gates them only with
    --quick-geglu (quick-GELU): for a language model with MLPs, --target-swiglu can only say what
    the source's arguments say."""
    swiglu = write_gpt_source(tmp_path / "swiglu", STACKED, args=swiglu_args())
    fsdp, report = to_fsdp(tmp_path, swiglu, name="on", target_swiglu="on")
    expect_fsdp_store(fsdp, HALVES)
    assert not {"swiglu", "quick_geglu"} & set(report["common"]["args_changes"])
    geglu = write_gpt_source(tmp_path / "geglu", STACKED, args=geglu_args())
    assert run("plan", geglu, "--to", "fsdp_dtensor", "--target-swiglu", "off") == 0
    capsys.readouterr()
    assert run("plan", geglu, "--to", "fsdp_dtensor", "--target-swiglu", "on") == 2
    err = capsys.readouterr().err
    assert (
        "error: --target-swiglu on: a job with --swiglu builds SiLU-gated MLPs, but the language "
        "model's are gated otherwise (args.swiglu=False, args.quick_geglu=True), so it builds "
        "another model; a job of this model keeps them fused"
    ) in err
    assert "hint: drop --target-swiglu" in err
    assert run("plan", swiglu, "--to", "fsdp_dtensor", "--target-swiglu", "off") == 2
    assert (
        "error: --target-swiglu off: a job without --swiglu gates MLPs only with --quick-geglu "
        "(quick-GELU), but the language model's MLPs are SwiGLU (args.swiglu=True)"
    ) in capsys.readouterr().err
    plain = write_gpt_source(tmp_path / "plain", stack_layers(gpt_tensors(LAYERS), "decoder"))
    assert run("plan", plain, "--to", "fsdp_dtensor", "--target-swiglu", "on") == 2
    assert "a job with --swiglu builds gated MLPs, but the language model's" in (
        capsys.readouterr().err
    )
    assert run("plan", fsdp, "--to", "torch_dist", "--target-swiglu", "off") == 2


ATTENTION_VLM = vision_language_tensors(hybrid_tensors("**"))
ATTENTION_VIT_HALVES, ATTENTION_VIT_FUSED = gated_units(ATTENTION_VLM, prefix="vision_model.")
"""A hybrid language model of attention layers only (no MLP) under a GLU vision encoder."""


def test_target_swiglu_splits_a_glu_encoder_under_a_language_model_without_mlps(tmp_path):
    """No MLP of the language model changes: ``on`` splits the gated encoder's linear_fc1 as a
    job with --swiglu does, the store's arguments record it, and converting back restores them."""
    source = stack_layers(ATTENTION_VIT_FUSED, "vision_model.decoder")
    src = write_gpt_source(tmp_path / "src", source, args=hybrid_args("**", swiglu=False))
    fsdp, report = to_fsdp(tmp_path, src, target_swiglu="on")
    expect_fsdp_store(fsdp, ATTENTION_VIT_HALVES)
    assert report["common"]["args_changes"]["swiglu"] == [False, True]
    derived = {d["field"]: d["value"] for d in report["derivations"]}
    assert derived["swiglu_split"] == {"vision_model.decoder": {"split": 2}}
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, source)
    assert read_common(out)["args"].swiglu is False
