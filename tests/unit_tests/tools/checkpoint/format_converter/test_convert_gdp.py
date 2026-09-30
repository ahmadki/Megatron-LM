# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Gated DeltaProduct mixers across formats, on synthetic stores.

Every component of a mixer's ``in_proj`` and ``conv1d`` weight (and bias) is
drawn separately (``gdp_mixer_tensors``, whose part order and sizes are
written out from ``GatedDeltaProductMixer``): the torch_dist store holds each
as ``<tensor>.<part>``, and the fsdp_dtensor store Megatron-FSDP writes holds
each tensor whole, as ``tp`` ranks' local tensors concatenated rank after
rank (``pack_by_ranks``: explicit loops over ranks and parts). The expected
stores never come from the converter.
"""

from __future__ import annotations

import pytest
import torch

from tools.checkpoint.format_converter.api import plan_conversion, verify_checkpoint
from tools.checkpoint.format_converter.options import PlanOptions, VerifyOptions

from ._synthetic import (
    dim0_grid,
    expert_grids,
    gated_units,
    gdp_hybrid_tensors,
    gdp_mixer_tensors,
    hybrid_args,
    hybrid_module_tensors,
    hybrid_mtp_tensors,
    layer_grids,
    moe_layer_tensors,
    packed_projections,
    packed_units,
    renumber_layers,
    stack_experts,
    stack_layers,
    swiglu_grids,
    torch_dist_parts,
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
from .test_convert_cross import read_items
from .test_convert_glu import assert_store, to_fsdp, to_torch_dist

pytestmark = pytest.mark.covers("gdp", "ground_truth")
PATTERN = "M*M-"
GDP_SPEC = ["megatron.core.models.hybrid.hybrid_layer_specs", "gated_delta_product_stack_spec"]
MIXER = {"d_inner": 8, "group_state": 12, "heads": 4, "householders": 2}
"""d = 4 heads x 2, GN = 4 groups x 3, H = 4, M = 2: every part divides by 1, 2 and 4 ranks."""
FUSED, OTHERS = gdp_hybrid_tensors(PATTERN, MIXER)
TORCH_DIST = {**torch_dist_parts(FUSED), **OTHERS}
PARTS = sum(len(parts) for parts in FUSED.values())
IN_PROJ = "decoder.layers.0.mixer.in_proj.weight"
SHARDED_BY_MEGATRON_FSDP = (".mixer.A_log", ".mixer.dt_bias", ".mixer.norm.weight")
"""Tensor-parallel without a partition dimension: Megatron-FSDP holds one rank's shard of each."""


def gdp_args(**overrides):
    fields = {
        "spec": GDP_SPEC,
        "mamba_num_heads": 4,
        "mamba_head_dim": 2,
        "mamba_num_groups": 4,
        "mamba_state_dim": 3,
        "gdp_num_householder": 2,
        "pipeline_model_parallel_size": 1,
        **overrides,
    }
    return hybrid_args(PATTERN, **fields)


def fsdp_expected(tp: int, fused=FUSED, others=OTHERS) -> dict[str, torch.Tensor]:
    return {**packed_projections(fused, tp), **others}


def tp_grids(tensors, tp: int) -> dict:
    """Megatron's grid of the parts saved at tensor-parallel size ``tp``: one chunk per rank."""
    return {
        key: dim0_grid(tuple(tensor.shape), tensor.shape[0] // tp)
        for key, tensor in tensors.items()
        if ".mixer.in_proj." in key or ".mixer.conv1d." in key
    }


def derived(report) -> dict:
    return {d["field"]: d["value"] for d in report["derivations"]}


def test_a_gdp_hybrid_is_fused_for_fsdp_dtensor(tmp_path):
    """Packed at tensor-parallel size 1, a fused tensor is its parts' concatenation."""
    src = write_gpt_source(tmp_path / "src", TORCH_DIST, args=gdp_args())
    out, report = to_fsdp(tmp_path, src)
    expected = fsdp_expected(1)
    parts = FUSED[IN_PROJ]
    assert list(parts) == ["z", "V0", "V1", "K0", "K1", "Q", "b0", "b1", "a"]
    assert torch.equal(expected[IN_PROJ], torch.cat(list(parts.values())))
    expect_fsdp_store(out, expected)
    assert report["claims"] == {"gdp": PARTS, "identity": len(OTHERS)}
    assert derived(report)["mixers"] == {"decoder": {"gdp": 2}}
    assert derived(report)["pack_tp"] == 1
    assert "tensor_model_parallel_size" not in report["common"]["args_changes"]


def test_fused_projections_split_back_into_megatrons_parts(tmp_path):
    src = write_gpt_source(tmp_path / "src", TORCH_DIST, args=gdp_args())
    fsdp, _ = to_fsdp(tmp_path, src)
    out, report = to_torch_dist(tmp_path, fsdp)
    assert_store(out, TORCH_DIST)
    assert grid_of(out, f"{IN_PROJ}.K1") == [((0, 0), (12, 8))]
    assert report["claims"]["gdp"] == 4  # the fused in_proj and conv1d of both mixers
    assert verify_checkpoint(out, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok


def test_a_tensor_parallel_source_is_packed_at_the_loading_jobs_size(tmp_path, capsys):
    """Saved at tensor-parallel size 2: each rank's local in_proj is ``[z_r; V0_r; V1_r; K0_r;
    K1_r; Q_r; b0_r; b1_r; a_r]``, so the fsdp_dtensor tensor is no concatenation of the parts."""
    args = gdp_args(tensor_model_parallel_size=2, tensor_parallel_num_weight_shards=2)
    src = write_gpt_source(tmp_path / "src", TORCH_DIST, grids=tp_grids(TORCH_DIST, 2), args=args)
    assert run("plan", src, "--to", "fsdp_dtensor") == 4
    err = capsys.readouterr().err
    assert (
        "error: FC-A01: fsdp_dtensor packs 4 Gated DeltaProduct in_proj/conv1d per rank of the "
        "job that loads it, and the source was saved with args.tensor_model_parallel_size=2"
    ) in err
    fsdp, report = to_fsdp(tmp_path, src, fsdp_tp=2)
    expected = fsdp_expected(2)
    parts = list(FUSED[IN_PROJ].values())
    rank0 = torch.cat([part[: part.shape[0] // 2] for part in parts])
    assert torch.equal(expected[IN_PROJ][:36], rank0)
    assert not torch.equal(expected[IN_PROJ], torch.cat(parts))
    expect_fsdp_store(fsdp, expected)
    run_stats = report["run"]
    assert run_stats["torch_load_calls"] == run_stats["items_touched"]  # each chunk read once
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, TORCH_DIST)


def test_packing_at_another_size_rewrites_and_restores_the_arguments(tmp_path):
    src = write_gpt_source(tmp_path / "src", TORCH_DIST, args=gdp_args())
    fsdp, report = to_fsdp(tmp_path, src, fsdp_tp=4)
    expect_fsdp_store(fsdp, fsdp_expected(4))
    assert report["common"]["args_changes"]["tensor_model_parallel_size"] == [1, 4]
    assert read_items(fsdp)["args"].tensor_model_parallel_size == 4
    out, report = to_torch_dist(tmp_path, fsdp)
    assert_store(out, TORCH_DIST)
    assert read_common(out)["args"].tensor_model_parallel_size == 1
    (pack,) = [d for d in report["derivations"] if d["field"] == "pack_tp"]
    assert pack["value"] == 4


@pytest.mark.parametrize("tp", [1, 2])
def test_a_megatron_fsdp_save_is_unpacked_at_its_size(tmp_path, tp):
    """A training save: fp32 main weights, fused projections with biases packed at ``tp``.

    At ``tp`` 2 the store is hypothetical: it holds every mixer's ``A_log``, ``dt_bias`` and
    ``norm.weight`` whole, which a current Megatron-FSDP job above TP 1 does not (it saves one
    rank's shard of each, refused below)."""
    fused, others = gdp_hybrid_tensors(PATTERN, MIXER, bias=True)
    fp32 = {
        name: {part: value.float() for part, value in parts.items()}
        for name, parts in fused.items()
    }
    store = {key: value.float() for key, value in fsdp_expected(tp, fp32, others).items()}
    args = gdp_args(ckpt_format="fsdp_dtensor", tensor_model_parallel_size=tp, bf16=False)
    fsdp = write_fsdp_store(tmp_path / "fsdp" / ITER, store, args=args)
    out, report = to_torch_dist(tmp_path, fsdp)
    assert_store(out, {**torch_dist_parts(fp32), **{k: v.float() for k, v in others.items()}})
    assert report["claims"]["gdp"] == 8  # in_proj and conv1d, weight and bias, of both mixers


def test_a_megatron_fsdp_save_of_one_ranks_shards_is_refused(tmp_path, capsys):
    """What a current Megatron-FSDP job at TP 2 saves: one rank's shard of every mixer's
    ``A_log``, ``dt_bias`` (H = 4) and ``norm.weight`` (d = 8), which it takes for replicated
    parameters. Half of each is gone: the store is refused, never converted to an incomplete
    torch_dist store."""
    shards = {
        key: value[: value.shape[0] // 2] if key.endswith(SHARDED_BY_MEGATRON_FSDP) else value
        for key, value in OTHERS.items()
    }
    args = gdp_args(ckpt_format="fsdp_dtensor", tensor_model_parallel_size=2)
    fsdp = write_fsdp_store(tmp_path / "fsdp" / ITER, fsdp_expected(2, others=shards), args=args)
    assert run("plan", fsdp, "--to", "torch_dist") == 3
    assert (
        "error: FC-R10: decoder[0].mixer: A_log (2,), dt_bias (2,), norm.weight (4,), but a Gated "
        "DeltaProduct mixer's A_log and dt_bias have 4 rows and its norm.weight 8"
    ) in capsys.readouterr().err


def test_only_a_store_packed_above_tp_1_is_warned_about(tmp_path, caplog):
    """No current Megatron-FSDP job above TP 1 holds the mixers' A_log, dt_bias and norm.weight
    whole, so it cannot load the store."""
    warning = (
        "Gated DeltaProduct mixers packed for --fsdp-tp 2: current Megatron-FSDP jobs above TP 1 "
        "hold one tensor-parallel rank's shard of a mixer's A_log, dt_bias and norm.weight"
    )
    src = write_gpt_source(tmp_path / "src", TORCH_DIST, args=gdp_args())
    assert run("plan", src, "--to", "fsdp_dtensor") == 0
    assert "Gated DeltaProduct mixers packed" not in caplog.text
    assert run("plan", src, "--to", "fsdp_dtensor", "--fsdp-tp", "2") == 0
    assert warning in caplog.text


def test_the_plan_names_the_rule_and_the_mixers(tmp_path, capsys):
    src = write_gpt_source(tmp_path / "src", TORCH_DIST, args=gdp_args())
    assert run("plan", src, "--to", "fsdp_dtensor") == 0
    text = capsys.readouterr().out
    assert f"model tensors claimed: gdp {PARTS}, identity {len(OTHERS)}" in text
    plan = plan_conversion(src, PlanOptions(to_format="fsdp_dtensor")).to_json()
    (mixers,) = [d for d in plan["derivations"] if d["field"] == "mixers"]
    assert mixers["evidence"].startswith(
        f"args.spec={'.'.join(GDP_SPEC)!r} builds Gated DeltaProduct mixers; 2 mixers stored as "
        "parts; d=8 (args.mamba_num_heads=4 x args.mamba_head_dim=2), GN=12 "
        "(args.mamba_num_groups=4 x args.mamba_state_dim=3), H=4, M=2 "
        "(args.gdp_num_householder=2): in_proj 72 rows, conv1d 52 rows"
    )
    consulted = plan["common"]["args_consulted"]
    assert consulted["gdp_num_householder"] == 2 and consulted["mamba_num_heads"] == 4


def test_a_mixer_store_without_the_gdp_spec_is_never_converted(tmp_path, capsys):
    """fsdp_dtensor's fused projections may be a legacy Mamba-2 store's: without the spec the
    store is a pending Mamba-2 model. torch_dist's parts name Gated DeltaProduct, and the spec
    builds Mamba-2 mixers: a contradiction."""
    args = gdp_args(ckpt_format="fsdp_dtensor", spec=None)
    fsdp = write_fsdp_store(tmp_path / "fsdp" / ITER, fsdp_expected(1), args=args)
    assert run("plan", fsdp, "--to", "torch_dist") == 3
    assert "error: FC-P08: not supported yet: Mamba-2 mixers" in capsys.readouterr().err
    src = write_gpt_source(tmp_path / "src", TORCH_DIST, args=gdp_args(spec=None))
    assert run("plan", src, "--to", "fsdp_dtensor") == 3
    err = capsys.readouterr().err
    assert (
        "error: FC-R10: decoder[0].mixer is a Gated DeltaProduct mixer by its in_proj parts z, "
        "V0, V1, K0, K1, Q, b0, b1, a, but args.spec=None builds Mamba-2 mixers"
    ) in err


def test_a_store_of_other_dimensions_is_refused_before_reading(tmp_path, capsys):
    src = write_gpt_source(tmp_path / "src", TORCH_DIST, args=gdp_args(mamba_num_groups=2))
    assert run("plan", src, "--to", "fsdp_dtensor") == 3
    err = capsys.readouterr().err
    assert (
        "error: FC-R10: decoder[0].mixer: its widths (conv1d.weight 52 rows, in_proj.weight 72 "
        "rows) are not what the arguments give Gated DeltaProduct mixers"
    ) in err


ARCHETYPE = "M-*EME-M/*E"
"""The language model of the vision-language archetype: Gated DeltaProduct mixers, dense MLPs,
shortcut-connected mixture-of-experts pairs after an attention layer and after a mixer, and a
multi-token prediction layer holding a pair."""
MODULES = ("M", "-", "*E", "ME", "-", "M")
DENSE_TO_TD = (0, 1, 2, 4, 6, 7)
"""The torch_dist index (pattern position) of each module of its decoder, written out by hand."""
EXPERTS = 4
LANGUAGE_MODEL = "language_model."


def archetype_language_model() -> tuple[dict, dict]:
    """The archetype's language model as fsdp_dtensor stores it (dense module indices, grouped
    experts): its mixers' fused projections by part, and every other tensor."""
    generator = torch.Generator().manual_seed(41)
    fused: dict = {}
    tensors = {
        "embedding.word_embeddings.weight": torch.randn(32, 8, generator=generator).bfloat16(),
        "decoder.final_norm.weight": torch.randn(8, generator=generator),
        "output_layer.weight": torch.randn(32, 8, generator=generator).bfloat16(),
    }
    for dense, symbols in enumerate(MODULES):
        module = f"decoder.layers.{dense}"
        if "M" in symbols:
            mixer = f"{module}.attn_layer.mixer" if len(symbols) == 2 else f"{module}.mixer"
            projections, rest = gdp_mixer_tensors(mixer, seed=10 + dense, **MIXER)
            fused.update(projections)
            tensors.update(rest)
            if len(symbols) == 2:
                tensors.update(
                    moe_layer_tensors(f"{module}.moe_layer", EXPERTS, bias=False, seed=dense)
                )
                norm = torch.randn(8, generator=generator)
                tensors[f"{module}.shortcut_pre_mlp_layernorm.weight"] = norm
        else:
            tensors.update(hybrid_module_tensors(module, symbols, seed=100 * dense))
    tensors.update(hybrid_mtp_tensors(("*E",), layers=1, experts=EXPERTS))
    return fused, tensors


def archetype(tp: int) -> tuple[dict, dict, dict]:
    """The vision-language archetype: its fsdp_dtensor store packed at ``tp`` (fused mixer
    projections and the gated vision encoder's linear_fc1 packed rank after rank), its torch_dist
    store (sparse decoder indices, stacked experts and vision layers, mixer parts) and that
    store's grid (one chunk per layer, expert and SwiGLU half)."""
    fused, language_model = archetype_language_model()
    fused = {LANGUAGE_MODEL + key: parts for key, parts in fused.items()}
    halves, glu = gated_units(vision_language_tensors(language_model), prefix="vision_model.")
    fsdp = {**packed_units(halves, lambda key: tp), **packed_projections(fused, tp)}
    source = {**glu, **torch_dist_parts(fused)}
    source = renumber_layers(source, f"{LANGUAGE_MODEL}decoder", DENSE_TO_TD)
    source = stack_layers(stack_experts(source), "vision_model.decoder")
    vision = {key: tensor for key, tensor in source.items() if key.startswith("vision_model.")}
    grids = {
        **layer_grids(source, "vision_model.decoder"),
        **expert_grids(source),
        **swiglu_grids(vision),
    }
    return fsdp, source, grids


def archetype_args(tp: int, **overrides):
    return gdp_args(
        hybrid_layer_pattern=ARCHETYPE,
        num_layers=8,
        mtp_num_layers=1,
        moe_shortcut_connection=True,
        num_experts=EXPERTS,
        tensor_model_parallel_size=tp,
        **overrides,
    )


@pytest.mark.parametrize("tp", [1, 2])
def test_the_vision_language_archetype_round_trips(tmp_path, tp):
    """Every family at once: the pair's mixer is found under its opaque child, the sparse indices
    map to dense ones, experts split, the vision layers unstack, and both the gated vision
    linear_fc1 and the mixers' projections are packed at the loading job's size."""
    fsdp_expected, source, grids = archetype(tp)
    src = write_gpt_source(tmp_path / "src", source, grids=grids, args=archetype_args(tp))
    fsdp, report = to_fsdp(tmp_path, src, fsdp_tp=tp)
    expect_fsdp_store(fsdp, fsdp_expected)
    assert "language_model.decoder.layers.3.attn_layer.mixer.in_proj.weight" in fsdp_expected
    assert "language_model.decoder.layers.4.attn_layer.mixer.conv1d.weight.K1" in source
    values = derived(report)
    assert values["mixers"] == {"language_model.decoder": {"gdp": 3}}
    assert values["swiglu_split"] == {"vision_model.decoder": {"fused": 2}}
    assert values["pack_tp"] == tp
    assert report["claims"]["gdp"] == 3 * 14  # 9 in_proj and 5 conv1d parts per mixer
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, source)
    assert verify_checkpoint(out, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok
