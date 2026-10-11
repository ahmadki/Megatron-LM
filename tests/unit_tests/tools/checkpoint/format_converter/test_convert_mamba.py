# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Mamba-2 mixers across formats, on synthetic stores.

Every component of a mixer's ``in_proj`` and ``conv1d`` weight (and bias) is
drawn separately (``mamba_mixer_tensors``, whose part order and sizes are
written out from ``MambaMixer``): the torch_dist store holds each as
``<tensor>.<part>`` (``conv1d.weight.x``), except an ``in_proj.bias``, which
``MambaMixer.sharded_state_dict`` would not split: it holds the ``tp`` ranks'
local biases one after the other (``MambaMixer`` asserts ``bias=False``, so
only a mixer built otherwise has one). The fsdp_dtensor store Megatron-FSDP
writes holds each tensor whole, packed rank after rank (``pack_by_ranks``:
explicit loops over ranks and parts), and names the convolution
``conv1d_weight``/``conv1d_bias`` (``mamba_fsdp_names``). The expected stores
never come from the converter.
"""

from __future__ import annotations

import pytest
import torch

from tools.checkpoint.format_converter.api import plan_conversion, verify_checkpoint
from tools.checkpoint.format_converter.options import PlanOptions, VerifyOptions

from ._synthetic import (
    dim0_grid,
    hybrid_args,
    mamba_fsdp_names,
    mamba_hybrid_tensors,
    pack_by_ranks,
    packed_projections,
    torch_dist_parts,
    write_fsdp_store,
)
from .test_convert_blocks import ITER, expect_fsdp_store, grid_of, run, write_gpt_source
from .test_convert_cross import read_items
from .test_convert_glu import assert_store, to_fsdp, to_torch_dist

pytestmark = pytest.mark.covers("mamba2", "ground_truth")
PATTERN = "M*M-"
MIXER = {"d_inner": 8, "group_state": 12, "heads": 4}
"""d = 4 heads x 2, GN = 4 groups x 3, H = 4: every part divides by 1, 2 and 4 ranks."""
DIMS = {"mamba_num_heads": 4, "mamba_head_dim": 2, "mamba_num_groups": 4, "mamba_state_dim": 3}
FUSED, OTHERS = mamba_hybrid_tensors(PATTERN, MIXER)
PARTS = sum(len(parts) for parts in FUSED.values())
IN_PROJ = "decoder.layers.0.mixer.in_proj.weight"
CONV = "decoder.layers.0.mixer.conv1d.weight"
IN_PROJ_BIAS = "decoder.layers.0.mixer.in_proj.bias"


def mamba_args(**overrides):
    return hybrid_args(PATTERN, **{**DIMS, "pipeline_model_parallel_size": 1, **overrides})


def torch_dist_store(fused=FUSED, others=OTHERS, tp: int = 1) -> dict[str, torch.Tensor]:
    """What Megatron's torch_dist save at tensor-parallel size ``tp`` holds: every part of the
    fused tensors, but ``in_proj.bias`` whole, its ``tp`` ranks' local biases one after the
    other."""
    split = {name: parts for name, parts in fused.items() if not name.endswith("in_proj.bias")}
    whole = {name: pack_by_ranks(tuple(parts.values()), tp) for name, parts in fused.items()}
    return {
        **torch_dist_parts(split),
        **{name: tensor for name, tensor in whole.items() if name not in split},
        **others,
    }


def fsdp_expected(tp: int, fused=FUSED, others=OTHERS) -> dict[str, torch.Tensor]:
    return mamba_fsdp_names({**packed_projections(fused, tp), **others})


def tp_grids(tensors, tp: int) -> dict:
    """Megatron's grid of the mixer tensors saved at tensor-parallel size ``tp``: one chunk per
    rank."""
    return {
        key: dim0_grid(tuple(tensor.shape), tensor.shape[0] // tp)
        for key, tensor in tensors.items()
        if ".mixer.in_proj." in key or ".mixer.conv1d." in key
    }


def derived(report) -> dict:
    return {d["field"]: d["value"] for d in report["derivations"]}


def test_a_mamba_hybrid_is_fused_for_fsdp_dtensor(tmp_path):
    """Packed at tensor-parallel size 1, a fused tensor is its parts' concatenation, and the
    convolution is the mixer's own conv1d_weight and conv1d_bias."""
    src = write_gpt_source(tmp_path / "src", torch_dist_store(), args=mamba_args())
    out, report = to_fsdp(tmp_path, src)
    expected = fsdp_expected(1)
    assert list(FUSED[IN_PROJ]) == ["z", "x", "B", "C", "dt"]
    assert list(FUSED[CONV]) == ["x", "B", "C"]
    assert torch.equal(expected[IN_PROJ], torch.cat(list(FUSED[IN_PROJ].values())))
    conv = expected["decoder.layers.0.mixer.conv1d_weight"]
    assert torch.equal(conv, torch.cat(list(FUSED[CONV].values())))
    assert "decoder.layers.2.mixer.conv1d_bias" in expected and CONV not in expected
    expect_fsdp_store(out, expected)
    assert report["claims"] == {"mamba2": PARTS, "identity": len(OTHERS)}
    assert derived(report)["mixers"] == {"decoder": {"mamba2": 2}}
    assert derived(report)["pack_tp"] == 1


def test_fused_projections_split_back_into_megatrons_parts(tmp_path):
    src = write_gpt_source(tmp_path / "src", torch_dist_store(), args=mamba_args())
    fsdp, _ = to_fsdp(tmp_path, src)
    out, report = to_torch_dist(tmp_path, fsdp)
    assert_store(out, torch_dist_store())
    assert grid_of(out, f"{CONV}.B") == [((0, 0, 0), (12, 1, 4))]
    assert report["claims"]["mamba2"] == 6  # the fused in_proj and conv1d weight and bias, x 2
    assert verify_checkpoint(out, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok


def test_a_tensor_parallel_source_is_packed_at_the_loading_jobs_size(tmp_path, capsys):
    """Saved at tensor-parallel size 2: each rank's local in_proj is ``[z_r; x_r; B_r; C_r;
    dt_r]`` and its conv1d ``[x_r; B_r; C_r]``, so the fsdp_dtensor tensors are no concatenation
    of the parts."""
    store = torch_dist_store()
    args = mamba_args(tensor_model_parallel_size=2, tensor_parallel_num_weight_shards=2)
    src = write_gpt_source(tmp_path / "src", store, grids=tp_grids(store, 2), args=args)
    assert run("plan", src, "--to", "fsdp_dtensor") == 4
    assert (
        "error: FC-A01: fsdp_dtensor packs 6 Mamba-2 in_proj/conv1d per rank of the job that "
        "loads it, and the source was saved with args.tensor_model_parallel_size=2"
    ) in capsys.readouterr().err
    fsdp, report = to_fsdp(tmp_path, src, fsdp_tp=2)
    expected = fsdp_expected(2)
    parts = list(FUSED[IN_PROJ].values())
    rank0 = torch.cat([part[: part.shape[0] // 2] for part in parts])
    assert torch.equal(expected[IN_PROJ][:22], rank0)
    assert not torch.equal(expected[IN_PROJ], torch.cat(parts))
    expect_fsdp_store(fsdp, expected)
    run_stats = report["run"]
    assert run_stats["torch_load_calls"] == run_stats["items_touched"]  # each chunk read once
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, store)


def test_packing_at_another_size_rewrites_and_restores_the_arguments(tmp_path):
    src = write_gpt_source(tmp_path / "src", torch_dist_store(), args=mamba_args())
    fsdp, report = to_fsdp(tmp_path, src, fsdp_tp=4)
    expect_fsdp_store(fsdp, fsdp_expected(4))
    assert report["common"]["args_changes"]["tensor_model_parallel_size"] == [1, 4]
    assert read_items(fsdp)["args"].tensor_model_parallel_size == 4
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, torch_dist_store())


@pytest.mark.parametrize("tp", [1, 2])
def test_an_in_proj_bias_converts_as_is_at_the_size_it_was_saved_at(tmp_path, tp):
    """TP-bound identity: torch_dist stores an in_proj.bias whole, the ranks' local biases
    ``[z_r; x_r; B_r; C_r; dt_r]`` one after the other, as fsdp_dtensor does at the same size."""
    fused, others = mamba_hybrid_tensors(PATTERN, MIXER, bias=True)
    store = torch_dist_store(fused, others, tp)
    assert IN_PROJ_BIAS in store and f"{IN_PROJ_BIAS}.z" not in store
    args = mamba_args(tensor_model_parallel_size=tp, tensor_parallel_num_weight_shards=tp)
    src = write_gpt_source(tmp_path / "src", store, grids=tp_grids(store, tp), args=args)
    fsdp, report = to_fsdp(tmp_path, src, fsdp_tp=tp)
    expected = fsdp_expected(tp, fused, others)
    assert torch.equal(expected[IN_PROJ_BIAS], store[IN_PROJ_BIAS])
    expect_fsdp_store(fsdp, expected)
    assert report["claims"]["mamba2"] == 2 * (5 + 1 + 3 + 3)
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, store)


@pytest.mark.parametrize(("saved", "packed"), [(2, 1), (2, 4), (1, 2)])
def test_an_in_proj_bias_is_never_packed_at_another_size(tmp_path, capsys, saved, packed):
    fused, others = mamba_hybrid_tensors(PATTERN, MIXER, bias=True)
    store = torch_dist_store(fused, others, saved)
    args = mamba_args(tensor_model_parallel_size=saved, tensor_parallel_num_weight_shards=saved)
    src = write_gpt_source(tmp_path / "src", store, grids=tp_grids(store, saved), args=args)
    assert run("plan", src, "--to", "fsdp_dtensor", "--fsdp-tp", packed) == 3
    assert (
        "error: FC-R10: 2 Mamba-2 mixer in_proj.bias tensors (such as decoder[0].mixer.in_proj."
        f"bias) are stored whole, as the {saved} tensor-parallel ranks' local tensors one after "
        f"the other, which no fsdp_dtensor store packed at {packed} holds"
    ) in capsys.readouterr().err


@pytest.mark.parametrize("tp", [1, 2])
def test_a_megatron_fsdp_save_is_unpacked_at_its_size(tmp_path, tp):
    """A training save: fp32 main weights, fused projections packed at ``tp``."""
    fused, others = mamba_hybrid_tensors(PATTERN, MIXER)
    fp32 = {
        name: {part: value.float() for part, value in parts.items()}
        for name, parts in fused.items()
    }
    others = {key: value.float() for key, value in others.items()}
    store = {key: value.float() for key, value in fsdp_expected(tp, fp32, others).items()}
    args = mamba_args(ckpt_format="fsdp_dtensor", tensor_model_parallel_size=tp, bf16=False)
    fsdp = write_fsdp_store(tmp_path / "fsdp" / ITER, store, args=args)
    out, report = to_torch_dist(tmp_path, fsdp)
    assert_store(out, torch_dist_store(fp32, others, tp))
    assert report["claims"]["mamba2"] == 6  # in_proj.weight, conv1d weight and bias, x 2


def test_a_store_saved_before_the_mixer_owned_its_convolution_is_read(tmp_path):
    """Megatron-FSDP once saved the convolution as the ``conv1d`` module's ``weight`` and
    ``bias``: without the Gated DeltaProduct spec those are a Mamba-2 mixer's."""
    legacy = {**packed_projections(FUSED, 1), **OTHERS}
    assert CONV in legacy
    fsdp = write_fsdp_store(
        tmp_path / "fsdp" / ITER, legacy, args=mamba_args(ckpt_format="fsdp_dtensor")
    )
    out, report = to_torch_dist(tmp_path, fsdp)
    assert_store(out, torch_dist_store())
    assert derived(report)["mixers"] == {"decoder": {"mamba2": 2}}


def test_a_convolution_stored_under_both_names_is_refused(tmp_path, capsys):
    store = fsdp_expected(1)
    store[CONV] = store["decoder.layers.0.mixer.conv1d_weight"]
    fsdp = write_fsdp_store(
        tmp_path / "fsdp" / ITER, store, args=mamba_args(ckpt_format="fsdp_dtensor")
    )
    assert run("plan", fsdp, "--to", "torch_dist") == 3
    assert (
        "error: FC-R10: decoder[0].mixer: conv1d.weight is stored twice, as "
        "model.module.decoder.layers.0.mixer.conv1d.weight and "
        "model.module.decoder.layers.0.mixer.conv1d_weight"
    ) in capsys.readouterr().err


def test_the_heads_default_to_twice_the_hidden_size_over_the_head_dimension(tmp_path):
    """Without ``mamba_num_heads``, MambaMixer's inner size is ``2 x hidden_size`` = 16 and its
    heads ``16 / mamba_head_dim`` = 8."""
    fused, others = mamba_hybrid_tensors(PATTERN, {"d_inner": 16, "group_state": 12, "heads": 8})
    store = torch_dist_store(fused, others)
    src = write_gpt_source(tmp_path / "src", store, args=mamba_args(mamba_num_heads=None))
    plan = plan_conversion(src, PlanOptions(to_format="fsdp_dtensor")).to_json()
    (mixers,) = [d for d in plan["derivations"] if d["field"] == "mixers"]
    assert "d=16, GN=12, H=8 (args.hidden_size=8 x 2 / args.mamba_head_dim)" in mixers["evidence"]
    fsdp, _ = to_fsdp(tmp_path, src)
    expect_fsdp_store(fsdp, fsdp_expected(1, fused, others))


def test_the_plan_names_the_rule_and_the_mixers(tmp_path, capsys):
    src = write_gpt_source(tmp_path / "src", torch_dist_store(), args=mamba_args())
    assert run("plan", src, "--to", "fsdp_dtensor") == 0
    assert (
        f"model tensors claimed: identity {len(OTHERS)}, mamba2 {PARTS}" in capsys.readouterr().out
    )
    plan = plan_conversion(src, PlanOptions(to_format="fsdp_dtensor")).to_json()
    (mixers,) = [d for d in plan["derivations"] if d["field"] == "mixers"]
    assert mixers["evidence"].startswith(
        "args.spec=None builds Mamba-2 mixers; 2 mixers stored as parts; d=8, GN=12, H=4 "
        "(args.mamba_num_heads=4): in_proj 44 rows, conv1d 32 rows, the only family whose widths "
        "the stored ones match"
    )


def test_a_store_of_other_dimensions_is_refused_before_reading(tmp_path, capsys):
    src = write_gpt_source(
        tmp_path / "src", torch_dist_store(), args=mamba_args(mamba_num_groups=2)
    )
    assert run("plan", src, "--to", "fsdp_dtensor") == 3
    assert (
        "error: FC-R10: decoder[0].mixer: its widths (conv1d.bias 32 rows, conv1d.weight 32 rows, "
        "in_proj.weight 44 rows) are not what the arguments give Mamba-2 mixers"
    ) in capsys.readouterr().err


def test_one_ranks_shard_of_the_per_head_parameters_is_refused(tmp_path, capsys):
    """A store that holds half of every mixer's ``A_log`` (H = 4) cannot be completed."""
    shards = {
        key: value[:2] if key.endswith(".mixer.A_log") else value for key, value in OTHERS.items()
    }
    args = mamba_args(ckpt_format="fsdp_dtensor", tensor_model_parallel_size=2)
    fsdp = write_fsdp_store(tmp_path / "fsdp" / ITER, fsdp_expected(2, others=shards), args=args)
    assert run("plan", fsdp, "--to", "torch_dist") == 3
    assert (
        "error: FC-R10: decoder[0].mixer: A_log (2,), but a Mamba-2 mixer's A_log and dt_bias have "
        "4 rows, its D 4 or 8 and its norm.weight 8"
    ) in capsys.readouterr().err
