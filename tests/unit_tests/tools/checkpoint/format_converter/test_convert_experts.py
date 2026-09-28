# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Routed experts of mixture-of-experts layers across formats, on synthetic stores.

Every expert's tensors are drawn one by one (named as fsdp_dtensor names
grouped experts) and the torch_dist source is built from them with
``torch.stack`` over experts, then over layers, so the expected tensors of
either format are the drawn ones; keys, chunk grids and derivations are
written out by hand.
"""

from __future__ import annotations

import argparse
import re

import pytest
import torch

from tools.checkpoint.format_converter.api import convert_checkpoint, verify_checkpoint
from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.options import ConvertOptions, VerifyOptions

from ._synthetic import (
    expert_grids,
    gpt_args,
    gpt_moe_tensors,
    hybrid_args,
    layer_grids,
    product_grid,
    sequential_experts,
    stack_experts,
    stack_layers,
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

pytestmark = pytest.mark.covers("identity", "ground_truth")
LAYERS, EXPERTS = 2, 4
GROUPED = gpt_moe_tensors(LAYERS, EXPERTS)
"""The GPT-MoE model one expert per key, as fsdp_dtensor names grouped experts."""
STACKED = stack_layers(stack_experts(GROUPED), "decoder")
"""The same model as torch_dist stores it: ``[L, E, ...]`` expert tensors."""
FC1 = "decoder.layers.mlp.experts.experts.linear_fc1.weight"


def moe_args(**overrides) -> argparse.Namespace:
    return gpt_args(LAYERS, **{"num_experts": EXPERTS, "moe_grouped_gemm": True, **overrides})


def stacked_grids() -> dict:
    """One chunk per (layer, expert) as Megatron saves them; fc1 split in two row halves."""
    grids = {**layer_grids(STACKED, "decoder"), **expert_grids(STACKED)}
    grids[FC1] = product_grid((LAYERS, EXPERTS, 6, 8), [[1], [1, 2, 3], [3], []])
    return grids


def fsdp_to_td(tmp_path, store, name="td", **options) -> tuple:
    report = convert_checkpoint(store, tmp_path / name / ITER, ConvertOptions(workers=0, **options))
    return tmp_path / name / ITER, report


@pytest.mark.parametrize(
    ("grouped_gemm", "expected"), [(True, GROUPED), (False, sequential_experts(GROUPED))]
)
def test_a_stacked_moe_store_converts_to_one_tensor_per_expert(tmp_path, grouped_gemm, expected):
    src = write_gpt_source(
        tmp_path / "src",
        STACKED,
        grids=stacked_grids(),
        args=moe_args(moe_grouped_gemm=grouped_gemm),
    )
    report = convert_checkpoint(
        src, tmp_path / "fsdp", ConvertOptions(to_format="fsdp_dtensor", workers=0)
    ).to_json()
    expect_fsdp_store(tmp_path / "fsdp" / ITER, expected)
    derived = {d["field"]: d for d in report["derivations"]}
    form = "grouped" if grouped_gemm else "sequential"
    assert derived["experts"]["value"] == {
        "num_experts": EXPERTS,
        "fsdp_form": form,
        "grouped_gemm": grouped_gemm,
    }
    assert report["claims"] == {"identity": len(expected)}
    run_stats = report["run"]
    assert run_stats["torch_load_calls"] == run_stats["items_touched"]  # each chunk read once
    assert run_stats["items_touched"] == sum(len(g) for g in stacked_grids().values()) + 3
    assert "moe_grouped_gemm" not in report["common"]["args_changes"]


def test_a_stacked_moe_store_round_trips_one_chunk_per_expert(tmp_path):
    src = write_gpt_source(tmp_path / "src", STACKED, grids=stacked_grids(), args=moe_args())
    convert_checkpoint(src, tmp_path / "fsdp", ConvertOptions(to_format="fsdp_dtensor", workers=0))
    out, report = fsdp_to_td(tmp_path, tmp_path / "fsdp" / ITER)
    assert set(MetadataView.load(out).tensors) == set(STACKED)
    for key, tensor in read_tensors(out).items():
        assert bitwise_equal(tensor, STACKED[key]), key
    # Megatron's own grid: one chunk per (layer, expert), the row split merged
    assert grid_of(out, FC1) == [
        ((layer, expert, 0, 0), (1, 1, 6, 8))
        for layer in range(LAYERS)
        for expert in range(EXPERTS)
    ]
    assert grid_of(out, "decoder.layers.mlp.router.expert_bias") == [
        ((layer, 0), (1, EXPERTS)) for layer in range(LAYERS)
    ]
    (block,) = [d for d in report.plan["derivations"] if d["field"] == "blocks.decoder"]
    assert block["value"]["td_layout"] == "stacked"
    assert verify_checkpoint(out, src / ITER, VerifyOptions(expected_diff="D")).ok


def test_expert_naming_chooses_the_fsdp_names_and_the_arguments_follow(tmp_path, caplog):
    src = write_gpt_source(tmp_path / "src", STACKED, grids=stacked_grids(), args=moe_args())
    fsdp = tmp_path / "fsdp"
    argv = ["convert", src, fsdp, "--to", "fsdp_dtensor", "--workers", "0"]
    assert run(*argv, "--expert-naming", "sequential") == 0
    expect_fsdp_store(fsdp / ITER, sequential_experts(GROUPED))
    assert "grouped expert biases" not in caplog.text  # sequential experts name them globally
    assert read_items(fsdp / ITER)["args"].moe_grouped_gemm is False
    out, _ = fsdp_to_td(tmp_path, fsdp / ITER)  # the store's arguments build SequentialMLP
    assert read_common(out)["args"].moe_grouped_gemm is True  # restored from the manifest
    assert verify_checkpoint(out, src / ITER, VerifyOptions(expected_diff="D")).ok


def test_grouped_expert_biases_are_right_only_without_expert_parallelism(tmp_path, caplog):
    src = write_gpt_source(tmp_path / "src", STACKED, args=moe_args())
    assert run("plan", src, "--to", "fsdp_dtensor") == 0
    assert (
        "grouped expert biases (2 tensors such as decoder.layers.mlp.experts.experts.linear_fc1."
        "bias): Megatron-FSDP names them by their expert-parallel-local index" in caplog.text
    )


def test_expert_parallel_grouped_biases_are_refused_in_both_directions(tmp_path, capsys):
    ep2 = moe_args(expert_model_parallel_size=2)
    src = write_gpt_source(tmp_path / "src", STACKED, args=ep2)
    assert run("plan", src, "--to", "fsdp_dtensor") == 3
    assert "error: FC-R05: grouped expert biases with args.expert_model_parallel_size=2" in (
        capsys.readouterr().err
    )
    assert run("plan", src, "--to", "fsdp_dtensor", "--expert-naming", "sequential") == 0
    fsdp_args = moe_args(expert_model_parallel_size=2, ckpt_format="fsdp_dtensor")
    local = {
        key: tensor
        for key, tensor in GROUPED.items()
        if not re.search(rf"\.experts\.linear_fc\d\.bias[{EXPERTS // 2}-9]$", key)
    }  # what Megatron-FSDP saves: the biases under the local indices of expert-parallel rank 0
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, local, args=fsdp_args)
    capsys.readouterr()
    assert run("plan", store, "--to", "torch_dist") == 3
    assert "error: FC-R05:" in capsys.readouterr().err


def test_moe_stores_saved_with_singleton_local_shards_are_refused(tmp_path, capsys):
    """Megatron then saves every expert under a key of its own (``experts.experts.<e>.<name>``)
    and every layer per layer: refused before planning, from content_metadata or the keys."""
    singleton = {
        re.sub(
            r"\.experts\.(linear_fc\d)\.(weight|bias)(\d+)$", r".experts.experts.\3.\1.\2", key
        ): t
        for key, t in GROUPED.items()
    }
    assert "decoder.layers.1.mlp.experts.experts.3.linear_fc2.bias" in singleton
    content = {"singleton_local_shards": True}
    flagged = write_gpt_source(
        tmp_path / "flagged", singleton, args=moe_args(), content_metadata=content
    )
    assert run("plan", flagged, "--to", "fsdp_dtensor") == 3
    err = capsys.readouterr().err
    assert "error: FC-R17: the checkpoint was saved with singleton_local_shards" in err
    assert "[content_metadata, decoder.layers.0.mlp.experts.experts.0.linear_fc1.bias" in err
    unflagged = write_gpt_source(tmp_path / "unflagged", singleton, args=moe_args())
    assert run("plan", unflagged, "--to", "fsdp_dtensor") == 3
    assert "(its experts are saved one key per expert)" in capsys.readouterr().err


def test_an_fsdp_moe_training_save_converts_to_a_stacked_store(tmp_path):
    """Megatron-FSDP stores fp32 main weights; the router's expert bias is fp32 anyway."""
    fp32 = {key: tensor.float() for key, tensor in sequential_experts(GROUPED).items()}
    args = moe_args(moe_grouped_gemm=False, ckpt_format="fsdp_dtensor", use_megatron_fsdp=True)
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, fp32, args=args)
    out, report = fsdp_to_td(tmp_path, store)
    expected = stack_layers(
        stack_experts({key: t.float() for key, t in GROUPED.items()}), "decoder"
    )
    assert set(MetadataView.load(out).tensors) == set(expected)
    for key, tensor in read_tensors(out).items():
        assert bitwise_equal(tensor, expected[key]), key
    assert grid_of(out, FC1) == [
        ((layer, expert, 0, 0), (1, 1, 6, 8))
        for layer in range(LAYERS)
        for expert in range(EXPERTS)
    ]
    (experts,) = [d for d in report.plan["derivations"] if d["field"] == "experts"]
    assert experts["evidence"] == (
        "args.num_experts=4; 32 fsdp_dtensor tensors: 8 expert weights and biases for each of "
        "experts 0-3; the store names them sequential (args.moe_grouped_gemm=False builds "
        "SequentialMLP experts)"
    )


def test_expert_names_of_an_fsdp_source_must_match_its_arguments(tmp_path, capsys):
    store = write_fsdp_store(
        tmp_path / "fsdp" / ITER, GROUPED, args=moe_args(moe_grouped_gemm=False)
    )
    assert run("plan", store, "--to", "torch_dist") == 4
    err = capsys.readouterr().err
    assert "error: FC-A04: the store names its routed experts grouped (TEGroupedMLP)" in err
    assert "hint: pass --expert-naming grouped if the store is right" in err
    assert run("plan", store, "--to", "torch_dist", "--expert-naming", "grouped") == 0
    missing = {k: v for k, v in GROUPED.items() if not k.endswith("linear_fc2.weight3")}
    partial = write_fsdp_store(tmp_path / "partial" / ITER, missing, args=moe_args())
    capsys.readouterr()
    assert run("plan", partial, "--to", "torch_dist") == 3
    assert (
        "error: FC-R13: args.num_experts=4, but the store holds decoder.layers.0.mlp.experts."
        "linear_fc2.weight for 3 experts (0-2)"
    ) in capsys.readouterr().err
    gapped = {k: v for k, v in GROUPED.items() if not re.search(r"\.(weight|bias)2$", k)}
    holed = write_fsdp_store(tmp_path / "holed" / ITER, gapped, args=moe_args())
    assert run("plan", holed, "--to", "torch_dist") == 3
    assert (
        "error: FC-R12: the store holds decoder.layers.0.mlp.experts.linear_fc1.bias for experts "
        "0-1, 3, not 0-3"
    ) in capsys.readouterr().err


def test_a_per_layer_moe_decoder_round_trips(tmp_path):
    """``--moe-layer-freq 2``: dense layer 0, MoE layer 1; torch_dist stores them per layer."""
    per_layer = gpt_moe_tensors(LAYERS, EXPERTS, moe_layers=[1])
    source = stack_experts(per_layer)
    assert source["decoder.layers.1.mlp.experts.experts.linear_fc2.weight"].shape == (4, 8, 6)
    content = {"non_homogeneous_layers": True, "singleton_local_shards": False}
    args = moe_args(moe_layer_freq=2)
    src = write_gpt_source(
        tmp_path / "src", source, grids=expert_grids(source), args=args, content_metadata=content
    )
    convert_checkpoint(src, tmp_path / "fsdp", ConvertOptions(to_format="fsdp_dtensor", workers=0))
    expect_fsdp_store(tmp_path / "fsdp" / ITER, per_layer)
    out, report = fsdp_to_td(tmp_path, tmp_path / "fsdp" / ITER)
    assert set(MetadataView.load(out).tensors) == set(source)
    assert grid_of(out, "decoder.layers.1.mlp.experts.experts.linear_fc2.bias") == [
        ((expert, 0), (1, 8)) for expert in range(EXPERTS)
    ]
    (block,) = [d for d in report.plan["derivations"] if d["field"] == "blocks.decoder"]
    assert block["value"]["td_layout"] == "per_layer"
    assert verify_checkpoint(out, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok


def hybrid_moe_tensors(*, bias: bool) -> dict[str, torch.Tensor]:
    """A hybrid ``*E*E`` model one expert per key, as fsdp_dtensor names grouped experts."""
    hybrid = {
        key.replace("decoder.layers.1.", "decoder.layers.3.").replace(
            "decoder.layers.0.", "decoder.layers.1."
        ): tensor
        for key, tensor in gpt_moe_tensors(LAYERS, EXPERTS, bias=bias).items()
        if ".mlp." in key or "pre_mlp" in key or not key.startswith("decoder.layers.")
    }
    attention = {
        key: tensor
        for key, tensor in gpt_moe_tensors(LAYERS, EXPERTS).items()
        if ".self_attention." in key
    }
    hybrid.update(
        {key.replace("decoder.layers.1.", "decoder.layers.2."): t for key, t in attention.items()}
    )
    return hybrid


def test_a_hybrid_moe_model_names_its_experts_grouped(tmp_path, capsys):
    """The hybrid stack spec builds TEGroupedMLP experts whatever ``moe_grouped_gemm`` says."""
    hybrid = hybrid_moe_tensors(bias=False)
    source = stack_experts(hybrid)
    args = hybrid_args("*E|*E", num_experts=EXPERTS, moe_grouped_gemm=False)
    src = write_gpt_source(tmp_path / "src", source, grids=expert_grids(source), args=args)
    assert run("plan", src, "--to", "fsdp_dtensor", "--expert-naming", "sequential") == 2
    assert (
        "error: --expert-naming sequential: the hybrid stack spec builds TEGroupedMLP experts "
        "whatever the arguments say, which names them grouped, so no job built from these "
        "arguments loads experts named sequential hint: drop --expert-naming"
    ) in capsys.readouterr().err
    report = convert_checkpoint(
        src, tmp_path / "fsdp", ConvertOptions(to_format="fsdp_dtensor", workers=0)
    ).to_json()
    expect_fsdp_store(tmp_path / "fsdp" / ITER, hybrid)
    (experts,) = [d for d in report["derivations"] if d["field"] == "experts"]
    assert experts["value"] == {
        "num_experts": EXPERTS,
        "fsdp_form": "grouped",
        "grouped_gemm": None,
    }
    assert "moe_grouped_gemm" not in report["common"]["args_changes"]
    out, _ = fsdp_to_td(tmp_path, tmp_path / "fsdp" / ITER)
    for key, tensor in read_tensors(out).items():
        assert bitwise_equal(tensor, source[key]), key
    assert verify_checkpoint(out, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok


def test_hybrid_grouped_expert_biases_have_no_sequential_way_out(tmp_path, caplog, capsys):
    """Sequential experts would name expert biases globally, but a hybrid model never builds
    them: its grouped biases load right only at expert-parallel size 1."""
    source = stack_experts(hybrid_moe_tensors(bias=True))
    args = hybrid_args("*E*E", num_experts=EXPERTS)
    src = write_gpt_source(tmp_path / "ep1", source, args=args)
    assert run("plan", src, "--to", "fsdp_dtensor") == 0
    assert (
        "so a job loads them right only with --expert-model-parallel-size 1 (the hybrid stack "
        "spec always builds grouped experts)"
    ) in caplog.text
    assert "--expert-naming" not in caplog.text
    ep2 = hybrid_args("*E*E", num_experts=EXPERTS, expert_model_parallel_size=2)
    src = write_gpt_source(tmp_path / "ep2", source, args=ep2)
    for naming in ("auto", "sequential"):
        capsys.readouterr()
        assert run("plan", src, "--to", "fsdp_dtensor", "--expert-naming", naming) == 3
        assert "error: FC-R05: grouped expert biases" in capsys.readouterr().err
