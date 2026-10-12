# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Gated DeltaNets (GDN and GDN2) across formats, on synthetic stores.

Every component of a Gated DeltaNet's ``in_proj`` and ``conv1d`` weight (and
bias) is drawn separately (``gdn_tensors``, whose part order and sizes are
written out from ``GatedDeltaNet`` and ``GatedDeltaNet2``): the torch_dist
store holds each as ``<tensor>.<part>`` (``self_attention.in_proj.weight.beta``),
except an ``in_proj.bias``, which ``_GDNBase.sharded_state_dict`` would not
split: it holds the ``tp`` ranks' local biases one after the other (no layer
spec builds one; a module built with ``bias=True`` has one). The fsdp_dtensor
store Megatron-FSDP writes holds each tensor whole, packed rank after rank
(``pack_by_ranks``: explicit loops over ranks and parts). The models are hybrid
models of ``G`` layers, and GPT models whose linear-attention layers are Gated
DeltaNets. The expected stores never come from the converter.
"""

from __future__ import annotations

import re

import pytest
import torch

from tools.checkpoint.format_converter.api import plan_conversion, verify_checkpoint
from tools.checkpoint.format_converter.options import PlanOptions, VerifyOptions

from ._synthetic import (
    dim0_grid,
    gdn_hybrid_tensors,
    gdn_tensors,
    gdp_mixer_tensors,
    gpt_args,
    gpt_tensors,
    hybrid_args,
    mixer_builder,
    mixer_hybrid_tensors,
    pack_by_ranks,
    packed_projections,
    torch_dist_parts,
    write_fsdp_store,
)
from .test_convert_blocks import ITER, expect_fsdp_store, grid_of, run, write_gpt_source
from .test_convert_cross import read_items
from .test_convert_glu import assert_store, to_fsdp, to_torch_dist

pytestmark = [pytest.mark.covers("gdn", "ground_truth"), pytest.mark.covers("gdn2", "ground_truth")]
PATTERN = "G*G-"
GDN = {"key_heads": 4, "key_dim": 3, "value_heads": 4, "value_dim": 2}
"""qk = 4 x 3 = 12, v = 4 x 2 = 8, 4 key and 4 value heads: every part divides by 1, 2 and 4
ranks. A Gated DeltaNet's in_proj has 2 x 12 + 2 x 8 + 2 x 4 = 48 rows, a Gated DeltaNet 2's
4 x 12 + 3 x 8 = 72; both convolutions 2 x 12 + 8 = 32."""
LINEAR = {
    "linear_key_head_dim": 3,
    "linear_num_key_heads": 4,
    "linear_value_head_dim": 2,
    "linear_num_value_heads": 4,
}
"""The same dimensions as Megatron's arguments."""
FUSED, OTHERS = gdn_hybrid_tensors(PATTERN, GDN)
PARTS = sum(len(parts) for parts in FUSED.values())
ATTENTION = "decoder.layers.0.self_attention"
IN_PROJ = f"{ATTENTION}.in_proj.weight"
CONV = f"{ATTENTION}.conv1d.weight"
IN_PROJ_BIAS = f"{ATTENTION}.in_proj.bias"


def gdn_args(**overrides):
    fields = {**LINEAR, "experimental_attention_variant": None, "pipeline_model_parallel_size": 1}
    return hybrid_args(PATTERN, **{**fields, **overrides})


def tp_args(tp: int, **overrides):
    return gdn_args(
        tensor_model_parallel_size=tp, tensor_parallel_num_weight_shards=tp, **overrides
    )


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
    return {**packed_projections(fused, tp), **others}


def tp_grids(tensors, tp: int) -> dict:
    """Megatron's grid of the Gated DeltaNet tensors saved at tensor-parallel size ``tp``: one
    chunk per rank."""
    return {
        key: dim0_grid(tuple(tensor.shape), tensor.shape[0] // tp)
        for key, tensor in tensors.items()
        if re.search(r"\.self_attention\.(?:in_proj|conv1d)\.(?:weight|bias)", key)
    }


def derived(report) -> dict:
    return {d["field"]: d["value"] for d in report["derivations"]}


def test_a_gated_deltanet_hybrid_is_fused_for_fsdp_dtensor(tmp_path):
    """Packed at tensor-parallel size 1, a fused tensor is its parts' concatenation, under the
    same names as torch_dist's parts' (``self_attention.conv1d.weight``)."""
    src = write_gpt_source(tmp_path / "src", torch_dist_store(), args=gdn_args())
    out, report = to_fsdp(tmp_path, src)
    expected = fsdp_expected(1)
    assert list(FUSED[IN_PROJ]) == ["query", "key", "value", "z", "beta", "alpha"]
    assert list(FUSED[CONV]) == ["query", "key", "value"]
    assert torch.equal(expected[IN_PROJ], torch.cat(list(FUSED[IN_PROJ].values())))
    assert expected[IN_PROJ].shape == (48, 8) and expected[CONV].shape == (32, 1, 4)
    expect_fsdp_store(out, expected)
    assert report["claims"] == {"gdn": PARTS, "identity": len(OTHERS)}
    assert derived(report)["mixers"] == {"decoder": {"gdn": 2}}
    assert derived(report)["pack_tp"] == 1


def test_fused_projections_split_back_into_megatrons_parts(tmp_path):
    src = write_gpt_source(tmp_path / "src", torch_dist_store(), args=gdn_args())
    fsdp, _ = to_fsdp(tmp_path, src)
    out, report = to_torch_dist(tmp_path, fsdp)
    assert_store(out, torch_dist_store())
    assert grid_of(out, f"{CONV}.value") == [((0, 0, 0), (8, 1, 4))]
    assert report["claims"]["gdn"] == 6  # the fused in_proj and conv1d weight and bias, x 2
    assert verify_checkpoint(out, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok


def test_a_tensor_parallel_source_is_packed_at_the_loading_jobs_size(tmp_path, capsys):
    """Saved at tensor-parallel size 2: each rank's local in_proj is ``[query_r; key_r; value_r;
    z_r; beta_r; alpha_r]`` and its conv1d ``[query_r; key_r; value_r]``, so the fsdp_dtensor
    tensors are no concatenation of the parts."""
    store = torch_dist_store()
    src = write_gpt_source(tmp_path / "src", store, grids=tp_grids(store, 2), args=tp_args(2))
    assert run("plan", src, "--to", "fsdp_dtensor") == 4
    assert (
        "error: FC-A01: fsdp_dtensor packs 6 Gated DeltaNet in_proj/conv1d per rank of the job "
        "that loads it, and the source was saved with args.tensor_model_parallel_size=2"
    ) in capsys.readouterr().err
    fsdp, report = to_fsdp(tmp_path, src, fsdp_tp=2)
    expected = fsdp_expected(2)
    parts = list(FUSED[IN_PROJ].values())
    rank0 = torch.cat([part[: part.shape[0] // 2] for part in parts])
    assert torch.equal(expected[IN_PROJ][:24], rank0)
    assert not torch.equal(expected[IN_PROJ], torch.cat(parts))
    expect_fsdp_store(fsdp, expected)
    run_stats = report["run"]
    assert run_stats["torch_load_calls"] == run_stats["items_touched"]  # each chunk read once
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, store)


def test_packing_at_another_size_rewrites_and_restores_the_arguments(tmp_path):
    src = write_gpt_source(tmp_path / "src", torch_dist_store(), args=gdn_args())
    fsdp, report = to_fsdp(tmp_path, src, fsdp_tp=4)
    expect_fsdp_store(fsdp, fsdp_expected(4))
    assert report["common"]["args_changes"]["tensor_model_parallel_size"] == [1, 4]
    assert read_items(fsdp)["args"].tensor_model_parallel_size == 4
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, torch_dist_store())


@pytest.mark.parametrize("tp", [1, 2])
def test_a_gated_deltanet_2_round_trips(tmp_path, tp):
    """GDN2's in_proj is ``query, key, value, z, f, b, w``; its A_log has a row per key head and its
    dt_bias one per key channel."""
    fused, others = gdn_hybrid_tensors(PATTERN, {**GDN, "variant": "gdn2"})
    assert list(fused[IN_PROJ]) == ["query", "key", "value", "z", "f", "b", "w"]
    assert others[f"{ATTENTION}.A_log"].shape == (4,) and others[f"{ATTENTION}.dt_bias"].shape == (
        12,
    )
    store = torch_dist_store(fused, others)
    args = tp_args(tp, experimental_attention_variant="gdn2")
    src = write_gpt_source(tmp_path / "src", store, grids=tp_grids(store, tp), args=args)
    fsdp, report = to_fsdp(tmp_path, src, fsdp_tp=tp)
    expected = fsdp_expected(tp, fused, others)
    assert expected[IN_PROJ].shape == (72, 8)
    expect_fsdp_store(fsdp, expected)
    assert derived(report)["mixers"] == {"decoder": {"gdn2": 2}}
    assert report["claims"]["gdn2"] == 2 * (7 + 3 + 3)
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, store)


@pytest.mark.parametrize("tp", [1, 2])
def test_an_in_proj_bias_converts_as_is_at_the_size_it_was_saved_at(tmp_path, tp):
    """TP-bound identity: torch_dist stores an in_proj.bias whole, the ranks' local biases one
    after the other, as fsdp_dtensor does at the same size."""
    fused, others = gdn_hybrid_tensors(PATTERN, GDN, bias=True)
    store = torch_dist_store(fused, others, tp)
    assert IN_PROJ_BIAS in store and f"{IN_PROJ_BIAS}.query" not in store
    src = write_gpt_source(tmp_path / "src", store, grids=tp_grids(store, tp), args=tp_args(tp))
    fsdp, report = to_fsdp(tmp_path, src, fsdp_tp=tp)
    expected = fsdp_expected(tp, fused, others)
    assert torch.equal(expected[IN_PROJ_BIAS], store[IN_PROJ_BIAS])
    expect_fsdp_store(fsdp, expected)
    assert report["claims"]["gdn"] == 2 * (6 + 1 + 3 + 3)
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, store)


@pytest.mark.parametrize(("saved", "packed"), [(2, 1), (2, 4), (1, 2)])
def test_an_in_proj_bias_is_never_packed_at_another_size(tmp_path, capsys, saved, packed):
    fused, others = gdn_hybrid_tensors(PATTERN, GDN, bias=True)
    store = torch_dist_store(fused, others, saved)
    args = tp_args(saved)
    src = write_gpt_source(tmp_path / "src", store, grids=tp_grids(store, saved), args=args)
    assert run("plan", src, "--to", "fsdp_dtensor", "--fsdp-tp", packed) == 3
    assert (
        "error: FC-R10: 2 Gated DeltaNet layer in_proj.bias tensors (such as decoder[0]."
        f"self_attention.in_proj.bias) are stored whole, as the {saved} tensor-parallel ranks' "
        f"local tensors one after the other, which no fsdp_dtensor store packed at {packed} holds"
    ) in capsys.readouterr().err


@pytest.mark.parametrize("tp", [1, 2])
def test_a_megatron_fsdp_save_is_unpacked_at_its_size(tmp_path, tp):
    """A training save: fp32 main weights, fused projections packed at ``tp``."""
    fp32 = {
        name: {part: value.float() for part, value in parts.items()}
        for name, parts in FUSED.items()
    }
    others = {key: value.float() for key, value in OTHERS.items()}
    store = {key: value.float() for key, value in fsdp_expected(tp, fp32, others).items()}
    args = gdn_args(ckpt_format="fsdp_dtensor", tensor_model_parallel_size=tp, bf16=False)
    fsdp = write_fsdp_store(tmp_path / "fsdp" / ITER, store, args=args)
    out, report = to_torch_dist(tmp_path, fsdp)
    assert_store(out, torch_dist_store(fp32, others, tp))
    assert report["claims"]["gdn"] == 6


def test_a_store_of_the_other_variant_is_refused_before_reading(tmp_path, capsys):
    """A Gated DeltaNet store whose arguments say ``gdn2``: its parts name the other one."""
    args = gdn_args(experimental_attention_variant="gdn2")
    src = write_gpt_source(tmp_path / "src", torch_dist_store(), args=args)
    assert run("plan", src, "--to", "fsdp_dtensor") == 3
    assert (
        "error: FC-R10: decoder[0].self_attention is a Gated DeltaNet layer by its in_proj parts "
        "query, key, value, z, beta, alpha, but args.experimental_attention_variant='gdn2' "
        "builds Gated DeltaNet 2 layers"
    ) in capsys.readouterr().err


def test_a_store_of_other_dimensions_is_refused_before_reading(tmp_path, capsys):
    args = gdn_args(linear_num_value_heads=2)
    src = write_gpt_source(tmp_path / "src", torch_dist_store(), args=args)
    assert run("plan", src, "--to", "fsdp_dtensor") == 3
    assert (
        "error: FC-R10: decoder[0].self_attention: its widths (conv1d.bias 32 rows, conv1d.weight "
        "32 rows, in_proj.weight 48 rows) are not what the arguments give Gated DeltaNet layers"
    ) in capsys.readouterr().err


def test_the_plan_names_the_rule_and_the_layers(tmp_path, capsys):
    src = write_gpt_source(tmp_path / "src", torch_dist_store(), args=gdn_args())
    assert run("plan", src, "--to", "fsdp_dtensor") == 0
    assert f"model tensors claimed: gdn {PARTS}, identity {len(OTHERS)}" in capsys.readouterr().out
    plan = plan_conversion(src, PlanOptions(to_format="fsdp_dtensor")).to_json()
    (mixers,) = [d for d in plan["derivations"] if d["field"] == "mixers"]
    assert mixers["evidence"].startswith(
        "args.experimental_attention_variant=None builds Gated DeltaNet layers; 2 layers stored "
        "as parts; qk=12, v=8, key heads=4, value heads=4 (args.linear_*): in_proj 48 rows, "
        "conv1d 32 rows, the only family whose widths the stored ones match"
    )


GDP_SPEC = ["megatron.core.models.hybrid.hybrid_layer_specs", "gated_delta_product_stack_spec"]
GDP_DIMS = {"mamba_num_heads": 4, "mamba_head_dim": 2, "mamba_num_groups": 4, "mamba_state_dim": 3}


def test_a_hybrid_of_gated_deltaproduct_mixers_and_gated_deltanets(tmp_path):
    """The Gated DeltaProduct stack spec builds its ``G`` layers as Gated DeltaNets: each mixer
    is converted by its own family's rule."""
    gdp = {"d_inner": 8, "group_state": 12, "heads": 4, "householders": 2}
    builders = {
        "M": mixer_builder(gdp_mixer_tensors, gdp, hidden=8, bias=False),
        "G": mixer_builder(gdn_tensors, GDN, module="self_attention", hidden=8, bias=False),
    }
    fused, others = mixer_hybrid_tensors("MG*G", builders, 8)
    store = torch_dist_store(fused, others)
    args = hybrid_args(
        "MG*G",
        **LINEAR,
        **GDP_DIMS,
        spec=GDP_SPEC,
        gdp_num_householder=2,
        pipeline_model_parallel_size=1,
    )
    src = write_gpt_source(tmp_path / "src", store, args=args)
    fsdp, report = to_fsdp(tmp_path, src)
    expect_fsdp_store(fsdp, fsdp_expected(1, fused, others))
    assert derived(report)["mixers"] == {"decoder": {"gdn": 2, "gdp": 1}}
    assert report["claims"]["gdp"] == 9 + 5 and report["claims"]["gdn"] == 2 * (6 + 3 + 3)
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, store)


LINEAR_ATTENTION_FREQ = 3
"""Layers 0 and 1 are Gated DeltaNets, layer 2 standard attention (``(i + 1) % 3 == 0``)."""


def linear_attention_gpt(variant: str = "gdn") -> tuple[dict, dict]:
    """A 3-layer GPT model whose layers 0 and 1 hold a Gated DeltaNet (``variant``) instead of
    standard attention, one key per layer: its Gated DeltaNets' fused projections by part, and
    every other tensor. The Gated DeltaNet fuses the layer's input norm into its ``in_proj``
    (``fuse_input_layernorm``)."""
    tensors = gpt_tensors(3)
    fused: dict = {}
    for layer in (0, 1):
        attention = f"decoder.layers.{layer}.self_attention"
        tensors = {
            key: value
            for key, value in tensors.items()
            if not key.startswith((f"{attention}.linear_qkv.", f"{attention}.linear_proj."))
        }
        projections, rest = gdn_tensors(attention, variant=variant, seed=50 + layer, **GDN)
        fused.update(projections)
        tensors.update(rest)
    return fused, tensors


@pytest.mark.parametrize("variant", ["gdn", "gdn2"])
@pytest.mark.parametrize("tp", [1, 2])
def test_a_gpt_model_with_linear_attention_round_trips(tmp_path, variant, tp):
    """GPT linear attention: ``linear_attention_freq`` makes Megatron save the decoder per layer,
    its Gated DeltaNets' projections as parts, beside a standard attention layer."""
    fused, others = linear_attention_gpt(variant)
    store = torch_dist_store(fused, others)
    args = gpt_args(
        3,
        **LINEAR,
        experimental_attention_variant=variant,
        linear_attention_freq=LINEAR_ATTENTION_FREQ,
        tensor_model_parallel_size=tp,
        tensor_parallel_num_weight_shards=tp,
    )
    assert "decoder.layers.2.self_attention.linear_qkv.weight" in store
    src = write_gpt_source(tmp_path / "src", store, grids=tp_grids(store, tp), args=args)
    fsdp, report = to_fsdp(tmp_path, src, fsdp_tp=tp)
    expect_fsdp_store(fsdp, fsdp_expected(tp, fused, others))
    assert derived(report)["mixers"] == {"decoder": {variant: 2}}
    out, _ = to_torch_dist(tmp_path, fsdp)
    assert_store(out, store)
    # The source's chunks are its ranks' shards; the converter writes one chunk per tensor.
    options = VerifyOptions(expected_diff="D", layout=tp == 1)
    assert verify_checkpoint(out, src / ITER, options).ok


@pytest.mark.parametrize(
    ("variant", "freq"), [(None, None), (None, LINEAR_ATTENTION_FREQ), ("gdn", None)]
)
def test_a_gpt_model_the_arguments_build_no_linear_attention_for_is_refused(
    tmp_path, capsys, variant, freq
):
    """A GPT decoder holds Gated DeltaNets only for a Gated DeltaNet variant with
    ``linear_attention_freq``: other arguments cannot have saved the store."""
    fused, others = linear_attention_gpt()
    args = gpt_args(3, **LINEAR, experimental_attention_variant=variant, linear_attention_freq=freq)
    src = write_gpt_source(tmp_path / "src", torch_dist_store(fused, others), args=args)
    assert run("plan", src, "--to", "fsdp_dtensor") == 3
    assert (
        "error: FC-R10: decoder[0].self_attention holds a Gated DeltaNet's in_proj and conv1d, but "
        "a TransformerBlock builds Gated DeltaNet layers only for "
        "args.experimental_attention_variant"
    ) in capsys.readouterr().err
