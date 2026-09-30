# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hybrid multi-token prediction (MTP) layers across formats, on synthetic stores.

Each MTP depth is a layer of the ``mtp`` block holding a hybrid layer list of
the pattern's MTP part under ``mtp_model_layer`` (the same name in both
formats). The modules are drawn one by one under their layer-list (dense)
indices, as fsdp_dtensor stores them; the torch_dist source renames each to
its first layer's pattern position, written out by hand, and stacks the
experts, so the expected tensors of either format are the drawn ones.
"""

from __future__ import annotations

import pytest

from tools.checkpoint.format_converter.api import convert_checkpoint, verify_checkpoint
from tools.checkpoint.format_converter.options import ConvertOptions, VerifyOptions

from ._synthetic import (
    expert_grids,
    gated_units,
    hybrid_args,
    hybrid_mtp_tensors,
    renumber_layers,
    shortcut_hybrid_tensors,
    stack_experts,
    swiglu_grids,
    write_fsdp_store,
)
from .test_convert_blocks import ITER, expect_fsdp_store, run, write_gpt_source
from .test_convert_cross import bitwise_equal, read_tensors

pytestmark = pytest.mark.covers("identity", "ground_truth")
EXPERTS = 4
INNER = "mtp.layers.0.mtp_model_layer"

PLAIN = {
    **shortcut_hybrid_tensors(("*", "-", "*", "-")),
    **hybrid_mtp_tensors(("*", "-"), layers=2),
}
"""A hybrid ``*-*-/*-/*-`` model: numbered alike in both formats, two MTP layers."""

SHORTCUT = {
    **shortcut_hybrid_tensors(("E", "*E", "-", "*E", "E"), experts=EXPERTS),
    **hybrid_mtp_tensors(("*E", "E"), layers=1, experts=EXPERTS),
}
"""A shortcut-MoE hybrid ``E*E-*EE/*EE/*EE`` model with one MTP layer for both depths
(``--mtp-use-repeated-layer``), as fsdp_dtensor stores it."""
SHORTCUT_TD = stack_experts(
    renumber_layers(renumber_layers(SHORTCUT, "decoder", (0, 1, 3, 4, 6)), INNER, (0, 2))
)
"""The same model as torch_dist stores it: the MTP layer's inner modules ``*E`` and ``E`` at the
positions 0 and 2 of its pattern ``*EE``."""


def mtp_args(pattern: str, **overrides):
    """Arguments of a hybrid MTP model (one pipeline stage unless overridden)."""
    fields = {"pipeline_model_parallel_size": 1, "mtp_num_layers": pattern.count("/")}
    return hybrid_args(pattern, **{**fields, **overrides})


SHORTCUT_ARGS = mtp_args(
    "E*E-*EE/*EE/*EE",
    moe_shortcut_connection=True,
    num_experts=EXPERTS,
    mtp_use_repeated_layer=True,
)


def convert(tmp_path, src, name: str, to_format: str):
    report = convert_checkpoint(
        src, tmp_path / name, ConvertOptions(to_format=to_format, workers=0)
    )
    return tmp_path / name / ITER, report.to_json()


def test_hybrid_mtp_layers_keep_their_names_in_both_formats(tmp_path):
    src = write_gpt_source(tmp_path / "src", PLAIN, args=mtp_args("*-*-/*-/*-"))
    out, report = convert(tmp_path, src, "fsdp", "fsdp_dtensor")
    expect_fsdp_store(out, PLAIN)
    assert "model.module.mtp.layers.1.mtp_model_layer.layers.1.mlp.linear_fc2.weight" in (
        read_tensors(out)
    )
    blocks = {
        d["field"]: d["value"]["kind"]
        for d in report["derivations"]
        if d["field"].startswith("blocks.")
    }
    assert blocks == {
        "blocks.decoder": "hybrid_stack",
        "blocks.mtp": "mtp_block",
        "blocks.mtp.layers.0.mtp_model_layer": "hybrid_mtp_inner",
        "blocks.mtp.layers.1.mtp_model_layer": "hybrid_mtp_inner",
    }
    back, _ = convert(tmp_path, out, "td", "torch_dist")
    assert verify_checkpoint(back, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok


def test_a_repeated_shortcut_mtp_layer_is_renumbered_like_the_decoder(tmp_path):
    src = write_gpt_source(
        tmp_path / "src", SHORTCUT_TD, grids=expert_grids(SHORTCUT_TD), args=SHORTCUT_ARGS
    )
    out, report = convert(tmp_path, src, "fsdp", "fsdp_dtensor")
    expect_fsdp_store(out, SHORTCUT)
    (inner,) = [d for d in report["derivations"] if d["field"] == f"blocks.{INNER}"]
    assert (inner["value"]["dense_to_td"], inner["value"]["symbols"]) == ([0, 2], ["*E", "E"])
    back, _ = convert(tmp_path, out, "td", "torch_dist")
    assert verify_checkpoint(back, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok


def test_an_fsdp_shortcut_mtp_store_converts_to_the_module_starts(tmp_path):
    """A Megatron-FSDP-like save (fp32) of the shortcut MTP model."""
    fp32 = {key: tensor.float() for key, tensor in SHORTCUT.items()}
    args = mtp_args(
        "E*E-*EE/*EE/*EE",
        moe_shortcut_connection=True,
        num_experts=EXPERTS,
        mtp_use_repeated_layer=True,
        ckpt_format="fsdp_dtensor",
        use_megatron_fsdp=True,
    )
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, fp32, args=args)
    out, _ = convert(tmp_path, store, "td", "torch_dist")
    expected = {key: tensor.float() for key, tensor in SHORTCUT_TD.items()}
    tensors = read_tensors(out)
    assert set(tensors) == set(expected)
    assert f"{INNER}.layers.2.mlp.experts.experts.linear_fc1.weight" in tensors
    for key, tensor in tensors.items():
        assert bitwise_equal(tensor, expected[key]), key


def test_the_legacy_mtp_pattern_argument_converts(tmp_path):
    """Older checkpoints give the MTP part as ``mtp_hybrid_override_pattern``."""
    args = mtp_args("*-*-", mtp_num_layers=2, mtp_hybrid_override_pattern="*-")
    src = write_gpt_source(tmp_path / "src", PLAIN, args=args)
    out, _ = convert(tmp_path, src, "fsdp", "fsdp_dtensor")
    expect_fsdp_store(out, PLAIN)


ATTENTION_DECODER = {
    **shortcut_hybrid_tensors(("*", "*")),
    **hybrid_mtp_tensors(("*", "-"), layers=1),
}
"""A hybrid ``**/*-`` model: the decoder has no MLP, its MTP layer has one."""


def test_the_mtp_layers_mlps_are_split_as_the_language_models(tmp_path):
    """MTP layers are built with the language model's configuration: under --swiglu,
    Megatron-FSDP splits their gated linear_fc1 into its halves as it splits the decoder's."""
    halves, fused = gated_units(PLAIN)
    assert f"{INNER}.layers.1.mlp.linear_fc1.weight_v" in halves
    args = mtp_args("*-*-/*-/*-", swiglu=True)
    src = write_gpt_source(tmp_path / "src", fused, grids=swiglu_grids(fused), args=args)
    out, report = convert(tmp_path, src, "fsdp", "fsdp_dtensor")
    expect_fsdp_store(out, halves)
    (split,) = [d["value"] for d in report["derivations"] if d["field"] == "swiglu_split"]
    assert split == {
        "decoder": {"split": 2},
        INNER: {"split": 1},
        "mtp.layers.1.mtp_model_layer": {"split": 1},
    }
    back, _ = convert(tmp_path, out, "td", "torch_dist")
    assert verify_checkpoint(back, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok


def test_mtp_mlps_gated_otherwise_than_the_arguments_are_refused(tmp_path, capsys):
    """Only the MTP layer has an MLP: gated, while the arguments build plain MLPs."""
    _, fused = gated_units(ATTENTION_DECODER, prefix="mtp.")
    src = write_gpt_source(tmp_path / "src", fused, args=mtp_args("**/*-", swiglu=False))
    assert run("plan", src, "--to", "fsdp_dtensor") == 3
    err = capsys.readouterr().err
    assert (
        "error: FC-R12: the language model's linear_fc1 weights (in decoder, mtp, "
        f"{INNER}) are gated (twice the rows linear_fc2 has inputs), but "
    ) in err
    assert f"[{INNER}.layers.1.mlp.linear_fc1.weight]" in err


def test_target_swiglu_may_not_change_the_mtp_layers_activation(tmp_path, capsys):
    """A decoder without MLPs does not free --target-swiglu: the MTP layer's MLP is plain."""
    src = write_gpt_source(
        tmp_path / "src", ATTENTION_DECODER, args=mtp_args("**/*-", swiglu=False)
    )
    assert run("plan", src, "--to", "fsdp_dtensor", "--target-swiglu", "on") == 2
    assert "error: --target-swiglu on: a job with --swiglu builds gated MLPs, but the " in (
        capsys.readouterr().err
    )
