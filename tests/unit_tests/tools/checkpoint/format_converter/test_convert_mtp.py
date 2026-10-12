# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Multi-token prediction (MTP) layers across formats, on synthetic stores.

Each hybrid MTP depth is a layer of the ``mtp`` block holding a hybrid layer
list of the pattern's MTP part under ``mtp_model_layer`` (the same name in both
formats). The modules are drawn one by one under their layer-list (dense)
indices, as fsdp_dtensor stores them; the torch_dist source renames each to
its first layer's pattern position, written out by hand, and stacks the
experts, so the expected tensors of either format are the drawn ones.

A GPT MTP depth holds one ``TransformerLayer``, which both formats store as
``transformer_layer``; a Megatron-FSDP store saved before
``handle_mtp_in_state_dict`` holds the live name ``mtp_model_layer``. The
expected stores are drawn under one name and renamed by hand
(:func:`live_inner_name`).
"""

from __future__ import annotations

import json
import logging

import pytest

from tools.checkpoint.format_converter.api import convert_checkpoint, verify_checkpoint
from tools.checkpoint.format_converter.options import ConvertOptions, VerifyOptions

from ._synthetic import (
    expert_grids,
    gated_units,
    gpt_args,
    gpt_moe_tensors,
    gpt_mtp_tensors,
    gpt_tensors,
    hybrid_args,
    hybrid_mtp_tensors,
    layer_grids,
    renumber_layers,
    shortcut_hybrid_tensors,
    stack_experts,
    stack_layers,
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


GPT_LAYERS, GPT_DEPTHS = 3, 2
GPT_MTP = {**gpt_tensors(GPT_LAYERS), **gpt_mtp_tensors(GPT_DEPTHS)}
"""A GPT model with two MTP layers, one layer per key (fsdp_dtensor without the prefix)."""
GPT_MTP_TD = stack_layers(GPT_MTP, "decoder")
"""The same model as torch_dist stores it: the decoder stacked, the MTP layers per layer."""


def gpt_mtp_args(**overrides):
    return gpt_args(**{"num_layers": GPT_LAYERS, "mtp_num_layers": GPT_DEPTHS, **overrides})


def live_inner_name(tensors):
    """``tensors`` with every GPT MTP layer's inner layer under the live name ``mtp_model_layer``
    (as Megatron-FSDP saved it before handle_mtp_in_state_dict)."""
    renamed = {}
    for key, tensor in tensors.items():
        head, found, rest = key.partition(".transformer_layer.")
        renamed[f"{head}.mtp_model_layer.{rest}" if found else key] = tensor
    return renamed


def write_gpt_mtp_source(tmp_path, **overrides):
    return write_gpt_source(
        tmp_path / "src",
        GPT_MTP_TD,
        grids=layer_grids(GPT_MTP_TD, "decoder"),
        args=gpt_mtp_args(**overrides),
    )


def fsdp_args(**overrides):
    return gpt_mtp_args(ckpt_format="fsdp_dtensor", use_megatron_fsdp=True, **overrides)


def derivation(report, field):
    (value,) = [d for d in report["derivations"] if d["field"] == field]
    return value


def test_gpt_mtp_layers_keep_transformer_layer_in_both_formats(tmp_path):
    src = write_gpt_mtp_source(tmp_path)
    out, report = convert(tmp_path, src, "fsdp", "fsdp_dtensor")
    expect_fsdp_store(out, GPT_MTP)
    assert "model.module.mtp.layers.1.transformer_layer.mlp.linear_fc2.weight" in (
        read_tensors(out)
    )
    assert derivation(report, "gpt_mtp_inner_layer")["value"] == {
        "torch_dist": "transformer_layer",
        "fsdp_dtensor": "transformer_layer",
    }
    assert derivation(report, "blocks.mtp")["value"]["kind"] == "mtp_block"
    back, _ = convert(tmp_path, out, "td", "torch_dist")
    assert verify_checkpoint(back, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok


def test_a_repeated_gpt_mtp_layer_converts_as_one_layer(tmp_path):
    """``--mtp-use-repeated-layer`` with two depths: one GPT MTP layer, ``mtp.layers.0``."""
    model = {**gpt_tensors(GPT_LAYERS), **gpt_mtp_tensors(1)}
    td = stack_layers(model, "decoder")
    src = write_gpt_source(
        tmp_path / "src",
        td,
        grids=layer_grids(td, "decoder"),
        args=gpt_mtp_args(mtp_use_repeated_layer=True),
    )
    out, report = convert(tmp_path, src, "fsdp", "fsdp_dtensor")
    expect_fsdp_store(out, model)
    assert derivation(report, "blocks.mtp")["value"]["num_modules"] == 1
    assert derivation(report, "gpt_mtp_inner_layer")["value"] == {
        "torch_dist": "transformer_layer",
        "fsdp_dtensor": "transformer_layer",
    }
    back, _ = convert(tmp_path, out, "td", "torch_dist")
    assert verify_checkpoint(back, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok


def test_a_legacy_fsdp_gpt_mtp_store_converts_to_transformer_layer(tmp_path, caplog):
    """A Megatron-FSDP store saved before handle_mtp_in_state_dict: the live name is read, and
    torch_dist gets transformer_layer, with a warning and a manifest record."""
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, live_inner_name(GPT_MTP), args=fsdp_args())
    with caplog.at_level(logging.WARNING):
        out, report = convert(tmp_path, store, "td", "torch_dist")
    tensors = read_tensors(out)
    assert set(tensors) == set(GPT_MTP_TD)
    for key, tensor in tensors.items():
        assert bitwise_equal(tensor, GPT_MTP_TD[key]), key
    reason = (
        "the fsdp_dtensor source stores the inner layer of its GPT MTP layers (24 keys) under the "
        "live name mtp_model_layer, as Megatron-FSDP saved it before handle_mtp_in_state_dict; "
        "torch_dist stores it as transformer_layer"
    )
    assert reason in caplog.text
    assert derivation(report, "gpt_mtp_inner_layer") == {
        "field": "gpt_mtp_inner_layer",
        "value": {"torch_dist": "transformer_layer", "fsdp_dtensor": "mtp_model_layer"},
        "evidence": reason,
    }
    manifest = json.loads((out / "conversion_manifest.json").read_text())
    assert derivation(manifest, "gpt_mtp_inner_layer")["value"]["fsdp_dtensor"] == (
        "mtp_model_layer"
    )
    replay = verify_checkpoint(store, out, VerifyOptions(map="convert"))
    assert replay.ok and replay.values_compared == len(GPT_MTP_TD)


def test_the_live_name_is_written_only_when_asked_and_read_back(tmp_path, caplog):
    src = write_gpt_mtp_source(tmp_path)
    with caplog.at_level(logging.WARNING):
        report = convert_checkpoint(
            src,
            tmp_path / "fsdp",
            ConvertOptions(
                to_format="fsdp_dtensor", workers=0, fsdp_gpt_mtp_inner_name="mtp_model_layer"
            ),
        ).to_json()
    out = tmp_path / "fsdp" / ITER
    expect_fsdp_store(out, live_inner_name(GPT_MTP))
    assert "--fsdp-gpt-mtp-inner-name mtp_model_layer: the fsdp_dtensor target stores" in (
        caplog.text
    )
    assert derivation(report, "gpt_mtp_inner_layer")["value"]["fsdp_dtensor"] == "mtp_model_layer"
    back, _ = convert(tmp_path, out, "td", "torch_dist")
    assert verify_checkpoint(back, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok


MOE_EXPERTS = 4
GPT_MOE_MTP = {**gpt_moe_tensors(2, MOE_EXPERTS), **gpt_mtp_tensors(1, experts=MOE_EXPERTS)}
"""A GPT mixture-of-experts model with one MTP layer whose MLP has the routed experts too."""


def test_a_legacy_fsdp_moe_mtp_store_stacks_the_mtp_experts_under_transformer_layer(tmp_path):
    args = fsdp_args(num_layers=2, mtp_num_layers=1, num_experts=MOE_EXPERTS, moe_grouped_gemm=True)
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, live_inner_name(GPT_MOE_MTP), args=args)
    out, _ = convert(tmp_path, store, "td", "torch_dist")
    expected = stack_layers(stack_experts(GPT_MOE_MTP), "decoder")
    tensors = read_tensors(out)
    assert set(tensors) == set(expected)
    assert tensors[
        "mtp.layers.0.transformer_layer.mlp.experts.experts.linear_fc1.weight"
    ].shape == (MOE_EXPERTS, 6, 8)
    for key, tensor in tensors.items():
        assert bitwise_equal(tensor, expected[key]), key


def test_a_store_with_both_gpt_mtp_names_is_refused(tmp_path, capsys):
    mixed = live_inner_name(GPT_MTP)
    mixed.update({k: v for k, v in GPT_MTP.items() if k.startswith("mtp.layers.1.transformer")})
    mixed = {k: v for k, v in mixed.items() if not k.startswith("mtp.layers.1.mtp_model_layer")}
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, mixed, args=fsdp_args())
    assert run("plan", store, "--to", "torch_dist") == 3
    err = capsys.readouterr().err
    assert (
        "FC-R09: the store holds the inner layer of its GPT MTP layers under both "
        "transformer_layer and mtp_model_layer"
    ) in err


def test_the_inner_name_flag_needs_a_torch_dist_source_and_gpt_mtp_layers(tmp_path, capsys, caplog):
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, GPT_MTP, args=fsdp_args())
    flag = ("--fsdp-gpt-mtp-inner-name", "mtp_model_layer")
    assert run("plan", store, "--to", "torch_dist", *flag) == 2
    assert "only torch_dist to fsdp_dtensor conversions take --fsdp-gpt-mtp-inner-name" in (
        capsys.readouterr().err
    )
    src = write_gpt_source(tmp_path / "hybrid", PLAIN, args=mtp_args("*-*-/*-/*-"))
    with caplog.at_level(logging.WARNING):
        assert run("plan", src, "--to", "fsdp_dtensor", *flag) == 0
    assert (
        "--fsdp-gpt-mtp-inner-name mtp_model_layer has no effect: the model has no GPT MTP layer"
    ) in caplog.text
