# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Multi-latent attention (MLA) across formats, on synthetic stores.

Both formats store MLA's projections unfused under the same names
(``FusedMLASelfAttention`` saves its fused down-projection as
``linear_q_down_proj`` and ``linear_kv_down_proj``, and its absorbed input
layer norm as ``input_layernorm``, in either format), so they convert
unchanged: in GPT decoders (stacked in torch_dist), as hybrid ``+`` layers, in
shortcut-connected pairs and in multi-token prediction layers. The tensors
are drawn per layer and stacked or renumbered by hand for torch_dist. A store
under the fused live name is refused (FC-R08).
"""

from __future__ import annotations

import pytest

from tools.checkpoint.format_converter.api import convert_checkpoint, verify_checkpoint
from tools.checkpoint.format_converter.options import ConvertOptions, VerifyOptions

from ._synthetic import (
    expert_grids,
    gpt_args,
    hybrid_args,
    hybrid_mtp_tensors,
    hybrid_tensors,
    layer_grids,
    mla_gpt_tensors,
    renumber_layers,
    shortcut_hybrid_tensors,
    stack_experts,
    stack_layers,
    write_fsdp_store,
)
from .test_convert_blocks import ITER, expect_fsdp_store, run, write_gpt_source
from .test_convert_cross import bitwise_equal, read_tensors

pytestmark = pytest.mark.covers("identity", "ground_truth")
LAYERS = 3
MLA = {
    "multi_latent_attention": True,
    "q_lora_rank": 6,
    "kv_lora_rank": 4,
    "qk_head_dim": 3,
    "qk_pos_emb_head_dim": 2,
    "v_head_dim": 3,
}
"""The arguments of the multi-latent attention of :data:`~._synthetic.MLA_SHAPES`."""
PER_LAYER = mla_gpt_tensors(LAYERS)
STACKED = stack_layers(PER_LAYER, "decoder")


def convert(tmp_path, src, name: str, to_format: str):
    report = convert_checkpoint(
        src, tmp_path / name, ConvertOptions(to_format=to_format, workers=0)
    )
    return tmp_path / name / ITER, report.to_json()


def round_trip_holds(tmp_path, src, fsdp) -> bool:
    back, _ = convert(tmp_path, fsdp, "back", "torch_dist")
    return verify_checkpoint(back, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok


@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "mla-down-proj-fusion"])
def test_a_stacked_mla_gpt_converts_unchanged(tmp_path, fused):
    """``--mla-down-proj-fusion`` changes the live modules, not the stored names."""
    args = gpt_args(LAYERS, **MLA, mla_down_proj_fusion=fused)
    src = write_gpt_source(
        tmp_path / "src", STACKED, grids=layer_grids(STACKED, "decoder"), args=args
    )
    out, report = convert(tmp_path, src, "fsdp", "fsdp_dtensor")
    expect_fsdp_store(out, PER_LAYER)
    assert "model.module.decoder.layers.2.self_attention.linear_kv_down_proj.weight" in (
        read_tensors(out)
    )
    assert report["claims"] == {"identity": len(PER_LAYER)}
    assert round_trip_holds(tmp_path, src, tmp_path / "fsdp")


HYBRID = hybrid_tensors("+-*+")


def test_hybrid_mla_layers_convert_unchanged_in_both_directions(tmp_path):
    src = write_gpt_source(tmp_path / "src", HYBRID, args=hybrid_args("+-*+", **MLA))
    out, report = convert(tmp_path, src, "fsdp", "fsdp_dtensor")
    expect_fsdp_store(out, HYBRID)
    (blocks,) = [d["value"] for d in report["derivations"] if d["field"] == "blocks.decoder"]
    assert blocks["symbols"] == ["+", "-", "*", "+"]
    assert round_trip_holds(tmp_path, src, tmp_path / "fsdp")


EXPERTS = 4
SHORTCUT = {
    **shortcut_hybrid_tensors(("+E", "-", "+"), experts=EXPERTS),
    **hybrid_mtp_tensors(("+E",), layers=1, experts=EXPERTS),
}
"""A shortcut-MoE hybrid ``+E-+/+E`` model, as fsdp_dtensor stores it: an MLA layer pairs with the
mixture-of-experts layer after it, in the decoder and in the MTP layer."""
SHORTCUT_TD = stack_experts(renumber_layers(SHORTCUT, "decoder", (0, 2, 3)))
"""The same model as torch_dist stores it: the decoder modules at pattern positions 0, 2 and 3."""


def test_mla_layers_pair_with_shortcut_experts_and_fill_mtp_layers(tmp_path):
    args = hybrid_args(
        "+E-+/+E",
        **MLA,
        moe_shortcut_connection=True,
        num_experts=EXPERTS,
        mtp_num_layers=1,
        pipeline_model_parallel_size=1,
    )
    src = write_gpt_source(
        tmp_path / "src", SHORTCUT_TD, grids=expert_grids(SHORTCUT_TD), args=args
    )
    out, _ = convert(tmp_path, src, "fsdp", "fsdp_dtensor")
    expect_fsdp_store(out, SHORTCUT)
    assert "model.module.decoder.layers.0.attn_layer.self_attention.linear_q_down_proj.weight" in (
        read_tensors(out)
    )
    assert round_trip_holds(tmp_path, src, tmp_path / "fsdp")


def test_an_fsdp_mla_store_converts_to_the_stacked_decoder(tmp_path):
    """A Megatron-FSDP-like save (fp32): the fused or unfused job saved the same names."""
    fp32 = {key: tensor.float() for key, tensor in PER_LAYER.items()}
    args = gpt_args(
        LAYERS, **MLA, mla_down_proj_fusion=True, ckpt_format="fsdp_dtensor", use_megatron_fsdp=True
    )
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, fp32, args=args)
    out, _ = convert(tmp_path, store, "td", "torch_dist")
    expected = stack_layers(fp32, "decoder")
    tensors = read_tensors(out)
    assert set(tensors) == set(expected)
    for key, tensor in tensors.items():
        assert bitwise_equal(tensor, expected[key]), key


FUSED_DOWN_PROJ = "decoder.layers.{}.self_attention.linear_qkv_down_proj.{}"


def test_a_legacy_fused_mla_fsdp_store_is_refused(tmp_path, capsys):
    """Saved before handle_mla_down_proj_in_state_dict: the fused down-projection and the input
    layer norm it absorbed, under their live names."""
    legacy = {
        key: tensor
        for key, tensor in PER_LAYER.items()
        if "_down_proj" not in key and "input_layernorm" not in key
    }
    for layer in range(LAYERS):
        legacy[FUSED_DOWN_PROJ.format(layer, "weight")] = PER_LAYER[
            f"decoder.layers.{layer}.self_attention.linear_q_down_proj.weight"
        ].repeat(2, 1)
        legacy[FUSED_DOWN_PROJ.format(layer, "layer_norm_weight")] = PER_LAYER[
            f"decoder.layers.{layer}.input_layernorm.weight"
        ]
    args = gpt_args(LAYERS, **MLA, mla_down_proj_fusion=True, ckpt_format="fsdp_dtensor")
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, legacy, args=args)
    assert run("plan", store, "--to", "torch_dist") == 3
    err = capsys.readouterr().err
    assert (
        "error: FC-R08: a Megatron-FSDP store saved before handle_mla_down_proj_in_state_dict: "
        "it holds multi-latent attention's fused linear_qkv_down_proj"
    ) in err
    assert f"[{FUSED_DOWN_PROJ.format(0, 'layer_norm_weight')}, " in err
    assert "in a job with --mla-down-proj-fusion, save it as torch_dist and convert that" in err
