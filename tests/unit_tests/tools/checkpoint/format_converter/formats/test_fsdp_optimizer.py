# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parsing a Megatron-FSDP v1 store's optimizer keys: planner paths, the strict split without
them, live names, and every refusal."""

from __future__ import annotations

import argparse

import pytest
import torch

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.errors import RefusedError
from tools.checkpoint.format_converter.formats.fsdp_optimizer import (
    MOMENTS,
    FsdpOptimizerState,
    live_names,
    parse_optimizer_state,
)

from .._synthetic import fsdp_optimizer_items, torch_save_bytes, write_fsdp_store

BF16 = "module.module.module."
FP32 = "module.module."
QKV = "decoder.layers.0.self_attention.linear_qkv.weight"
FC1_W = "decoder.layers.1.mlp.linear_fc1.weight_w"
FC1_V = "decoder.layers.1.mlp.linear_fc1.weight_v"
FC1 = "decoder.layers.1.mlp.linear_fc1.weight"
MODEL = {QKV: torch.zeros(6, 2), FC1_W: torch.zeros(4, 2), FC1_V: torch.zeros(4, 2)}
LIVE = {QKV: QKV, FC1_W: FC1, FC1_V: FC1}
META = {"lr": 1e-3, "betas": (0.9, 0.95), "wd_mult": 1.0, "is_expert_parallel": False}


def _args(**values) -> ArgsView:
    return ArgsView(argparse.Namespace(**values))


def _state(depth: str = BF16, *, step: bool = False) -> dict[str, dict[str, torch.Tensor]]:
    leaves = (*MOMENTS, "step") if step else MOMENTS
    return {
        name: {
            leaf: torch.zeros(()) if leaf == "step" else torch.full_like(t, i) for leaf in leaves
        }
        for i, (name, t) in enumerate(MODEL.items())
    }


def _store(tmp_path, *, depth=BF16, paths=True, state=None, meta=None, extra_bytes=None):
    tensors, byte_items, planner = fsdp_optimizer_items(
        _state(depth) if state is None else state,
        {QKV: META, FC1: META} if meta is None else meta,
        depth,
    )
    byte_items.update(extra_bytes or {})
    iter_dir = write_fsdp_store(
        tmp_path / "iter_0000007",
        MODEL,
        extra_tensors=tensors,
        extra_byte_items=byte_items,
        planner_paths=planner if paths else {key: (key,) for key in planner},
    )
    view = MetadataView.load(iter_dir)
    keys = [k for k in (*view.tensors, *view.byte_items) if k.startswith("optimizer.")]
    return view, keys


def _parse(view, keys, depth=BF16) -> FsdpOptimizerState:
    return parse_optimizer_state(view, keys, depth=depth, live=LIVE)


@pytest.mark.parametrize("paths", [True, False], ids=["planner-paths", "strict-split"])
@pytest.mark.parametrize("depth", [BF16, FP32])
def test_parses_moments_and_group_meta(tmp_path, paths, depth):
    view, keys = _store(tmp_path, depth=depth, paths=paths)
    parsed = _parse(view, keys, depth)
    assert parsed.depth == depth
    for moment in MOMENTS:
        assert set(parsed.moments[moment]) == {QKV, FC1_W, FC1_V}
        entry = parsed.moments[moment][FC1_W]
        assert entry.key == f"optimizer.state.{depth}{FC1_W}.{moment}"
        assert entry.shape == (4, 2)
    assert parsed.steps == {}
    assert parsed.trainable == {QKV, FC1_W, FC1_V}
    assert set(parsed.group_meta) == {QKV, FC1}  # the halves share their live name's group
    assert parsed.group_meta[FC1] == {
        hparam: f"optimizer.param_to_group_meta.{depth}{FC1}.{hparam}" for hparam in sorted(META)
    }


def test_keeps_per_parameter_steps_apart(tmp_path):
    view, keys = _store(tmp_path, state=_state(step=True))
    parsed = _parse(view, keys)
    assert set(parsed.steps) == {QKV, FC1_W, FC1_V}
    assert parsed.steps[QKV].key == f"optimizer.state.{BF16}{QKV}.step"


def _refused(view, keys, depth=BF16) -> RefusedError:
    with pytest.raises(RefusedError) as caught:
        _parse(view, keys, depth)
    return caught.value


@pytest.mark.parametrize("paths", [True, False])
def test_refuses_other_module_levels_than_the_arguments_give(tmp_path, paths):
    view, keys = _store(tmp_path, depth=FP32, paths=paths)
    error = _refused(view, keys, BF16)
    assert error.codes == (errors.FC_R13,)
    assert "module.module. module. levels" in error.refusal.message
    assert "gives module.module.module." in error.refusal.message


def test_refuses_state_that_is_not_adams(tmp_path):
    state = _state()
    state[QKV]["momentum_buffer"] = torch.zeros(6, 2)
    view, keys = _store(tmp_path, state=state)
    error = _refused(view, keys)
    assert error.codes == (errors.FC_R16,)
    assert error.refusal.keys == (f"optimizer.state.{BF16}{QKV}.momentum_buffer",)


def test_refuses_unknown_leaves_without_planner_paths(tmp_path):
    state = _state()
    state[QKV]["momentum_buffer"] = torch.zeros(6, 2)
    view, keys = _store(tmp_path, state=state, paths=False)
    error = _refused(view, keys)
    assert error.codes == (errors.FC_R13,)
    assert error.refusal.keys == (f"optimizer.state.{BF16}{QKV}.momentum_buffer",)
    assert "no planner paths" in error.refusal.message


@pytest.mark.parametrize("paths", [True, False])
def test_refuses_parameters_the_model_section_lacks(tmp_path, paths):
    meta = {QKV: META, FC1: META, "decoder.layers.9.mlp.linear_fc2.weight": META}
    view, keys = _store(tmp_path, meta=meta, paths=paths)
    error = _refused(view, keys)
    assert error.codes == (errors.FC_R13,)
    stray = f"optimizer.param_to_group_meta.{BF16}decoder.layers.9.mlp.linear_fc2.weight"
    assert all(key.startswith(stray) for key in error.refusal.keys)
    assert len(error.refusal.keys) == len(META)


def test_refuses_a_planner_path_that_does_not_spell_its_key(tmp_path):
    key = f"optimizer.state.{BF16}{QKV}.exp_avg"
    tensors, byte_items, planner = fsdp_optimizer_items(_state(), {}, BF16)
    planner[key] = ("optimizer", "state", f"{BF16}decoder.layers.0", "x.exp_avg")
    view = MetadataView.load(
        write_fsdp_store(
            tmp_path / "it",
            MODEL,
            extra_tensors=tensors,
            extra_byte_items=byte_items,
            planner_paths=planner,
        )
    )
    error = _refused(view, [k for k in view.tensors if k.startswith("optimizer.")])
    assert error.refusal.keys == (key,)
    assert "planner paths" in error.refusal.message


def test_refuses_state_stored_as_byte_items(tmp_path):
    key = f"optimizer.state.{BF16}{QKV}.exp_avg"
    state = _state()
    del state[QKV]["exp_avg"]
    view, keys = _store(tmp_path, state=state, extra_bytes={key: torch_save_bytes(0.0)})
    error = _refused(view, keys)
    assert error.refusal.keys == (key,)


def test_live_names_merge_swiglu_halves():
    assert live_names([FC1_W, FC1_V, QKV], _args()) == {FC1_W: FC1, FC1_V: FC1, QKV: QKV}


def test_live_names_of_gpt_mtp_inner_layers():
    keys = [
        "mtp.layers.0.transformer_layer.mlp.linear_fc1.weight_w",
        "mtp.layers.1.mtp_model_layer.self_attention.linear_qkv.weight",
        "mtp.layers.0.enorm.weight",
        "language_model.mtp.layers.0.transformer_layer.input_layernorm.weight",
    ]
    assert live_names(keys, _args(mtp_num_layers=2)) == {
        keys[0]: "mtp.layers.0.mtp_model_layer.mlp.linear_fc1.weight",
        keys[1]: keys[1],  # a legacy store's live name already
        keys[2]: keys[2],
        keys[3]: "language_model.mtp.layers.0.mtp_model_layer.input_layernorm.weight",
    }


def test_live_names_of_hybrid_mtp_inner_layers_are_their_stored_names():
    keys = ["mtp.layers.0.mtp_model_layer.layers.1.mlp.linear_fc1.weight"]
    assert live_names(keys, _args(hybrid_layer_pattern="*-/*-")) == {keys[0]: keys[0]}


MLA_KEYS = [
    "decoder.layers.0.input_layernorm.weight",
    "decoder.layers.0.self_attention.linear_q_down_proj.weight",
    "decoder.layers.0.self_attention.linear_kv_down_proj.weight",
    "decoder.layers.0.self_attention.linear_q_up_proj.layer_norm_weight",
    "decoder.layers.1.input_layernorm.weight",  # a layer without multi-latent attention
    "decoder.layers.1.self_attention.linear_qkv.weight",
    "mtp.layers.0.transformer_layer.input_layernorm.weight",
    "mtp.layers.0.transformer_layer.self_attention.linear_q_down_proj.weight",
]


def test_live_names_of_fused_mla_in_a_gpt_model():
    names = live_names(MLA_KEYS, _args(mla_down_proj_fusion=True, mtp_num_layers=1))
    assert names == {
        MLA_KEYS[0]: "decoder.layers.0.self_attention.linear_qkv_down_proj.layer_norm_weight",
        MLA_KEYS[1]: "decoder.layers.0.self_attention.linear_qkv_down_proj.weight",
        MLA_KEYS[2]: "decoder.layers.0.self_attention.linear_qkv_down_proj.weight",
        MLA_KEYS[3]: MLA_KEYS[3],
        MLA_KEYS[4]: MLA_KEYS[4],
        MLA_KEYS[5]: MLA_KEYS[5],
        MLA_KEYS[6]: "mtp.layers.0.mtp_model_layer.self_attention.linear_qkv_down_proj"
        ".layer_norm_weight",
        MLA_KEYS[7]: "mtp.layers.0.mtp_model_layer.self_attention.linear_qkv_down_proj.weight",
    }


def test_live_names_of_fused_mla_in_a_hybrid_model_keep_the_input_layer_norm():
    keys = MLA_KEYS[:3]
    names = live_names(keys, _args(mla_down_proj_fusion=True, hybrid_layer_pattern="+-"))
    assert names[keys[0]] == keys[0]
    assert names[keys[1]] == "decoder.layers.0.self_attention.linear_qkv_down_proj.weight"


def test_live_names_of_unfused_mla_are_their_stored_names():
    assert live_names(MLA_KEYS, _args(mtp_num_layers=1))[MLA_KEYS[1]] == MLA_KEYS[1]
