# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Weights-only args sanitation: only existing fields change, and every change is reported."""

from __future__ import annotations

import argparse

import torch

from tools.checkpoint.format_converter.mcore_state.args_sanitize import (
    sanitize_for_fsdp_dtensor,
    sanitize_for_torch_dist,
    sanitize_weights_only,
)


def test_existing_fields_are_set_and_reported():
    args = argparse.Namespace(no_save_optim=False, no_save_rng=None, params_dtype=torch.bfloat16)
    result = sanitize_weights_only(args)
    assert result.changes == {"no_save_optim": (False, True), "no_save_rng": (None, True)}
    assert result.consulted == {"no_save_optim": False, "no_save_rng": None}
    assert vars(result.args) == {
        "no_save_optim": True,
        "no_save_rng": True,
        "params_dtype": torch.bfloat16,
    }
    assert vars(args) == {
        "no_save_optim": False,
        "no_save_rng": None,
        "params_dtype": torch.bfloat16,
    }


def test_missing_fields_are_not_added_and_unchanged_ones_not_reported():
    args = argparse.Namespace(no_save_optim=True, num_layers=2)
    result = sanitize_weights_only(args)
    assert result.changes == {} and result.consulted == {"no_save_optim": True}
    assert vars(result.args) == {"no_save_optim": True, "num_layers": 2}
    assert result.args is not args


def _training_args(**overrides):
    fields = {
        "ckpt_format": "torch_dist",
        "use_megatron_fsdp": False,
        "no_save_optim": False,
        "no_save_rng": False,
        "pipeline_model_parallel_size": 2,
        "virtual_pipeline_model_parallel_size": 2,
        "num_layers_per_virtual_pipeline_stage": None,
        "hybrid_layer_pattern": "*-|*-|*-|*-",
        "params_dtype": torch.bfloat16,
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)


def test_fsdp_dtensor_arguments_describe_a_single_stage_megatron_fsdp_store():
    args = _training_args()
    result = sanitize_for_fsdp_dtensor(args)
    assert result.changes == {
        "no_save_optim": (False, True),
        "no_save_rng": (False, True),
        "ckpt_format": ("torch_dist", "fsdp_dtensor"),
        "use_megatron_fsdp": (False, True),
        "pipeline_model_parallel_size": (2, 1),
        "virtual_pipeline_model_parallel_size": (2, None),
        "hybrid_layer_pattern": ("*-|*-|*-|*-", "*-*-*-*-"),
    }
    assert vars(result.args)["params_dtype"] is torch.bfloat16
    assert "transformer_pipeline_model_parallel_size" not in vars(result.args)  # never added
    assert vars(args) == vars(_training_args())


def test_fsdp_dtensor_arguments_build_the_experts_module_the_store_names():
    """``--expert-naming`` against the arguments: the store's arguments must build its naming."""
    args = _training_args(moe_grouped_gemm=True)
    assert "moe_grouped_gemm" not in sanitize_for_fsdp_dtensor(args, grouped_gemm=True).changes
    result = sanitize_for_fsdp_dtensor(args, grouped_gemm=False)
    assert result.changes["moe_grouped_gemm"] == (True, False)
    assert "moe_grouped_gemm" not in sanitize_for_fsdp_dtensor(args).changes  # not constrained
    unset = sanitize_for_fsdp_dtensor(_training_args(), grouped_gemm=True)
    assert "moe_grouped_gemm" not in vars(unset.args)  # only existing fields change


def test_fsdp_dtensor_arguments_record_the_sizes_fused_tensors_are_packed_at():
    """The pack size replaces the (expert) tensor-parallel size, with the weight shards a job
    saves as many of; an unset shard count stays unset (validate_args derives it)."""
    args = _training_args(
        tensor_model_parallel_size=2,
        tensor_parallel_num_weight_shards=2,
        expert_tensor_parallel_size=None,
        expert_tensor_parallel_num_weight_shards=None,
    )
    result = sanitize_for_fsdp_dtensor(args, pack_tp=1, pack_etp=4)
    assert {k: v for k, v in result.changes.items() if "tensor" in k} == {
        "tensor_model_parallel_size": (2, 1),
        "tensor_parallel_num_weight_shards": (2, 1),
        "expert_tensor_parallel_size": (None, 4),
    }
    untouched = sanitize_for_fsdp_dtensor(args)  # nothing is packed: any size loads the store
    assert not {k for k in untouched.changes if "tensor" in k}
    same = sanitize_for_fsdp_dtensor(args, pack_tp=2)
    assert not {k for k in same.changes if "tensor" in k}
    bare = sanitize_for_fsdp_dtensor(_training_args(), pack_tp=2, pack_etp=2)
    assert "tensor_model_parallel_size" not in vars(bare.args)  # only existing fields change
    back = sanitize_for_torch_dist(result.args, {f: old for f, (old, _) in result.changes.items()})
    assert vars(back.args)["tensor_model_parallel_size"] == 2
    assert vars(back.args)["expert_tensor_parallel_size"] is None


def test_torch_dist_arguments_restore_a_manifest_but_never_clear_no_save_flags():
    fsdp = sanitize_for_fsdp_dtensor(_training_args())
    restore = {field: old for field, (old, _) in fsdp.changes.items()}
    back = sanitize_for_torch_dist(fsdp.args, restore)
    assert back.changes == {
        "ckpt_format": ("fsdp_dtensor", "torch_dist"),
        "use_megatron_fsdp": (True, False),
        "pipeline_model_parallel_size": (1, 2),
        "virtual_pipeline_model_parallel_size": (None, 2),
        "hybrid_layer_pattern": ("*-*-*-*-", "*-|*-|*-|*-"),
    }
    assert vars(back.args) == {**vars(_training_args()), "no_save_optim": True, "no_save_rng": True}


def test_torch_dist_arguments_without_a_manifest():
    fsdp_args = _training_args(ckpt_format="fsdp_dtensor", use_megatron_fsdp=True, no_save_rng=True)
    result = sanitize_for_torch_dist(fsdp_args)
    assert result.changes == {
        "ckpt_format": ("fsdp_dtensor", "torch_dist"),
        "use_megatron_fsdp": (True, False),
        "no_save_optim": (False, True),
    }
