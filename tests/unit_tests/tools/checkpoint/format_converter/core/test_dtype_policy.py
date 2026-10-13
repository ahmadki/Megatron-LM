# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Weights-only conversions preserve every dtype, integers and fp32 buffers included; with
masters, a trainable parameter's model weight takes the parameter dtype unless Megatron keeps it
in fp32."""

from __future__ import annotations

import pytest
import torch

from tools.checkpoint.format_converter.core.dtype_policy import (
    DtypePolicy,
    keeps_fp32,
    model_weight_dtype,
)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64, torch.int64, torch.bool]
)
@pytest.mark.parametrize("policy", list(DtypePolicy))
def test_rules_copy_every_dtype_as_stored(policy, dtype):
    assert policy.output_dtype(dtype) is dtype


@pytest.mark.parametrize(
    "name",
    [
        "self_attention_hyper_connection.mapping_proj.weight",
        "mlp_hyper_connection.alpha_pre",
        "hyper_connection.alpha_post",
        "mlp_hyper_connection.alpha_res",
        "self_attention_hyper_connection.bias",
        "decoder.hc_head_fn",
        "hc_head_base",
        "language_model.mtp.layers.0.hc_head_scale",
        "self_attention.core_attention.compressor.ape",
        "self_attention.core_attention.attn_sink",
    ],
)
def test_the_registry_keeps_megatrons_fp32_parameters(name):
    assert keeps_fp32(name)
    assert model_weight_dtype(name, torch.bfloat16) is torch.float32


@pytest.mark.parametrize(
    "name",
    [
        "self_attention.linear_qkv.bias",  # not a hyper-connection's bias
        "mlp.router.weight",  # created fp32, cast by Float16Module
        "mapping_proj.weight",
        "decoder.final_layernorm.weight",
        "mlp.experts.linear_fc1.weight3",
        "shape",  # ends like ape, but is no compressor's
    ],
)
def test_every_other_parameter_takes_the_parameter_dtype(name):
    assert not keeps_fp32(name)
    assert model_weight_dtype(name, torch.bfloat16) is torch.bfloat16
    assert model_weight_dtype(name, torch.float16) is torch.float16
