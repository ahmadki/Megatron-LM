# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The identity rule copies every unreserved slice whole and leaves reserved ones declined."""

from __future__ import annotations

import argparse

import pytest
import torch

from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.context import build_arch_context
from tools.checkpoint.format_converter.core.logical import LogicalCopy, LogicalId, LogicalSlice
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, TORCH_DIST
from tools.checkpoint.format_converter.core.registry import REGISTRY
from tools.checkpoint.format_converter.core.rule import LogicalPool, Stage
from tools.checkpoint.format_converter.engine.regions import Region
from tools.checkpoint.format_converter.rules.identity import IdentityRule

pytestmark = pytest.mark.covers("identity", "ground_truth")


def _context(source: str, target: str):
    args = ArgsView(argparse.Namespace(bf16=True))
    return build_arch_context(source=source, target=target, args=args, model={})


def _slice(key: str, shape: tuple[int, ...], dtype=torch.bfloat16) -> LogicalSlice:
    path, _, leaf = key.rpartition(".")
    return LogicalSlice(LogicalId(None, None, path, leaf), key, Region.full(shape), dtype)


def test_identity_is_registered_last():
    assert isinstance(REGISTRY.rules["identity"], IdentityRule)
    assert IdentityRule.stage is Stage.IDENTITY


def test_unreserved_slices_are_copied_whole_with_their_dtype():
    norm = _slice("decoder.final_norm.weight", (8,), torch.float32)
    qkv = _slice("embedding.word_embeddings.weight", (16, 8))
    pool = LogicalPool([norm, qkv])
    instances = list(IdentityRule().claim(pool, _context(TORCH_DIST, FSDP_DTENSOR)))
    assert [instance.inputs for instance in instances] == [(norm,), (qkv,)]
    output = instances[0].outputs[0]
    assert (output.lid, output.shape, output.dtype) == (norm.lid, (8,), torch.float32)
    assert output.copies == (LogicalCopy(norm.lid, Region.full((8,)), Region.full((8,))),)
    for instance in instances:
        pool.consume(instance)
    assert pool.remaining() == [] and pool.unconsumed_claims() == []


def test_a_layer_of_a_stacked_tensor_is_copied_with_the_layer_shape():
    lid = LogicalId("decoder", 1, "mlp.linear_fc1", "weight")
    layer = LogicalSlice(
        lid,
        "decoder.layers.mlp.linear_fc1.weight",
        Region((1, 0, 0), (1, 16, 8)),
        torch.bfloat16,
        1,
    )
    (instance,) = IdentityRule().claim(LogicalPool([layer]), _context(TORCH_DIST, FSDP_DTENSOR))
    (output,) = instance.outputs
    assert (output.lid, output.shape) == (lid, (16, 8))
    assert output.copies == (LogicalCopy(lid, Region.full((16, 8)), Region.full((16, 8))),)


def test_one_expert_of_an_expert_tensor_is_copied_as_that_experts_tensor():
    lid = LogicalId("decoder", 1, "mlp.experts.linear_fc1", "weight", 3)
    expert = LogicalSlice(
        lid,
        "decoder.layers.mlp.experts.experts.linear_fc1.weight",
        Region((1, 3, 0, 0), (1, 1, 6, 8)),
        torch.bfloat16,
        2,
    )
    (instance,) = IdentityRule().claim(LogicalPool([expert]), _context(TORCH_DIST, FSDP_DTENSOR))
    (output,) = instance.outputs
    assert (output.lid, output.shape, output.dtype) == (lid, (6, 8), torch.bfloat16)


@pytest.mark.parametrize(
    ("key", "dtype"),
    [
        ("decoder.layers.1.mlp.router.expert_bias", torch.float32),
        ("decoder.layers.1.mlp.router.qb_beta", torch.float32),
        ("decoder.layers.1.mlp.router.tid2eid", torch.int32),
    ],
)
def test_router_buffers_keep_their_dtype_in_a_bf16_model(key, dtype):
    buffer = _slice(key, (16, 2), dtype)
    (instance,) = IdentityRule().claim(LogicalPool([buffer]), _context(FSDP_DTENSOR, TORCH_DIST))
    assert instance.outputs[0].dtype == dtype


def test_sentinel_keys_are_declined_with_their_reason():
    part = _slice("decoder.layers.0.mixer.in_proj.weight.z", (8, 8))
    plain = _slice("output_layer.weight", (16, 8))
    pool = LogicalPool([part, plain])
    taken = list(IdentityRule().claim(pool, _context(TORCH_DIST, FSDP_DTENSOR)))
    assert [instance.inputs for instance in taken] == [(plain,)]
    assert pool.remaining() == [part]
    (reason,) = pool.declined(part.lid)
    assert reason.startswith("a parameter stored as parts") and "gdn, gdn2, gdp, mamba2" in reason


def test_sentinels_apply_to_their_source_format_only():
    half = _slice("model.module.decoder.layers.1.mlp.linear_fc1.weight_w", (8, 8))
    pool = LogicalPool([half])
    assert list(IdentityRule().claim(pool, _context(FSDP_DTENSOR, TORCH_DIST))) == []
    assert "(owned by glu)" in pool.declined(half.lid)[0]
    pool = LogicalPool([half])
    assert len(list(IdentityRule().claim(pool, _context(TORCH_DIST, FSDP_DTENSOR)))) == 1
