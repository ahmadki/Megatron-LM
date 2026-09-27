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


def test_sentinel_keys_are_declined_with_their_reason():
    stacked = _slice("decoder.layers.mlp.linear_fc1.weight", (2, 8, 8))
    part = _slice("decoder.layers.0.mixer.in_proj.weight.z", (8, 8))
    plain = _slice("output_layer.weight", (16, 8))
    pool = LogicalPool([stacked, part, plain])
    taken = list(IdentityRule().claim(pool, _context(TORCH_DIST, FSDP_DTENSOR)))
    assert [instance.inputs for instance in taken] == [(plain,)]
    assert pool.remaining() == [stacked, part]
    (reason,) = pool.declined(stacked.lid)
    assert reason.startswith("a tensor stacked over the layers") and "FC-P01" in reason
    assert "FC-P05, FC-P08, FC-P09" in pool.declined(part.lid)[0]


def test_sentinels_apply_to_their_source_format_only():
    half = _slice("model.module.decoder.layers.1.mlp.linear_fc1.weight_w", (8, 8))
    pool = LogicalPool([half])
    assert list(IdentityRule().claim(pool, _context(FSDP_DTENSOR, TORCH_DIST))) == []
    assert "FC-P07" in pool.declined(half.lid)[0]
    pool = LogicalPool([half])
    assert len(list(IdentityRule().claim(pool, _context(TORCH_DIST, FSDP_DTENSOR)))) == 1
