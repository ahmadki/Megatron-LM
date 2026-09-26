# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Plans compile into pieces that read every source element once and tile every output.

Expected pieces are written out by hand from the chunk grids.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator

import pytest
import torch

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.context import build_arch_context
from tools.checkpoint.format_converter.core.logical import LogicalCopy, LogicalId, LogicalOutput
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, TORCH_DIST
from tools.checkpoint.format_converter.core.planner import plan_model
from tools.checkpoint.format_converter.core.registry import REGISTRY
from tools.checkpoint.format_converter.core.rule import FamilyInstance, LeafRule, Stage
from tools.checkpoint.format_converter.engine.metadata import StoredChunk, TensorEntry
from tools.checkpoint.format_converter.engine.pieces import CopyOp
from tools.checkpoint.format_converter.engine.regions import Region
from tools.checkpoint.format_converter.errors import InternalError, RefusedError
from tools.checkpoint.format_converter.formats.fsdp_mfsdp_v1 import FsdpMfsdpV1Adapter
from tools.checkpoint.format_converter.formats.torch_dist import TorchDistAdapter
from tools.checkpoint.format_converter.rules.identity import IdentityRule

QKV = "decoder.layers.0.self_attention.linear_qkv.weight"
FC1 = "decoder.layers.1.mlp.linear_fc1.weight"
NORM = "decoder.final_norm.weight"


def entry(key: str, shape: tuple[int, ...], grid: list[Region] | None = None) -> TensorEntry:
    """A float32 tensor entry stored as ``grid`` (default: one chunk)."""
    chunks = tuple(
        StoredChunk(key, r.offsets, r.sizes, torch.float32, "__0_0.distcp", 64 * i, 8, ())
        for i, r in enumerate(grid or [Region.full(shape)])
    )
    return TensorEntry(key, shape, torch.float32, chunks)


def source(prefix: str = "", **grids) -> dict[str, TensorEntry]:
    """``*-`` hybrid model tensors, stored under ``prefix``."""
    shapes = {QKV: (6, 4), FC1: (4, 4), NORM: (4,)}
    return {rel: entry(prefix + rel, shape, grids.get(rel)) for rel, shape in shapes.items()}


def context(src: str, dst: str, model):
    args = ArgsView(argparse.Namespace(hybrid_layer_pattern="*-", num_layers=2))
    return build_arch_context(source=src, target=dst, args=args, model=model)


def test_torch_dist_to_fsdp_pieces_are_split_and_read_the_right_boxes():
    split = [Region((0, 0), (6, 2)), Region((0, 2), (6, 2))]  # a column-split source grid
    model = source(**{QKV: split})
    ctx = context(TORCH_DIST, FSDP_DTENSOR, model)
    plan = plan_model(ctx, model, FsdpMfsdpV1Adapter(), rules=[IdentityRule()], max_chunk_bytes=32)
    assert plan.claims == {"identity": 3} and plan.outputs == 3
    qkv = [p for p in plan.pieces if p.key == "model.module." + QKV]
    chunk_a, chunk_b = model[QKV].chunks
    expected = []
    for rows in (0, 2, 4):  # 16-byte rows, 2 per 32-byte chunk
        expected.append(
            (
                Region((rows, 0), (2, 4)),
                (
                    CopyOp(chunk_a, Region((rows, 0), (2, 2)), Region((0, 0), (2, 2))),
                    CopyOp(chunk_b, Region((rows, 2), (2, 2)), Region((0, 2), (2, 2))),
                ),
            )
        )
    assert [(p.chunk, p.ops) for p in qkv] == expected
    assert {p.planner_path for p in qkv} == {("model", "module." + QKV)}
    assert all(p.dtype == torch.float32 and p.global_shape == (6, 4) for p in qkv)
    norm = [p for p in plan.pieces if p.key.endswith(NORM)]
    assert [(p.chunk, p.ops) for p in norm] == [
        (
            Region((0,), (4,)),
            (CopyOp(model[NORM].chunks[0], Region((0,), (4,)), Region((0,), (4,))),),
        )
    ]


def test_fsdp_to_torch_dist_merges_chunks_into_one():
    rows = [Region((0, 0), (3, 4)), Region((3, 0), (3, 4))]
    model = source("model.module.", **{QKV: rows})
    ctx = context(FSDP_DTENSOR, TORCH_DIST, model)
    plan = plan_model(ctx, model, TorchDistAdapter(), rules=[IdentityRule()], max_chunk_bytes=8)
    (qkv,) = [p for p in plan.pieces if p.key == QKV]
    first, second = model[QKV].chunks
    assert qkv.chunk == Region.full((6, 4)) and qkv.planner_path is None
    assert qkv.ops == (
        CopyOp(first, Region((0, 0), (3, 4)), Region((0, 0), (3, 4))),
        CopyOp(second, Region((3, 0), (3, 4)), Region((3, 0), (3, 4))),
    )
    assert sorted(p.key for p in plan.pieces) == sorted([QKV, FC1, NORM])


def test_unclaimed_slices_are_refused_with_the_reasons():
    model = source()
    ctx = context(TORCH_DIST, FSDP_DTENSOR, model)
    stacked = "encoder.layers.mlp.linear_fc2.weight"  # outside every block, so a sentinel sees it
    extended = {**model, stacked: entry(stacked, (2, 4, 4))}
    with pytest.raises(RefusedError) as excinfo:
        plan_model(ctx, extended, FsdpMfsdpV1Adapter(), rules=[IdentityRule()], max_chunk_bytes=64)
    refusal = excinfo.value.refusal
    assert refusal.code == errors.FC_R13 and refusal.keys == (stacked,)
    assert "not handled by any rule: a tensor stacked over the layers" in refusal.message


@pytest.mark.parametrize(
    "stray", ["decoder.layers.5.mlp.linear_fc2.weight", "decoder.layers.mlp.linear_fc2.weight"]
)
def test_keys_outside_the_layout_are_refused(stray):
    model = source()
    ctx = context(TORCH_DIST, FSDP_DTENSOR, model)
    with pytest.raises(RefusedError) as excinfo:
        plan_model(
            ctx,
            {**model, stray: entry(stray, (4,))},
            FsdpMfsdpV1Adapter(),
            rules=[IdentityRule()],
            max_chunk_bytes=64,
        )
    assert excinfo.value.refusal.code == errors.FC_R13
    assert excinfo.value.refusal.keys == (stray,)


class _Rule(LeafRule):
    """A test rule over the ``linear_qkv`` slice, broken in the ways the planner must catch."""

    name = "broken"
    stage = Stage.FUSED

    def __init__(self, bug: str) -> None:
        self.bug = bug

    def claim(self, pool, ctx) -> Iterator[FamilyInstance]:
        (piece,) = pool.take(self.name, lambda s: s.key == QKV)
        if self.bug == "unconsumed":
            return
        half = Region((0, 0), (3, 4))
        if self.bug == "drops rows":
            output = LogicalOutput(
                piece.lid, (3, 4), piece.dtype, (LogicalCopy(piece.lid, half, half),)
            )
            yield FamilyInstance(self.name, (piece,), (output,))
        elif self.bug == "same key":
            full = Region.full((6, 4))
            copy = (LogicalCopy(piece.lid, full, full),)
            twin = LogicalId(None, None, "decoder.layers.0.self_attention.linear_qkv", "weight")
            yield FamilyInstance(
                self.name,
                (piece,),
                (
                    LogicalOutput(piece.lid, (6, 4), piece.dtype, copy),
                    LogicalOutput(twin, (6, 4), piece.dtype, copy),
                ),
            )
        elif self.bug == "misnamed":
            yield FamilyInstance("someone else", (piece,), ())


@pytest.mark.parametrize(
    ("bug", "message"),
    [
        ("unconsumed", "did not consume"),
        (
            "drops rows",
            "does not read decoder.layers.0.self_attention.linear_qkv.weight exactly once",
        ),
        ("same key", "would both be written as"),
        ("misnamed", "returned an instance of someone else"),
    ],
)
def test_rule_bugs_are_internal_errors(bug, message):
    model = source()
    ctx = context(TORCH_DIST, FSDP_DTENSOR, model)
    with pytest.raises(InternalError, match=message):
        plan_model(
            ctx, model, TorchDistAdapter(), rules=[_Rule(bug), IdentityRule()], max_chunk_bytes=64
        )


def test_the_registered_rules_plan_a_hybrid_model():
    model = source()
    ctx = context(TORCH_DIST, FSDP_DTENSOR, model)
    plan = plan_model(
        ctx, model, FsdpMfsdpV1Adapter(), rules=REGISTRY.rules_for(ctx), max_chunk_bytes=2**30
    )
    assert plan.claims == {"identity": 3}
    assert [p.key for p in plan.pieces] == ["model.module." + k for k in sorted([QKV, FC1, NORM])]
