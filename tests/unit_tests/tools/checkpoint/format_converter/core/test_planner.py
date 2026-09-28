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


def dim0_regions(shape: tuple[int, ...], rows: int) -> list[Region]:
    return Region.full(shape).split_dim0(rows)


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


def test_a_stacked_decoder_is_written_one_layer_per_key():
    """A GPT decoder stored stacked, each layer split in two column chunks (tensor parallel)."""
    fc1, norm = (
        "decoder.layers.mlp.linear_fc1.weight",
        "decoder.layers.mlp.linear_fc1.layer_norm_weight",
    )
    grid = [Region((layer, 0, col), (1, 4, 2)) for layer in (0, 1) for col in (0, 2)]
    model = {fc1: entry(fc1, (2, 4, 4), grid), norm: entry(norm, (2, 4))}
    args = ArgsView(argparse.Namespace(num_layers=2))
    ctx = build_arch_context(source=TORCH_DIST, target=FSDP_DTENSOR, args=args, model=model)
    (block,) = ctx.blocks
    assert (block.prefix, block.td_layout, block.num_modules) == ("decoder", "stacked", 2)
    plan = plan_model(ctx, model, FsdpMfsdpV1Adapter(), rules=[IdentityRule()], max_chunk_bytes=32)
    assert plan.claims == {"identity": 4} and plan.outputs == 4
    chunks = model[fc1].chunks
    expected = []
    for layer in (0, 1):
        for rows in (0, 2):  # 16-byte rows, 2 per 32-byte chunk
            left, right = chunks[2 * layer], chunks[2 * layer + 1]
            expected.append(
                (
                    f"model.module.decoder.layers.{layer}.mlp.linear_fc1.weight",
                    Region((rows, 0), (2, 4)),
                    (
                        CopyOp(left, Region((layer, rows, 0), (1, 2, 2)), Region((0, 0), (2, 2))),
                        CopyOp(right, Region((layer, rows, 2), (1, 2, 2)), Region((0, 2), (2, 2))),
                    ),
                )
            )
    weights = [p for p in plan.pieces if p.key.endswith("linear_fc1.weight")]
    assert [(p.key, p.chunk, p.ops) for p in weights] == expected
    assert {p.global_shape for p in weights} == {(4, 4)}
    norms = [p for p in plan.pieces if p.key.endswith("layer_norm_weight")]
    (norm_chunk,) = model[norm].chunks
    assert [(p.key, p.ops) for p in norms] == [
        (
            f"model.module.decoder.layers.{layer}.mlp.linear_fc1.layer_norm_weight",
            (CopyOp(norm_chunk, Region((layer, 0), (1, 4)), Region((0,), (4,))),),
        )
        for layer in (0, 1)
    ]


def test_layers_of_an_fsdp_store_are_stacked_one_chunk_per_layer():
    """Each layer of the stacked target is its own chunk, filled from that layer's stored rows."""
    prefix = "model.module.decoder.layers"
    model = {}
    for layer in (0, 1):
        key = f"{prefix}.{layer}.mlp.linear_fc1.weight"
        model[key.removeprefix("model.module.")] = entry(key, (4, 4), dim0_regions((4, 4), 2))
    args = ArgsView(argparse.Namespace(num_layers=2))
    ctx = build_arch_context(source=FSDP_DTENSOR, target=TORCH_DIST, args=args, model=model)
    assert [(b.prefix, b.td_layout) for b in ctx.blocks] == [("decoder", "stacked")]
    plan = plan_model(ctx, model, TorchDistAdapter(), rules=[IdentityRule()], max_chunk_bytes=8)
    stacked = "decoder.layers.mlp.linear_fc1.weight"
    expected = []
    for layer in (0, 1):
        top, bottom = model[f"decoder.layers.{layer}.mlp.linear_fc1.weight"].chunks
        expected.append(
            (
                Region((layer, 0, 0), (1, 4, 4)),
                (
                    CopyOp(top, Region((0, 0), (2, 4)), Region((0, 0, 0), (1, 2, 4))),
                    CopyOp(bottom, Region((2, 0), (2, 4)), Region((0, 2, 0), (1, 2, 4))),
                ),
            )
        )
    assert [(p.key, p.global_shape) for p in plan.pieces] == [(stacked, (2, 4, 4))] * 2
    assert [(p.chunk, p.ops) for p in plan.pieces] == expected
    assert all(p.planner_path is None for p in plan.pieces)


EXPERTS_FC1 = "decoder.layers.mlp.experts.experts.linear_fc1.weight"


@pytest.mark.parametrize(
    ("naming", "name"),
    [("grouped", "linear_fc1.weight{e}"), ("sequential", "local_experts.{e}.linear_fc1.weight")],
)
def test_stacked_experts_are_written_one_tensor_per_layer_and_expert(naming, name):
    """torch_dist's ``[L, E, 4, 4]`` saved one chunk per (layer, expert), split in two columns."""
    grid = [
        Region((layer, expert, 0, col), (1, 1, 4, 2))
        for layer in (0, 1)
        for expert in (0, 1, 2)
        for col in (0, 2)
    ]
    model = {EXPERTS_FC1: entry(EXPERTS_FC1, (2, 3, 4, 4), grid)}
    args = ArgsView(argparse.Namespace(num_layers=2, num_experts=3))
    ctx = build_arch_context(
        source=TORCH_DIST, target=FSDP_DTENSOR, args=args, model=model, expert_naming=naming
    )
    plan = plan_model(ctx, model, FsdpMfsdpV1Adapter(), rules=[IdentityRule()], max_chunk_bytes=64)
    assert plan.claims == {"identity": 6} and plan.outputs == 6
    chunks = iter(model[EXPERTS_FC1].chunks)
    expected = []
    for layer in (0, 1):
        for expert in (0, 1, 2):
            left, right = next(chunks), next(chunks)
            expected.append(
                (
                    f"model.module.decoder.layers.{layer}.mlp.experts.{name.format(e=expert)}",
                    (4, 4),
                    Region((0, 0), (4, 4)),
                    (
                        CopyOp(
                            left,
                            Region((layer, expert, 0, 0), (1, 1, 4, 2)),
                            Region((0, 0), (4, 2)),
                        ),
                        CopyOp(
                            right,
                            Region((layer, expert, 0, 2), (1, 1, 4, 2)),
                            Region((0, 2), (4, 2)),
                        ),
                    ),
                )
            )
    assert sorted((p.key, p.global_shape, p.chunk, p.ops) for p in plan.pieces) == sorted(expected)


@pytest.mark.parametrize(
    ("naming", "name"),
    [("grouped", "linear_fc1.weight{e}"), ("sequential", "local_experts.{e}.linear_fc1.weight")],
)
def test_experts_of_an_fsdp_store_are_stacked_one_chunk_per_layer_and_expert(naming, name):
    model = {}
    for layer in (0, 1):
        for expert in (0, 1, 2):
            key = f"model.module.decoder.layers.{layer}.mlp.experts.{name.format(e=expert)}"
            model[key.removeprefix("model.module.")] = entry(key, (4, 4), dim0_regions((4, 4), 2))
    args = ArgsView(
        argparse.Namespace(num_layers=2, num_experts=3, moe_grouped_gemm=naming == "grouped")
    )
    ctx = build_arch_context(source=FSDP_DTENSOR, target=TORCH_DIST, args=args, model=model)
    assert ctx.experts is not None and ctx.experts.fsdp_form == naming
    plan = plan_model(ctx, model, TorchDistAdapter(), rules=[IdentityRule()], max_chunk_bytes=8)
    expected = []
    for layer in (0, 1):
        for expert in (0, 1, 2):
            relative = f"decoder.layers.{layer}.mlp.experts.{name.format(e=expert)}"
            top, bottom = model[relative].chunks
            expected.append(
                (
                    Region((layer, expert, 0, 0), (1, 1, 4, 4)),
                    (
                        CopyOp(top, Region((0, 0), (2, 4)), Region((0, 0, 0, 0), (1, 1, 2, 4))),
                        CopyOp(bottom, Region((2, 0), (2, 4)), Region((0, 0, 2, 0), (1, 1, 2, 4))),
                    ),
                )
            )
    assert {(p.key, p.global_shape) for p in plan.pieces} == {(EXPERTS_FC1, (2, 3, 4, 4))}
    assert [(p.chunk, p.ops) for p in plan.pieces] == expected


def test_unclaimed_slices_are_refused_with_the_reasons():
    model = source()
    ctx = context(TORCH_DIST, FSDP_DTENSOR, model)
    part = "decoder.layers.0.mixer.in_proj.weight.z"  # a part a sentinel reserves
    extended = {**model, part: entry(part, (4, 4))}
    with pytest.raises(RefusedError) as excinfo:
        plan_model(ctx, extended, FsdpMfsdpV1Adapter(), rules=[IdentityRule()], max_chunk_bytes=64)
    refusal = excinfo.value.refusal
    assert refusal.code == errors.FC_R13 and refusal.keys == (part,)
    assert "not handled by any rule: a parameter stored as parts" in refusal.message


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
        elif self.bug == "same key":  # both halves written over one another under one key
            other = Region((3, 0), (3, 4))
            twin = LogicalId(None, None, "decoder.layers.0.self_attention.linear_qkv", "weight")
            yield FamilyInstance(
                self.name,
                (piece,),
                (
                    LogicalOutput(
                        piece.lid, (3, 4), piece.dtype, (LogicalCopy(piece.lid, half, half),)
                    ),
                    LogicalOutput(
                        twin, (3, 4), piece.dtype, (LogicalCopy(piece.lid, other, half),)
                    ),
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
        ("same key", "the planned chunks of .*linear_qkv.weight do not tile it"),
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
