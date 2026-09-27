# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Planning the model section of a cross-format conversion, from metadata only.

:func:`plan_model` runs the rule pipeline and compiles its result into
engine pieces:

1. **Decompose** every source model tensor into logical slices (the key
   grammar and the blocks' layer numbering, ``core/structure.py``): one per
   tensor, or one per layer of a tensor stacked over a block's layers.
2. **Claim**: the rules run in stage order over one :class:`~.rule.LogicalPool`;
   every slice is claimed and consumed exactly once. Slices left over
   (including those a sentinel reserves) are refused with FC-R13 and the
   rules' reasons.
3. **Compose** every output's target key and chunk grid (the target adapter).
4. **Compile** each output chunk into a :class:`~..engine.pieces.Piece` whose
   copy operations read boxes of the source's stored chunks.
5. **Coverage**: the copies must read every element of every consumed source
   tensor exactly once, every piece must be tiled by its copies and every
   output tensor by its pieces, and no two outputs may share a key. Any
   violation is a bug in a rule (exit 1), never silently written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from .. import errors
from ..engine.metadata import TensorEntry
from ..engine.pieces import CopyOp, Piece, expected_tensors
from ..engine.regions import Region, corresponding, find_tiling_defect
from ..errors import InternalError, Refusal
from .context import ArchContext
from .keys import KeyGrammarError
from .logical import LogicalId, LogicalOutput, LogicalSlice
from .rule import FamilyInstance, LeafRule, LogicalPool
from .structure import compose, decompose


class TargetLayout(Protocol):
    """What the planner needs from the target format's adapter."""

    fmt: str

    def stored_key(self, relative: str) -> str:
        """The store key of a model-relative key."""

    def planner_path(self, key: str) -> tuple[str, ...] | None:
        """The planner path of an output model key (``None``: not recorded)."""

    def output_grid(
        self, shape: tuple[int, ...], dtype: torch.dtype, *, max_chunk_bytes: int
    ) -> tuple[Region, ...]:
        """The chunk grid an output tensor is written with."""


@dataclass(frozen=True)
class ModelPlan:
    """The planned model section of a conversion.

    Attributes:
        pieces: The output pieces, by (key, offsets).
        claims: Source slices claimed per rule.
        outputs: Number of output tensors.
    """

    pieces: tuple[Piece, ...]
    claims: dict[str, int]
    outputs: int


def plan_model(
    ctx: ArchContext,
    source: Mapping[str, TensorEntry],
    target: TargetLayout,
    *,
    rules: Sequence[LeafRule],
    max_chunk_bytes: int,
) -> ModelPlan:
    """Plan the model tensors of ``source`` as ``target`` tensors.

    Args:
        ctx: The architecture context.
        source: The source's model tensors by model-relative key.
        target: The target format's adapter.
        rules: The enabled rules, in stage order.
        max_chunk_bytes: Chunk-size limit passed to the target's grid.

    Raises:
        RefusedError: A source key does not fit the key grammar or the block
            layouts, or slices are left unclaimed (FC-R13).
        InternalError: A rule claimed or consumed slices wrongly, or its
            copies do not cover the sources and outputs exactly.
    """
    pool = LogicalPool(_decompose_all(ctx, source))
    instances: list[FamilyInstance] = []
    for rule in rules:
        for instance in rule.claim(pool, ctx):
            if instance.rule != rule.name:
                raise InternalError(f"rule {rule.name} returned an instance of {instance.rule}")
            pool.consume(instance)
            instances.append(instance)
    _refuse_leftovers(pool)
    unconsumed = pool.unconsumed_claims()
    if unconsumed:
        raise InternalError(
            f"rules claimed {len(unconsumed)} slices they did not consume, e.g. {unconsumed[0].lid}"
        )
    slices = {piece.lid: piece for instance in instances for piece in instance.inputs}
    outputs = [output for instance in instances for output in instance.outputs]
    pieces = _compile(ctx, outputs, slices, source, target, max_chunk_bytes)
    _check_source_coverage(pieces, slices.values(), source)
    expected_tensors(pieces)  # every output tensor is tiled exactly by its pieces
    return ModelPlan(tuple(pieces), pool.claims(), len(outputs))


def _decompose_all(ctx: ArchContext, source: Mapping[str, TensorEntry]) -> list[LogicalSlice]:
    slices, malformed = [], []
    for relative, entry in source.items():
        try:
            slices.extend(decompose(relative, entry, ctx.blocks, ctx.source[0]))
        except KeyGrammarError as err:
            malformed.append((entry.key, str(err)))
    if malformed:
        raise Refusal(
            errors.FC_R13,
            f"model keys do not fit the key grammar or the layer layout ({malformed[0][1]})",
            keys=tuple(key for key, _ in malformed),
            hint="check that the arguments belong to this checkpoint",
        ).to_error()
    return slices


def _refuse_leftovers(pool: LogicalPool) -> None:
    leftovers = pool.remaining()
    if not leftovers:
        return
    reasons = sorted({reason for piece in leftovers for reason in pool.declined(piece.lid)})
    raise Refusal(
        errors.FC_R13,
        f"{len(leftovers)} model tensors are not handled by any rule"
        + (f": {'; '.join(reasons)}" if reasons else ""),
        keys=tuple(piece.key for piece in leftovers),
        hint="the model uses a layer layout this version cannot convert",
    ).to_error()


def _compile(
    ctx: ArchContext,
    outputs: Sequence[LogicalOutput],
    slices: Mapping[LogicalId, LogicalSlice],
    source: Mapping[str, TensorEntry],
    target: TargetLayout,
    max_chunk_bytes: int,
) -> list[Piece]:
    entries = {entry.key: entry for entry in source.values()}
    pieces: list[Piece] = []
    owners: dict[str, LogicalId] = {}
    for output in outputs:
        key = target.stored_key(compose(output.lid, ctx.blocks, target.fmt))
        owner = owners.setdefault(key, output.lid)
        if owner != output.lid:
            raise InternalError(f"{owner} and {output.lid} would both be written as {key}")
        for chunk in target.output_grid(
            output.shape, output.dtype, max_chunk_bytes=max_chunk_bytes
        ):
            ops = tuple(_chunk_ops(output, chunk, slices, entries))
            pieces.append(
                Piece(key, output.shape, output.dtype, chunk, ops, target.planner_path(key))
            )
    pieces.sort(key=lambda piece: (piece.key, piece.chunk.offsets))
    return pieces


def _chunk_ops(
    output: LogicalOutput,
    chunk: Region,
    slices: Mapping[LogicalId, LogicalSlice],
    entries: Mapping[str, TensorEntry],
) -> list[CopyOp]:
    """The copies of stored source boxes that fill output chunk ``chunk``.

    Each copy's destination box is cut to the chunk, mapped to the box of the
    stored source tensor that holds it (through the source slice, which may
    be one layer of a stacked tensor) and split along the source's stored
    chunks; every piece of it becomes a copy into the chunk's buffer.
    """
    ops = []
    for copy in output.copies:
        part = copy.dst_region.intersect(chunk)
        if part is None:
            continue
        source = slices[copy.src]
        src_box = source.stored_box(corresponding(part, copy.dst_region, copy.src_region))
        dst_box = part.relative_to(chunk)
        for stored in entries[source.key].chunks:
            read = src_box.intersect(stored.region)
            if read is not None:
                ops.append(CopyOp(stored, read, corresponding(read, src_box, dst_box)))
    return ops


def _check_source_coverage(
    pieces: Sequence[Piece], slices: Sequence[LogicalSlice], source: Mapping[str, TensorEntry]
) -> None:
    """Every consumed source tensor must be read exactly once, element for element."""
    shapes = {entry.key: entry.shape for entry in source.values()}
    read: dict[str, list[Region]] = {piece.key: [] for piece in slices}
    for piece in pieces:
        for op in piece.ops:
            read.setdefault(op.src.key, []).append(op.src_region)
    for key, boxes in read.items():
        defect = find_tiling_defect(boxes, shapes[key])
        if defect is not None:
            raise InternalError(f"the plan does not read {key} exactly once: {defect}")
