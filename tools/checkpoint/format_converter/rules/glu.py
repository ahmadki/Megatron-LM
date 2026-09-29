# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The GLU rule: gated ``linear_fc1`` that fsdp_dtensor stores as SwiGLU halves.

torch_dist stores a gated first projection whole, ``[W; V]``; Megatron-FSDP
stores the ones its SwiGLU predicate splits (``core/glu.py``) as two tensors,
the gate half ``W`` and the value half ``V``, both TP-canonical. The rule
takes the weight and bias of every such first projection and writes each half
as an output of its own: cut out of the whole tensor (rows ``[0, F)`` and
``[F, 2F)``), or copied as they are from an fsdp_dtensor source. The target
format places them (``core/structure.py``): fsdp_dtensor as ``<name>_w`` and
``<name>_v``, torch_dist as the two row blocks of the whole tensor, each
written as a chunk of its own (Megatron's SwiGLU factory saves each half as
chunks of its own, one per tensor-parallel rank).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence

from ..core.context import ArchContext
from ..core.keys import GLU_PARTS
from ..core.logical import LogicalCopy, LogicalId, LogicalOutput, LogicalSlice
from ..core.part_layout import PartLayout
from ..core.registry import REGISTRY
from ..core.rule import FamilyInstance, LeafRule, LogicalPool, Stage
from ..engine.regions import Region


@REGISTRY.rule
class GluRule(LeafRule):
    """Write the weight and bias of a SwiGLU-split ``linear_fc1`` as its two halves."""

    name = "glu"
    stage = Stage.GLU

    def claim(self, pool: LogicalPool, ctx: ArchContext) -> Iterator[FamilyInstance]:
        """Take every tensor (or half) of a split first projection, grouped by whole tensor."""

        def split(piece: LogicalSlice) -> bool:
            fc1 = ctx.glu.of(piece.lid)
            return fc1 is not None and fc1.split

        tensors: dict[LogicalId, list[LogicalSlice]] = {}
        for piece in pool.take(self.name, split):
            tensors.setdefault(piece.lid.whole, []).append(piece)
        for whole, pieces in tensors.items():
            yield FamilyInstance(self.name, tuple(pieces), _halves(whole, pieces, ctx))


def _halves(
    whole: LogicalId, pieces: Sequence[LogicalSlice], ctx: ArchContext
) -> tuple[LogicalOutput, ...]:
    """The two halves of ``whole``: cut out of the whole tensor, or its stored halves as is."""
    if len(pieces) == 1 and pieces[0].lid.part is None:
        (piece,) = pieces
        rows, rest = piece.shape[0], piece.shape[1:]
        layout = PartLayout(GLU_PARTS, (rows // len(GLU_PARTS),) * len(GLU_PARTS))
        return tuple(
            LogicalOutput(
                dataclasses.replace(whole, part=run.part),
                (run.rows, *rest),
                ctx.dtype_policy.output_dtype(piece.dtype),
                (
                    LogicalCopy(
                        piece.lid,
                        _row_box(run.fused_start, run.rows, rest),
                        _row_box(run.part_start, run.rows, rest),
                    ),
                ),
            )
            for run in layout.rank_major_map(1)
        )
    return tuple(
        LogicalOutput(
            piece.lid,
            piece.shape,
            ctx.dtype_policy.output_dtype(piece.dtype),
            (LogicalCopy(piece.lid, Region.full(piece.shape), Region.full(piece.shape)),),
        )
        for piece in pieces
    )


def _row_box(start: int, rows: int, rest: tuple[int, ...]) -> Region:
    """Rows ``[start, start + rows)`` of a tensor whose other dimensions are ``rest``."""
    return Region((start,) + (0,) * len(rest), (rows, *rest))
