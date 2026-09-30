# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The GLU rule: the gated ``linear_fc1`` that the two formats store differently.

torch_dist stores a gated first projection whole and TP-canonical,
``[W; V]`` (Megatron's SwiGLU factory saves each half as chunks of its own,
one per tensor-parallel rank; the converter writes one chunk per half).
Megatron-FSDP (``core/glu.py``) stores it

* *split*, where its SwiGLU predicate holds: the gate half ``W`` and the value
  half ``V`` as two tensors, both TP-canonical; or
* *packed*: whole, as the concatenation of the tensor-parallel ranks' local
  tensors, ``[W_0; V_0; W_1; V_1; ...]``, at the context's pack size (the
  expert pack size for a routed expert; ``[W; V]`` at size 1).

The rule takes the weight and bias of every such first projection. Each half
is an output of its own wherever a format stores it by halves: fsdp_dtensor
names a split one ``<name>_w``/``<name>_v``, and torch_dist places each half at
its rows of the whole tensor (``core/structure.py``). The copies follow the
halves' :class:`~..core.part_layout.PartLayout`: cutting the halves out of a
tensor packed at some size (the canonical tensor is packed at 1), or packing
the canonical tensor at the pack size.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence

from ..core.context import ArchContext
from ..core.glu import Fc1
from ..core.logical import LogicalCopy, LogicalId, LogicalOutput, LogicalSlice
from ..core.names import FSDP_DTENSOR
from ..core.part_layout import rows_box, unpack_copies
from ..core.registry import REGISTRY
from ..core.rule import FamilyInstance, LeafRule, LogicalPool, Stage
from ..engine.regions import Region
from ..errors import InternalError


@REGISTRY.rule
class GluRule(LeafRule):
    """Split, merge, pack and unpack the weight and bias of gated ``linear_fc1``."""

    name = "glu"
    stage = Stage.GLU

    def claim(self, pool: LogicalPool, ctx: ArchContext) -> Iterator[FamilyInstance]:
        """Take every tensor (or half) of a split or packed first projection, grouped by whole
        tensor."""

        def handled(piece: LogicalSlice) -> bool:
            fc1 = ctx.glu.of(piece.lid)
            return fc1 is not None and (fc1.split or fc1.packed)

        tensors: dict[LogicalId, list[LogicalSlice]] = {}
        for piece in pool.take(self.name, handled):
            tensors.setdefault(piece.lid.whole, []).append(piece)
        for whole, pieces in tensors.items():
            fc1 = ctx.glu.of(whole)
            assert fc1 is not None  # taken by ``handled``
            yield FamilyInstance(self.name, tuple(pieces), _outputs(whole, pieces, fc1, ctx))


def _outputs(
    whole: LogicalId, pieces: Sequence[LogicalSlice], fc1: Fc1, ctx: ArchContext
) -> tuple[LogicalOutput, ...]:
    """The output tensors of one weight or bias of ``fc1``, stored as ``pieces`` in the source."""
    if len(pieces) > 1 or pieces[0].lid.part is not None:
        return tuple(_copied(piece, ctx) for piece in pieces)  # fsdp_dtensor halves
    (piece,) = pieces
    if ctx.target[0] == FSDP_DTENSOR and fc1.packed:
        return (_packed(whole, piece, fc1, _pack_size(fc1, ctx), ctx),)
    source_pack = 1 if ctx.target[0] == FSDP_DTENSOR else _pack_size(fc1, ctx)
    return _unpacked(whole, piece, fc1, source_pack, ctx)


def _pack_size(fc1: Fc1, ctx: ArchContext) -> int:
    """The size fsdp_dtensor packs ``fc1`` at: the expert size for a routed expert."""
    size = ctx.pack_size(fc1.lid)
    if size is None:
        raise InternalError(f"{fc1.lid} is packed, but the context has no pack size for it")
    return size


def _copied(piece: LogicalSlice, ctx: ArchContext) -> LogicalOutput:
    """A stored half as it is."""
    full = Region.full(piece.shape)
    dtype = ctx.dtype_policy.output_dtype(piece.dtype)
    return LogicalOutput(piece.lid, piece.shape, dtype, (LogicalCopy(piece.lid, full, full),))


def _unpacked(
    whole: LogicalId, piece: LogicalSlice, fc1: Fc1, pack: int, ctx: ArchContext
) -> tuple[LogicalOutput, ...]:
    """The two halves cut out of the whole tensor ``piece`` packed at ``pack``."""
    rest = piece.shape[1:]
    dtype = ctx.dtype_policy.output_dtype(piece.dtype)
    return tuple(
        LogicalOutput(
            dataclasses.replace(whole, part=part), (fc1.parts.size(part), *rest), dtype, copies
        )
        for part, copies in unpack_copies(fc1.parts, pack, piece.lid, rest).items()
    )


def _packed(
    whole: LogicalId, piece: LogicalSlice, fc1: Fc1, pack: int, ctx: ArchContext
) -> LogicalOutput:
    """The canonical whole tensor ``piece`` packed at ``pack``."""
    rest = piece.shape[1:]
    return LogicalOutput(
        whole,
        piece.shape,
        ctx.dtype_policy.output_dtype(piece.dtype),
        tuple(
            LogicalCopy(
                piece.lid,
                rows_box(move.src_start, move.rows, rest),
                rows_box(move.dst_start, move.rows, rest),
            )
            for move in fc1.parts.repack_map(1, pack)
        ),
    )
