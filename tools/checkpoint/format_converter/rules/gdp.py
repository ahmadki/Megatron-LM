# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The Gated DeltaProduct rule: a mixer's fused ``in_proj`` and ``conv1d``, component by component.

torch_dist stores the weight and bias of a Gated DeltaProduct mixer's
``in_proj`` (``z, V0..V(M-1), K0..K(M-1), Q, b0..b(M-1), a``) and ``conv1d``
(``V0..V(M-1), K0..K(M-1), Q``) as one TP-canonical tensor per component
(Megatron's split factory). fsdp_dtensor stores each of them whole: the
tensor-parallel ranks' local tensors one after the other, each rank's holding
its shard of every component in order (``core/mixers.py`` derives the
components and their sizes from the arguments). The rule takes every tensor
of these projections and copies it along the projection's per-component
:class:`~..core.part_layout.PartLayout`: towards fsdp_dtensor it packs the
parts rank-major at the context's tensor-parallel pack size, and back it cuts
the parts out of the tensor packed at the fsdp_dtensor store's size. At pack
size 1 the fused tensor is the parts' concatenation.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence

from ..core.context import ArchContext
from ..core.logical import LogicalId, LogicalOutput, LogicalSlice
from ..core.mixers import GDP, FusedProjection
from ..core.names import FSDP_DTENSOR
from ..core.part_layout import pack_copies, unpack_copies
from ..core.registry import REGISTRY
from ..core.rule import FamilyInstance, LeafRule, LogicalPool, Stage
from ..errors import InternalError


@REGISTRY.rule
class GdpRule(LeafRule):
    """Fuse and split the ``in_proj`` and ``conv1d`` weights and biases of Gated DeltaProduct
    mixers."""

    name = "gdp"
    stage = Stage.FUSED

    def claim(self, pool: LogicalPool, ctx: ArchContext) -> Iterator[FamilyInstance]:
        """Take every part (torch_dist) or fused tensor (fsdp_dtensor) of a Gated DeltaProduct
        mixer's projections, grouped by whole tensor."""

        def handled(piece: LogicalSlice) -> bool:
            fused = ctx.mixers.of(piece.lid)
            return fused is not None and fused.family == GDP

        tensors: dict[LogicalId, list[LogicalSlice]] = {}
        for piece in pool.take(self.name, handled):
            tensors.setdefault(piece.lid.whole, []).append(piece)
        for whole, pieces in tensors.items():
            fused = ctx.mixers.of(whole)
            assert fused is not None  # taken by ``handled``
            pack = _pack_size(fused, ctx)
            if ctx.target[0] == FSDP_DTENSOR:
                outputs: tuple[LogicalOutput, ...] = (_fused(whole, pieces, fused, pack, ctx),)
            else:
                outputs = _split(whole, pieces, fused, pack, ctx)
            yield FamilyInstance(self.name, tuple(pieces), outputs)


def _pack_size(fused: FusedProjection, ctx: ArchContext) -> int:
    """The tensor-parallel size fsdp_dtensor packs ``fused`` at."""
    if ctx.pack_tp is None:
        raise InternalError(f"{fused.lid} is packed, but the context has no pack size for it")
    return ctx.pack_tp


def _fused(
    whole: LogicalId,
    pieces: Sequence[LogicalSlice],
    fused: FusedProjection,
    pack: int,
    ctx: ArchContext,
) -> LogicalOutput:
    """The fused tensor packed rank-major at ``pack`` from its torch_dist parts ``pieces``."""
    parts = {piece.lid.part: piece.lid for piece in pieces if piece.lid.part is not None}
    if len(parts) != len(pieces) or set(parts) != set(fused.parts.parts):
        raise InternalError(f"{whole} is not stored as the parts of {fused.parts}")
    rest = pieces[0].shape[1:]
    return LogicalOutput(
        whole,
        (fused.parts.fused_rows, *rest),
        ctx.dtype_policy.output_dtype(pieces[0].dtype),
        pack_copies(fused.parts, pack, parts, rest),
    )


def _split(
    whole: LogicalId,
    pieces: Sequence[LogicalSlice],
    fused: FusedProjection,
    pack: int,
    ctx: ArchContext,
) -> tuple[LogicalOutput, ...]:
    """The torch_dist parts cut out of the fused fsdp_dtensor tensor (``pieces``: the whole one)
    packed rank-major at ``pack``."""
    if len(pieces) != 1 or pieces[0].lid.part is not None:
        raise InternalError(f"{whole} is not stored whole: {[str(p.lid) for p in pieces]}")
    (piece,) = pieces
    rest = piece.shape[1:]
    dtype = ctx.dtype_policy.output_dtype(piece.dtype)
    return tuple(
        LogicalOutput(
            dataclasses.replace(whole, part=part), (fused.parts.size(part), *rest), dtype, copies
        )
        for part, copies in unpack_copies(fused.parts, pack, piece.lid, rest).items()
    )
