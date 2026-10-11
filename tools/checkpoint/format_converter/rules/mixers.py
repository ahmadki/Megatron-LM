# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The mixer rules: a mixer's fused ``in_proj`` and ``conv1d``, component by component.

torch_dist stores the weight and bias of a state-space mixer's ``in_proj`` and
``conv1d`` as one TP-canonical tensor per component (Megatron's split factory):
a Gated DeltaProduct mixer's ``in_proj`` is ``z, V0..V(M-1), K0..K(M-1), Q,
b0..b(M-1), a`` and its ``conv1d`` ``V0..V(M-1), K0..K(M-1), Q``; a Mamba-2
mixer's are ``z, x, B, C, dt`` and ``x, B, C``. fsdp_dtensor stores each of
them whole: the tensor-parallel ranks' local tensors one after the other, each
rank's holding its shard of every component in order (``core/mixers.py``
derives the components and their sizes from the arguments). A rule takes
every tensor of its family's projections and copies it along the projection's
per-component :class:`~..core.part_layout.PartLayout`: towards fsdp_dtensor it
packs the parts rank-major at the context's tensor-parallel pack size, and
back it cuts the parts out of the tensor packed at the fsdp_dtensor store's
size. At pack size 1 the fused tensor is the parts' concatenation.

The names follow the target format (a Mamba-2 mixer's convolution is its own
``conv1d_weight`` and ``conv1d_bias`` in fsdp_dtensor), and a tensor torch_dist
stores whole (a Mamba-2 ``in_proj.bias``) is copied as is: the context has
checked that it is packed at the size it was saved at.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence

from ..core.context import ArchContext
from ..core.logical import LogicalCopy, LogicalId, LogicalOutput, LogicalSlice
from ..core.mixers import GDP, MAMBA2, FusedProjection
from ..core.names import FSDP_DTENSOR, TORCH_DIST
from ..core.part_layout import pack_copies, unpack_copies
from ..core.registry import REGISTRY
from ..core.rule import FamilyInstance, LeafRule, LogicalPool, Stage
from ..engine.regions import Region
from ..errors import InternalError


class _MixerRule(LeafRule):
    """Fuse and split the ``in_proj`` and ``conv1d`` weights and biases of the mixers of
    ``family``."""

    family: str
    stage = Stage.FUSED

    def claim(self, pool: LogicalPool, ctx: ArchContext) -> Iterator[FamilyInstance]:
        """Take every part (torch_dist) or fused tensor (fsdp_dtensor) of the family's mixers'
        projections, grouped by whole tensor."""

        def handled(piece: LogicalSlice) -> bool:
            fused = ctx.mixers.of(piece.lid)
            return fused is not None and fused.family == self.family

        tensors: dict[LogicalId, list[LogicalSlice]] = {}
        for piece in pool.take(self.name, handled):
            tensors.setdefault(piece.lid.whole, []).append(piece)
        for whole, pieces in tensors.items():
            fused = ctx.mixers.of(whole)
            assert fused is not None  # taken by ``handled``
            yield FamilyInstance(self.name, tuple(pieces), _outputs(fused, pieces, ctx))


@REGISTRY.rule
class GdpRule(_MixerRule):
    """Fuse and split the ``in_proj`` and ``conv1d`` weights and biases of Gated DeltaProduct
    mixers."""

    name = "gdp"
    family = GDP


@REGISTRY.rule
class Mamba2Rule(_MixerRule):
    """Fuse and split the ``in_proj`` and ``conv1d`` weights and biases of Mamba-2 mixers."""

    name = "mamba2"
    family = MAMBA2


def _outputs(
    fused: FusedProjection, pieces: Sequence[LogicalSlice], ctx: ArchContext
) -> tuple[LogicalOutput, ...]:
    """The target tensors of the fused projection tensor ``fused`` stored as ``pieces``."""
    target = ctx.target[0]
    if fused.torch_dist_whole:
        return (_whole(fused, pieces, ctx),)
    pack = _pack_size(fused, ctx)
    if target == FSDP_DTENSOR:
        return (_fused(fused, pieces, pack, ctx),)
    return _split(fused, pieces, pack, ctx)


def _pack_size(fused: FusedProjection, ctx: ArchContext) -> int:
    """The tensor-parallel size fsdp_dtensor packs ``fused`` at."""
    if ctx.pack_tp is None:
        raise InternalError(f"{fused.lid} is packed, but the context has no pack size for it")
    return ctx.pack_tp


def _the_whole(fused: FusedProjection, pieces: Sequence[LogicalSlice]) -> LogicalSlice:
    """The one slice of ``pieces``, which holds ``fused`` whole."""
    if len(pieces) != 1 or pieces[0].lid.part is not None:
        raise InternalError(f"{fused.lid} is not stored whole: {[str(p.lid) for p in pieces]}")
    return pieces[0]


def _whole(
    fused: FusedProjection, pieces: Sequence[LogicalSlice], ctx: ArchContext
) -> LogicalOutput:
    """A tensor both formats store whole, packed at the same size (``check_tp_bound``), as is."""
    piece = _the_whole(fused, pieces)
    full = Region.full(piece.shape)
    return LogicalOutput(
        fused.target_lid(ctx.target[0]),
        piece.shape,
        ctx.dtype_policy.output_dtype(piece.dtype),
        (LogicalCopy(piece.lid, full, full),),
    )


def _fused(
    fused: FusedProjection, pieces: Sequence[LogicalSlice], pack: int, ctx: ArchContext
) -> LogicalOutput:
    """The fused tensor packed rank-major at ``pack`` from its torch_dist parts ``pieces``."""
    parts = {piece.lid.part: piece.lid for piece in pieces if piece.lid.part is not None}
    if len(parts) != len(pieces) or set(parts) != set(fused.parts.parts):
        raise InternalError(f"{fused.lid} is not stored as the parts of {fused.parts}")
    rest = pieces[0].shape[1:]
    return LogicalOutput(
        fused.target_lid(FSDP_DTENSOR),
        (fused.parts.fused_rows, *rest),
        ctx.dtype_policy.output_dtype(pieces[0].dtype),
        pack_copies(fused.parts, pack, parts, rest),
    )


def _split(
    fused: FusedProjection, pieces: Sequence[LogicalSlice], pack: int, ctx: ArchContext
) -> tuple[LogicalOutput, ...]:
    """The torch_dist parts cut out of the fused fsdp_dtensor tensor (``pieces``: the whole one)
    packed rank-major at ``pack``."""
    piece = _the_whole(fused, pieces)
    rest = piece.shape[1:]
    dtype = ctx.dtype_policy.output_dtype(piece.dtype)
    whole = fused.target_lid(TORCH_DIST)
    return tuple(
        LogicalOutput(
            dataclasses.replace(whole, part=part), (fused.parts.size(part), *rest), dtype, copies
        )
        for part, copies in unpack_copies(fused.parts, pack, piece.lid, rest).items()
    )
