# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fused tensors made of parts, and how tensor parallelism lays the parts out.

Some parameters concatenate several *parts* along their first axis: a gated
linear unit's ``linear_fc1`` holds the gate ``W`` and the value ``V``
projections, a state-space mixer's ``in_proj`` holds one projection per
component. A column-parallel layer shards every part over the
tensor-parallel ranks, so rank ``r`` holds the concatenation of the ``r``-th
shard of each part. A store that keeps such a tensor TP-canonical holds each
part whole, one after the other (``[W; V]``); a store that concatenates the
ranks' local tensors holds it *rank-major*: ``[W_0; V_0; W_1; V_1; ...]``
for ``tp`` ranks, which is the canonical order only at ``tp = 1``.

A :class:`PartLayout` lists the parts of one fused tensor in the order each
rank concatenates them, with their global sizes along the first axis, one
entry per component (never an aggregate of components, whose rank shards
would interleave). :meth:`PartLayout.rank_major_map` gives, for a pack size
``tp``, where each rank's shard of each part sits in the rank-major tensor::

    every size divides by tp (else FC-R10)
    rows per rank L = sum(size_p / tp)
    for r in range(tp):
        o = r * L
        for each part p, in order:
            part rows [r * size_p/tp, (r+1) * size_p/tp) at fused rows [o, o + size_p/tp)
            o += size_p / tp

Packing copies part rows into the fused rows, unpacking copies them back,
along the same map; :meth:`PartLayout.repack_map` moves the rows of a fused
tensor packed at one size into the tensor packed at another.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import errors
from ..errors import InternalError, Refusal


@dataclass(frozen=True, slots=True)
class PartRows:
    """A run of rows of one part and the rows of the fused tensor that hold them.

    Attributes:
        part: The part's name.
        part_start: First row of the run inside the part.
        fused_start: First row of the run inside the fused tensor.
        rows: Number of rows.
    """

    part: str
    part_start: int
    fused_start: int
    rows: int


@dataclass(frozen=True, slots=True)
class RowMove:
    """A run of rows moved from one fused tensor into another.

    Attributes:
        src_start: First row in the source tensor.
        dst_start: First row in the destination tensor.
        rows: Number of rows.
    """

    src_start: int
    dst_start: int
    rows: int


@dataclass(frozen=True, slots=True)
class PartLayout:
    """The parts of a fused tensor, in the order each tensor-parallel rank concatenates them.

    Attributes:
        parts: The part names, one per component.
        sizes: Each part's global (unsharded) size along the first axis.

    Raises:
        InternalError: The layout has no parts, names a part twice, has
            another number of sizes than parts, or a size that is not positive.
    """

    parts: tuple[str, ...]
    sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.parts or len(set(self.parts)) != len(self.parts):
            raise InternalError(f"a part layout needs distinct part names, not {self.parts}")
        if len(self.sizes) != len(self.parts) or any(size < 1 for size in self.sizes):
            raise InternalError(f"part layout {self.parts}: bad sizes {self.sizes}")

    @property
    def fused_rows(self) -> int:
        """Rows of the fused tensor (the sum of the part sizes)."""
        return sum(self.sizes)

    def size(self, part: str) -> int:
        """The global size of ``part``.

        Raises:
            InternalError: The layout has no such part.
        """
        if part not in self.parts:
            raise InternalError(f"part layout {self.parts} has no part {part!r}")
        return self.sizes[self.parts.index(part)]

    def rank_major_map(self, tp: int) -> tuple[PartRows, ...]:
        """Where each rank's shard of each part sits in the tensor packed rank-major at ``tp``.

        Returns:
            One run per (rank, part), in fused-row order; together they tile
            both the fused tensor and every part. At ``tp = 1`` the fused
            tensor is the parts' concatenation.

        Raises:
            RefusedError: A part size does not divide by ``tp`` (FC-R10).
            InternalError: ``tp`` is not positive.
        """
        self.check_divisible(tp)
        runs = []
        fused = 0
        for rank in range(tp):
            for part, size in zip(self.parts, self.sizes):
                shard = size // tp
                runs.append(PartRows(part, rank * shard, fused, shard))
                fused += shard
        return tuple(runs)

    def repack_map(self, src_tp: int, dst_tp: int) -> tuple[RowMove, ...]:
        """How the tensor packed rank-major at ``src_tp`` becomes the one packed at ``dst_tp``.

        Returns:
            Row runs in destination order; together they tile both tensors.

        Raises:
            RefusedError: A part size does not divide by ``src_tp`` or ``dst_tp`` (FC-R10).
            InternalError: A pack size is not positive.
        """
        sources: dict[str, list[PartRows]] = {part: [] for part in self.parts}
        for run in self.rank_major_map(src_tp):
            sources[run.part].append(run)
        moves = []
        for dst in self.rank_major_map(dst_tp):
            for src in sources[dst.part]:
                start = max(src.part_start, dst.part_start)
                end = min(src.part_start + src.rows, dst.part_start + dst.rows)
                if start < end:
                    moves.append(
                        RowMove(
                            src.fused_start + start - src.part_start,
                            dst.fused_start + start - dst.part_start,
                            end - start,
                        )
                    )
        return tuple(moves)

    def check_divisible(self, tp: int) -> None:
        """Every part divides by the pack size ``tp``.

        Raises:
            RefusedError: A part size does not (FC-R10).
            InternalError: ``tp`` is not positive.
        """
        if tp < 1:
            raise InternalError(f"a tensor-parallel pack size must be positive, not {tp}")
        uneven = [f"{part}={size}" for part, size in zip(self.parts, self.sizes) if size % tp]
        if uneven:
            raise Refusal(
                errors.FC_R10,
                f"parts {', '.join(uneven)} of a fused tensor do not divide by the "
                f"tensor-parallel size {tp} they are packed at",
                hint="pack at a tensor-parallel size that divides every part",
            ).to_error()
