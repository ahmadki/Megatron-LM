# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Axis-aligned boxes in the global index space of a tensor.

A :class:`Region` is the unit every read and compare is planned in: a stored
DCP chunk, the part of it a caller needs, or a destination buffer. Coordinates
are global (relative to the full tensor) unless a function says otherwise.
A 0-d tensor has exactly one region, ``Region((), ())``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Region:
    """A box ``[offsets, offsets + sizes)`` in a tensor's index space.

    Attributes:
        offsets: Start index per dimension (non-negative).
        sizes: Extent per dimension (non-negative; zero means empty).
    """

    offsets: tuple[int, ...]
    sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        offsets = tuple(int(o) for o in self.offsets)
        sizes = tuple(int(s) for s in self.sizes)
        if len(offsets) != len(sizes):
            raise ValueError(f"Region offsets {offsets} and sizes {sizes} differ in rank")
        if any(o < 0 for o in offsets) or any(s < 0 for s in sizes):
            raise ValueError(f"Region offsets {offsets} and sizes {sizes} must be non-negative")
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "sizes", sizes)

    @classmethod
    def full(cls, shape: Sequence[int]) -> Region:
        """The region covering a whole tensor of ``shape``."""
        return cls((0,) * len(shape), tuple(shape))

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.sizes)

    @property
    def ends(self) -> tuple[int, ...]:
        """Exclusive end index per dimension."""
        return tuple(o + s for o, s in zip(self.offsets, self.sizes))

    def numel(self) -> int:
        """Number of elements (1 for a 0-d region)."""
        return math.prod(self.sizes)

    def intersect(self, other: Region) -> Region | None:
        """The common sub-box, or ``None`` when the two regions share no element."""
        self._check_rank(other)
        offsets = tuple(max(a, b) for a, b in zip(self.offsets, other.offsets))
        ends = tuple(min(a, b) for a, b in zip(self.ends, other.ends))
        if any(e <= o for o, e in zip(offsets, ends)):
            return None
        return Region(offsets, tuple(e - o for o, e in zip(offsets, ends)))

    def shift(self, delta: Sequence[int]) -> Region:
        """The same box moved by ``delta`` (which may be negative)."""
        if len(delta) != self.ndim:
            raise ValueError(f"shift {tuple(delta)} does not match rank {self.ndim}")
        return Region(tuple(o + d for o, d in zip(self.offsets, delta)), self.sizes)

    def relative_to(self, container: Region) -> Region:
        """This region in the local coordinates of ``container``, which must contain it."""
        if not container.contains(self):
            raise ValueError(f"{self} is not inside {container}")
        return self.shift(tuple(-o for o in container.offsets))

    def contains(self, other: Region) -> bool:
        """True if every element of ``other`` lies inside this region."""
        self._check_rank(other)
        return all(
            so <= oo and oe <= se
            for so, se, oo, oe in zip(self.offsets, self.ends, other.offsets, other.ends)
        )

    def is_full(self, shape: Sequence[int]) -> bool:
        """True if the region covers the whole tensor of ``shape``."""
        return self == Region.full(shape)

    def split_dim0(self, max_rows: int) -> list[Region]:
        """Split along dimension 0 into consecutive slabs of at most ``max_rows`` rows."""
        if max_rows < 1:
            raise ValueError(f"max_rows must be positive, got {max_rows}")
        if self.ndim == 0 or self.sizes[0] <= max_rows:
            return [self]
        slabs = []
        for start in range(self.offsets[0], self.ends[0], max_rows):
            rows = min(max_rows, self.ends[0] - start)
            slabs.append(Region((start, *self.offsets[1:]), (rows, *self.sizes[1:])))
        return slabs

    def slices(self) -> tuple[slice, ...]:
        """Index expression selecting this region from a tensor laid out in its coordinates."""
        return tuple(slice(o, e) for o, e in zip(self.offsets, self.ends))

    def __str__(self) -> str:
        return "[" + ", ".join(f"{o}:{e}" for o, e in zip(self.offsets, self.ends)) + "]"

    def _check_rank(self, other: Region) -> None:
        if other.ndim != self.ndim:
            raise ValueError(f"regions of rank {self.ndim} and {other.ndim} are not comparable")


def bounding_box(regions: Iterable[Region]) -> Region:
    """The smallest region containing every region in ``regions`` (at least one)."""
    items = list(regions)
    if not items:
        raise ValueError("bounding_box() needs at least one region")
    ndim = items[0].ndim
    if any(r.ndim != ndim for r in items):
        raise ValueError("bounding_box() regions differ in rank")
    offsets = tuple(min(r.offsets[d] for r in items) for d in range(ndim))
    ends = tuple(max(r.ends[d] for r in items) for d in range(ndim))
    return Region(offsets, tuple(e - o for o, e in zip(offsets, ends)))


def dim0_only(chunks: Iterable[Region], shape: Sequence[int]) -> bool:
    """True if every chunk spans all of dimensions 1.. (the grid splits dimension 0 only)."""
    rest = tuple(shape[1:])
    return all(c.offsets[1:] == (0,) * len(rest) and c.sizes[1:] == rest for c in chunks)


def same_up_to_leading_ones(a: Sequence[int], b: Sequence[int]) -> bool:
    """Whether boxes of sizes ``a`` and ``b`` hold the same elements in the same row-major order.

    That is the case when the sizes are equal once aligned on their last axes
    and the extra leading axes of the longer one are all 1: a layer
    ``[l:l+1, :, :]`` of a stacked tensor and a tensor of its own of the
    layer's shape, e.g. ``(1, 4, 8)`` and ``(4, 8)``.
    """
    common = min(len(a), len(b))
    longer = a if len(a) > len(b) else b
    return tuple(a[len(a) - common :]) == tuple(b[len(b) - common :]) and all(
        size == 1 for size in longer[: len(longer) - common]
    )


def corresponding(part: Region, of: Region, onto: Region) -> Region:
    """The box of ``onto`` that holds the elements ``part`` holds in ``of``.

    ``of`` and ``onto`` hold the same elements (their sizes are equal up to
    leading unit axes, see :func:`same_up_to_leading_ones`) and may live in
    tensors of different ranks; ``part`` lies inside ``of``.

    Raises:
        ValueError: ``of`` and ``onto`` differ in more than leading unit axes,
            or ``part`` is not inside ``of``.
    """
    if not same_up_to_leading_ones(of.sizes, onto.sizes):
        raise ValueError(f"{of} and {onto} do not hold the same elements")
    if not of.contains(part):
        raise ValueError(f"{part} is not inside {of}")
    common = min(of.ndim, onto.ndim)
    lead, skip = onto.ndim - common, of.ndim - common
    offsets = onto.offsets[:lead] + tuple(
        o + p - f for o, p, f in zip(onto.offsets[lead:], part.offsets[skip:], of.offsets[skip:])
    )
    return Region(offsets, onto.sizes[:lead] + part.sizes[skip:])


def find_tiling_defect(regions: Sequence[Region], shape: Sequence[int]) -> str | None:
    """Check that ``regions`` tile a tensor of ``shape`` exactly.

    Returns:
        ``None`` if the regions are in bounds, pairwise disjoint and cover every
        element; otherwise a one-line description of the first defect found.
    """
    full = Region.full(shape)
    for region in regions:
        if region.ndim != full.ndim or not full.contains(region):
            return f"chunk {region} lies outside shape {tuple(shape)}"
    ordered = sorted((r for r in regions if r.numel() > 0), key=lambda r: r.offsets)
    active: list[Region] = []
    for region in ordered:
        if region.ndim > 0:
            active = [a for a in active if a.ends[0] > region.offsets[0]]
        for other in active:
            if other.intersect(region) is not None:
                return f"chunks {other} and {region} overlap"
        active.append(region)
    covered = sum(r.numel() for r in regions)
    if covered != full.numel():
        return f"chunks cover {covered} of {full.numel()} elements"
    return None
