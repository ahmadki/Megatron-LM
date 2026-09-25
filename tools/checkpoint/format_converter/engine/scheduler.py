# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Packing pieces into units under a memory budget, and the order units run in.

1. **Split.** A piece larger than half the unit budget ``U`` that reads several
   stored items is cut along dimension 0 into row slabs of at most ``U/2``, so
   the slabs can run in different units. A single row above ``U/2`` cannot be
   bounded (exit 8). A piece that reads one stored item is never split: the DCP
   reader decodes that item whole in any case, so splitting could only add
   staging (and it keeps a copied chunk grid identical to its source).
2. **Cluster.** Pieces that read the same stored item are grouped (union-find)
   so every stored item is read by one unit, once. A cluster costs its piece
   buffers, its staging boxes and the largest item it decodes; a cluster above
   the hard cap exits 8.
3. **Pack.** Clusters, larger ones first and then in (data file, offset) order,
   are packed next-fit into units whose estimated peak stays within ``U``; a
   cluster above ``U`` (and within the hard cap) is a unit on its own. Byte
   items go into unit 0.
4. **Dispatch.** Workers claim units largest estimate first (LPT), ties broken
   by data file and offset.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..errors import BudgetExceeded, InternalError
from .metadata import StoredChunk
from .pieces import BytePiece, CopyOp, Piece, Unit
from .reader_dcp import staging_bytes
from .regions import Region
from .sizes import format_bytes

_Item = tuple[str, int]
"""A stored item: (data file, byte offset)."""


@dataclass
class _Cluster:
    """Pieces that read the same stored items, with their memory cost."""

    pieces: list[Piece]
    piece_bytes: int = 0
    staging: int = 0
    largest: StoredChunk | None = None
    read_bytes: int = 0
    first: _Item = ("", 0)

    @property
    def decode_bytes(self) -> int:
        """Size of the largest stored item decoded whole."""
        return 0 if self.largest is None else self.largest.nbytes

    @property
    def cost(self) -> int:
        """Peak memory of reading and holding the cluster alone."""
        return self.piece_bytes + self.staging + self.decode_bytes


@dataclass
class _OpenUnit:
    clusters: list[_Cluster] = field(default_factory=list)
    byte_pieces: tuple[BytePiece, ...] = ()
    held: int = 0
    decode: int = 0

    def estimate(self, extra: _Cluster | None = None) -> int:
        """Peak memory of the unit (with ``extra`` added, if given)."""
        held = self.held + sum(2 * len(item.payload) for item in self.byte_pieces)
        decode = self.decode
        if extra is not None:
            held += extra.piece_bytes + extra.staging
            decode = max(decode, extra.decode_bytes)
        return held + decode

    def add(self, cluster: _Cluster) -> None:
        """Put ``cluster`` into the unit."""
        self.clusters.append(cluster)
        self.held += cluster.piece_bytes + cluster.staging
        self.decode = max(self.decode, cluster.decode_bytes)


def pack_units(
    pieces: Sequence[Piece], byte_pieces: Sequence[BytePiece], *, unit_bytes: int, hard_cap: int
) -> tuple[Unit, ...]:
    """Pack pieces and byte items into units (uids ``0..n-1``).

    Args:
        pieces: The output pieces.
        byte_pieces: The output byte items (all go into unit 0).
        unit_bytes: Target peak memory of a unit (``U``).
        hard_cap: Largest peak memory a single unit may need.

    Raises:
        BudgetExceeded: A row of a split piece exceeds ``U/2``, or the pieces
            reading one stored item cannot be held within ``hard_cap``.
    """
    if unit_bytes <= 0 or hard_cap <= 0:
        raise InternalError(f"unit budgets must be positive, got {unit_bytes} and {hard_cap}")
    split = [slab for piece in pieces for slab in split_piece(piece, unit_bytes // 2)]
    clusters = _clusters(split)
    for cluster in clusters:
        _check_hard_cap(cluster, hard_cap)
    clusters.sort(key=lambda c: (c.cost <= unit_bytes // 2, c.first))
    packed = [_OpenUnit(byte_pieces=tuple(byte_pieces))]
    for cluster in clusters:
        current = packed[-1]
        if (current.clusters or current.byte_pieces) and current.estimate(cluster) > unit_bytes:
            current = _OpenUnit()
            packed.append(current)
        current.add(cluster)
    return tuple(
        Unit(
            uid=uid,
            pieces=tuple(piece for cluster in unit.clusters for piece in cluster.pieces),
            byte_pieces=unit.byte_pieces,
            est_peak_bytes=unit.estimate(),
            read_bytes=sum(cluster.read_bytes for cluster in unit.clusters),
        )
        for uid, unit in enumerate(packed)
        if unit.clusters or unit.byte_pieces
    )


def dispatch_order(units: Sequence[Unit]) -> tuple[int, ...]:
    """Unit ids, largest estimated peak first; ties by first data file and offset."""
    return tuple(unit.uid for unit in sorted(units, key=lambda u: (-u.est_peak_bytes, _first(u))))


def split_piece(piece: Piece, max_bytes: int) -> tuple[Piece, ...]:
    """Cut a piece that reads several stored items into dim-0 slabs of at most ``max_bytes``.

    Raises:
        BudgetExceeded: One row of the piece is larger than ``max_bytes``.
    """
    items = {(op.src.relative_path, op.src.offset) for op in piece.ops}
    if piece.nbytes <= max_bytes or piece.chunk.ndim == 0 or len(items) <= 1:
        return (piece,)
    row_bytes = piece.nbytes // piece.chunk.sizes[0]
    if row_bytes > max_bytes:
        raise BudgetExceeded(
            f"one row of {piece.key} {piece.chunk} is {format_bytes(row_bytes)}, above half "
            f"the unit budget ({format_bytes(max_bytes)})",
            hint=f"pass --unit-bytes {2 * row_bytes} or more",
        )
    rows = max(1, max_bytes // row_bytes)
    return tuple(_slab(piece, local) for local in Region.full(piece.chunk.sizes).split_dim0(rows))


def _slab(piece: Piece, local: Region) -> Piece:
    """The part of ``piece`` in box ``local`` (piece coordinates) as a piece of its own."""
    ops = []
    for op in piece.ops:
        part = op.dst_region.intersect(local)
        if part is None:
            continue
        delta = tuple(p - d for p, d in zip(part.offsets, op.dst_region.offsets))
        source = Region(tuple(s + d for s, d in zip(op.src_region.offsets, delta)), part.sizes)
        ops.append(CopyOp(op.src, source, part.relative_to(local)))
    chunk = Region(tuple(c + o for c, o in zip(piece.chunk.offsets, local.offsets)), local.sizes)
    return Piece(piece.key, piece.global_shape, piece.dtype, chunk, tuple(ops), piece.planner_path)


def _clusters(pieces: Sequence[Piece]) -> list[_Cluster]:
    parent = list(range(len(pieces)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    owner: dict[_Item, int] = {}
    for index, piece in enumerate(pieces):
        for op in piece.ops:
            other = owner.setdefault((op.src.relative_path, op.src.offset), index)
            parent[find(index)] = find(other)
    groups: dict[int, list[Piece]] = {}
    for index, piece in enumerate(pieces):
        groups.setdefault(find(index), []).append(piece)
    return [_cluster(group) for group in groups.values()]


def _cluster(pieces: list[Piece]) -> _Cluster:
    reads: dict[_Item, tuple[StoredChunk, list[Region]]] = {}
    for piece in pieces:
        for op in piece.ops:
            item = (op.src.relative_path, op.src.offset)
            reads.setdefault(item, (op.src, []))[1].append(op.src_region)
    chunks = [chunk for chunk, _ in reads.values()]
    return _Cluster(
        pieces=sorted(pieces, key=lambda p: (p.key, p.chunk.offsets)),
        piece_bytes=sum(p.nbytes for p in pieces),
        staging=sum(staging_bytes(chunk, regions) for chunk, regions in reads.values()),
        largest=max(chunks, key=lambda c: c.nbytes, default=None),
        read_bytes=sum(chunk.length for chunk in chunks),
        first=min(reads, default=("", 0)),
    )


def _check_hard_cap(cluster: _Cluster, hard_cap: int) -> None:
    if cluster.cost <= hard_cap:
        return
    chunk = cluster.largest
    what = "its pieces" if chunk is None else f"stored chunk {chunk.offsets} of {chunk.key}"
    raise BudgetExceeded(
        f"reading {what} with the pieces it feeds needs {format_bytes(cluster.cost)} at once "
        f"({format_bytes(cluster.piece_bytes)} of pieces, {format_bytes(cluster.staging)} of "
        f"staging, {format_bytes(cluster.decode_bytes)} decoded), above the hard unit cap of "
        f"{format_bytes(hard_cap)}",
        hint=f"pass --max-unit-bytes-hard {cluster.cost} or more (host memory permitting)",
    )


def _first(unit: Unit) -> _Item:
    items = [(op.src.relative_path, op.src.offset) for piece in unit.pieces for op in piece.ops]
    return min(items, default=("", 0))
