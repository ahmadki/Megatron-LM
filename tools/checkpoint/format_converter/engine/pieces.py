# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The write program of a conversion: output pieces and the copies that fill them.

A :class:`Piece` is one output DCP chunk: a box of an output tensor, backed at
run time by one standalone CPU buffer of exactly its size. Its :class:`CopyOp`
list says where every element comes from (a box of a stored source chunk) and
tiles the piece exactly. A :class:`BytePiece` is an output byte item whose
payload is already encoded (or, in a dry-run plan, only sized). Pieces are pure
descriptions (no tensor data), so they can be planned from metadata alone and
pickled to worker processes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from ..errors import InternalError
from .metadata import StoredChunk, TensorEntry
from .regions import Region, find_tiling_defect, same_up_to_leading_ones


@dataclass(frozen=True, slots=True)
class CopyOp:
    """Copy one box of a stored source chunk into a box of a piece buffer.

    The two boxes hold the same elements in the same row-major order: their
    sizes are equal up to leading size-1 axes, so a layer ``[l:l+1, ...]`` of
    a tensor stacked over layers can be copied into a tensor of its own and
    back (see :func:`~.regions.same_up_to_leading_ones`).

    Attributes:
        src: The stored source chunk read from.
        src_region: The box to copy, in the global coordinates of ``src.key``;
            it lies inside ``src``.
        dst_region: Where the box goes, in the local coordinates of the piece
            buffer; same sizes as ``src_region`` up to leading size-1 axes.
    """

    src: StoredChunk
    src_region: Region
    dst_region: Region

    def __post_init__(self) -> None:
        if self.src_region.ndim != len(self.src.sizes) or not self.src.region.contains(
            self.src_region
        ):
            raise InternalError(
                f"copy from {self.src.key}: {self.src_region} is outside stored chunk "
                f"{self.src.region}"
            )
        if not same_up_to_leading_ones(self.src_region.sizes, self.dst_region.sizes):
            raise InternalError(
                f"copy from {self.src.key}: source box {self.src_region} and destination box "
                f"{self.dst_region} differ in shape beyond leading size-1 axes"
            )


@dataclass(frozen=True, slots=True)
class Piece:
    """One output chunk of a tensor and the copies that fill it.

    Attributes:
        key: Output tensor key.
        global_shape: Shape of the whole output tensor.
        dtype: Element type of the output tensor.
        chunk: The box this piece stores, in global coordinates.
        ops: Copies that tile the piece exactly (in its local coordinates).
        planner_path: ``Metadata.planner_data`` entry of the key, if the target
            format records one.
        cast: Copies convert their source's floating-point type to ``dtype``
            (rounding to nearest even) where the two differ; without it every
            source has ``dtype``.

    Raises:
        InternalError: The chunk is outside the tensor, a copy's source has
            another type without ``cast`` (or either type is not a
            floating-point type), or the copies do not tile the chunk.
    """

    key: str
    global_shape: tuple[int, ...]
    dtype: torch.dtype
    chunk: Region
    ops: tuple[CopyOp, ...]
    planner_path: tuple[str, ...] | None = None
    cast: bool = False

    def __post_init__(self) -> None:
        full = Region.full(self.global_shape)
        if self.chunk.ndim != full.ndim or not full.contains(self.chunk):
            raise InternalError(
                f"piece of {self.key}: {self.chunk} is outside shape {self.global_shape}"
            )
        for op in self.ops:
            if op.src.dtype == self.dtype:
                continue
            if not self.cast:
                raise InternalError(
                    f"piece {self.chunk} of {self.key}: copy from {op.src.key} is "
                    f"{op.src.dtype}, the piece is {self.dtype} (no cast is planned)"
                )
            if not (op.src.dtype.is_floating_point and self.dtype.is_floating_point):
                raise InternalError(
                    f"piece {self.chunk} of {self.key}: cannot cast {op.src.dtype} from "
                    f"{op.src.key} to {self.dtype} (only floating-point types are converted)"
                )
        defect = find_tiling_defect([op.dst_region for op in self.ops], self.chunk.sizes)
        if defect is not None:
            raise InternalError(
                f"piece {self.chunk} of {self.key}: copies do not tile it: {defect}"
            )

    @property
    def nbytes(self) -> int:
        """Size of the piece buffer."""
        return self.chunk.numel() * self.dtype.itemsize


@dataclass(frozen=True, slots=True)
class BytePiece:
    """One output byte item with its encoded payload.

    A dry-run plan may record only the payload's (estimated) size, without
    reading or encoding it (:meth:`sized`); such a piece can be packed and
    budgeted but never written.

    Attributes:
        key: Output item key.
        payload: The bytes to store, written verbatim; ``None`` if only sized.
        planner_path: ``Metadata.planner_data`` entry of the key, if the target
            format records one.
        size: The payload size of a piece without a payload.

    Raises:
        InternalError: Neither or both of ``payload`` and ``size`` are given.
    """

    key: str
    payload: bytes | None
    planner_path: tuple[str, ...] | None = None
    size: int | None = None

    def __post_init__(self) -> None:
        if (self.payload is None) == (self.size is None):
            raise InternalError(f"byte item {self.key} needs exactly one of a payload and a size")

    @classmethod
    def sized(cls, key: str, size: int, planner_path: tuple[str, ...] | None = None) -> BytePiece:
        """A byte item of a dry-run plan: its size is known, its payload is not read."""
        return cls(key, None, planner_path, size)

    @property
    def nbytes(self) -> int:
        """Size of the payload."""
        return len(self.payload) if self.payload is not None else int(self.size or 0)


def copy_pieces(entry: TensorEntry) -> tuple[Piece, ...]:
    """One piece per stored chunk of ``entry``, copying the chunk as it is.

    The output tensor has the source's key, shape, dtype and chunk grid.
    """
    return tuple(
        Piece(
            key=entry.key,
            global_shape=entry.shape,
            dtype=entry.dtype,
            chunk=chunk.region,
            ops=(CopyOp(chunk, chunk.region, Region.full(chunk.sizes)),),
        )
        for chunk in entry.chunks
    )


@dataclass(frozen=True, slots=True)
class Unit:
    """The pieces and byte items that one worker reads and writes as one part store.

    Attributes:
        uid: Unit id; parts are merged in uid order.
        pieces: Output pieces (all their buffers are resident at once).
        byte_pieces: Output byte items.
        est_peak_bytes: Estimated peak memory of executing the unit (piece
            buffers, read staging and the largest decoded stored item).
        read_bytes: On-disk size of the stored items the unit reads.
    """

    uid: int
    pieces: tuple[Piece, ...]
    byte_pieces: tuple[BytePiece, ...] = ()
    est_peak_bytes: int = 0
    read_bytes: int = 0

    @property
    def source_keys(self) -> frozenset[str]:
        """Keys of the source tensors the unit reads."""
        return frozenset(op.src.key for piece in self.pieces for op in piece.ops)


@dataclass(frozen=True, slots=True)
class ExpectedTensor:
    """What the merged output must hold for one tensor.

    Attributes:
        key: Output tensor key.
        global_shape: Its shape.
        dtype: Its element type.
        chunks: The planned chunk grid, sorted by offsets; it tiles the tensor.
    """

    key: str
    global_shape: tuple[int, ...]
    dtype: torch.dtype
    chunks: tuple[Region, ...]


def expected_tensors(pieces: Iterable[Piece]) -> dict[str, ExpectedTensor]:
    """The output tensors a set of pieces describes, by key (sorted).

    Raises:
        InternalError: Pieces of one key disagree on its shape or dtype, or
            their chunks do not tile the tensor exactly.
    """
    grouped: dict[str, list[Piece]] = {}
    for piece in pieces:
        grouped.setdefault(piece.key, []).append(piece)
    expected = {}
    for key in sorted(grouped):
        first = grouped[key][0]
        for piece in grouped[key]:
            if (piece.global_shape, piece.dtype) != (first.global_shape, first.dtype):
                raise InternalError(
                    f"pieces of {key} disagree: {first.global_shape} {first.dtype} "
                    f"and {piece.global_shape} {piece.dtype}"
                )
        chunks = tuple(sorted((p.chunk for p in grouped[key]), key=lambda r: r.offsets))
        defect = find_tiling_defect(chunks, first.global_shape)
        if defect is not None:
            raise InternalError(f"the planned chunks of {key} do not tile it: {defect}")
        expected[key] = ExpectedTensor(key, first.global_shape, first.dtype, chunks)
    return expected
