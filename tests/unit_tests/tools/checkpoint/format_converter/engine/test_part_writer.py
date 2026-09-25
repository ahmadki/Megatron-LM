# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pieces and the partial-tensor save planner on synthetic stores."""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

from tools.checkpoint.format_converter.engine.metadata import MetadataView, StoredChunk
from tools.checkpoint.format_converter.engine.part_writer import PieceSavePlanner
from tools.checkpoint.format_converter.engine.pieces import BytePiece, CopyOp, Piece, copy_pieces
from tools.checkpoint.format_converter.engine.regions import Region
from tools.checkpoint.format_converter.errors import InternalError

from .._synthetic import dim0_grid, torch_save_bytes, write_store

W = torch.arange(32, dtype=torch.float32).reshape(8, 4)
B = torch.arange(6, dtype=torch.bfloat16)


@pytest.fixture(name="source")
def fixture_source(tmp_path) -> MetadataView:
    write_store(tmp_path / "src", {"w": W, "b": B}, grids={"w": dim0_grid((8, 4), 2)})
    return MetadataView.load(tmp_path / "src")


def _buffers(pieces, tensors):
    """Standalone buffers holding each piece's box of the known tensors."""
    return [(p, tensors[p.key][p.chunk.slices()].clone()) for p in pieces]


def _save(part_dir: Path, pieces, byte_pieces=()) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="torch.distributed is disabled")
        dcp.save(
            {},
            storage_writer=FileSystemWriter(part_dir, per_thread_copy_ahead=0),
            planner=PieceSavePlanner(pieces, byte_pieces),
            no_dist=True,
        )


def _stored_items(part_dir: Path) -> dict:
    """Every stored item of a store, decoded independently from its byte range."""
    metadata = FileSystemReader(part_dir).read_metadata()
    items = {}
    for index, info in metadata.storage_data.items():
        with open(part_dir / info.relative_path, "rb") as f:
            f.seek(info.offset)
            raw = f.read(info.length)
        offsets = None if index.offset is None else tuple(index.offset)
        items[(index.fqn, offsets)] = raw if offsets is None else torch.load(io.BytesIO(raw))
    return items


def test_parts_hold_partial_chunks_of_one_tensor(source, tmp_path):
    pieces = copy_pieces(source.tensors["w"])
    assert [p.chunk for p in pieces] == list(dim0_grid((8, 4), 2))
    _save(tmp_path / "a", _buffers([pieces[0], pieces[2]], {"w": W}))
    _save(tmp_path / "b", _buffers([pieces[1], pieces[3]], {"w": W}))
    for part, rows in (("a", (0, 4)), ("b", (2, 6))):
        metadata = FileSystemReader(tmp_path / part).read_metadata()
        entry = metadata.state_dict_metadata["w"]
        assert tuple(entry.size) == (8, 4) and entry.properties.dtype == torch.float32
        assert sorted(tuple(c.offsets) for c in entry.chunks) == [(r, 0) for r in rows]
        items = _stored_items(tmp_path / part)
        assert set(items) == {("w", (r, 0)) for r in rows}
        for row in rows:
            assert torch.equal(items[("w", (row, 0))], W[row : row + 2])
        assert metadata.planner_data is None


def test_byte_items_are_written_verbatim(source, tmp_path):
    payload = torch_save_bytes([{"iteration": 3}])
    pieces = copy_pieces(source.tensors["b"])
    _save(tmp_path / "part", _buffers(pieces, {"b": B}), [BytePiece("blob", payload)])
    items = _stored_items(tmp_path / "part")
    assert items[("blob", None)] == payload
    assert torch.equal(items[("b", (0,))], B)


def test_standalone_buffers_are_written_without_a_copy(source, tmp_path, monkeypatch):
    buffers = _buffers(copy_pieces(source.tensors["w"]), {"w": W})
    saved_pointers = []
    clones = []
    real_save, real_clone = torch.save, torch.Tensor.clone

    def recording_save(obj, *args, **kwargs):
        if isinstance(obj, torch.Tensor):
            saved_pointers.append(obj.data_ptr())
        return real_save(obj, *args, **kwargs)

    def counting_clone(tensor, *args, **kwargs):
        clones.append(tuple(tensor.shape))
        return real_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "save", recording_save)
    monkeypatch.setattr(torch.Tensor, "clone", counting_clone)
    _save(tmp_path / "part", buffers)
    assert sorted(saved_pointers) == sorted(buffer.data_ptr() for _, buffer in buffers)
    assert clones == []


def test_planner_data_collects_planner_paths(source, tmp_path):
    pieces = [
        Piece("model.w", p.global_shape, p.dtype, p.chunk, p.ops, planner_path=("model", "w"))
        for p in copy_pieces(source.tensors["w"])
    ]
    byte_pieces = [BytePiece("args", b"x", planner_path=("args",))]
    _save(tmp_path / "part", _buffers(pieces, {"model.w": W}), byte_pieces)
    metadata = FileSystemReader(tmp_path / "part").read_metadata()
    assert metadata.planner_data == {"model.w": ("model", "w"), "args": ("args",)}


def _piece_w(source, index=0):
    return copy_pieces(source.tensors["w"])[index]


@pytest.mark.parametrize(
    "make, message",
    [
        (lambda p: [(p, W[0:2])], "does not own exactly its contiguous storage"),
        (
            lambda p: [(p, W[0:2].clone().t().contiguous().t())],
            "does not own exactly its contiguous storage",
        ),
        (lambda p: [(p, torch.zeros(4, 2))], r"is \(4, 2\) torch.float32, expected"),
        (lambda p: [(p, torch.zeros(2, 4, dtype=torch.float64))], "expected"),
        (lambda p: [(p, torch.zeros(2, 4, device="meta"))], "is on meta"),
        (lambda p: [(p, torch.zeros(2, 4)), (p, torch.zeros(2, 4))], "written twice"),
    ],
)
def test_planner_rejects_bad_buffers(source, make, message):
    with pytest.raises(InternalError, match=message):
        PieceSavePlanner(make(_piece_w(source)), [])


def test_planner_rejects_key_clashes(source):
    piece = _piece_w(source)
    with pytest.raises(InternalError, match="byte item w is written twice"):
        PieceSavePlanner([(piece, torch.zeros(2, 4))], [BytePiece("w", b"")])
    with pytest.raises(InternalError, match="byte item x is written twice"):
        PieceSavePlanner([], [BytePiece("x", b""), BytePiece("x", b"")])
    other = Piece(piece.key, piece.global_shape, piece.dtype, piece.chunk, piece.ops, ("a",))
    second = _piece_w(source, 1)
    second = Piece(second.key, second.global_shape, second.dtype, second.chunk, second.ops, ("b",))
    with pytest.raises(InternalError, match="two planner paths"):
        PieceSavePlanner([(other, torch.zeros(2, 4)), (second, torch.zeros(2, 4))], [])


def _chunk(offsets=(0, 0), sizes=(8, 4), dtype=torch.float32) -> StoredChunk:
    return StoredChunk("src", offsets, sizes, dtype, "__0_0.distcp", 0, 100, ())


def test_pieces_are_validated():
    full = Region((0, 0), (8, 4))
    top, bottom = Region((0, 0), (4, 4)), Region((4, 0), (4, 4))
    Piece("w", (8, 4), torch.float32, full, (CopyOp(_chunk(), full, full),))
    Piece(
        "w",
        (8, 4),
        torch.float32,
        full,
        (CopyOp(_chunk(), bottom, top), CopyOp(_chunk(), top, bottom)),
    )
    Piece("s", (), torch.float32, Region((), ()), (CopyOp(_chunk((), ()), *[Region((), ())] * 2),))
    Piece("e", (0, 4), torch.float32, Region((0, 0), (0, 4)), ())
    with pytest.raises(InternalError, match="do not tile it: chunks cover 16 of 32"):
        Piece("w", (8, 4), torch.float32, full, (CopyOp(_chunk(), top, top),))
    with pytest.raises(InternalError, match="overlap"):
        Piece("w", (8, 4), torch.float32, full, (CopyOp(_chunk(), full, full),) * 2)
    with pytest.raises(InternalError, match="casts are not supported"):
        Piece("w", (8, 4), torch.bfloat16, full, (CopyOp(_chunk(), full, full),))
    with pytest.raises(InternalError, match="outside shape"):
        Piece("w", (4, 4), torch.float32, full, (CopyOp(_chunk(), full, full),))
    with pytest.raises(InternalError, match="outside stored chunk"):
        CopyOp(_chunk(sizes=(4, 4)), full, full)
    with pytest.raises(InternalError, match="differ in shape"):
        CopyOp(_chunk(), top, full)


def test_copy_pieces_mirror_the_stored_grid(source):
    entry = source.tensors["w"]
    pieces = copy_pieces(entry)
    assert [p.chunk for p in pieces] == [c.region for c in entry.chunks]
    for piece, chunk in zip(pieces, entry.chunks):
        (op,) = piece.ops
        assert op.src == chunk and op.src_region == chunk.region
        assert op.dst_region == Region.full(chunk.sizes)
        assert piece.nbytes == chunk.nbytes and piece.planner_path is None
