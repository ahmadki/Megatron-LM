# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit packing: clusters of shared stored items, dim-0 splitting, budgets and dispatch order."""

from __future__ import annotations

import pytest
import torch

from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.engine.pieces import (
    BytePiece,
    CopyOp,
    Piece,
    Unit,
    copy_pieces,
)
from tools.checkpoint.format_converter.engine.reader import ReadReq, SourceReader
from tools.checkpoint.format_converter.engine.regions import Region
from tools.checkpoint.format_converter.engine.runner import RunOptions, run_program
from tools.checkpoint.format_converter.engine.scheduler import (
    dispatch_order,
    pack_units,
    split_piece,
)
from tools.checkpoint.format_converter.engine.staging import PublishTarget
from tools.checkpoint.format_converter.errors import BudgetExceeded, ExitCode, InternalError

from .._programs import packed_program
from .._synthetic import dim0_grid, product_grid, write_store

W = torch.arange(32, dtype=torch.float32).reshape(8, 4)  # 4 stored chunks of 2 rows (32 B each)
BIG = torch.arange(256, dtype=torch.float32).reshape(16, 16)  # one stored chunk of 1 KiB
B = torch.arange(6, dtype=torch.bfloat16)


@pytest.fixture(name="view")
def fixture_view(tmp_path) -> MetadataView:
    write_store(tmp_path / "src", {"w": W, "big": BIG, "b": B}, grids={"w": dim0_grid((8, 4), 2)})
    return MetadataView.load(tmp_path / "src")


def _copies(view) -> list[Piece]:
    return [p for key in sorted(view.tensors) for p in copy_pieces(view.tensors[key])]


def _whole(view, key, out_key) -> Piece:
    """An output piece covering all of ``key``, built from every one of its stored chunks."""
    entry = view.tensors[key]
    ops = tuple(CopyOp(c, c.region, c.region) for c in entry.chunks)
    return Piece(out_key, entry.shape, entry.dtype, Region.full(entry.shape), ops)


def _rows(view, key, out_key, start, stop) -> Piece:
    """An output tensor holding rows [start, stop) of single-chunk tensor ``key``."""
    entry = view.tensors[key]
    (chunk,) = entry.chunks
    box = Region((start, 0), (stop - start, entry.shape[1]))
    local = Region.full(box.sizes)
    return Piece(out_key, box.sizes, entry.dtype, local, (CopyOp(chunk, box, local),))


def _items(unit: Unit) -> set:
    return {(op.src.relative_path, op.src.offset) for p in unit.pieces for op in p.ops}


def test_copies_are_packed_next_fit_within_the_budget(view):
    pieces = _copies(view)
    units = pack_units(pieces, [], unit_bytes=160, hard_cap=4096)
    packed = [p for unit in units for p in unit.pieces]
    assert sorted((p.key, p.chunk.offsets) for p in packed) == sorted(
        (p.key, p.chunk.offsets) for p in pieces
    )
    for unit in units:
        chunks = {op.src for p in unit.pieces for op in p.ops}
        expected = sum(p.nbytes for p in unit.pieces) + max(c.nbytes for c in chunks)
        assert unit.est_peak_bytes == expected  # copies read each item once: no staging
        assert unit.read_bytes == sum(c.length for c in chunks)
        assert unit.est_peak_bytes <= 160 or [p.key for p in unit.pieces] == ["big"]
    assert units[0].pieces[0].key == "big"  # larger clusters are packed first
    first_items = [_items(unit) for unit in units]
    assert all(a.isdisjoint(b) for i, a in enumerate(first_items) for b in first_items[i + 1 :])


def test_byte_items_go_to_unit_zero(view):
    blob = BytePiece("blob", b"x" * 10)
    units = pack_units(copy_pieces(view.tensors["b"]), [blob], unit_bytes=10**6, hard_cap=10**6)
    assert [u.byte_pieces for u in units] == [(blob,)]
    assert units[0].est_peak_bytes == 12 + 12 + 2 * 10
    only_bytes = pack_units([], [blob], unit_bytes=10, hard_cap=10)
    assert [(u.uid, u.pieces, u.byte_pieces) for u in only_bytes] == [(0, (), (blob,))]
    assert pack_units([], [], unit_bytes=10, hard_cap=10) == ()


def test_pieces_reading_one_stored_item_share_a_unit_and_its_staging(view, tmp_path):
    top, bottom = _rows(view, "big", "top", 0, 8), _rows(view, "big", "bottom", 8, 16)
    units = pack_units(
        [top, bottom, *copy_pieces(view.tensors["b"])], [], unit_bytes=64, hard_cap=4096
    )
    (shared,) = [u for u in units if {p.key for p in u.pieces} == {"top", "bottom"}]
    assert shared.est_peak_bytes == 512 + 512 + 1024 + 1024  # pieces, staging box, decode
    target = PublishTarget(tmp_path / "out")
    program = packed_program(view, target, [top, bottom], unit_bytes=64, hard_cap=4096)
    result = run_program(program, target, RunOptions())
    (record,) = result.journal
    assert record["items_touched"] == record["torch_load_calls"] == 1  # read once
    out = MetadataView.load(tmp_path / "out")
    reader = SourceReader(out, max_item_bytes=4096)
    for key, rows in (("top", BIG[:8]), ("bottom", BIG[8:])):
        buffer = torch.empty(8, 16)
        reader.read_into([ReadReq(key, Region.full((8, 16)), buffer)])
        assert torch.equal(buffer, rows)


def test_cluster_above_the_hard_cap_exits_8(view):
    with pytest.raises(
        BudgetExceeded, match=r"stored chunk \(0, 0\) of big .* needs 2.00 KiB"
    ) as e:
        pack_units(copy_pieces(view.tensors["big"]), [], unit_bytes=64, hard_cap=2047)
    assert e.value.exit_code == ExitCode.BUDGET_EXCEEDED
    assert "--max-unit-bytes-hard 2048" in e.value.hint
    assert len(pack_units(copy_pieces(view.tensors["big"]), [], unit_bytes=64, hard_cap=2048)) == 1


def test_piece_reading_several_items_is_split_along_dim0(view, tmp_path):
    whole = _whole(view, "w", "w_all")
    slabs = split_piece(whole, 64)
    assert [s.chunk for s in slabs] == [Region((0, 0), (4, 4)), Region((4, 0), (4, 4))]
    for slab in slabs:
        rows = slab.chunk.offsets[0]
        assert [(op.src.offsets, op.src_region, op.dst_region) for op in slab.ops] == [
            ((rows + r, 0), Region((rows + r, 0), (2, 4)), Region((r, 0), (2, 4))) for r in (0, 2)
        ]
    target = PublishTarget(tmp_path / "out")
    program = packed_program(view, target, [whole], unit_bytes=128, hard_cap=4096)
    assert len(program.units) == 2
    run_program(program, target, RunOptions())
    out = MetadataView.load(tmp_path / "out")
    assert [c.region for c in out.tensors["w_all"].chunks] == [s.chunk for s in slabs]
    buffer = torch.empty(8, 4)
    SourceReader(out, max_item_bytes=4096).read_into(
        [ReadReq("w_all", Region.full((8, 4)), buffer)]
    )
    assert torch.equal(buffer, W)


def test_layer_copies_of_a_stacked_tensor_are_split_and_run(tmp_path):
    """A tensor of its own filled from one layer of a stacked tensor (rank 3 into rank 2)."""
    stacked = torch.arange(96, dtype=torch.float32).reshape(3, 8, 4)
    grid = product_grid((3, 8, 4), [[1, 2], [4], []])  # one chunk per layer and row half
    write_store(tmp_path / "src", {"s": stacked}, grids={"s": grid})
    view = MetadataView.load(tmp_path / "src")
    halves = [c for c in view.tensors["s"].chunks if c.offsets[0] == 1]
    ops = tuple(CopyOp(c, c.region, Region((c.offsets[1], 0), (4, 4))) for c in reversed(halves))
    piece = Piece("layer1", (8, 4), torch.float32, Region.full((8, 4)), ops)
    slabs = split_piece(piece, 32)  # 16-byte rows: two rows per slab
    assert [s.chunk for s in slabs] == [Region((r, 0), (2, 4)) for r in (0, 2, 4, 6)]
    assert [(op.src_region, op.dst_region) for op in slabs[3].ops] == [
        (Region((1, 6, 0), (1, 2, 4)), Region((0, 0), (2, 4)))
    ]
    target = PublishTarget(tmp_path / "out")
    run_program(
        packed_program(view, target, [piece], unit_bytes=64, hard_cap=4096), target, RunOptions()
    )
    buffer = torch.empty(8, 4)
    out = MetadataView.load(tmp_path / "out")
    SourceReader(out, max_item_bytes=4096).read_into(
        [ReadReq("layer1", Region.full((8, 4)), buffer)]
    )
    assert torch.equal(buffer, stacked[1])


def test_a_layer_chunk_of_a_stacked_tensor_is_split_after_its_unit_axis(view):
    """A chunk [l:l+1, ...] reading several items is cut along its second axis."""
    ops = tuple(
        CopyOp(c, c.region, Region((0, *c.offsets), (1, *c.sizes)))
        for c in view.tensors["w"].chunks
    )
    piece = Piece("stacked", (3, 8, 4), torch.float32, Region((1, 0, 0), (1, 8, 4)), ops)
    slabs = split_piece(piece, 64)
    assert [s.chunk for s in slabs] == [Region((1, 0, 0), (1, 4, 4)), Region((1, 4, 0), (1, 4, 4))]
    assert [(op.src_region, op.dst_region) for op in slabs[1].ops] == [
        (Region((rows, 0), (2, 4)), Region((0, r, 0), (1, 2, 4))) for r, rows in ((0, 4), (2, 6))
    ]
    with pytest.raises(BudgetExceeded, match=r"one row of stacked .* is 16 B"):
        split_piece(piece, 8)


def test_piece_reading_one_item_is_never_split(view):
    (big,) = copy_pieces(view.tensors["big"])
    assert split_piece(big, 64) == (big,)
    (scalar_like,) = copy_pieces(view.tensors["b"])
    assert split_piece(scalar_like, 1) == (scalar_like,)


def test_row_above_half_the_budget_exits_8(view):
    whole = _whole(view, "w", "w_all")
    with pytest.raises(BudgetExceeded, match=r"one row of w_all .* is 16 B, above half") as e:
        pack_units([whole], [], unit_bytes=30, hard_cap=4096)
    assert "--unit-bytes 32" in e.value.hint


def test_dispatch_order_is_largest_first_then_file_order(view):
    units = pack_units(_copies(view), [BytePiece("blob", b"")], unit_bytes=100, hard_cap=4096)
    order = dispatch_order(units)
    estimates = [next(u for u in units if u.uid == uid).est_peak_bytes for uid in order]
    assert estimates == sorted(estimates, reverse=True)
    ties = [
        Unit(0, (), est_peak_bytes=5),
        Unit(1, (), est_peak_bytes=9),
        Unit(2, (), est_peak_bytes=5),
    ]
    assert dispatch_order(ties) == (1, 0, 2)


def test_budgets_must_be_positive(view):
    with pytest.raises(InternalError, match="unit budgets must be positive"):
        pack_units(_copies(view), [], unit_bytes=0, hard_cap=1)


def _cast(piece: Piece, dtype: torch.dtype, key: str) -> Piece:
    return Piece(key, piece.global_shape, dtype, piece.chunk, piece.ops, cast=True)


def test_a_piece_casts_only_when_it_plans_to(view):
    exact = _whole(view, "w", "w")
    with pytest.raises(InternalError, match="no cast is planned"):
        Piece("w16", exact.global_shape, torch.bfloat16, exact.chunk, exact.ops)
    with pytest.raises(InternalError, match="only floating-point types"):
        _cast(exact, torch.int32, "wi")
    assert _cast(exact, torch.bfloat16, "w16").cast


def test_a_cast_and_a_copy_of_one_item_share_its_staging_and_run_once(view, tmp_path):
    """A master and its rounded weight: one unit, one decode, the rounded copy to nearest even."""
    (chunk,) = view.tensors["big"].chunks
    exact = _whole(view, "big", "master")
    rounded = _cast(exact, torch.bfloat16, "weight")
    (unit,) = pack_units([exact, rounded], [], unit_bytes=1 << 20, hard_cap=1 << 20)
    assert unit.est_peak_bytes == exact.nbytes + rounded.nbytes + 2 * chunk.nbytes
    (alone,) = pack_units([rounded], [], unit_bytes=1 << 20, hard_cap=1 << 20)
    assert alone.est_peak_bytes == rounded.nbytes + 2 * chunk.nbytes  # staged, then decoded
    target = PublishTarget(tmp_path / "out" / "iter_0000001", None, None)
    program = packed_program(view, target, [exact, rounded], unit_bytes=1 << 20)
    (record,) = run_program(program, target, RunOptions()).journal
    assert record["torch_load_calls"] == record["items_touched"] == 1
    out = MetadataView.load(target.iter_dir)
    master, weight = torch.empty(16, 16), torch.empty(16, 16, dtype=torch.bfloat16)
    SourceReader(out, max_item_bytes=1 << 20).read_into(
        [
            ReadReq("master", Region.full((16, 16)), master),
            ReadReq("weight", Region.full((16, 16)), weight),
        ]
    )
    assert torch.equal(master, BIG) and torch.equal(weight, BIG.bfloat16())
