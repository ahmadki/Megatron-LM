# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Merging part stores: the happy path, every validation failure, link fallbacks, post-check."""

from __future__ import annotations

import errno
import hashlib
import os
import pickle
import warnings

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

from tools.checkpoint.format_converter.engine import torch_compat
from tools.checkpoint.format_converter.engine.merge import (
    ItemDigest,
    MetadataMerger,
    PartStore,
    post_check,
)
from tools.checkpoint.format_converter.engine.metadata import MetadataView, StoredChunk
from tools.checkpoint.format_converter.engine.part_writer import write_part
from tools.checkpoint.format_converter.engine.pieces import (
    BytePiece,
    CopyOp,
    Piece,
    copy_pieces,
    expected_tensors,
)
from tools.checkpoint.format_converter.engine.regions import Region
from tools.checkpoint.format_converter.errors import (
    ExitCode,
    IntegrityError,
    InternalError,
    RefusedError,
)

from .._synthetic import dim0_grid, torch_save_bytes, write_store

W = torch.arange(32, dtype=torch.float32).reshape(8, 4)
B = torch.arange(6, dtype=torch.bfloat16) - 2.5
TENSORS = {"w": W, "b": B}
BLOB = BytePiece("blob", torch_save_bytes([{"iteration": 3}]))


@pytest.fixture(name="pieces")
def fixture_pieces(tmp_path) -> dict[str, tuple[Piece, ...]]:
    write_store(tmp_path / "src", TENSORS, grids={"w": dim0_grid((8, 4), 2)})
    view = MetadataView.load(tmp_path / "src")
    return {key: copy_pieces(view.tensors[key]) for key in TENSORS}


def _piece(key, shape, dtype, region) -> Piece:
    """A piece of an arbitrary box, copied from a same-shaped (fabricated) source chunk."""
    source = StoredChunk(key, region.offsets, region.sizes, dtype, "__0_0.distcp", 0, 1, ())
    return Piece(key, shape, dtype, region, (CopyOp(source, region, Region.full(region.sizes)),))


def _part(tmp_path, uid, pieces, byte_pieces=(), tensors=None) -> PartStore:
    tensors = tensors or TENSORS
    path = tmp_path / "parts" / f"u{uid:05d}"
    buffers = [(p, tensors[p.key][p.chunk.slices()].to(p.dtype).clone()) for p in pieces]
    write_part(path, buffers, byte_pieces, threads=1)
    return PartStore(uid, path)


def _standard_parts(tmp_path, pieces) -> list[PartStore]:
    w = pieces["w"]
    return [
        _part(tmp_path, 0, [w[0], w[2]], [BLOB]),
        _part(tmp_path, 1, [w[1], w[3], *pieces["b"]]),
    ]


def _merger(pieces, byte_keys=("blob",)) -> MetadataMerger:
    return MetadataMerger(
        expected_tensors(p for group in pieces.values() for p in group), frozenset(byte_keys)
    )


def _merge(merger, parts, tmp_path):
    return merger.merge(
        parts,
        tmp_path / "final",
        journal=tmp_path / "merge.journal",
        checkpoint_id="/out/iter_0000007",
        save_id="1234",
    )


def _load_all(store) -> dict[str, torch.Tensor]:
    """Read the merged store back with torch's own reader and default planner."""
    metadata = FileSystemReader(store).read_metadata()
    targets = {
        key: torch.empty(tuple(item.size), dtype=item.properties.dtype)
        for key, item in metadata.state_dict_metadata.items()
        if hasattr(item, "size")
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dcp.load(targets, storage_reader=FileSystemReader(store), no_dist=True)
    return targets


def test_merge_links_parts_into_one_valid_store(tmp_path, pieces):
    parts = _standard_parts(tmp_path, pieces)
    summary = _merge(_merger(pieces), parts, tmp_path)
    final = tmp_path / "final"
    assert (
        summary.files,
        summary.renamed,
        summary.tensors,
        summary.chunks,
        summary.byte_items,
    ) == (2, 0, 2, 5, 1)
    assert sorted(p.name for p in final.iterdir()) == [
        ".metadata",
        "u00000___0_0.distcp",
        "u00001___0_0.distcp",
    ]
    for part in parts:
        linked = final / f"u{part.uid:05d}___0_0.distcp"
        assert os.path.samefile(part.path / "__0_0.distcp", linked)
        assert linked.stat().st_nlink == 2
    loaded = _load_all(final)
    assert torch.equal(loaded["w"], W) and torch.equal(loaded["b"], B)
    view = MetadataView.load(final)
    assert [c.offsets for c in view.tensors["w"].chunks] == [(0, 0), (2, 0), (4, 0), (6, 0)]
    assert view.planner_data is None
    assert set(view.byte_items) == {"blob"}
    metadata = pickle.loads((final / ".metadata").read_bytes())
    assert metadata.storage_meta.checkpoint_id == "/out/iter_0000007"
    assert metadata.storage_meta.save_id == "1234"
    assert not hasattr(metadata, "all_local_plans")
    records = torch_compat.storage_records(metadata)
    hints = {(r.key, r.offsets): r.index.index for r in records}
    assert [hints[("w", (row, 0))] for row in (0, 2, 4, 6)] == [0, 1, 2, 3]


def test_merge_is_idempotent_and_deterministic(tmp_path, pieces):
    parts = _standard_parts(tmp_path, pieces)
    merger = _merger(pieces)
    _merge(merger, parts, tmp_path)
    first = (tmp_path / "final" / ".metadata").read_bytes()
    (tmp_path / "final" / ".metadata").unlink()
    _merge(merger, parts, tmp_path)
    assert (tmp_path / "final" / ".metadata").read_bytes() == first


def test_merge_records_the_dcp_version_only_where_torch_has_the_field(tmp_path, pieces):
    parts = _standard_parts(tmp_path, pieces)
    path = parts[0].path / ".metadata"
    metadata = pickle.loads(path.read_bytes())
    metadata.version = "9.9.9"
    path.write_bytes(pickle.dumps(metadata))
    _merge(_merger(pieces), parts, tmp_path)
    merged = pickle.loads((tmp_path / "final" / ".metadata").read_bytes())
    has_field = torch_compat.metadata_has_version_field()
    assert getattr(merged, "version", None) == ("9.9.9" if has_field else None)


def test_merge_keeps_planner_paths(tmp_path, pieces):
    with_paths = [
        Piece("model.w", p.global_shape, p.dtype, p.chunk, p.ops, ("model", "w"))
        for p in pieces["w"]
    ]
    tensors = {"model.w": W}
    parts = [
        _part(tmp_path, 0, with_paths[:2], [BytePiece("args", b"a", ("args",))], tensors),
        _part(tmp_path, 1, with_paths[2:], tensors=tensors),
    ]
    merger = MetadataMerger(expected_tensors(with_paths), frozenset({"args"}))
    _merge(merger, parts, tmp_path)
    view = MetadataView.load(tmp_path / "final")
    assert view.planner_data == {"model.w": ("model", "w"), "args": ("args",)}


def _expect_integrity_error(merger, parts, tmp_path, match):
    with pytest.raises(IntegrityError, match=match) as excinfo:
        _merge(merger, parts, tmp_path)
    assert excinfo.value.exit_code == ExitCode.INTEGRITY
    assert not (tmp_path / "final" / ".metadata").exists()


def test_gap_is_refused(tmp_path, pieces):
    w = pieces["w"]
    parts = [_part(tmp_path, 0, [w[0], w[2]], [BLOB]), _part(tmp_path, 1, [w[3], *pieces["b"]])]
    _expect_integrity_error(_merger(pieces), parts, tmp_path, "cover 24 of 32 elements")


def test_duplicate_chunk_is_an_overlap(tmp_path, pieces):
    w = pieces["w"]
    parts = _standard_parts(tmp_path, pieces) + [_part(tmp_path, 2, [w[0]])]
    _expect_integrity_error(_merger(pieces), parts, tmp_path, r"chunks \[0:2, 0:4\] and .* overlap")


def test_overlapping_chunk_is_refused(tmp_path, pieces):
    w = pieces["w"]
    shifted = _piece("w", (8, 4), torch.float32, Region((1, 0), (2, 4)))
    parts = [
        _part(tmp_path, 0, [w[0], shifted, w[2]], [BLOB]),
        _part(tmp_path, 1, [w[3], *pieces["b"]]),
    ]
    _expect_integrity_error(_merger(pieces), parts, tmp_path, "overlap")


def test_unplanned_chunk_grid_is_refused(tmp_path, pieces):
    w = pieces["w"]
    top = _piece("w", (8, 4), torch.float32, Region((0, 0), (4, 4)))
    parts = [_part(tmp_path, 0, [top, w[2], w[3], *pieces["b"]], [BLOB])]
    _expect_integrity_error(
        _merger(pieces), parts, tmp_path, r"differ from the plan: 3 chunks, first unplanned \[0:4"
    )


def test_missing_and_unexpected_keys_are_refused(tmp_path, pieces):
    w = pieces["w"]
    missing_b = [_part(tmp_path, 0, list(w), [BLOB])]
    _expect_integrity_error(_merger(pieces), missing_b, tmp_path, r"missing \['b'\]")
    extra = _standard_parts(tmp_path / "x", pieces)
    merger = MetadataMerger(expected_tensors(w), frozenset({"blob"}))
    _expect_integrity_error(merger, extra, tmp_path / "x", r"unexpected \['b'\]")


def test_byte_items_are_checked(tmp_path, pieces):
    parts = _standard_parts(tmp_path, pieces)
    _expect_integrity_error(
        _merger(pieces, ("blob", "args")), parts, tmp_path, r"byte items .* missing \['args'\]"
    )
    parts = _standard_parts(tmp_path / "dup", pieces) + [_part(tmp_path / "dup", 2, [], [BLOB])]
    _expect_integrity_error(
        _merger(pieces), parts, tmp_path / "dup", "blob is stored by unit 0 and unit 2"
    )


def test_dtype_differing_from_the_plan_is_refused(tmp_path, pieces):
    as_double = [_piece("w", (8, 4), torch.float64, p.chunk) for p in pieces["w"]]
    parts = [_part(tmp_path, 0, as_double, [BLOB]), _part(tmp_path, 1, list(pieces["b"]))]
    _expect_integrity_error(_merger(pieces), parts, tmp_path, "is torch.float64, the plan says")


def test_parts_disagreeing_on_a_shape_are_refused(tmp_path, pieces):
    longer = _piece("b", (7,), torch.bfloat16, Region((6,), (1,)))
    parts = _standard_parts(tmp_path, pieces)
    parts.append(_part(tmp_path, 2, [longer], tensors={"b": torch.zeros(7)}))
    _expect_integrity_error(_merger(pieces), parts, tmp_path, r"parts disagree on b: \(6,\)")


def test_short_data_file_is_refused(tmp_path, pieces):
    parts = _standard_parts(tmp_path, pieces)
    data = parts[1].path / "__0_0.distcp"
    os.truncate(data, data.stat().st_size - 10)
    _expect_integrity_error(_merger(pieces), parts, tmp_path, "u00001___0_0.distcp holds")


def test_missing_part_file_is_refused(tmp_path, pieces):
    parts = _standard_parts(tmp_path, pieces)
    (parts[1].path / "__0_0.distcp").unlink()
    _expect_integrity_error(_merger(pieces), parts, tmp_path, "part file .* is missing")


def test_foreign_file_at_the_target_is_refused(tmp_path, pieces):
    parts = _standard_parts(tmp_path, pieces)
    (tmp_path / "final").mkdir()
    (tmp_path / "final" / "u00000___0_0.distcp").write_bytes(b"stale")
    _expect_integrity_error(_merger(pieces), parts, tmp_path, "exists and is not a link")


@pytest.mark.parametrize("code", [errno.EPERM, errno.EOPNOTSUPP])
def test_rename_fallback_is_journaled_and_resumable(tmp_path, pieces, monkeypatch, code):
    parts = _standard_parts(tmp_path, pieces)

    def refuse_links(src, dst):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(os, "link", refuse_links)
    summary = _merge(_merger(pieces), parts, tmp_path)
    assert summary.renamed == 2
    assert not (parts[0].path / "__0_0.distcp").exists()
    journal = (tmp_path / "merge.journal").read_text().splitlines()
    assert len(journal) == 2 and '"dst"' in journal[0]
    first = (tmp_path / "final" / ".metadata").read_bytes()
    assert torch.equal(_load_all(tmp_path / "final")["w"], W)
    (tmp_path / "final" / ".metadata").unlink()
    summary = _merge(_merger(pieces), parts, tmp_path)
    assert summary.renamed == 2 and (tmp_path / "final" / ".metadata").read_bytes() == first


def test_cross_device_link_is_refused_with_fc_e02(tmp_path, pieces, monkeypatch):
    parts = _standard_parts(tmp_path, pieces)

    def cross_device(src, dst):
        raise OSError(errno.EXDEV, os.strerror(errno.EXDEV))

    monkeypatch.setattr(os, "link", cross_device)
    with pytest.raises(RefusedError) as excinfo:
        _merge(_merger(pieces), parts, tmp_path)
    assert excinfo.value.exit_code == ExitCode.REFUSED
    assert excinfo.value.format().startswith("FC-E02: cannot hard-link")


def _digests(parts_pieces) -> list[ItemDigest]:
    digests = []
    for piece in parts_pieces:
        buffer = TENSORS[piece.key][piece.chunk.slices()].contiguous()
        sha = hashlib.sha256(buffer.reshape(-1).view(torch.uint8).numpy()).hexdigest()
        digests.append(ItemDigest(piece.key, piece.chunk.offsets, sha))
    digests.append(ItemDigest("blob", None, hashlib.sha256(BLOB.payload).hexdigest()))
    return digests


def test_post_check_accepts_the_store_and_catches_corruption(tmp_path, pieces):
    parts = _standard_parts(tmp_path, pieces)
    _merge(_merger(pieces), parts, tmp_path)
    digests = _digests([pieces["w"][1], pieces["w"][2], *pieces["b"]])
    post_check(tmp_path / "final", digests)
    wrong_payload = [*digests[:-1], ItemDigest("blob", None, "0" * 64)]
    with pytest.raises(IntegrityError, match="post-check: 1 items .* differ .*: blob"):
        post_check(tmp_path / "final", wrong_payload)
    data = tmp_path / "final" / "u00001___0_0.distcp"
    raw = bytearray(data.read_bytes())
    view = MetadataView.load(tmp_path / "final")
    chunk = next(c for c in view.tensors["w"].chunks if c.offsets == (2, 0))
    item = bytes(raw[chunk.offset : chunk.offset + chunk.length])
    position = chunk.offset + item.index(W[2:4].numpy().tobytes())  # the chunk's element bytes
    raw[position] ^= 0x01
    data.write_bytes(bytes(raw))
    with pytest.raises(IntegrityError, match=r"differ .*: w@\(2, 0\)"):
        post_check(tmp_path / "final", digests)


def test_expected_tensors_are_validated(pieces):
    w = pieces["w"]
    with pytest.raises(InternalError, match="do not tile it: chunks cover 24 of 32"):
        expected_tensors([w[0], w[1], w[3]])
    other = Piece("w", (9, 4), torch.float32, Region((0, 0), (2, 4)), w[0].ops)
    with pytest.raises(InternalError, match="pieces of w disagree"):
        expected_tensors([w[0], other])
    expected = expected_tensors(reversed(w))
    assert list(expected) == ["w"] and expected["w"].chunks == tuple(p.chunk for p in w)
