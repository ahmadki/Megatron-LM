# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Merging part stores into one DCP store, validated against the plan.

Modelled on the ``.metadata`` rewrite of Megatron's ``remove_sharded_tensors``:
every data file of every part is hard-linked into the final directory under a
unit-unique name (``u<uid>_<file>``), the parts' metadata tables are unioned
with their storage entries pointing at the new names, and the union is checked
against the plan before a fresh ``.metadata`` is written atomically. Nothing is
copied, and the parts stay intact until the store is published.

If the file system refuses hard links (EPERM, ENOTSUP), a file is renamed
instead; the move is appended to a merge journal first, so a merge resumed after
a crash accepts files that already moved. A hard link across file systems
(EXDEV) is refused with FC-E02: staging always lives on the destination's file
system.

Every check failure is an :class:`~..errors.IntegrityError` (exit 6).
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)

from .. import errors
from ..errors import IntegrityError, Refusal
from . import torch_compat
from .faults import inject_fault
from .fs import append_line, write_atomic
from .hashing import byte_view
from .metadata import METADATA_FILE, MetadataView, read_metadata_file
from .pieces import ExpectedTensor
from .reader import ReadReq, SourceReader
from .regions import Region, find_tiling_defect

_NO_HARD_LINKS = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP})
_MAX_LISTED = 8


@dataclass(frozen=True, slots=True)
class PartStore:
    """A complete part store of one unit.

    Attributes:
        uid: The unit id (parts are merged in uid order).
        path: The part store directory.
    """

    uid: int
    path: Path


@dataclass(frozen=True, slots=True)
class MergeSummary:
    """What a merge produced.

    Attributes:
        files: Data files placed in the final directory.
        renamed: Of those, files moved because hard links were refused.
        tensors: Tensors in the merged store.
        chunks: Stored chunks in the merged store.
        byte_items: Byte items in the merged store.
    """

    files: int
    renamed: int
    tensors: int
    chunks: int
    byte_items: int


@dataclass(frozen=True, slots=True)
class ItemDigest:
    """SHA-256 of what was written for one piece or byte item.

    Attributes:
        key: Tensor or byte-item key.
        offsets: Chunk offsets of a piece, ``None`` for a byte item.
        sha256: Hex digest of the piece buffer's logical bytes, or of the payload.
    """

    key: str
    offsets: tuple[int, ...] | None
    sha256: str


@dataclass(frozen=True, slots=True)
class _Stored:
    record: torch_compat.StorageRecord
    name: str
    uid: int


@dataclass
class _TensorUnion:
    meta: TensorStorageMetadata
    chunks: list[tuple[Any, _Stored]] = field(default_factory=list)


class MetadataMerger:
    """Merges part stores into a final store holding exactly the planned output.

    Args:
        expected: The planned output tensors by key.
        expected_bytes: The planned output byte-item keys.
    """

    def __init__(
        self, expected: Mapping[str, ExpectedTensor], expected_bytes: frozenset[str]
    ) -> None:
        self._expected = dict(expected)
        self._expected_bytes = frozenset(expected_bytes)

    def merge(
        self,
        parts: Sequence[PartStore],
        final_dir: Path,
        *,
        journal: Path,
        checkpoint_id: str,
        save_id: str,
    ) -> MergeSummary:
        """Place the parts' data files in ``final_dir`` and write its ``.metadata``.

        The merge is idempotent: running it again (for example after a crash)
        accepts files that are already linked or journaled as moved and writes
        the same ``.metadata``.

        Args:
            parts: The complete part stores, one per unit.
            final_dir: The merged store directory (created if missing).
            journal: Merge journal recording rename fallbacks.
            checkpoint_id: ``storage_meta.checkpoint_id`` of the merged store.
            save_id: ``storage_meta.save_id`` of the merged store.

        Raises:
            IntegrityError: A part is unreadable or inconsistent, or the union
                does not match the plan (missing, unexpected or duplicate keys,
                shape, dtype, gaps, overlaps, chunks other than planned, short
                data files).
            RefusedError: A hard link crosses file systems (FC-E02).
        """
        final_dir.mkdir(parents=True, exist_ok=True)
        moved = _journaled_moves(journal)
        tensors: dict[str, _TensorUnion] = {}
        byte_items: dict[str, _Stored] = {}
        planner_data: dict[str, Any] = {}
        version = None
        files = renamed = 0
        for position, part in enumerate(sorted(parts, key=lambda p: p.uid)):
            metadata = _read_part(part)
            if position == 0:
                version = torch_compat.metadata_version(metadata)
            records = torch_compat.storage_records(metadata)
            for relative_path in sorted({r.location.relative_path for r in records}):
                target = final_dir / _final_name(part.uid, relative_path)
                renamed += _place(part.path / relative_path, target, journal, moved)
                files += 1
                inject_fault(f"merge:link:{files}")
            _add_part(part, metadata, records, tensors, byte_items)
            _add_planner_data(part, metadata, planner_data)
        self._validate(tensors, byte_items)
        stored = [item for union in tensors.values() for _, item in union.chunks]
        _check_file_sizes(final_dir, [*stored, *byte_items.values()])
        state_dict_metadata, storage_data = _merged_tables(tensors, byte_items)
        merged = torch_compat.build_metadata(
            state_dict_metadata,
            storage_data,
            planner_data or None,
            checkpoint_id=checkpoint_id,
            save_id=save_id,
            version=version,
        )
        write_atomic(final_dir / METADATA_FILE, pickle.dumps(merged))
        return MergeSummary(
            files=files,
            renamed=renamed,
            tensors=len(tensors),
            chunks=sum(len(t.chunks) for t in tensors.values()),
            byte_items=len(byte_items),
        )

    def _validate(self, tensors: Mapping[str, _TensorUnion], byte_items: Mapping[str, Any]) -> None:
        _check_key_set("tensors", tensors.keys(), self._expected.keys())
        _check_key_set("byte items", byte_items.keys(), self._expected_bytes)
        for key, expected in self._expected.items():
            union = tensors[key]
            shape = tuple(int(s) for s in union.meta.size)
            if shape != expected.global_shape:
                raise IntegrityError(
                    f"merged {key} has shape {shape}, the plan says {expected.global_shape}"
                )
            if union.meta.properties.dtype != expected.dtype:
                raise IntegrityError(
                    f"merged {key} is {union.meta.properties.dtype}, the plan says {expected.dtype}"
                )
            regions = [_region(chunk) for chunk, _ in union.chunks]
            defect = find_tiling_defect(regions, shape)
            if defect is not None:
                raise IntegrityError(f"merged chunk grid of {key} is invalid: {defect}")
            if set(regions) != set(expected.chunks):
                unplanned = sorted(set(regions) - set(expected.chunks), key=lambda r: r.offsets)
                raise IntegrityError(
                    f"merged chunks of {key} differ from the plan: {len(regions)} chunks, "
                    f"first unplanned {unplanned[0] if unplanned else None}"
                )


def post_check(final_dir: Path, digests: Sequence[ItemDigest]) -> None:
    """Re-read a merged store and compare the given pieces and byte items with their digests.

    The installed torch must parse the new ``.metadata`` and the safe metadata
    view must accept it (its chunk grids tile every tensor); each listed piece
    is then read back through DCP one at a time and hashed.

    Raises:
        IntegrityError: The store cannot be read, or an item's bytes differ from
            what was written.
    """
    try:
        FileSystemReader(final_dir).read_metadata()
    except Exception as err:  # any failure means the running torch cannot open the store
        raise IntegrityError(f"torch cannot read {final_dir}/{METADATA_FILE}: {err}") from err
    view = MetadataView.load(final_dir)
    tensor_digests = [d for d in digests if d.offsets is not None]
    byte_digests = [d for d in digests if d.offsets is None]
    chunks = {(c.key, c.offsets): c for d in tensor_digests for c in view.tensors[d.key].chunks}
    largest = max((c.nbytes for c in chunks.values()), default=1)
    reader = SourceReader(view, max_item_bytes=max(largest, 1))
    differing = []
    for digest in tensor_digests:
        chunk = chunks.get((digest.key, digest.offsets))
        if chunk is None:
            raise IntegrityError(f"post-check: {digest.key} has no chunk at {digest.offsets}")
        buffer = torch.empty(chunk.sizes, dtype=chunk.dtype)
        reader.read_into([ReadReq(digest.key, chunk.region, buffer)])
        if hashlib.sha256(byte_view(buffer)).hexdigest() != digest.sha256:
            differing.append(f"{digest.key}@{digest.offsets}")
    payloads = reader.read_bytes([d.key for d in byte_digests])
    for digest in byte_digests:
        if hashlib.sha256(payloads[digest.key]).hexdigest() != digest.sha256:
            differing.append(digest.key)
    if differing:
        raise IntegrityError(
            f"post-check: {len(differing)} items of {final_dir} differ from what was written: "
            + ", ".join(differing[:_MAX_LISTED])
        )


def _read_part(part: PartStore) -> Metadata:
    loaded = read_metadata_file(part.path)
    if loaded.stubbed:
        raise IntegrityError(f"part store {part.path} references {sorted(loaded.stubbed)}")
    return loaded.metadata


def _journaled_moves(journal: Path) -> set[tuple[str, str]]:
    if not journal.is_file():
        return set()
    moves = set()
    for line in journal.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            moves.add((entry["src"], entry["dst"]))
    return moves


def _place(src: Path, dst: Path, journal: Path, moved: set[tuple[str, str]]) -> bool:
    """Hard-link ``src`` to ``dst`` (or rename it); return whether it was renamed."""
    if not src.exists():
        if (str(src), str(dst)) in moved and dst.is_file():
            return True
        raise IntegrityError(f"part file {src} is missing")
    try:
        os.link(src, dst)
        return False
    except FileExistsError as err:
        if os.path.samefile(src, dst):
            return False
        raise IntegrityError(f"{dst} exists and is not a link to {src}") from err
    except OSError as err:
        if err.errno == errno.EXDEV:
            raise Refusal(
                errors.FC_E02,
                f"cannot hard-link {src} to {dst}: they are on different file systems",
                hint="put --staging-dir on the destination's file system",
            ).to_error() from err
        if err.errno not in _NO_HARD_LINKS:
            raise IntegrityError(f"cannot link {src} to {dst}: {err.strerror or err}") from err
    if dst.exists():
        raise IntegrityError(f"{dst} exists and is not {src}")
    append_line(journal, json.dumps({"src": str(src), "dst": str(dst)}))
    os.rename(src, dst)
    moved.add((str(src), str(dst)))
    return True


def _add_part(
    part: PartStore,
    metadata: Metadata,
    records: Sequence[torch_compat.StorageRecord],
    tensors: dict[str, _TensorUnion],
    byte_items: dict[str, _Stored],
) -> None:
    located = {(r.key, r.offsets): r for r in records}
    for key, item in metadata.state_dict_metadata.items():
        if key in byte_items:
            raise IntegrityError(
                f"{key} is stored by unit {byte_items[key].uid} and unit {part.uid}"
            )
        if isinstance(item, BytesStorageMetadata):
            if key in tensors:
                raise IntegrityError(
                    f"{key} is a tensor in another unit and a byte item in {part.uid}"
                )
            byte_items[key] = _stored(part, located, key, None)
        elif isinstance(item, TensorStorageMetadata):
            union = tensors.setdefault(key, _TensorUnion(item))
            if union.meta.size != item.size or union.meta.properties != item.properties:
                raise IntegrityError(
                    f"parts disagree on {key}: {tuple(union.meta.size)} "
                    f"{union.meta.properties.dtype} and {tuple(item.size)} "
                    f"{item.properties.dtype} (unit {part.uid})"
                )
            for chunk in item.chunks:
                stored = _stored(part, located, key, tuple(int(o) for o in chunk.offsets))
                union.chunks.append((chunk, stored))
        else:
            raise IntegrityError(f"part {part.path}: {key} has unknown storage type {item!r}")
    undeclared = {r.key for r in records} - metadata.state_dict_metadata.keys()
    if undeclared:
        raise IntegrityError(f"part {part.path} stores undeclared items {sorted(undeclared)}")


def _final_name(uid: int, relative_path: str) -> str:
    """Name of a part's data file in the merged store (unique across units)."""
    return f"u{uid:05d}_{relative_path}"


def _stored(
    part: PartStore,
    located: Mapping[tuple[str, tuple[int, ...] | None], torch_compat.StorageRecord],
    key: str,
    offsets: tuple[int, ...] | None,
) -> _Stored:
    record = located.get((key, offsets))
    if record is None:
        raise IntegrityError(f"part {part.path}: {key} {offsets or ''} has no storage entry")
    return _Stored(record, _final_name(part.uid, record.location.relative_path), part.uid)


def _add_planner_data(part: PartStore, metadata: Metadata, planner_data: dict[str, Any]) -> None:
    for key, path in (metadata.planner_data or {}).items():
        known = planner_data.setdefault(key, path)
        if known != path:
            raise IntegrityError(f"parts disagree on the planner path of {key} (unit {part.uid})")


def _check_key_set(what: str, found: Any, planned: Any) -> None:
    missing = sorted(set(planned) - set(found))
    unexpected = sorted(set(found) - set(planned))
    if missing or unexpected:
        raise IntegrityError(
            f"merged {what} differ from the plan: missing {missing[:_MAX_LISTED]}, "
            f"unexpected {unexpected[:_MAX_LISTED]}"
        )


def _region(chunk: Any) -> Region:
    return Region(tuple(int(o) for o in chunk.offsets), tuple(int(s) for s in chunk.sizes))


def _merged_tables(
    tensors: Mapping[str, _TensorUnion], byte_items: Mapping[str, _Stored]
) -> tuple[dict[str, Any], dict[Any, Any]]:
    state_dict_metadata: dict[str, Any] = {}
    storage_data: dict[Any, Any] = {}
    for key in sorted(tensors.keys() | byte_items.keys()):
        if key in byte_items:
            state_dict_metadata[key] = BytesStorageMetadata()
            index, info = torch_compat.relocated_record(
                byte_items[key].record, byte_items[key].name, None
            )
            storage_data[index] = info
            continue
        union = tensors[key]
        ordered = sorted(union.chunks, key=lambda item: _region(item[0]).offsets)
        state_dict_metadata[key] = TensorStorageMetadata(
            properties=union.meta.properties,
            size=union.meta.size,
            chunks=[chunk for chunk, _ in ordered],
        )
        for position, (_, stored) in enumerate(ordered):
            index, info = torch_compat.relocated_record(stored.record, stored.name, position)
            storage_data[index] = info
    return state_dict_metadata, storage_data


def _check_file_sizes(final_dir: Path, items: Sequence[_Stored]) -> None:
    ends: dict[str, int] = {}
    for item in items:
        end = item.record.location.offset + item.record.location.length
        ends[item.name] = max(ends.get(item.name, 0), end)
    for name, end in sorted(ends.items()):
        path = final_dir / name
        size = path.stat().st_size if path.is_file() else 0
        if size < end:
            raise IntegrityError(f"{path} holds {size} bytes, its items need {end}")
