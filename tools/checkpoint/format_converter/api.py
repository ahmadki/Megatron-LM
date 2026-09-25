# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Public Python API of format_converter."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engine.hashing import TensorDigest, compute_digests
from .engine.metadata import MetadataView
from .engine.reader import ReadStats, SourceReader
from .engine.regions import Region
from .engine.sizes import format_bytes
from .formats.common_summary import CommonSummary, summarize_common_state
from .formats.detect import Detection, detect_format, open_store
from .formats.sections import Census, Section, classify_key, take_census
from .options import InspectOptions

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class KeyListing:
    """One listed key of an ``inspect`` report.

    Attributes:
        key: The key.
        section: Its coarse section.
        shape: Tensor shape (``None`` for byte items).
        dtype: Tensor dtype as text (``None`` for byte items).
        nbytes: Decoded tensor size, or payload length for byte items.
        num_chunks: Number of stored chunks (0 for byte items).
        chunks: Stored chunk boxes, when chunk listing was requested.
    """

    key: str
    section: Section
    shape: tuple[int, ...] | None
    dtype: str | None
    nbytes: int
    num_chunks: int
    chunks: tuple[Region, ...] | None

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable form."""
        record: dict[str, Any] = {
            "key": self.key,
            "section": self.section.value,
            "shape": None if self.shape is None else list(self.shape),
            "dtype": self.dtype,
            "nbytes": self.nbytes,
            "num_chunks": self.num_chunks,
        }
        if self.chunks is not None:
            record["chunks"] = [
                {"offsets": list(c.offsets), "sizes": list(c.sizes)} for c in self.chunks
            ]
        return record


@dataclass(frozen=True)
class InspectReport:
    """Result of :func:`inspect_checkpoint`.

    Attributes:
        detection: Format, dialect, location and evidence.
        census: Key, byte, dtype and chunk counts.
        dcp_version: DCP format version recorded by the writer, if any.
        has_planner_data: Whether ``.metadata`` carries planner data.
        has_legacy_mcore_data: Whether ``.metadata`` carries old Megatron ``mcore_data``.
        metadata_stubs: Classes stubbed while unpickling ``.metadata``.
        common: Common-state summary, if the store has common state.
        keys: Listed keys (empty unless listing was requested).
        digests: Tensor digests, if requested.
        read_stats: What reading the store cost.
    """

    detection: Detection
    census: Census
    dcp_version: str | None
    has_planner_data: bool
    has_legacy_mcore_data: bool
    metadata_stubs: tuple[str, ...]
    common: CommonSummary | None
    keys: tuple[KeyListing, ...]
    digests: Mapping[str, TensorDigest] | None
    read_stats: ReadStats

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable form (digests are written to their own file)."""
        detection = self.detection
        census = self.census
        largest = census.largest_chunk
        return {
            "iter_dir": str(detection.iter_dir),
            "root": None if detection.root is None else str(detection.root),
            "iteration": detection.iteration,
            "format": detection.fmt,
            "dialect": detection.dialect,
            "evidence": list(detection.evidence),
            "dcp_version": self.dcp_version,
            "has_planner_data": self.has_planner_data,
            "has_legacy_mcore_data": self.has_legacy_mcore_data,
            "metadata_stubs": list(self.metadata_stubs),
            "tensors": census.tensors,
            "byte_items": census.byte_items,
            "tensor_bytes": census.tensor_bytes,
            "sections": {
                section.value: {"keys": totals.keys, "tensor_bytes": totals.tensor_bytes}
                for section, totals in census.sections.items()
            },
            "dtypes": census.dtypes,
            "chunks_per_tensor": {str(n): count for n, count in census.chunk_counts.items()},
            "largest_chunk": (
                None
                if largest is None
                else {
                    "key": largest.key,
                    "offsets": list(largest.offsets),
                    "sizes": list(largest.sizes),
                    "nbytes": largest.nbytes,
                    "file": largest.relative_path,
                }
            ),
            "common": None if self.common is None else self.common.to_json(),
            "keys": [listing.to_json() for listing in self.keys],
            "digested_tensors": None if self.digests is None else len(self.digests),
        }


def inspect_checkpoint(src: Path, opts: InspectOptions) -> InspectReport:
    """Summarize a checkpoint store from its metadata (and optionally hash it).

    Args:
        src: A store directory or a checkpoint root.
        opts: What to list and compute.

    Raises:
        ConverterError: The store cannot be resolved, is refused, is corrupt,
            or a digest does not fit the memory budget.
    """
    _log_budget(opts.max_item_bytes, opts.workers)
    store = open_store(src, iteration=opts.iteration)
    detection = detect_format(store, from_format=opts.from_format)
    view = store.view
    reader = SourceReader(view, max_item_bytes=opts.max_item_bytes)
    common = summarize_common_state(detection, reader)
    digests = None
    read_stats = ReadStats()
    if opts.digests:
        selected = [key for key in sorted(view.tensors) if opts.selects(key)]
        digests, read_stats = compute_digests(reader, selected, workers=opts.workers)
        _log_reads(f"digests of {len(digests)} tensors", read_stats)
    return InspectReport(
        detection=detection,
        census=take_census(view, detection.fmt),
        dcp_version=view.version,
        has_planner_data=view.planner_data is not None,
        has_legacy_mcore_data=view.mcore_data is not None,
        metadata_stubs=tuple(sorted(view.stubbed)),
        common=common,
        keys=_list_keys(detection.fmt, store.view, opts) if opts.lists_keys else (),
        digests=digests,
        read_stats=read_stats,
    )


def verify_checkpoint(a: Path, b: Path | None, opts: object) -> object:
    """Compare two checkpoint stores (not implemented yet)."""
    raise NotImplementedError(f"verify is not implemented yet (a={a}, b={b}, opts={opts})")


def _log_budget(max_item_bytes: int, workers: int) -> None:
    _LOGGER.info("per-item budget %s, %d workers", format_bytes(max_item_bytes), workers)


def _log_reads(what: str, stats: ReadStats) -> None:
    _LOGGER.info(
        "%s read %s in %d stored items (amplification %.2f)",
        what,
        format_bytes(stats.bytes_read),
        stats.items_touched,
        stats.amplification,
    )


def _list_keys(fmt: str, view: MetadataView, opts: InspectOptions) -> tuple[KeyListing, ...]:
    listings = []
    for key in sorted(view.tensors.keys() | view.byte_items.keys()):
        if not opts.selects(key):
            continue
        entry = view.tensors.get(key)
        if entry is None:
            section = classify_key(fmt, key, is_bytes=True)
            length = view.byte_items[key].length
            listings.append(KeyListing(key, section, None, None, length, 0, None))
            continue
        section = classify_key(fmt, key, is_bytes=False)
        chunks = tuple(c.region for c in entry.chunks) if opts.chunks else None
        listings.append(
            KeyListing(
                key, section, entry.shape, str(entry.dtype), entry.nbytes, len(entry.chunks), chunks
            )
        )
    return tuple(listings)
