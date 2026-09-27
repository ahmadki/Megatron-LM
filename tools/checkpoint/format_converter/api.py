# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Public Python API of format_converter."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core.manifest import MANIFEST_FILE, Manifest
from .engine.fs import write_atomic
from .engine.hashing import TensorDigest, compute_digests
from .engine.metadata import METADATA_FILE, MetadataView
from .engine.reader import ReadStats, SourceReader
from .engine.regions import Region
from .engine.runner import RunOptions, run_program
from .engine.sizes import format_bytes
from .errors import IntegrityError, UsageError
from .formats.common_summary import CommonSummary, summarize_common_state
from .formats.detect import Detection, Store, detect_format, open_store, publish_target
from .formats.sections import Census, Section, classify_key, take_census
from .options import ConvertOptions, InspectOptions, PlanOptions, VerifyOptions
from .planning import ConversionPlan, build_plan
from .report import ConvertReport, peak_rss
from .sidecars import INTEGRITY_MANIFEST, copy_sidecars, write_integrity_manifest
from .verify.compare import compare_stores
from .verify.oracle import verify_against_oracle
from .verify.replay import replay_conversion
from .verify.report import VerifyReport

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


def verify_checkpoint(a: Path, b: Path | None, opts: VerifyOptions) -> VerifyReport:
    """Verify store ``a`` bitwise against store ``b`` or against digest files.

    Args:
        a: The store to verify (a store directory or a checkpoint root).
        b: The store to compare with; ``None`` when ``opts.oracle`` is set.
        opts: What to compare.

    With ``opts.map == "convert"``, ``b`` is compared with the replayed
    conversion of ``a`` (see ``verify/replay.py``; not an independent check).

    Raises:
        UsageError: Neither or both of ``b`` and ``opts.oracle`` are given.
        ConverterError: A store cannot be resolved or is corrupt, an oracle file
            is malformed, a read unit does not fit the memory budget, or the
            conversion to replay is refused.
    """
    if b is None and opts.oracle is None:
        raise UsageError(
            "verify needs a second store B or --oracle GLOB",
            hint="pass B to compare two stores, or --oracle 'digests*.json' to check digests",
        )
    if b is not None and opts.oracle is not None:
        raise UsageError(
            "verify takes either a second store B or --oracle GLOB, not both",
            hint="drop B to check A against the digests, or drop --oracle",
        )
    _log_budget(opts.max_item_bytes, opts.workers)
    store_a = open_store(a, iteration=opts.iteration)
    if b is None:
        report = verify_against_oracle(store_a, opts)
    elif opts.map == "convert":
        report = _replay(store_a, open_store(b, iteration=opts.iteration), opts)
    else:
        report = compare_stores(store_a, open_store(b, iteration=opts.iteration), opts)
    _log_reads(f"verify ({len(report.findings)} findings)", report.read_stats)
    return report


def _replay(source: Store, converted: Store, opts: VerifyOptions) -> VerifyReport:
    """``converted`` against the dry-run plan of converting ``source`` as it was converted."""
    plan = build_plan(source.iter_dir, _replay_options(converted), dry_run=True)
    pieces = [piece for unit in plan.units for piece in unit.pieces]
    byte_keys = frozenset(item.key for unit in plan.units for item in unit.byte_pieces)
    return replay_conversion(source, converted, pieces, byte_keys, opts)


def _pairs(mapping: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(mapping.items())


_REPLAYED_OPTIONS = {
    "drop_common_keys": frozenset,
    "unit_bytes": int,
    "optimizer": str,
    "allow_roots": tuple,
    "max_output_chunk_bytes": int,
    "allow_missing_fp8_extra_state": bool,
    "ignore_manifest": bool,
    "block_layouts": _pairs,
}
"""Options a conversion manifest records that shape the replayed plan, and their parsers."""


def _replay_options(converted: Store) -> PlanOptions:
    """The planning options ``converted`` was written with (from its manifest, if it has one).

    Raises:
        IntegrityError: The manifest is unreadable or malformed.
    """
    fmt = detect_format(converted).fmt
    path = converted.iter_dir / MANIFEST_FILE
    if not path.is_file():
        return PlanOptions(to_format=fmt, workers=0)
    try:
        manifest = Manifest.from_json(json.loads(path.read_text()), label=str(path))
    except (OSError, ValueError) as err:
        raise IntegrityError(f"cannot read {path}: {err}") from err
    recorded = {
        name: kind(manifest.options[name])
        for name, kind in _REPLAYED_OPTIONS.items()
        if name in manifest.options
    }
    return PlanOptions(to_format=fmt, workers=0, **recorded)


def plan_conversion(src: Path, opts: PlanOptions) -> ConversionPlan:
    """Dry-run a conversion of ``src`` from its metadata and common state (nothing is written).

    Every refusal ``convert`` would raise is raised here. Byte items are sized
    from the metadata rather than read. On a host without Megatron, a common
    state that references Megatron classes is decoded with stubs (codec
    ``stub``): the plan is complete, but ``convert`` needs the Megatron
    container. The plan cannot be executed.

    Args:
        src: A store directory or a checkpoint root.
        opts: Target format and budgets.

    Raises:
        ConverterError: The source is refused, unreadable, or does not fit the
            memory budgets.
    """
    return _plan(src, opts, dry_run=True)


def _plan(src: Path, opts: PlanOptions, *, dry_run: bool) -> ConversionPlan:
    plan = build_plan(Path(src), opts, dry_run=dry_run)
    _LOGGER.info(
        "planned %d units, about %s of host memory",
        len(plan.units),
        format_bytes(plan.memory.total),
    )
    return plan


def convert_checkpoint(src: Path, dst: Path, opts: ConvertOptions) -> ConvertReport:
    """Convert ``src`` into a new store under ``dst`` (a checkpoint root or an iteration directory).

    Args:
        src: A store directory or a checkpoint root.
        dst: Checkpoint root (gets ``iter_%07d`` and the tracker) or, if its
            name is ``iter_<N>``/``release`` or ``opts.dst_is_iter_dir``, the
            iteration directory itself.
        opts: Planning, staging and publishing options.

    Raises:
        ConverterError: The source is refused or unreadable, a budget is
            exceeded, the target overlaps the source or the staging directory,
            the target exists (without ``overwrite``), staging belongs to
            another conversion, or writing or checking the output fails.
    """
    started = time.time()
    plan = _plan(src, opts, dry_run=False)
    target = publish_target(
        Path(dst), plan.iteration, iter_dir_mode=opts.dst_is_iter_dir, tracker=opts.tracker
    )

    def finalize(final_dir: Path) -> None:
        for name, payload in plan.adapter.marker_files().items():
            write_atomic(final_dir / name, payload)
        if opts.copy_sidecars:
            copy_sidecars(plan.sidecars, plan.store.iter_dir, final_dir)
        if plan.manifest is not None:
            digest = hashlib.sha256((final_dir / METADATA_FILE).read_bytes()).hexdigest()
            record = plan.manifest.for_output(digest).to_json()
            # Unsorted: the recorded content_metadata keeps the source's key order.
            write_atomic(final_dir / MANIFEST_FILE, json.dumps(record, indent=2).encode())
        plan.adapter.validate_published(final_dir)
        if INTEGRITY_MANIFEST in plan.sidecars.regenerated:
            write_integrity_manifest(final_dir)  # last: it hashes every other file

    run = run_program(
        plan.program(target),
        target,
        RunOptions(
            workers=opts.workers,
            resume=opts.resume,
            restart=opts.restart,
            keep_staging=opts.keep_staging,
            overwrite=opts.overwrite,
            staging_dir=opts.staging_dir,
        ),
        finalize=finalize,
    )
    return ConvertReport(plan.to_json(), run, time.time() - started, peak_rss())


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
