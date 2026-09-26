# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Running a write program end to end: stage, execute units, merge, check, publish.

A :class:`WriteProgram` is a fully planned output store (units of pieces and
byte items over one source store). :func:`run_program` executes it through a
staging directory next to the target, so an interrupted run can be resumed
with the same fingerprint, and publishes the merged store atomically. The
format-specific parts (marker files, sidecars) are added by a ``finalize``
callback before the merged store is checked and published.

Before anything is created, the target, the source store and the staging
directory must be disjoint (none inside another), an existing target is
replaced (``overwrite``) only if it is a single store directory, and the
target's file system must have room for the output (exit 8 otherwise).
"""

from __future__ import annotations

import functools
import random
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import InternalError, UsageError
from .memory import check_free_disk, estimate_output_bytes
from .merge import MergeSummary, MetadataMerger, post_check
from .metadata import MetadataView
from .pieces import ExpectedTensor, Unit, expected_tensors
from .staging import (
    PublishTarget,
    StagedUnit,
    Staging,
    StagingPlan,
    TrackerUpdate,
    check_replaceable,
    ensure_same_file_system,
    target_exists_error,
)
from .workers import run_units

DEFAULT_CHECKED_PIECES = 16
"""Pieces re-read and compared with their digests after the merge."""


@dataclass(frozen=True)
class WriteProgram:
    """A planned output store.

    Attributes:
        source: Metadata view of the source store.
        units: The units (uids ``0..n-1``).
        dispatch_order: Unit ids in the order workers claim them.
        checked: ``(key, offsets)`` of the pieces re-read after the merge.
        writer_threads: Writer threads (data files) per part store.
        max_item_bytes: Per-item read budget of the workers.
        fingerprint: Identity of the conversion; staging is resumed only by a
            run with the same fingerprint.
        info: Human-readable description recorded in ``PLAN.json``.
    """

    source: MetadataView
    units: tuple[Unit, ...]
    dispatch_order: tuple[int, ...]
    checked: frozenset[tuple[str, tuple[int, ...]]]
    writer_threads: int
    max_item_bytes: int
    fingerprint: str
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unread = [p.key for u in self.units for p in u.byte_pieces if p.payload is None]
        if unread:
            raise InternalError(
                f"byte items {unread[:4]} have no payload (a dry-run plan cannot be executed)"
            )

    @functools.cached_property
    def expected(self) -> dict[str, ExpectedTensor]:
        """The output tensors (validated to tile exactly)."""
        return expected_tensors(piece for unit in self.units for piece in unit.pieces)

    @property
    def expected_bytes(self) -> frozenset[str]:
        """The output byte-item keys."""
        return frozenset(item.key for unit in self.units for item in unit.byte_pieces)

    def staging_plan(self) -> StagingPlan:
        """The ``PLAN.json`` record of this program."""
        return StagingPlan(
            fingerprint=self.fingerprint,
            source_dir=self.source.iter_dir.absolute(),
            writer_threads=self.writer_threads,
            max_item_bytes=self.max_item_bytes,
            dispatch_order=self.dispatch_order,
            info=self.info,
        )

    def staged_units(self) -> list[StagedUnit]:
        """Every unit with the source metadata tables it reads."""
        return [
            StagedUnit(
                unit,
                self.source.subset(unit.source_keys, tables_only=True),
                tuple(
                    index
                    for index, piece in enumerate(unit.pieces)
                    if (piece.key, piece.chunk.offsets) in self.checked
                ),
            )
            for unit in self.units
        ]


@dataclass(frozen=True)
class RunOptions:
    """How to run a program.

    Attributes:
        workers: Worker processes executing units; 0 runs them in this process.
        resume: ``auto`` continues existing staging of the same conversion;
            ``never`` refuses to run when staging exists.
        restart: Discard existing staging (of any conversion) first.
        keep_staging: Keep the staging directory (with its part stores) after
            publishing.
        overwrite: Replace an existing target iteration directory.
        staging_dir: Directory in which the staging directory
            ``.fc-staging-<name>`` is created (default: the target's parent).
            Only that subdirectory is ever written or deleted.
    """

    workers: int = 0
    resume: str = "auto"
    restart: bool = False
    keep_staging: bool = False
    overwrite: bool = False
    staging_dir: Path | None = None


@dataclass(frozen=True)
class RunResult:
    """What a run did.

    Attributes:
        target: Where the store was published.
        staging_dir: The staging directory used.
        resumed: Whether existing staging was continued.
        units_run: Units executed by this run.
        merge: The merge summary, or ``None`` if a resumed run found the store
            already merged and in place.
        journal: The journal record of every unit (this and earlier runs).
        tracker: What publishing did to the tracker (``None`` without one).
    """

    target: PublishTarget
    staging_dir: Path
    resumed: bool
    units_run: int
    merge: MergeSummary | None
    journal: tuple[dict[str, Any], ...]
    tracker: TrackerUpdate | None = None


def run_program(
    program: WriteProgram,
    target: PublishTarget,
    options: RunOptions,
    *,
    finalize: Callable[[Path], None] | None = None,
) -> RunResult:
    """Execute ``program`` and publish its store at ``target``.

    Args:
        program: The planned store.
        target: Where to publish it.
        options: Resume, restart, overwrite and staging settings.
        finalize: Called with the merged store directory before it is checked
            and published (adds marker files and sidecars).

    Raises:
        UsageError: The target, the source store and the staging directory
            overlap, or ``overwrite`` would replace something other than a
            single store directory.
        IntegrityError: The target exists without ``overwrite``, staging is
            locked or exists with ``resume='never'``, or any read, write, merge
            or post-check fails.
        FingerprintMismatch: Staging of another conversion exists (exit 7).
        RefusedError: Staging is not on the target's file system (FC-E02).
        BudgetExceeded: The target's file system has less than 1.02 x the
            estimated output free (exit 8).
    """
    staging = Staging(target.default_staging_dir(options.staging_dir))
    check_disjoint(program.source.iter_dir, target, staging.root)
    if target.iter_dir.exists() and not staging.published():  # before anything is created
        if not options.overwrite:
            raise target_exists_error(target)
        check_replaceable(target.iter_dir)
    check_free_disk(_existing_ancestor(target.iter_dir), estimate_output_bytes(program.units))
    target.iter_dir.parent.mkdir(parents=True, exist_ok=True)
    with staging.locked():
        ensure_same_file_system(staging.root, target.iter_dir.parent)
        resumed = staging.prepare(
            program.fingerprint, resume=options.resume, restart=options.restart
        )
        units_run, merge = 0, None
        if not staging.published():
            if target.iter_dir.exists() and not options.overwrite:
                raise target_exists_error(target)
            if not resumed:
                staging.stage(program.staging_plan(), program.staged_units())
            units_run = run_units(staging, workers=options.workers)
            merge = MetadataMerger(program.expected, program.expected_bytes).merge(
                staging.part_stores(),
                staging.final_dir,
                journal=staging.merge_journal,
                checkpoint_id=str(target.iter_dir),
                save_id=save_id(program.fingerprint),
            )
            if finalize is not None:
                finalize(staging.final_dir)
            post_check(staging.final_dir, staging.digests())
        journal = tuple(staging.journal_records())
        tracker = staging.publish(target, overwrite=options.overwrite)
        if not options.keep_staging:
            staging.remove()
    return RunResult(target, staging.root, resumed, units_run, merge, journal, tracker)


def check_disjoint(source_dir: Path, target: PublishTarget, staging_dir: Path) -> None:
    """Refuse a target or staging directory that is, contains or lies inside another.

    The source store is never written, the target is replaced as a whole, and
    the staging directory is deleted as a whole, so none of the three may hold
    another.

    Raises:
        UsageError: Two of them overlap.
    """
    source, output, staging = (
        path.resolve() for path in (source_dir, target.iter_dir, staging_dir)
    )
    if output == source:
        raise UsageError(f"DST {target.iter_dir} is the source store", hint="pick another DST")
    if _overlap(output, source):
        relation = "lies inside" if source in output.parents else "contains"
        raise UsageError(
            f"the target {target.iter_dir} {relation} the source store {source_dir}",
            hint="pick a DST outside the source store (a checkpoint root gets iter_NNNNNNN)",
        )
    for other, what in ((source, "the source store"), (output, "the target")):
        if staging == other or _overlap(staging, other):
            raise UsageError(
                f"the staging directory {staging_dir} overlaps {what} {other}",
                hint="pass a --staging-dir outside the source and the target",
            )


def _overlap(a: Path, b: Path) -> bool:
    return a in b.parents or b in a.parents


def _existing_ancestor(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


def save_id(fingerprint: str) -> str:
    """The ``storage_meta.save_id`` of a merged store: derived from the fingerprint.

    Runs of the same conversion (any worker count, resumed or not) therefore
    write byte-identical ``.metadata`` files.
    """
    return str(uuid.UUID(fingerprint[:32]))


def choose_checked(
    units: Iterable[Unit], *, seed: str, count: int = DEFAULT_CHECKED_PIECES
) -> frozenset[tuple[str, tuple[int, ...]]]:
    """A deterministic sample of ``count`` pieces (key, offsets) to re-read after the merge."""
    slots = sorted((piece.key, piece.chunk.offsets) for unit in units for piece in unit.pieces)
    return frozenset(random.Random(seed).sample(slots, min(count, len(slots))))
