# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The staging directory of a conversion: locking, resumable units and atomic publish.

It lives next to the published store, or in the directory given as
``--staging-dir``, and always on the target's file system (so part files can be
hard-linked and the merged store renamed into place). The tool owns only the
``.fc-staging-<name>`` directory it creates, never the directory it is created
in::

    .fc-staging-<name>/
      LOCK              flock held by the driver for the whole run
      RUN               id of the current run, recorded in claims
      PLAN.json         fingerprint, source, writer settings, dispatch order (written last)
      units/uNNNNN.pkl  one unit each: pieces, byte items, subset metadata
      claims/uNNNNN     O_EXCL claim of a unit by a worker
      parts/uNNNNN.tmp/ a part store being written
      parts/uNNNNN/     a complete part store with its DONE record (the rename commits it)
      journal.jsonl     one line per completed unit
      merge.journal     rename fallbacks of the merge
      publish.json      publish in progress: target and trash names
      final/            the merged store, renamed into place on publish

**Resume.** A new run takes the lock. If ``PLAN.json`` holds the same
fingerprint, complete parts are kept and every ``*.tmp`` part and every claim is
deleted: while the lock is held no other process of this conversion runs (worker
processes inherit the locked descriptor, so the lock outlives a killed driver
until its workers exit), so all claims are stale (process ids are not compared;
containers have separate pid namespaces). A different fingerprint exits 7
unless the run restarts.

**Publish.** The merged store is renamed onto the target iteration directory,
then the tracker (if any) is replaced atomically, last, and only if that moves
it forward (:func:`tracker_moves_forward`). With ``overwrite`` an existing
target is first renamed into a trash directory beside it and deleted after
success; only a single store directory is ever replaced
(:func:`check_replaceable`), and nothing else under the target's parent is
touched. ``publish.json`` makes each step restartable.

Unit files are pickles of engine data classes, torch dtypes, bytes and DCP
metadata tables only, and are loaded with an allowlist unpickler.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import pickle
import re
import shutil
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch.distributed.checkpoint.metadata import Metadata

from .. import errors
from ..errors import FingerprintMismatch, IntegrityError, InternalError, Refusal, UsageError
from .faults import inject_fault
from .fs import append_line, fsync_dir, write_atomic
from .merge import ItemDigest, PartStore
from .metadata import METADATA_FILE, SafeMetadataUnpickler, StoredChunk
from .pieces import BytePiece, CopyOp, Piece, Unit
from .regions import Region

STAGING_PREFIX = ".fc-staging-"
TRASH_PREFIX = ".fc-trash-"
RESUME_MODES = ("auto", "never")

_LOCK = "LOCK"
_RUN = "RUN"
_PLAN = "PLAN.json"
_UNITS = "units"
_CLAIMS = "claims"
_PARTS = "parts"
_DONE = "DONE"
_JOURNAL = "journal.jsonl"
_MERGE_JOURNAL = "merge.journal"
_PUBLISH = "publish.json"
_FINAL = "final"
_ITERATION = re.compile(r"^\d+$")
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PublishTarget:
    """Where a converted store is published.

    Attributes:
        iter_dir: The store directory to create.
        tracker: Tracker file to replace after the store is in place, if any.
        tracker_value: The tracker's content (an iteration number or ``release``).
    """

    iter_dir: Path
    tracker: Path | None = None
    tracker_value: str | None = None

    def default_staging_dir(self, parent: Path | None = None) -> Path:
        """``<parent>/.fc-staging-<name>``, an absolute path.

        Args:
            parent: The directory to create it in (default: the target's
                parent, i.e. beside the target). The tool owns only the
                ``.fc-staging-*`` directory, never ``parent``.
        """
        base = self.iter_dir.parent if parent is None else Path(parent)
        return base.absolute() / f"{STAGING_PREFIX}{self.iter_dir.name}"


@dataclass(frozen=True, slots=True)
class TrackerUpdate:
    """What publishing did to the tracker file.

    Attributes:
        path: The tracker file.
        before: Its content before publishing (``None`` if it did not exist).
        after: Its content afterwards.
    """

    path: Path
    before: str | None
    after: str | None

    @property
    def written(self) -> bool:
        """Whether the tracker now names the published iteration because of this run."""
        return self.after != self.before

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {"path": str(self.path), "before": self.before, "after": self.after}


@dataclass(frozen=True)
class StagedUnit:
    """What a worker needs to execute one unit.

    Attributes:
        unit: The unit.
        subset: The source metadata tables of the keys the unit reads.
        checked: Indices of the pieces whose buffers are digested for the
            post-merge check.
    """

    unit: Unit
    subset: Metadata
    checked: tuple[int, ...] = ()


@dataclass(frozen=True)
class StagingPlan:
    """The contents of ``PLAN.json``.

    Attributes:
        fingerprint: Identity of the conversion (source, options, versions).
        source_dir: The source store directory.
        writer_threads: Writer threads per part store.
        max_item_bytes: Per-item read budget of the workers.
        dispatch_order: Unit ids in the order workers claim them.
        info: Human-readable description (options, versions, estimates).
    """

    fingerprint: str
    source_dir: Path
    writer_threads: int
    max_item_bytes: int
    dispatch_order: tuple[int, ...]
    info: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {
            "fingerprint": self.fingerprint,
            "source_dir": str(self.source_dir),
            "writer_threads": self.writer_threads,
            "max_item_bytes": self.max_item_bytes,
            "dispatch_order": list(self.dispatch_order),
            "info": self.info,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> StagingPlan:
        """Parse the JSON form."""
        return cls(
            fingerprint=data["fingerprint"],
            source_dir=Path(data["source_dir"]),
            writer_threads=int(data["writer_threads"]),
            max_item_bytes=int(data["max_item_bytes"]),
            dispatch_order=tuple(int(u) for u in data["dispatch_order"]),
            info=dict(data.get("info", {})),
        )


_UNIT_CLASSES = frozenset(
    (cls.__module__, cls.__qualname__)
    for cls in (Region, StoredChunk, CopyOp, Piece, BytePiece, Unit, StagedUnit)
)


class _UnitUnpickler(SafeMetadataUnpickler):
    """Metadata unpickler that also allows the engine's own unit data classes."""

    def find_class(self, module: str, name: str) -> Any:
        """Resolve the unit data classes; defer everything else to the metadata allowlist."""
        if (module, name) in _UNIT_CLASSES:
            return super(SafeMetadataUnpickler, self).find_class(module, name)
        return super().find_class(module, name)


class Staging:
    """One staging directory.

    Args:
        root: The staging directory (created by :meth:`locked`). Its name must
            start with ``.fc-staging-``: only such directories are ever wiped
            or deleted.

    Raises:
        InternalError: ``root`` is not named like a staging directory.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).absolute()
        if not self.root.name.startswith(STAGING_PREFIX):
            raise InternalError(
                f"refusing to use {self.root} as a staging directory: its name does not start "
                f"with {STAGING_PREFIX}"
            )
        self._plan: StagingPlan | None = None
        self._lock_fd: int | None = None

    @property
    def final_dir(self) -> Path:
        """The merged store before it is published."""
        return self.root / _FINAL

    @property
    def merge_journal(self) -> Path:
        """Journal of the merge's rename fallbacks."""
        return self.root / _MERGE_JOURNAL

    @property
    def journal(self) -> Path:
        """Journal of completed units (JSON lines)."""
        return self.root / _JOURNAL

    @property
    def lock_fd(self) -> int:
        """The descriptor holding the lock (inside :meth:`locked`).

        Worker processes inherit it, so the lock is held while any of them
        runs, even if the driver is killed.

        Raises:
            InternalError: The lock is not held.
        """
        if self._lock_fd is None:
            raise InternalError(f"the lock of {self.root} is not held")
        return self._lock_fd

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the staging lock (creating the directory) for the duration of the block.

        Raises:
            IntegrityError: Another conversion holds the lock.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / _LOCK, "a+b") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as err:
                raise IntegrityError(
                    f"another conversion is using {self.root} (or a worker it left running is)",
                    hint="wait for it to finish, or remove the staging directory if it died",
                ) from err
            self._lock_fd = lock.fileno()
            try:
                yield
            finally:
                self._lock_fd = None

    def prepare(self, fingerprint: str, *, resume: str, restart: bool) -> bool:
        """Get the (locked) staging directory ready for a run of ``fingerprint``.

        Returns:
            Whether existing staging of the same conversion is resumed.

        Raises:
            IntegrityError: Staging exists and ``resume`` is ``never``.
            FingerprintMismatch: Staging of a different conversion exists and
                ``restart`` is not set.
        """
        if resume not in RESUME_MODES:
            raise InternalError(f"resume must be one of {RESUME_MODES}, got {resume!r}")
        resumed = self._reuse(fingerprint, resume=resume, restart=restart)
        for name in (_UNITS, _CLAIMS, _PARTS):
            (self.root / name).mkdir(exist_ok=True)
        write_atomic(self.root / _RUN, uuid.uuid4().hex.encode())
        return resumed

    def stage(self, plan: StagingPlan, units: Sequence[StagedUnit]) -> None:
        """Write the unit files, then ``PLAN.json`` (which commits the staging)."""
        for staged in units:
            payload = pickle.dumps(staged, protocol=pickle.HIGHEST_PROTOCOL)
            write_atomic(self._unit_file(staged.unit.uid), payload)
        write_atomic(self.root / _PLAN, json.dumps(plan.to_json(), indent=1).encode())
        self._plan = plan

    def plan(self) -> StagingPlan:
        """The staged plan.

        Raises:
            IntegrityError: ``PLAN.json`` is missing or unreadable.
        """
        if self._plan is None:
            try:
                self._plan = StagingPlan.from_json(json.loads((self.root / _PLAN).read_text()))
            except (OSError, ValueError, KeyError) as err:
                raise IntegrityError(f"cannot read {self.root / _PLAN}: {err}") from err
        return self._plan

    def load_unit(self, uid: int) -> StagedUnit:
        """Unpickle one unit file with the allowlist unpickler.

        Raises:
            IntegrityError: The file is missing or not a unit.
        """
        path = self._unit_file(uid)
        try:
            with open(path, "rb") as stream:
                staged = _UnitUnpickler(stream).load()
        except Exception as err:  # a damaged staging file can fail in any way
            raise IntegrityError(f"cannot load {path}: {type(err).__name__}: {err}") from err
        if not isinstance(staged, StagedUnit) or staged.unit.uid != uid:
            raise IntegrityError(f"{path} does not hold unit {uid}")
        return staged

    def claim(self, uid: int, worker_id: int) -> bool:
        """Claim a unit for a worker; ``False`` if it is claimed already."""
        record = {
            "run": (self.root / _RUN).read_text(),
            "worker": worker_id,
            "pid": os.getpid(),
            "t0": time.time(),
        }
        try:
            descriptor = os.open(
                self.root / _CLAIMS / f"u{uid:05d}", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
            )
        except FileExistsError:
            return False
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream)
        return True

    def is_complete(self, uid: int) -> bool:
        """Whether unit ``uid`` has a committed part store."""
        return (self._part_dir(uid) / _DONE).is_file()

    def fresh_part_dir(self, uid: int) -> Path:
        """The (not yet existing) directory to write unit ``uid``'s part store into."""
        tmp = self._part_dir(uid).with_suffix(".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        return tmp

    def complete(self, uid: int, digests: Sequence[ItemDigest], record: dict[str, Any]) -> None:
        """Commit unit ``uid``: record its digests, rename its part store, journal it."""
        tmp = self._part_dir(uid).with_suffix(".tmp")
        done = {
            "fingerprint": self.plan().fingerprint,
            "uid": uid,
            "digests": [_digest_record(digest) for digest in digests],
        }
        write_atomic(tmp / _DONE, json.dumps(done).encode())
        os.rename(tmp, self._part_dir(uid))
        fsync_dir(self.root / _PARTS)
        append_line(self.journal, json.dumps(record, sort_keys=True))

    def part_stores(self) -> list[PartStore]:
        """The complete part store of every planned unit.

        Raises:
            IntegrityError: A unit has no complete part store.
        """
        uids = sorted(self.plan().dispatch_order)
        missing = [uid for uid in uids if not self.is_complete(uid)]
        if missing:
            raise IntegrityError(f"units {missing[:8]} of {self.root} have not completed")
        return [PartStore(uid, self._part_dir(uid)) for uid in uids]

    def digests(self) -> list[ItemDigest]:
        """The digests recorded by every complete unit, in uid order."""
        fingerprint = self.plan().fingerprint
        digests = []
        for part in self.part_stores():
            done = json.loads((part.path / _DONE).read_text())
            if done["fingerprint"] != fingerprint or done["uid"] != part.uid:
                raise IntegrityError(f"{part.path / _DONE} belongs to another conversion or unit")
            for entry in done["digests"]:
                offsets = None if entry["offsets"] is None else tuple(entry["offsets"])
                digests.append(ItemDigest(entry["key"], offsets, entry["sha256"]))
        return digests

    def journal_records(self) -> list[dict[str, Any]]:
        """Every unit record in the journal (of this and earlier runs)."""
        if not self.journal.is_file():
            return []
        lines = self.journal.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def published(self) -> bool:
        """Whether the merged store was already renamed into place (tracker may be pending)."""
        return (self.root / _PUBLISH).is_file() and not self.final_dir.exists()

    def publish(self, target: PublishTarget, *, overwrite: bool) -> TrackerUpdate | None:
        """Rename the merged store onto ``target`` and write the tracker last.

        Restartable at every step: a run resumed after a crash in here finishes
        the same publish.

        Returns:
            What happened to the tracker (``None`` if the target has none).

        Raises:
            IntegrityError: The target exists and ``overwrite`` is not set.
            UsageError: The existing target is not a single store directory.
        """
        record = self._publish_record(target)
        trash = Path(record["trash"])
        parent = target.iter_dir.parent
        if self.final_dir.exists():
            if target.iter_dir.exists():
                if not overwrite:
                    raise target_exists_error(target)
                check_replaceable(target.iter_dir)
                os.rename(target.iter_dir, trash)
                fsync_dir(parent)
            inject_fault("publish:before_rename")
            os.rename(self.final_dir, target.iter_dir)
            fsync_dir(parent)
        inject_fault("publish:before_tracker")
        update = None
        if target.tracker is not None:
            update = _update_tracker(target.tracker, target.tracker_value or "")
        if trash.exists():
            shutil.rmtree(trash)
        (self.root / _PUBLISH).unlink()
        return update

    def remove(self) -> None:
        """Delete the staging directory."""
        shutil.rmtree(self.root)

    def _reuse(self, fingerprint: str, *, resume: str, restart: bool) -> bool:
        if not any(p.name != _LOCK for p in self.root.iterdir()):
            return False
        if restart:
            self._wipe()
            return False
        if resume == "never":
            raise IntegrityError(
                f"staging directory {self.root} exists",
                hint="pass --resume auto to continue it, or --restart to discard it",
            )
        if not (self.root / _PLAN).is_file():
            self._wipe()  # a run died before staging completed; nothing to keep
            return False
        staged = self.plan().fingerprint
        if staged != fingerprint:
            raise FingerprintMismatch(
                f"{self.root} holds a different conversion (fingerprint {staged[:12]}, "
                f"this run {fingerprint[:12]})",
                hint="pass --restart to discard it, or rerun with the original source and options",
            )
        for tmp in (self.root / _PARTS).glob("u*.tmp"):
            shutil.rmtree(tmp)
        for claim in (self.root / _CLAIMS).iterdir():
            claim.unlink()
        return True

    def _wipe(self) -> None:
        self._plan = None
        for entry in self.root.iterdir():
            if entry.name == _LOCK:
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    def _publish_record(self, target: PublishTarget) -> dict[str, str]:
        path = self.root / _PUBLISH
        if path.is_file():
            record = json.loads(path.read_text())
            if record["target"] == str(target.iter_dir):
                return record
        record = {
            "target": str(target.iter_dir),
            "trash": str(target.iter_dir.parent / f"{TRASH_PREFIX}{uuid.uuid4().hex}"),
        }
        write_atomic(path, json.dumps(record).encode())
        return record

    def _unit_file(self, uid: int) -> Path:
        return self.root / _UNITS / f"u{uid:05d}.pkl"

    def _part_dir(self, uid: int) -> Path:
        return self.root / _PARTS / f"u{uid:05d}"


def target_exists_error(target: PublishTarget) -> IntegrityError:
    """The error for a target store directory that already exists."""
    return IntegrityError(
        f"{target.iter_dir} already exists",
        hint="pass --overwrite to replace that iteration directory (only it)",
    )


def check_replaceable(iter_dir: Path) -> None:
    """Refuse to replace (``--overwrite``) anything but one store directory.

    ``iter_dir`` must be a directory holding a ``.metadata`` file and no other
    store (no subdirectory with a ``.metadata``), so a checkpoint root, the
    source's parent or an unrelated directory is never moved to the trash.

    Raises:
        UsageError: ``iter_dir`` is not a single store directory.
    """
    if iter_dir.is_symlink() or not iter_dir.is_dir():
        raise UsageError(
            f"--overwrite replaces a store directory, and {iter_dir} is not a directory",
            hint="remove it yourself or pick another DST",
        )
    if not (iter_dir / METADATA_FILE).is_file():
        raise UsageError(
            f"--overwrite replaces a store directory, and {iter_dir} holds no {METADATA_FILE}",
            hint="remove it yourself if it may be deleted, or pick another DST",
        )
    nested = sorted(
        child.name
        for child in iter_dir.iterdir()
        if child.is_dir() and (child / METADATA_FILE).is_file()
    )
    if nested:
        raise UsageError(
            f"--overwrite replaces a single store directory, and {iter_dir} holds other stores "
            f"({', '.join(nested[:4])})",
            hint="pass the checkpoint root as DST (without --dst-is-iter-dir)",
        )


def tracker_moves_forward(current: str | None, new: str) -> bool:
    """Whether a tracker holding ``current`` is rewritten to name ``new``.

    The tracker names the checkpoint a training job resumes from, so it only
    moves forward: it is written when absent or when both values are iteration
    numbers and ``new`` is not older. A tracker naming a newer iteration,
    ``release``, or anything unreadable is left alone.
    """
    if current is None:
        return True
    if _ITERATION.match(current) and _ITERATION.match(new):
        return int(new) >= int(current)
    return current == new


def _update_tracker(tracker: Path, value: str) -> TrackerUpdate:
    try:
        current: str | None = tracker.read_text().strip()
    except FileNotFoundError:
        current = None
    if not tracker_moves_forward(current, value):
        _LOGGER.warning(
            "%s names %s; left unchanged instead of moving it back to %s", tracker, current, value
        )
        return TrackerUpdate(tracker, current, current)
    write_atomic(tracker, value.encode())
    return TrackerUpdate(tracker, current, value)


def _digest_record(digest: ItemDigest) -> dict[str, Any]:
    offsets = None if digest.offsets is None else list(digest.offsets)
    return {"key": digest.key, "offsets": offsets, "sha256": digest.sha256}


def ensure_same_file_system(staging_dir: Path, target_parent: Path) -> None:
    """Refuse (FC-E02) a staging directory on another file system than the target's parent.

    Both directories must exist.
    """
    if os.stat(staging_dir).st_dev != os.stat(target_parent).st_dev:
        raise Refusal(
            errors.FC_E02,
            f"staging directory {staging_dir} is not on the file system of {target_parent}",
            hint="put --staging-dir on the destination's file system",
        ).to_error()
