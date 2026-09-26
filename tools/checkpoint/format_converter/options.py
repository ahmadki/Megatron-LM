# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validated, immutable options of the commands (inspect, verify, plan, convert)."""

from __future__ import annotations

import os
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .engine.memory import available_memory_bytes
from .errors import UsageError
from .formats.names import FORMATS, TORCH_DIST

DEFAULT_WORKERS = 4
MAX_ITEM_FRACTION_OF_AVAILABLE = 0.25
_FALLBACK_MAX_ITEM_BYTES = 4 * 2**30


def default_max_item_bytes() -> int:
    """A quarter of the available host memory (4 GiB when it cannot be read)."""
    available = available_memory_bytes()
    if available is None:
        return _FALLBACK_MAX_ITEM_BYTES
    return max(1, int(available * MAX_ITEM_FRACTION_OF_AVAILABLE))


def _check_common(iteration: int | None, max_item_bytes: int, workers: int) -> None:
    if iteration is not None and iteration < 0:
        raise UsageError(f"--iteration must be non-negative, got {iteration}")
    if max_item_bytes <= 0:
        raise UsageError(f"--max-item-bytes must be positive, got {max_item_bytes}")
    if workers < 1:
        raise UsageError(f"--workers must be at least 1, got {workers}")


@dataclass(frozen=True, slots=True)
class InspectOptions:
    """Options of ``inspect``.

    Attributes:
        iteration: Iteration to open under a checkpoint root.
        from_format: Format to assume when the store has no format markers.
        keys: List the keys matching this pattern (``re.search``).
        chunks: List keys with their stored chunk grids.
        digests: Compute SHA-256 digests of the listed (or all) tensors.
        max_item_bytes: Per-item read budget.
        workers: Threads used for digests.
    """

    iteration: int | None = None
    from_format: str | None = None
    keys: re.Pattern[str] | None = None
    chunks: bool = False
    digests: bool = False
    max_item_bytes: int = field(default_factory=default_max_item_bytes)
    workers: int = DEFAULT_WORKERS

    def __post_init__(self) -> None:
        _check_common(self.iteration, self.max_item_bytes, self.workers)
        if self.from_format is not None and self.from_format not in FORMATS:
            raise UsageError(f"--from must be one of {', '.join(FORMATS)}")

    @property
    def lists_keys(self) -> bool:
        """Whether the report lists individual keys."""
        return self.keys is not None or self.chunks

    def selects(self, key: str) -> bool:
        """Whether ``key`` is selected by ``keys`` (all keys when unset)."""
        return self.keys is None or self.keys.search(key) is not None


COMMON_MODES = ("skip", "bytes")
_MODE_PATTERN = re.compile(r"^(?:(?P<kind>meta|full)|sample:(?P<size>[1-9]\d*))$")


@dataclass(frozen=True, slots=True)
class CompareMode:
    """How much of each tensor ``verify`` compares.

    Attributes:
        kind: ``meta`` (keys, shapes, dtypes), ``full`` (all values) or
            ``sample`` (values of ``sample_size`` keys chosen with a seeded RNG).
        sample_size: Number of sampled keys (``sample`` only).
    """

    kind: str
    sample_size: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("meta", "full", "sample"):
            raise UsageError(f"--mode must be meta, full or sample:N, got {self.kind!r}")
        if (self.kind == "sample") != (self.sample_size > 0):
            raise UsageError("--mode sample needs a positive sample size (sample:N)")

    @classmethod
    def parse(cls, text: str) -> CompareMode:
        """Parse ``meta``, ``full`` or ``sample:N``.

        Raises:
            ValueError: ``text`` is none of those.
        """
        match = _MODE_PATTERN.match(text)
        if match is None:
            raise ValueError(f"invalid mode {text!r}; use meta, full or sample:N")
        if match.group("kind"):
            return cls(match.group("kind"))
        return cls("sample", int(match.group("size")))

    def __str__(self) -> str:
        return f"sample:{self.sample_size}" if self.kind == "sample" else self.kind


@dataclass(frozen=True, slots=True)
class VerifyOptions:
    """Options of ``verify``.

    Attributes:
        iteration: Iteration to open under checkpoint roots (both sides).
        mode: What to compare (see :class:`CompareMode`).
        seed: Seed of the key sample for ``sample:N``.
        layout: Also require identical chunk grids.
        common: ``bytes`` compares byte items (common state) byte for byte;
            ``skip`` ignores them.
        ignore_keys: Keys matching any of these (``re.search``) are ignored.
        oracle: Glob of digest files to verify against instead of a second store.
        oracle_key_map: Python file mapping oracle keys to store keys.
        max_item_bytes: Per-item read budget (one side of one compare unit).
        workers: Threads used for reading and comparing.
    """

    iteration: int | None = None
    mode: CompareMode = CompareMode("full")
    seed: int = 0
    layout: bool = False
    common: str = "bytes"
    ignore_keys: tuple[re.Pattern[str], ...] = ()
    oracle: str | None = None
    oracle_key_map: Path | None = None
    max_item_bytes: int = field(default_factory=default_max_item_bytes)
    workers: int = DEFAULT_WORKERS

    def __post_init__(self) -> None:
        _check_common(self.iteration, self.max_item_bytes, self.workers)
        if self.common not in COMMON_MODES:
            raise UsageError(f"--common must be one of {', '.join(COMMON_MODES)}")
        if self.oracle_key_map is not None and self.oracle is None:
            raise UsageError("--oracle-key-map needs --oracle")
        if self.layout and self.oracle is not None:
            raise UsageError("--layout compares two stores; it cannot be used with --oracle")

    def ignores(self, key: str) -> bool:
        """Whether ``key`` is excluded by ``ignore_keys``."""
        return any(pattern.search(key) for pattern in self.ignore_keys)

    def value_keys(self, comparable: Sequence[str]) -> list[str]:
        """The keys whose values ``mode`` compares, from ``comparable`` (sorted)."""
        if self.mode.kind == "meta":
            return []
        if self.mode.kind == "full":
            return sorted(comparable)
        count = min(self.mode.sample_size, len(comparable))
        return sorted(random.Random(self.seed).sample(sorted(comparable), count))


DEFAULT_UNIT_BYTES = 4 * 2**30
DEFAULT_WRITER_THREADS = 4
HOST_BUDGET_FRACTION = 0.8
"""Default ``--max-host-bytes``: this fraction of ``MemAvailable``."""
HARD_CAP_FRACTION = 0.6
"""Default ``--max-unit-bytes-hard``: this fraction of ``MemAvailable``, split among the workers."""
_FALLBACK_AVAILABLE_BYTES = 16 * 2**30
CONVERT_FORMATS = (TORCH_DIST,)
"""Target formats ``plan`` and ``convert`` support."""
RESUME_CHOICES = ("auto", "never")


def default_convert_workers() -> int:
    """``min(4, cores // 8)`` worker processes (0: the driver runs the units itself)."""
    return min(4, (os.cpu_count() or 1) // 8)


def _available_or_fallback() -> int:
    available = available_memory_bytes()
    return _FALLBACK_AVAILABLE_BYTES if available is None else available


@dataclass(frozen=True, slots=True)
class PlanOptions:
    """Options of ``plan`` (and the planning part of ``convert``).

    Attributes:
        to_format: Target format.
        from_format: Source format to assume when the store has no markers.
        iteration: Iteration to open under a checkpoint root.
        drop_common_keys: Unknown common-state keys to drop instead of refusing.
        workers: Worker processes (0: units run in the driver).
        unit_bytes: Target peak memory of one unit.
        max_unit_bytes_hard: Largest peak memory any single unit may need
            (default: 0.6 x ``MemAvailable`` / workers).
        writer_threads: Writer threads (data files) per part store.
        max_host_bytes: Host-memory budget of the whole conversion
            (default: 0.8 x ``MemAvailable``).
    """

    to_format: str = TORCH_DIST
    from_format: str | None = None
    iteration: int | None = None
    drop_common_keys: frozenset[str] = frozenset()
    workers: int = field(default_factory=default_convert_workers)
    unit_bytes: int = DEFAULT_UNIT_BYTES
    max_unit_bytes_hard: int | None = None
    writer_threads: int = DEFAULT_WRITER_THREADS
    max_host_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.to_format not in CONVERT_FORMATS:
            raise UsageError(
                f"--to {self.to_format} is not supported yet; this version writes "
                f"{', '.join(CONVERT_FORMATS)} only"
            )
        if self.from_format is not None and self.from_format not in FORMATS:
            raise UsageError(f"--from must be one of {', '.join(FORMATS)}")
        if self.iteration is not None and self.iteration < 0:
            raise UsageError(f"--iteration must be non-negative, got {self.iteration}")
        if self.workers < 0:
            raise UsageError(f"--workers must be non-negative, got {self.workers}")
        if self.writer_threads < 1:
            raise UsageError(f"--writer-threads must be at least 1, got {self.writer_threads}")
        if self.max_unit_bytes_hard is None:
            hard_cap = int(HARD_CAP_FRACTION * _available_or_fallback() / max(self.workers, 1))
            object.__setattr__(self, "max_unit_bytes_hard", hard_cap)
        if self.max_host_bytes is None:
            budget = int(HOST_BUDGET_FRACTION * _available_or_fallback())
            object.__setattr__(self, "max_host_bytes", budget)
        for flag, value in (
            ("--unit-bytes", self.unit_bytes),
            ("--max-unit-bytes-hard", self.max_unit_bytes_hard),
            ("--max-host-bytes", self.max_host_bytes),
        ):
            if value <= 0:
                raise UsageError(f"{flag} must be positive, got {value}")

    def output_fields(self) -> dict[str, Any]:
        """The options that determine the output bytes (part of the conversion fingerprint)."""
        return {
            "to_format": self.to_format,
            "drop_common_keys": sorted(self.drop_common_keys),
            "unit_bytes": self.unit_bytes,
            "writer_threads": self.writer_threads,
        }


@dataclass(frozen=True, slots=True)
class ConvertOptions(PlanOptions):
    """Options of ``convert``: planning options plus staging and publishing.

    Attributes:
        staging_dir: Directory in which the staging directory ``.fc-staging-<name>``
            is created, on the target's file system (default: the target's
            parent). Only that subdirectory is ever deleted.
        resume: ``auto`` continues staging of the same conversion; ``never``
            refuses to run when staging exists.
        restart: Discard existing staging first.
        keep_staging: Keep the staging directory (and its part stores) afterwards.
        overwrite: Replace an existing target iteration directory (only it).
        dst_is_iter_dir: Treat DST as the iteration directory itself (no tracker).
        tracker: Write ``latest_checkpointed_iteration.txt`` in root mode.
        copy_sidecars: Copy ``run_config.yaml`` and tokenizer assets.
    """

    staging_dir: Path | None = None
    resume: str = "auto"
    restart: bool = False
    keep_staging: bool = False
    overwrite: bool = False
    dst_is_iter_dir: bool = False
    tracker: bool = True
    copy_sidecars: bool = True

    def __post_init__(self) -> None:
        super(ConvertOptions, self).__post_init__()
        if self.resume not in RESUME_CHOICES:
            raise UsageError(f"--resume must be one of {', '.join(RESUME_CHOICES)}")

    def output_fields(self) -> dict[str, Any]:
        """The options that determine the output bytes (part of the conversion fingerprint)."""
        return {**super(ConvertOptions, self).output_fields(), "copy_sidecars": self.copy_sidecars}
