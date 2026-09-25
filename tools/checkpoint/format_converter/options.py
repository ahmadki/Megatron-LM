# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validated, immutable options of the ``inspect`` and ``verify`` commands."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .errors import UsageError
from .formats.names import FORMATS

DEFAULT_WORKERS = 4
MAX_ITEM_FRACTION_OF_AVAILABLE = 0.25
_FALLBACK_MAX_ITEM_BYTES = 4 * 2**30


def available_memory_bytes() -> int | None:
    """``MemAvailable`` from ``/proc/meminfo`` (Linux), else ``None``."""
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


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
