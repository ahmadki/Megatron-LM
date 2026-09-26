# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The memory model of a conversion, its preflight checks, and process memory probes.

Per unit ``u`` (see :func:`~.scheduler.pack_units`):

* ``P_u``: the piece buffers, all resident while the unit runs (the writer
  keeps them referenced until their data file is closed);
* ``S_u``: read staging, i.e. the largest stored item decoded whole by DCP
  (``C``) plus a staging box for every stored chunk that several copies read.

A worker peaks at ``max_u (P_u + S_u) + M_w`` where ``M_w`` is its baseline
(interpreter, torch, unit metadata). With ``N`` worker processes the host peak
is ``N * worker_peak + M_driver``; with ``--workers 0`` the driver runs the
units itself. Page cache is not counted: it is reclaimable.

This module does not import torch, so option parsing can use its probes.
"""

from __future__ import annotations

import os
import resource
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import BudgetExceeded
from .sizes import format_bytes

if TYPE_CHECKING:
    from .pieces import Unit

WORKER_BASELINE_BYTES = 768 * 2**20
"""``M_w``: resident memory of a worker before it allocates buffers (measured ~0.6 GiB)."""

STORED_ITEM_OVERHEAD_BYTES = 4096
"""Upper bound of the per-chunk overhead on disk: ``torch.save`` framing and its metadata entry."""

DISK_HEADROOM = 1.02
"""Free space required on the destination, as a multiple of the estimated output size."""

MERGE_BYTES_PER_CHUNK = 2048
"""Driver memory per output chunk while merging (parsed metadata entries; ~1.5 KiB measured)."""


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    """Estimated host-memory peak of a conversion.

    Attributes:
        unit_peak: ``max_u (P_u + S_u)`` over all units.
        worker_baseline: ``M_w``, per worker process.
        workers: Worker processes (0: units run in the driver).
        driver: ``M_driver``: the driver's resident set (source metadata, plan)
            plus what the merge adds.
    """

    unit_peak: int
    worker_baseline: int
    workers: int
    driver: int

    @property
    def worker_peak(self) -> int:
        """Peak of one worker process (or of the unit work inside the driver)."""
        return self.unit_peak + (self.worker_baseline if self.workers else 0)

    @property
    def total(self) -> int:
        """Peak of the driver and all workers together."""
        return max(self.workers, 1) * self.worker_peak + self.driver

    def to_json(self) -> dict[str, int]:
        """JSON-serializable form."""
        return {
            "unit_peak": self.unit_peak,
            "worker_baseline": self.worker_baseline,
            "workers": self.workers,
            "driver": self.driver,
            "worker_peak": self.worker_peak,
            "total": self.total,
        }


def estimate_memory(units: Iterable[Unit], *, workers: int, driver: int) -> MemoryEstimate:
    """The memory estimate of running ``units`` with ``workers`` worker processes."""
    unit_peak = max((unit.est_peak_bytes for unit in units), default=0)
    return MemoryEstimate(unit_peak, WORKER_BASELINE_BYTES, workers, driver)


def driver_bytes(output_chunks: int) -> int:
    """``M_driver``: the driver's resident set now plus what merging ``output_chunks`` adds."""
    return (process_rss_bytes() or 0) + output_chunks * MERGE_BYTES_PER_CHUNK


def estimate_output_bytes(units: Iterable[Unit]) -> int:
    """Estimated on-disk size of the merged store (data files and metadata)."""
    total = 0
    for unit in units:
        total += sum(p.nbytes + STORED_ITEM_OVERHEAD_BYTES for p in unit.pieces)
        total += sum(item.nbytes for item in unit.byte_pieces)
    return total


def check_host_memory(estimate: MemoryEstimate, budget: int) -> None:
    """Refuse a conversion whose estimated peak exceeds ``budget`` (exit 8)."""
    if estimate.total > budget:
        raise BudgetExceeded(
            f"the conversion needs about {format_bytes(estimate.total)} of host memory "
            f"({max(estimate.workers, 1)} x {format_bytes(estimate.worker_peak)} for units, "
            f"{format_bytes(estimate.driver)} for the driver), above the budget of "
            f"{format_bytes(budget)}",
            hint="lower --workers or --unit-bytes, or raise --max-host-bytes",
        )


def check_free_disk(path: Path, output_bytes: int) -> None:
    """Refuse a conversion whose output does not fit on ``path``'s file system (exit 8)."""
    needed = int(output_bytes * DISK_HEADROOM)
    free = free_disk_bytes(path)
    if free < needed:
        raise BudgetExceeded(
            f"the output needs about {format_bytes(needed)} on the file system of {path}, "
            f"which has {format_bytes(free)} free",
            hint="free disk space or write to another destination",
        )


def free_disk_bytes(path: Path) -> int:
    """Bytes available to this user on ``path``'s file system."""
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize


def available_memory_bytes() -> int | None:
    """``MemAvailable`` from ``/proc/meminfo`` (Linux), else ``None``."""
    return _proc_bytes("/proc/meminfo", "MemAvailable:")


def process_rss_bytes() -> int | None:
    """Current resident set size of this process (``VmRSS``), if readable."""
    return _proc_bytes("/proc/self/status", "VmRSS:")


def process_peak_rss_bytes() -> int | None:
    """Peak resident set size of this process so far (``VmHWM``), if readable."""
    return _proc_bytes("/proc/self/status", "VmHWM:")


def children_peak_rss_bytes() -> int:
    """Largest peak resident set size of any waited-for child process."""
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024


def _proc_bytes(path: str, field: str) -> int | None:
    try:
        with open(path, encoding="ascii") as stream:
            for line in stream:
                if line.startswith(field):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None
