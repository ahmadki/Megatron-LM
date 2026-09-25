# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Running staged units in worker processes with a sanitized environment.

``--workers 0`` executes the units in the calling process. ``--workers N``
starts ``N`` fresh interpreters (``python -m ...engine.worker STAGING ID``)
whose environment is built before they start: no visible GPU, no launcher
variables (so nothing in them can join or create a process group, even when
the driver runs under ``torchrun``), the repository on ``PYTHONPATH`` and a
bounded number of OpenMP threads. Workers coordinate only through claim files
in the staging directory. They inherit the descriptor that holds the staging
lock, so the lock stays held while any of them runs (a rerun cannot start
beside a worker orphaned by a killed driver), and each stops before its next
unit once the driver is gone. The driver polls them; the first non-zero exit
stops the others and fails the run, which stays resumable.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from ..errors import IntegrityError
from .staging import Staging
from .worker import run_worker

LAUNCHER_VARIABLES = frozenset(
    {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE"}
)
"""Launcher variables removed from a worker's environment."""

LAUNCHER_PREFIXES = ("ROLE_", "MASTER_", "TORCHELASTIC_", "TORCH_NCCL_")
"""Prefixes of launcher variables removed from a worker's environment."""

WORKER_MODULE = f"{__package__}.worker"
REPO_ROOT = Path(__file__).resolve().parents[len(__package__.split("."))]
"""The directory from which the package is importable (``tools/..``)."""

_POLL_SECONDS = 0.05


def child_env(base: Mapping[str, str], repo_root: Path, *, threads: int) -> dict[str, str]:
    """A worker's environment: ``base`` without launcher variables, GPUs hidden.

    Args:
        base: The driver's environment.
        repo_root: Directory put first on ``PYTHONPATH``.
        threads: ``OMP_NUM_THREADS`` of the worker.
    """
    env = {
        name: value
        for name, value in base.items()
        if name not in LAUNCHER_VARIABLES and not name.startswith(LAUNCHER_PREFIXES)
    }
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = str(threads)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + pythonpath if pythonpath else "")
    return env


def worker_threads(workers: int) -> int:
    """CPU threads per worker: half the cores, split between the workers."""
    return max(1, (os.cpu_count() or 1) // max(workers, 1) // 2)


def run_units(staging: Staging, *, workers: int) -> int:
    """Execute every incomplete staged unit.

    Args:
        staging: The staged conversion (its lock held by the caller).
        workers: Worker processes; 0 runs the units in this process.

    Returns:
        The number of units executed.

    Raises:
        IntegrityError: A worker exited with a non-zero status (the run can be
            resumed).
        KeyboardInterrupt: The run was interrupted; the workers are stopped.
    """
    if workers == 0:
        return run_worker(staging, worker_id=0)
    uids = staging.plan().dispatch_order
    done_before = sum(staging.is_complete(uid) for uid in uids)
    env = child_env(os.environ, REPO_ROOT, threads=worker_threads(workers))
    command = [sys.executable, "-m", WORKER_MODULE, str(staging.root)]
    processes = [
        subprocess.Popen(  # pylint: disable=consider-using-with
            [*command, str(worker_id), "--driver-pid", str(os.getpid())],
            env=env,
            cwd=REPO_ROOT,
            pass_fds=(staging.lock_fd,),
        )
        for worker_id in range(workers)
    ]
    try:
        _wait_for(processes)
    finally:
        _stop(processes)
    return sum(staging.is_complete(uid) for uid in uids) - done_before


def _wait_for(processes: list[subprocess.Popen]) -> None:
    running = dict(enumerate(processes))
    while running:
        for worker_id, process in list(running.items()):
            status = process.poll()
            if status is None:
                continue
            del running[worker_id]
            if status != 0:
                raise IntegrityError(
                    f"worker {worker_id} exited with status {status}",
                    hint="the staging directory is kept; rerun the same command to resume",
                )
        if running:
            time.sleep(_POLL_SECONDS)


def _stop(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        process.wait()
