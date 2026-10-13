# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Executing staged units: read the sources, write the part store, commit.

A worker claims units in the staged dispatch order and, for each one:

1. allocates one standalone buffer per piece;
2. reads every copy's source box into its piece buffer with a single batched
   ``dcp.load`` (each touched stored item is decoded once), converting the
   floating-point type where the piece plans a cast;
3. writes the buffers and byte items as the unit's part store (``*.tmp``);
4. digests the pieces selected for the post-merge check and commits the unit
   (``DONE`` record, rename, journal line).

Worker processes run ``python -m tools.checkpoint.format_converter.engine.worker
STAGING WORKER_ID --driver-pid PID`` (see :mod:`.workers`) and stop before their
next unit once the driver is no longer their parent; the module imports only
the engine (standard library, torch and DCP) and never uses CUDA or
``torch.distributed``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from ..errors import ConverterError, ExitCode, IntegrityError
from .faults import inject_fault
from .hashing import byte_view
from .memory import process_peak_rss_bytes
from .merge import ItemDigest
from .metadata import MetadataView
from .part_writer import write_part
from .pieces import Piece, Unit
from .reader import ReadReq, SourceReader
from .staging import Staging, StagingPlan


def run_worker(staging: Staging, worker_id: int, *, driver_pid: int | None = None) -> int:
    """Claim and execute units in dispatch order until none is left.

    Args:
        staging: The staged conversion.
        worker_id: Id recorded in claims and the journal.
        driver_pid: The driving process, which must stay this process's parent
            (``None``: the units run in the driver itself).

    Returns:
        The number of units this worker executed.

    Raises:
        IntegrityError: The driver exited (checked before each unit).
    """
    plan = staging.plan()
    executed = 0
    for uid in plan.dispatch_order:
        if driver_pid is not None and os.getppid() != driver_pid:
            raise IntegrityError(
                f"the driver (pid {driver_pid}) is gone; stopping before unit {uid}",
                hint="rerun the conversion to resume it",
            )
        if staging.is_complete(uid) or not staging.claim(uid, worker_id):
            continue
        execute_unit(staging, plan, uid, worker_id)
        executed += 1
    return executed


def execute_unit(staging: Staging, plan: StagingPlan, uid: int, worker_id: int) -> None:
    """Read, write and commit one claimed unit."""
    started = time.time()
    staged = staging.load_unit(uid)
    unit = staged.unit
    view = MetadataView.of_metadata(plan.source_dir, staged.subset)
    reader = SourceReader(view, max_item_bytes=plan.max_item_bytes)
    buffers = [torch.empty(piece.chunk.sizes, dtype=piece.dtype) for piece in unit.pieces]
    stats = reader.read_into(read_requests(unit.pieces, buffers))
    part_dir = staging.fresh_part_dir(uid)
    write_part(
        part_dir, list(zip(unit.pieces, buffers)), unit.byte_pieces, threads=plan.writer_threads
    )
    digests = _digests(unit, buffers, staged.checked)
    inject_fault(f"unit:{uid}")
    record = {
        "uid": uid,
        "worker": worker_id,
        "pid": os.getpid(),
        "t0": started,
        "t1": time.time(),
        "bytes_read": stats.bytes_read,
        "bytes_written": _directory_bytes(part_dir),
        "items_touched": stats.items_touched,
        "torch_load_calls": stats.torch_load_calls,
        "vmhwm_bytes": process_peak_rss_bytes(),
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    staging.complete(uid, digests, record)


def read_requests(pieces: Sequence[Piece], buffers: Sequence[torch.Tensor]) -> list[ReadReq]:
    """One read request per copy, targeting the copy's box of its piece buffer.

    The box is viewed with the source box's shape (they differ at most in
    leading size-1 axes, which a view adds or removes without copying). A copy
    from a source of another type than its piece's converts it (the piece
    plans the cast).
    """
    return [
        ReadReq(
            op.src.key,
            op.src_region,
            buffer[op.dst_region.slices()].view(op.src_region.sizes),
            cast=op.src.dtype != piece.dtype,
        )
        for piece, buffer in zip(pieces, buffers)
        for op in piece.ops
    ]


def _digests(
    unit: Unit, buffers: Sequence[torch.Tensor], checked: Sequence[int]
) -> list[ItemDigest]:
    digests = [
        ItemDigest(
            unit.pieces[index].key,
            unit.pieces[index].chunk.offsets,
            hashlib.sha256(byte_view(buffers[index])).hexdigest(),
        )
        for index in checked
    ]
    digests.extend(
        ItemDigest(item.key, None, hashlib.sha256(item.payload).hexdigest())
        for item in unit.byte_pieces
    )
    return digests


def _directory_bytes(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.iterdir() if entry.is_file())


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of a worker process; returns its exit status."""
    parser = argparse.ArgumentParser(description="format_converter unit worker")
    parser.add_argument("staging", type=Path, help="staging directory of the conversion")
    parser.add_argument("worker_id", type=int, help="id recorded in claims and the journal")
    parser.add_argument(
        "--driver-pid", type=int, help="stop before the next unit once this process is gone"
    )
    args = parser.parse_args(argv)
    try:
        run_worker(Staging(args.staging), args.worker_id, driver_pid=args.driver_pid)
    except ConverterError as err:
        sys.stderr.write(f"worker {args.worker_id}: error: {err.format()}\n")
        return int(err.exit_code)
    except KeyboardInterrupt:
        return int(ExitCode.INTERRUPTED)
    return int(ExitCode.OK)


if __name__ == "__main__":
    sys.exit(main())
