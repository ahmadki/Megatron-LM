# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Executing staged units: read the sources, write the part store, commit.

A worker claims units in the staged dispatch order and, for each one:

1. allocates one standalone buffer per piece;
2. reads every copy's source box into its piece buffer with a single batched
   ``dcp.load`` (each touched stored item is decoded once);
3. writes the buffers and byte items as the unit's part store (``*.tmp``);
4. digests the pieces selected for the post-merge check and commits the unit
   (``DONE`` record, rename, journal line).

It imports only the engine (standard library, torch and DCP).
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from .faults import inject_fault
from .hashing import byte_view
from .merge import ItemDigest
from .metadata import MetadataView
from .part_writer import write_part
from .pieces import Piece, Unit
from .reader import ReadReq, SourceReader
from .staging import Staging, StagingPlan


def run_worker(staging: Staging, worker_id: int) -> int:
    """Claim and execute units in dispatch order until none is left.

    Returns:
        The number of units this worker executed.
    """
    plan = staging.plan()
    executed = 0
    for uid in plan.dispatch_order:
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
    }
    staging.complete(uid, digests, record)


def read_requests(pieces: Sequence[Piece], buffers: Sequence[torch.Tensor]) -> list[ReadReq]:
    """One read request per copy, targeting the copy's box of its piece buffer."""
    return [
        ReadReq(op.src.key, op.src_region, buffer[op.dst_region.slices()])
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
