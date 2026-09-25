# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Streaming SHA-256 digests of tensors over their logical bytes.

The digest of a tensor is SHA-256 over its elements in row-major order, each
element in its native in-memory encoding (``tensor.contiguous()`` viewed as
bytes; bf16 and other dtypes without a numpy equivalent included). It does not
depend on how the tensor is chunked on disk or on the ``torch.save`` framing,
so digests of the same tensor from different stores and formats are equal.

Reading plan per tensor:

* small tensors are read whole, many per ``dcp.load`` batch;
* larger tensors whose chunk grid splits only dimension 0 are streamed as
  consecutive row slabs, so memory is bounded by the batch size or the largest
  stored chunk;
* other grids are materialized whole if they fit ``max_item_bytes``;
  otherwise the digest is refused with exit 8 before anything is read.

Digest files map keys to ``{"shape": [...], "dtype": "torch.bfloat16", "sha256": hex}``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..errors import BudgetExceeded, IntegrityError
from .metadata import TensorEntry
from .reader import ReadReq, ReadStats, SourceReader
from .regions import Region, bounding_box, dim0_only
from .sizes import format_bytes

_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True, slots=True)
class TensorDigest:
    """Shape, dtype and SHA-256 of a tensor's logical bytes.

    Attributes:
        shape: Global shape.
        dtype: ``str(torch.dtype)``, e.g. ``torch.bfloat16``.
        sha256: Hex digest.
    """

    shape: tuple[int, ...]
    dtype: str
    sha256: str

    def to_json(self) -> dict[str, Any]:
        """The digest-file record of this digest."""
        return {"shape": list(self.shape), "dtype": self.dtype, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class DigestJob:
    """One unit of hashing work.

    Attributes:
        keys: Keys hashed by this job. Several keys means each is read whole in
            one batch; a single key with ``slabs`` means it is streamed.
        slabs: Consecutive boxes of the single key, in row-major order.
    """

    keys: tuple[str, ...]
    slabs: tuple[Region, ...] = ()


def byte_view(tensor: torch.Tensor) -> Any:
    """The tensor's logical bytes (row-major, native encoding) as a buffer."""
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy()


def plan_digest_jobs(
    entries: Iterable[TensorEntry], *, max_item_bytes: int, batch_bytes: int
) -> list[DigestJob]:
    """Group tensors into hashing jobs whose resident size is bounded.

    Raises:
        BudgetExceeded: A tensor cannot be hashed within ``max_item_bytes``
            (a stored chunk is larger, or its grid forces a full read).
    """
    jobs: list[DigestJob] = []
    batch: list[str] = []
    batch_size = 0
    for entry in entries:
        if entry.nbytes <= batch_bytes:
            if batch and batch_size + entry.nbytes > batch_bytes:
                jobs.append(DigestJob(tuple(batch)))
                batch, batch_size = [], 0
            batch.append(entry.key)
            batch_size += entry.nbytes
        elif entry.shape and dim0_only((c.region for c in entry.chunks), entry.shape):
            jobs.append(DigestJob((entry.key,), _row_slabs(entry, max_item_bytes, batch_bytes)))
        elif entry.nbytes <= max_item_bytes:
            jobs.append(DigestJob((entry.key,), (Region.full(entry.shape),)))
        else:
            raise BudgetExceeded(
                f"hashing {entry.key} needs the whole tensor in memory "
                f"({format_bytes(entry.nbytes)}; its chunk grid splits more than dimension 0), "
                f"above the per-item budget of {format_bytes(max_item_bytes)}",
                hint=f"pass --max-item-bytes {entry.nbytes} or more (host memory permitting)",
            )
    if batch:
        jobs.append(DigestJob(tuple(batch)))
    return jobs


def compute_digests(
    reader: SourceReader, keys: Sequence[str], *, workers: int
) -> tuple[dict[str, TensorDigest], ReadStats]:
    """Digest tensors ``keys`` of ``reader``'s store with up to ``workers`` threads.

    Returns:
        Digests by key (sorted) and the combined read statistics.

    Raises:
        BudgetExceeded: Before any read, if a tensor cannot be hashed within
            the reader's per-item budget.
    """
    entries = [reader.view.tensors[key] for key in keys]
    jobs = plan_digest_jobs(
        entries, max_item_bytes=reader.max_item_bytes, batch_bytes=reader.batch_bytes
    )
    digests: dict[str, TensorDigest] = {}
    stats = ReadStats()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for job_digests, job_stats in pool.map(lambda job: _run_job(reader, job), jobs):
            digests.update(job_digests)
            stats += job_stats
    return dict(sorted(digests.items())), stats


def write_digest_file(path: Path, digests: Mapping[str, TensorDigest]) -> None:
    """Write a digest file atomically (temporary file, then rename)."""
    payload = {key: digest.to_json() for key, digest in sorted(digests.items())}
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def read_digest_file(path: Path) -> dict[str, TensorDigest]:
    """Read and validate a digest file.

    Raises:
        IntegrityError: The file is unreadable or does not follow the schema.
    """
    try:
        records = json.loads(path.read_text())
    except (OSError, ValueError) as err:
        raise IntegrityError(f"cannot read digest file {path}: {err}") from err
    if not isinstance(records, dict):
        raise IntegrityError(f"digest file {path} must hold a JSON object of key records")
    return {key: _digest_from_record(path, key, record) for key, record in records.items()}


def _digest_from_record(path: Path, key: str, record: Any) -> TensorDigest:
    shape = record.get("shape") if isinstance(record, dict) else None
    dtype = record.get("dtype") if isinstance(record, dict) else None
    sha256 = record.get("sha256") if isinstance(record, dict) else None
    if (
        not isinstance(shape, list)
        or not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape)
        or not isinstance(dtype, str)
        or not isinstance(sha256, str)
        or _SHA256_HEX.match(sha256) is None
    ):
        raise IntegrityError(
            f"digest file {path}: record of {key!r} must be "
            '{"shape": [int, ...], "dtype": "torch.<type>", "sha256": "<64 hex digits>"}'
        )
    return TensorDigest(tuple(shape), dtype, sha256.lower())


def _row_slabs(entry: TensorEntry, max_item_bytes: int, batch_bytes: int) -> tuple[Region, ...]:
    slabs: list[Region] = []
    group: list[Region] = []
    group_bytes = 0
    for chunk in sorted(entry.chunks, key=lambda c: c.offsets[0]):
        if chunk.nbytes > max_item_bytes:
            raise BudgetExceeded(
                f"stored chunk {chunk.offsets} of {entry.key} decodes to "
                f"{format_bytes(chunk.nbytes)}, above the per-item budget of "
                f"{format_bytes(max_item_bytes)}",
                hint=f"pass --max-item-bytes {chunk.nbytes} or more (host memory permitting)",
            )
        if group and group_bytes + chunk.nbytes > batch_bytes:
            slabs.append(bounding_box(group))
            group, group_bytes = [], 0
        group.append(chunk.region)
        group_bytes += chunk.nbytes
    if group:
        slabs.append(bounding_box(group))
    return tuple(slabs)


def _run_job(reader: SourceReader, job: DigestJob) -> tuple[dict[str, TensorDigest], ReadStats]:
    tensors = reader.view.tensors
    if not job.slabs:
        reqs = [
            ReadReq(
                k,
                Region.full(tensors[k].shape),
                torch.empty(tensors[k].shape, dtype=tensors[k].dtype),
            )
            for k in job.keys
        ]
        stats = reader.read_into(reqs)
        return {
            r.key: _digest(tensors[r.key], hashlib.sha256(byte_view(r.dst))) for r in reqs
        }, stats
    (key,) = job.keys
    entry = tensors[key]
    hasher = hashlib.sha256()
    stats = ReadStats()
    for slab in job.slabs:
        buffer = torch.empty(slab.sizes, dtype=entry.dtype)
        stats += reader.read_into([ReadReq(key, slab, buffer)])
        hasher.update(byte_view(buffer))
    return {key: _digest(entry, hasher)}, stats


def _digest(entry: TensorEntry, hasher: Any) -> TensorDigest:
    return TensorDigest(entry.shape, str(entry.dtype), hasher.hexdigest())
