# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Read-once region reads through ``torch.distributed.checkpoint.load``.

DCP loads each stored chunk with one whole-item ``torch.load`` per ``ReadItem``
and plans one ``ReadItem`` per (requested box, stored chunk) overlap. To read
every touched stored chunk exactly once per batch, the planner below asks for
exactly one box per touched stored chunk: the bounding box of what the batch
needs from it, clipped to the chunk.

* If one request needs exactly that box, DCP copies straight into a view of
  the request's destination (no staging).
* Otherwise the box is loaded into a staging buffer (at most the chunk size)
  and narrow-copied into each destination afterwards.
* A request for a whole tensor passes the destination tensor itself.

All requests of a batch are served by a single ``dcp.load`` through a
:class:`~.metadata.CachedMetadataFileSystemReader` over a subset ``Metadata``.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed.checkpoint import DefaultLoadPlanner

from . import torch_compat
from .metadata import CachedMetadataFileSystemReader, MetadataView, StoredChunk
from .regions import Region, bounding_box

if TYPE_CHECKING:
    from .reader import ReadReq


class RawBytesLoadPlanner(DefaultLoadPlanner):
    """Load planner that keeps byte items as raw payloads and records tensor reads.

    Byte items are never unpickled (the stock planner ``torch.load``s them with
    ``weights_only=False``). Each committed tensor ``ReadItem`` corresponds to
    one whole-item ``torch.load`` in ``FileSystemReader.read_data``.

    Attributes:
        payloads: Raw byte-item payloads by key.
        tensor_reads: ``(key, chunk offsets)`` of the stored chunk behind every
            committed tensor ``ReadItem``, in execution order.
    """

    def __init__(self) -> None:
        super().__init__(
            flatten_state_dict=False, flatten_sharded_tensors=False, allow_partial_load=False
        )
        self.payloads: dict[str, bytes] = {}
        self.tensor_reads: list[tuple[str, tuple[int, ...]]] = []

    def load_bytes(self, read_item: Any, value: io.BytesIO) -> None:
        """Keep the payload as bytes instead of unpickling it."""
        self.payloads[torch_compat.read_item_key(read_item)] = value.getvalue()

    def commit_tensor(self, read_item: Any, tensor: torch.Tensor) -> None:
        """Record which stored chunk was loaded for this read."""
        del tensor
        self.tensor_reads.append(torch_compat.read_item_source(read_item))


@dataclass(frozen=True, slots=True)
class RegionLoadPlan:
    """One batch of region reads, ready for a single ``dcp.load``.

    Attributes:
        targets: The state dict handed to ``dcp.load`` (destination tensors or
            :class:`~.torch_compat.RegionTarget` objects).
        copies: ``(destination view, staging view)`` pairs to copy after the load.
        touched: Every stored chunk the batch reads, once each.
    """

    targets: dict[str, torch.Tensor]
    copies: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    touched: tuple[StoredChunk, ...]


def plan_region_load(view: MetadataView, reqs: Sequence[ReadReq]) -> RegionLoadPlan:
    """Plan the reads of ``reqs`` (already validated) so each stored chunk is read once."""
    by_key: dict[str, list[ReadReq]] = {}
    for req in reqs:
        if req.region.numel() > 0:
            by_key.setdefault(req.key, []).append(req)
    targets: dict[str, torch.Tensor] = {}
    copies: list[tuple[torch.Tensor, torch.Tensor]] = []
    touched: list[StoredChunk] = []
    for key, key_reqs in by_key.items():
        entry = view.tensors[key]
        if len(key_reqs) == 1 and key_reqs[0].region.is_full(entry.shape):
            targets[key] = key_reqs[0].dst
            touched.extend(c for c in entry.chunks if c.region.numel() > 0)
            continue
        parts: list[tuple[Region, torch.Tensor]] = []
        for chunk in entry.chunks:
            needs = [
                (req, overlap)
                for req in key_reqs
                if (overlap := req.region.intersect(chunk.region)) is not None
            ]
            if not needs:
                continue
            touched.append(chunk)
            box = bounding_box(overlap for _, overlap in needs)
            if staging_bytes(chunk, [overlap for _, overlap in needs]) == 0:
                req = needs[0][0]
                parts.append((box, req.dst[box.relative_to(req.region).slices()]))
                continue
            staging = torch.empty(box.sizes, dtype=entry.dtype)
            parts.append((box, staging))
            for req, overlap in needs:
                copies.append(
                    (
                        req.dst[overlap.relative_to(req.region).slices()],
                        staging[overlap.relative_to(box).slices()],
                    )
                )
        targets[key] = torch_compat.RegionTarget(entry.shape, entry.dtype, parts)
    return RegionLoadPlan(targets, tuple(copies), tuple(touched))


def staging_bytes(chunk: StoredChunk, overlaps: Sequence[Region]) -> int:
    """Size of the staging buffer a batch needs for one stored chunk.

    ``overlaps`` are the parts of the chunk the batch's requests need. A single
    request is read straight into its destination; several share one staging
    buffer covering their bounding box, which is copied out after the load.
    """
    if len(overlaps) <= 1:
        return 0
    return bounding_box(overlaps).numel() * chunk.dtype.itemsize


def execute_region_load(
    view: MetadataView, plan: RegionLoadPlan
) -> list[tuple[str, tuple[int, ...]]]:
    """Run ``plan`` with one ``dcp.load``; return the stored chunk behind each ``torch.load``."""
    if not plan.targets:
        return []
    planner = RawBytesLoadPlanner()
    reader = CachedMetadataFileSystemReader(view.iter_dir, view.subset(plan.targets))
    torch_compat.load_without_process_group(dict(plan.targets), reader, planner)
    for dst, staged in plan.copies:
        dst.copy_(staged)
    return planner.tensor_reads


def load_byte_items(view: MetadataView, keys: Sequence[str]) -> dict[str, bytes]:
    """Read the raw payloads of byte items ``keys`` with one ``dcp.load``."""
    if not keys:
        return {}
    planner = RawBytesLoadPlanner()
    reader = CachedMetadataFileSystemReader(view.iter_dir, view.subset(keys))
    torch_compat.load_without_process_group(dict.fromkeys(keys), reader, planner)
    return planner.payloads
