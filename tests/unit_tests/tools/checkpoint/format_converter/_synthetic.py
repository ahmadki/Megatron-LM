# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Torch-only builders of synthetic DCP stores for the format_converter tests.

Stores are written with ``torch.distributed.checkpoint.save(no_dist=True)``
through :class:`ChunkedSavePlanner`, a TEST-ONLY planner that emits explicit
``SHARD`` write items, so a tensor can be stored as any chunk grid. Nothing here
imports megatron; Megatron classes that stores reference are written under
their real global names by :class:`AliasPickler` without importing them.
"""

from __future__ import annotations

import io
import pickle
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.default_planner import create_default_global_save_plan
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import (
    SavePlan,
    SavePlanner,
    TensorWriteData,
    WriteItem,
    WriteItemType,
)

from tools.checkpoint.format_converter.engine.regions import Region


class ChunkedSavePlanner(SavePlanner):
    """TEST-ONLY save planner writing each tensor as an explicit chunk grid."""

    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor],
        grids: Mapping[str, Sequence[Region]],
        byte_items: Mapping[str, bytes],
        planner_data: Mapping[str, tuple[str, ...]] | None,
    ) -> None:
        self.tensors = tensors
        self.grids = grids
        self.byte_items = byte_items
        self.planner_data = planner_data

    def set_up_planner(self, state_dict, storage_meta=None, is_coordinator=False) -> None:
        del state_dict, storage_meta, is_coordinator

    def create_local_plan(self) -> SavePlan:
        items = []
        for key, tensor in self.tensors.items():
            for index, region in enumerate(self.grids[key]):
                chunk = ChunkStorageMetadata(
                    offsets=torch.Size(region.offsets), sizes=torch.Size(region.sizes)
                )
                items.append(
                    WriteItem(
                        index=MetadataIndex(key, region.offsets, index),
                        type=WriteItemType.SHARD,
                        tensor_data=TensorWriteData(
                            chunk=chunk,
                            properties=TensorProperties.create_from_tensor(tensor),
                            size=tensor.size(),
                        ),
                    )
                )
        for key in self.byte_items:
            items.append(WriteItem(index=MetadataIndex(key), type=WriteItemType.BYTE_IO))
        return SavePlan(items)

    def create_global_plan(self, all_plans):
        plans, metadata = create_default_global_save_plan(all_plans)
        if self.planner_data is not None:
            metadata.planner_data = dict(self.planner_data)
        return plans, metadata

    def finish_plan(self, new_plan):
        return new_plan

    def resolve_data(self, write_item):
        if write_item.type == WriteItemType.BYTE_IO:
            return io.BytesIO(self.byte_items[write_item.index.fqn])
        chunk = write_item.tensor_data.chunk
        region = Region(tuple(chunk.offsets), tuple(chunk.sizes))
        return self.tensors[write_item.index.fqn][region.slices()].clone()


def write_store(
    iter_dir: Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    grids: Mapping[str, Sequence[Region]] | None = None,
    byte_items: Mapping[str, bytes] | None = None,
    planner_data: Mapping[str, tuple[str, ...]] | None = None,
    thread_count: int = 1,
) -> Path:
    """Write a DCP store; tensors without a grid are stored as one chunk."""
    grids = dict(grids or {})
    for key, tensor in tensors.items():
        grids.setdefault(key, (Region.full(tuple(tensor.shape)),))
    iter_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="torch.distributed is disabled")
        dcp.save(
            {},
            storage_writer=FileSystemWriter(
                iter_dir, thread_count=thread_count, per_thread_copy_ahead=0
            ),
            planner=ChunkedSavePlanner(tensors, grids, dict(byte_items or {}), planner_data),
            no_dist=True,
        )
    return iter_dir


def dim0_grid(shape: Sequence[int], rows: int) -> tuple[Region, ...]:
    """Chunks of ``rows`` rows along dimension 0 (the last one may be shorter)."""
    return tuple(Region.full(tuple(shape)).split_dim0(rows))


def product_grid(shape: Sequence[int], cuts: Sequence[Sequence[int]]) -> tuple[Region, ...]:
    """The grid whose per-dimension boundaries are ``cuts[d]`` (interior cut points)."""
    axes = []
    for extent, points in zip(shape, cuts):
        bounds = [0, *points, extent]
        axes.append([(lo, hi - lo) for lo, hi in zip(bounds, bounds[1:])])
    regions = [((), ())]
    for axis in axes:
        regions = [(o + (lo,), s + (n,)) for o, s in regions for lo, n in axis]
    return tuple(Region(o, s) for o, s in regions)


def torch_save_bytes(obj: Any) -> bytes:
    """``torch.save(obj)`` into memory, as DCP stores byte items."""
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


class AliasPickler(pickle._Pickler):  # pylint: disable=protected-access
    """Pickler that writes chosen classes or functions under another global name.

    Lets tests produce pickles that reference ``megatron.*`` (or forbidden)
    globals without importing or even having them.
    """

    def __init__(self, file, aliases: Mapping[Any, tuple[str, str]], protocol: int = 2) -> None:
        super().__init__(file, protocol=protocol)
        self.aliases = dict(aliases)

    def save_global(self, obj, name=None):
        alias = self.aliases.get(obj)
        if alias is None:
            super().save_global(obj, name)
            return
        module, qualname = alias
        self.write(pickle.GLOBAL + f"{module}\n{qualname}\n".encode())
        self.memoize(obj)


def alias_dumps(obj: Any, aliases: Mapping[Any, tuple[str, str]]) -> bytes:
    """Pickle ``obj`` (protocol 2) with :class:`AliasPickler`."""
    buffer = io.BytesIO()
    AliasPickler(buffer, aliases).dump(obj)
    return buffer.getvalue()


def rewrite_metadata(
    iter_dir: Path,
    mutate: Callable[[Metadata], Any],
    aliases: Mapping[Any, tuple[str, str]] | None = None,
) -> None:
    """Load ``.metadata`` with plain pickle, let ``mutate`` edit or replace it, write it back."""
    path = iter_dir / ".metadata"
    metadata = pickle.loads(path.read_bytes())
    replacement = mutate(metadata)
    path.write_bytes(alias_dumps(metadata if replacement is None else replacement, aliases or {}))
