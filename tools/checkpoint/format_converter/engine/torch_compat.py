# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The only module that touches private or version-dependent DCP internals.

Everything the engine needs from ``torch.distributed.checkpoint`` beyond its
public, stable surface goes through the small functions below, so a torch
upgrade that moves an internal breaks exactly one file and the pins in
``test_torch_contract.py``. Covered here:

* The DCP classes a ``.metadata`` pickle may name (:data:`PICKLED_DCP_CLASSES`),
  including the private ``filesystem._StorageInfo`` and ``_StoragePrefix``.
* ``Metadata.storage_data``: a dict of ``MetadataIndex`` to the private
  ``filesystem._StorageInfo`` (``relative_path``, ``offset``, ``length`` and,
  on newer torch, ``transform_descriptors``).
* ``Metadata.version`` (added after torch 2.8) and attributes that savers
  attach to ``Metadata`` instances (``mcore_data`` from old Megatron,
  ``all_local_plans`` from the nvidia-resiliency-ext async saver).
* The checkpointable-tensor protocol (``__create_chunk_list__`` and
  ``__get_tensor_shard__`` on a ``torch.Tensor._make_wrapper_subclass``), which
  lets one load request cover several disjoint boxes of a tensor.
* ``ReadItem`` index fields seen by load-planner hooks, the warning
  ``dcp.load`` emits when it runs without a process group, and the
  ``CheckpointException`` (a ``BaseException``) it wraps every failure in.
* The write side of a custom ``SavePlanner``: ``WriteItem`` construction for
  partial tensor chunks (``SHARD`` items) and byte items, the global plan of
  ``create_default_global_save_plan``, which, unlike ``DefaultSavePlanner``,
  does not validate the chunk grid, and ``dcp.save`` without a process group.

The public, stable fields ``Metadata.state_dict_metadata`` and
``Metadata.planner_data`` are read directly by ``metadata.py``.
"""

from __future__ import annotations

import copy
import threading
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import CheckpointException
from torch.distributed.checkpoint.default_planner import create_default_global_save_plan
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import (
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    TensorWriteData,
    WriteItem,
    WriteItemType,
)
from torch.distributed.checkpoint.storage import StorageReader, StorageWriter

from ..errors import IntegrityError
from .regions import Region

_SAVER_PLANS_ATTRIBUTE = "all_local_plans"
_LEGACY_MCORE_ATTRIBUTE = "mcore_data"
_METADATA_FIELDS = ("state_dict_metadata", "planner_data", "storage_data", "storage_meta")
_NO_PROCESS_GROUP_WARNING = r"torch\.distributed is (disabled|unavailable)"
_WARNING_FILTER_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class StorageLocation:
    """Where one stored item lives: a byte range of a data file in the store.

    Attributes:
        relative_path: Data file name relative to the iteration directory.
        offset: Byte offset of the item in that file.
        length: Byte length of the item (a ``torch.save`` payload for tensors).
        transforms: Stream transform descriptors (empty for plain items).
    """

    relative_path: str
    offset: int
    length: int
    transforms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StorageRecord:
    """One entry of ``Metadata.storage_data``.

    Attributes:
        key: State-dict key (fqn) of the item.
        offsets: Global chunk offsets for tensor items, ``None`` for byte items.
        location: Decoded storage location.
        index: The original ``MetadataIndex`` (opaque outside this module).
        info: The original ``_StorageInfo`` (opaque outside this module).
    """

    key: str
    offsets: tuple[int, ...] | None
    location: StorageLocation
    index: Any
    info: Any


def storage_records(metadata: Metadata) -> list[StorageRecord]:
    """Decode every ``storage_data`` entry of ``metadata``."""
    records = []
    for index, info in (metadata.storage_data or {}).items():
        offsets = None if index.offset is None else tuple(int(o) for o in index.offset)
        location = StorageLocation(
            relative_path=str(info.relative_path),
            offset=int(info.offset),
            length=int(info.length),
            transforms=tuple(getattr(info, "transform_descriptors", None) or ()),
        )
        records.append(StorageRecord(index.fqn, offsets, location, index, info))
    return records


def metadata_version(metadata: Metadata) -> str | None:
    """The DCP format version recorded by the writer (``None`` before torch 2.9)."""
    return getattr(metadata, "version", None)


def legacy_mcore_data(metadata: Metadata) -> Mapping[str, Any] | None:
    """Old Megatron N-D reformulation data attached to the metadata, if any."""
    return getattr(metadata, _LEGACY_MCORE_ATTRIBUTE, None)


def drop_saver_plans(metadata: Metadata) -> bool:
    """Remove the saver-side ``all_local_plans`` attachment; return whether it existed."""
    return vars(metadata).pop(_SAVER_PLANS_ATTRIBUTE, None) is not None


PICKLED_DCP_CLASSES = frozenset(
    {
        ("torch.distributed.checkpoint.metadata", name)
        for name in (
            "BytesStorageMetadata",
            "ChunkStorageMetadata",
            "Metadata",
            "MetadataIndex",
            "StorageMeta",
            "TensorProperties",
            "TensorStorageMetadata",
            "_MEM_FORMAT_ENCODING",
        )
    }
    | {
        ("torch.distributed.checkpoint.planner", name)
        for name in (
            "BytesIOWriteData",
            "SavePlan",
            "TensorWriteData",
            "WriteItem",
            "WriteItemType",
        )
    }
    | {
        ("torch.distributed.checkpoint.filesystem", name)
        for name in ("_StorageInfo", "_StoragePrefix")
    }
)
"""The DCP classes (all dataclasses or enums) that a ``.metadata`` pickle may name.

``Metadata`` and its tables, plus the save-plan classes that savers attach as
``all_local_plans``. Every other global under ``torch.distributed.checkpoint``
(its functions, readers, writers and the modules they import) is refused.
"""


def metadata_from_fields(fields: Mapping[str, Any]) -> Metadata:
    """Rebuild a ``Metadata`` from the attribute dict of a legacy metadata object.

    Old Megatron saved ``.metadata`` as its own ``Metadata`` subclass; once that
    class is stubbed, only its attribute dict survives. Known fields become
    constructor arguments; the rest (for example ``version`` or ``mcore_data``)
    are kept as attributes, as ``pickle`` would have restored them.
    """
    known = {name: fields[name] for name in _METADATA_FIELDS if name in fields}
    metadata = Metadata(**known)
    for name, value in fields.items():
        if name not in known:
            setattr(metadata, name, value)
    return metadata


def with_tables(
    metadata: Metadata,
    *,
    state_dict_metadata: dict[str, Any],
    storage_data: dict[Any, Any],
    planner_data: Any,
) -> Metadata:
    """A shallow copy of ``metadata`` with its three per-key tables replaced.

    Every other attribute (``storage_meta``, ``version``, legacy attachments)
    is carried over unchanged, whatever the installed torch defines.
    """
    subset = copy.copy(metadata)
    subset.state_dict_metadata = state_dict_metadata
    subset.storage_data = storage_data
    subset.planner_data = planner_data
    return subset


class RegionTarget(torch.Tensor):
    """Load destination covering disjoint boxes of one tensor, each with its own buffer.

    ``dcp.load`` asks a checkpointable object for its local chunks, plans one
    read per (local chunk, stored chunk) overlap, and fetches the buffer of a
    local chunk through ``__get_tensor_shard__``. The object reports the full
    global shape and supports no tensor operations.
    """

    @staticmethod
    def __new__(
        cls, shape: Sequence[int], dtype: torch.dtype, parts: Sequence[tuple[Region, torch.Tensor]]
    ) -> RegionTarget:
        del parts
        return torch.Tensor._make_wrapper_subclass(  # pylint: disable=protected-access
            cls, torch.Size(shape), dtype=dtype, device="cpu"
        )

    def __init__(
        self, shape: Sequence[int], dtype: torch.dtype, parts: Sequence[tuple[Region, torch.Tensor]]
    ) -> None:
        del shape, dtype
        self._regions = tuple(region for region, _ in parts)
        self._buffers = {region.offsets: buffer for region, buffer in parts}

    def __create_chunk_list__(self) -> list[ChunkStorageMetadata]:
        """The boxes this target wants, as DCP chunk metadata."""
        return [
            ChunkStorageMetadata(offsets=torch.Size(r.offsets), sizes=torch.Size(r.sizes))
            for r in self._regions
        ]

    def __get_tensor_shard__(self, index: Any) -> torch.Tensor:
        """The buffer of the local chunk that starts at ``index.offset``."""
        return self._buffers[tuple(int(o) for o in index.offset)]

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise NotImplementedError(f"{cls.__name__} is a load target only; {func} is unsupported")


def read_item_key(read_item: ReadItem) -> str:
    """The state-dict key a ``ReadItem`` loads into."""
    return read_item.dest_index.fqn


def read_item_source(read_item: ReadItem) -> tuple[str, tuple[int, ...]]:
    """The stored chunk a tensor ``ReadItem`` reads: its key and global offsets."""
    index = read_item.storage_index
    return index.fqn, tuple(int(o) for o in index.offset)


def tensor_write_item(
    key: str, chunk: Region, global_shape: Sequence[int], dtype: torch.dtype
) -> WriteItem:
    """A write item storing box ``chunk`` of tensor ``key`` as one stored chunk.

    The item is of type ``SHARD``, so a save may hold any subset of a tensor's
    chunks (several items of one key are allowed).
    """
    return WriteItem(
        index=MetadataIndex(key, torch.Size(chunk.offsets)),
        type=WriteItemType.SHARD,
        tensor_data=TensorWriteData(
            chunk=ChunkStorageMetadata(
                offsets=torch.Size(chunk.offsets), sizes=torch.Size(chunk.sizes)
            ),
            properties=TensorProperties(dtype=dtype),
            size=torch.Size(global_shape),
        ),
    )


def bytes_write_item(key: str) -> WriteItem:
    """A write item storing the byte item ``key`` (an opaque payload)."""
    return WriteItem(index=MetadataIndex(key), type=WriteItemType.BYTE_IO)


def write_item_target(item: WriteItem) -> tuple[str, tuple[int, ...] | None]:
    """The key a write item stores and its chunk offsets (``None`` for byte items)."""
    if item.type == WriteItemType.BYTE_IO:
        return item.index.fqn, None
    return item.index.fqn, tuple(int(o) for o in item.index.offset)


def global_save_plan(plans: list[SavePlan]) -> tuple[list[SavePlan], Metadata]:
    """The global plan and metadata of ``plans``, without coverage validation.

    ``DefaultSavePlanner`` rejects a save whose tensors are not complete
    (``_validate_global_plan``); a part store holds partial tensors, so the
    engine validates coverage when it merges the parts instead.
    """
    return create_default_global_save_plan(plans, rewrite_index_hints=True)


def load_without_process_group(
    state_dict: dict[str, Any], reader: StorageReader, planner: LoadPlanner
) -> None:
    """``dcp.load`` in this process only, whether or not a process group exists.

    The first call adds one process-wide ``warnings`` filter that ignores the
    "torch.distributed is disabled" warning ``dcp.load(no_dist=True)`` emits;
    later calls leave the warning filters untouched.

    Raises:
        IntegrityError: DCP failed to read the store (for example a missing or
            truncated data file); the message names the underlying error.
        KeyboardInterrupt: The read was interrupted (DCP wraps the interrupt,
            like any other failure, in a ``CheckpointException``).
    """
    _ignore_no_process_group_warning()
    try:
        dcp.load(state_dict, storage_reader=reader, planner=planner, no_dist=True)
    except CheckpointException as err:
        raise _failure(err, f"reading {reader.checkpoint_id} failed") from err


def save_without_process_group(writer: StorageWriter, planner: SavePlanner) -> None:
    """``dcp.save({})`` in this process only; ``planner`` supplies everything to write.

    Raises:
        IntegrityError: DCP failed to write the store (for example a full disk).
        KeyboardInterrupt: The write was interrupted.
    """
    _ignore_no_process_group_warning()
    try:
        dcp.save({}, storage_writer=writer, planner=planner, no_dist=True)
    except CheckpointException as err:
        raise _failure(err, f"writing {writer.checkpoint_id} failed") from err


def _failure(err: CheckpointException, what: str) -> BaseException:
    """The ``KeyboardInterrupt`` DCP wrapped, else an ``IntegrityError`` naming the cause."""
    failure = next(iter(err.failures.values()), None)
    cause = failure[0] if isinstance(failure, tuple) else failure
    if isinstance(cause, KeyboardInterrupt):
        return cause
    return IntegrityError(f"{what}: {type(cause).__name__}: {cause}")


def _ignore_no_process_group_warning() -> None:
    with _WARNING_FILTER_LOCK:  # warnings.filters is process-global; readers run in threads
        if not any(_is_no_process_group_filter(f) for f in warnings.filters):
            warnings.filterwarnings(
                "ignore", message=_NO_PROCESS_GROUP_WARNING, category=UserWarning
            )


def _is_no_process_group_filter(entry: tuple[Any, ...]) -> bool:
    action, message, category = entry[:3]
    pattern = getattr(message, "pattern", None)
    return action == "ignore" and pattern == _NO_PROCESS_GROUP_WARNING and category is UserWarning
