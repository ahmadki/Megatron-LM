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

The public, stable fields ``Metadata.state_dict_metadata`` and
``Metadata.planner_data`` are read directly by ``metadata.py``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch.distributed.checkpoint.metadata import Metadata

_SAVER_PLANS_ATTRIBUTE = "all_local_plans"
_LEGACY_MCORE_ATTRIBUTE = "mcore_data"
_METADATA_FIELDS = ("state_dict_metadata", "planner_data", "storage_data", "storage_meta")


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
