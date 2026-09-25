# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Host-safe view of a DCP store's ``.metadata`` and a reader that reuses it.

``.metadata`` is a plain pickle. Unpickling it with ``pickle.load`` would import
whatever classes it names, which on a host without Megatron fails (old stores
reference Megatron classes) and in general executes arbitrary code. The
:class:`SafeMetadataUnpickler` allows a fixed set of standard-library types, the
DCP metadata and save-plan classes listed in
:data:`torch_compat.PICKLED_DCP_CLASSES` (classes only, never functions), and
the tensor rebuild helpers ``torch.save`` payloads need. It replaces
``megatron.*`` and ``nvidia_resiliency_ext.*`` classes by inert :class:`Stub`
objects and refuses everything else.

A :class:`MetadataView` unpickles ``.metadata`` exactly once and exposes typed,
torch-light descriptions of every stored item. :class:`CachedMetadataFileSystemReader`
hands an already loaded ``Metadata`` to ``torch.distributed.checkpoint.load`` so
that repeated loads never unpickle ``.metadata`` again.
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import _compat_pickle
import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)

from ..errors import IntegrityError, InternalError
from . import torch_compat
from .regions import Region, find_tiling_defect

METADATA_FILE = ".metadata"

STUBBED_PACKAGES = ("megatron", "nvidia_resiliency_ext")
"""Top-level packages whose classes are replaced by :class:`Stub` when unpickling."""

_ALLOWED_GLOBALS = frozenset(
    {
        ("builtins", name)
        for name in (
            "bool",
            "bytearray",
            "bytes",
            "complex",
            "dict",
            "float",
            "frozenset",
            "int",
            "list",
            "object",
            "range",
            "set",
            "slice",
            "str",
            "tuple",
        )
    }
    | {("collections", name) for name in ("Counter", "OrderedDict", "defaultdict", "deque")}
    | {("pathlib", name) for name in ("Path", "PosixPath", "PurePath", "PurePosixPath")}
    | {("pathlib._local", name) for name in ("Path", "PosixPath", "PurePath", "PurePosixPath")}
    | {("datetime", name) for name in ("date", "datetime", "time", "timedelta", "timezone")}
    | {
        ("_codecs", "encode"),
        ("argparse", "Namespace"),
        ("copyreg", "_reconstructor"),
        ("types", "SimpleNamespace"),
        ("uuid", "UUID"),
        ("torch", "Size"),
        ("torch", "device"),
        ("torch.serialization", "_get_layout"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch._utils", "_rebuild_parameter"),
        ("torch._utils", "_rebuild_parameter_with_state"),
        ("torch._tensor", "_rebuild_from_type_v2"),
    }
)
_TORCH_VALUE_TYPES = (torch.dtype, torch.layout, torch.memory_format)


class Stub:
    """Inert stand-in for an instance of a class that is not imported on this host.

    It records how pickle built the object (constructor arguments, ``BUILD``
    state, appended items and set items) and supports nothing else.

    Attributes:
        args: Positional constructor arguments (``REDUCE``/``NEWOBJ``).
        kwargs: Keyword constructor arguments (``NEWOBJ_EX``).
        state: The ``BUILD`` state, typically the instance ``__dict__``.
        items: Items added with ``APPEND``/``APPENDS`` (list subclasses).
        entries: Items added with ``SETITEM(S)`` (dict subclasses).
    """

    stub_module: str = ""
    stub_name: str = ""
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = MappingProxyType({})
    state: Any = None

    def __new__(cls, *args: Any, **kwargs: Any) -> Stub:
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = MappingProxyType(kwargs)
        return instance

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs  # captured by __new__, which pickle may call on its own

    def __setstate__(self, state: Any) -> None:
        self.state = state

    @property
    def items(self) -> list[Any]:
        """Items added by pickle with ``APPEND``/``APPENDS``."""
        return vars(self).setdefault("_items", [])

    @property
    def entries(self) -> dict[Any, Any]:
        """Items added by pickle with ``SETITEM``/``SETITEMS``."""
        return vars(self).setdefault("_entries", {})

    def append(self, item: Any) -> None:
        """Record an item appended by pickle (``APPEND``)."""
        self.items.append(item)

    def extend(self, items: Iterable[Any]) -> None:
        """Record items appended by pickle (``APPENDS``)."""
        self.items.extend(items)

    def __setitem__(self, key: Any, value: Any) -> None:
        self.entries[key] = value

    @property
    def qualified_name(self) -> str:
        """``module.Class`` of the stubbed class."""
        return f"{self.stub_module}.{self.stub_name}"

    def __repr__(self) -> str:
        args = ", ".join(repr(a) for a in self.args)
        return f"<stub {self.qualified_name}({args})>"


_STUB_CLASSES: dict[tuple[str, str], type[Stub]] = {}


def _stub_class(module: str, name: str) -> type[Stub]:
    key = (module, name)
    if key not in _STUB_CLASSES:
        _STUB_CLASSES[key] = type(
            name.replace(".", "_"), (Stub,), {"stub_module": module, "stub_name": name}
        )
    return _STUB_CLASSES[key]


def _python3_name(module: str, name: str) -> tuple[str, str]:
    """Map Python 2 names that protocol 0-2 pickles may use (``__builtin__.set``)."""
    if (module, name) in _compat_pickle.NAME_MAPPING:
        return _compat_pickle.NAME_MAPPING[(module, name)]
    return _compat_pickle.IMPORT_MAPPING.get(module, module), name


def _is_allowed_global(module: str, name: str) -> bool:
    """Whether ``module.name`` may be imported: an exact allowlist entry, never a dotted path.

    ``pickle`` resolves a dotted ``name`` attribute by attribute from ``module``,
    so ``name`` must be a plain identifier to name the object that was checked.
    """
    if "." in name:
        return False
    if (module, name) in _ALLOWED_GLOBALS or _is_dcp_class(module, name):
        return True
    return module == "torch" and isinstance(getattr(torch, name, None), _TORCH_VALUE_TYPES)


def _is_dcp_class(module: str, name: str) -> bool:
    return (module, name) in torch_compat.PICKLED_DCP_CLASSES


class SafeMetadataUnpickler(pickle.Unpickler):
    """Unpickler for DCP metadata and small common-state payloads.

    Args:
        file: Binary file to read the pickle from.
        stub_unknown: Replace every global that is not allow-listed by a stub
            instead of refusing it (for display-only decoding).
        **kwargs: Passed to ``pickle.Unpickler``.

    Attributes:
        stubbed: Qualified names of the classes that were replaced by stubs.
    """

    def __init__(self, file: Any, *, stub_unknown: bool = False, **kwargs: Any) -> None:
        super().__init__(file, **kwargs)
        self.stub_unknown = stub_unknown
        self.stubbed: set[str] = set()

    def find_class(self, module: str, name: str) -> Any:
        """Resolve a global: allow-listed, stubbed, or refused."""
        module, name = _python3_name(module, name)
        if _is_allowed_global(module, name):
            resolved = super().find_class(module, name)
            if _is_dcp_class(module, name) and not isinstance(resolved, type):
                raise pickle.UnpicklingError(f"refusing to unpickle non-class {module}.{name}")
            return resolved
        if self.stub_unknown or module.split(".")[0] in STUBBED_PACKAGES:
            self.stubbed.add(f"{module}.{name}")
            return _stub_class(module, name)
        raise pickle.UnpicklingError(f"refusing to unpickle global {module}.{name}")


@dataclass(frozen=True)
class DecodedPayload:
    """A byte-item payload decoded for display.

    Attributes:
        value: The decoded object; unknown classes are :class:`Stub` objects.
        stubbed: Qualified names of the classes that were stubbed.
    """

    value: Any
    stubbed: frozenset[str]


def decode_payload(payload: bytes, *, label: str) -> DecodedPayload:
    """Decode a ``torch.save`` payload without importing or calling unknown globals.

    Every global outside the allowlist becomes an inert :class:`Stub`, so the
    result is safe to display but must not be written back.

    Raises:
        IntegrityError: The payload is not a readable ``torch.save`` blob.
    """
    stubbed: set[str] = set()

    class _Unpickler(SafeMetadataUnpickler):
        def __init__(self, file: Any, **kwargs: Any) -> None:
            super().__init__(file, stub_unknown=True, **kwargs)
            self.stubbed = stubbed

    pickle_module = SimpleNamespace(
        __name__="format_converter_stub_pickle",
        Unpickler=_Unpickler,
        load=lambda file, **kwargs: _Unpickler(file, **kwargs).load(),
    )
    try:
        value = torch.load(
            io.BytesIO(payload), map_location="cpu", weights_only=False, pickle_module=pickle_module
        )
    except Exception as err:  # torch.load of untrusted bytes can fail in any way
        raise IntegrityError(f"cannot decode {label}: {type(err).__name__}: {err}") from err
    return DecodedPayload(value, frozenset(stubbed))


@dataclass(frozen=True, slots=True)
class StoredChunk:
    """One stored chunk of a tensor: its box in the tensor and its bytes on disk.

    Attributes:
        key: Tensor key.
        offsets: Global offsets of the chunk.
        sizes: Chunk extent.
        dtype: Element type.
        relative_path: Data file relative to the iteration directory.
        offset: Byte offset of the stored item in that file.
        length: Byte length of the stored item.
        transforms: Stream transform descriptors (empty for plain items).
    """

    key: str
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]
    dtype: torch.dtype
    relative_path: str
    offset: int
    length: int
    transforms: tuple[str, ...]

    @property
    def region(self) -> Region:
        """The chunk's box in global coordinates."""
        return Region(self.offsets, self.sizes)

    @property
    def nbytes(self) -> int:
        """Decoded size of the chunk in memory."""
        return self.region.numel() * self.dtype.itemsize


@dataclass(frozen=True, slots=True)
class TensorEntry:
    """A tensor of the store with its chunk grid (chunks sorted by offsets).

    Attributes:
        key: Tensor key.
        shape: Global shape.
        dtype: Element type.
        chunks: Stored chunks; they tile ``shape`` exactly (checked at load).
    """

    key: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    chunks: tuple[StoredChunk, ...]

    @property
    def numel(self) -> int:
        """Number of elements of the full tensor."""
        return Region.full(self.shape).numel()

    @property
    def nbytes(self) -> int:
        """Size of the full tensor in memory."""
        return self.numel * self.dtype.itemsize


@dataclass(frozen=True, slots=True)
class ByteEntry:
    """A byte item of the store (an opaque payload such as a ``torch.save`` blob).

    Attributes:
        key: Item key.
        relative_path: Data file relative to the iteration directory.
        offset: Byte offset in that file.
        length: Payload length.
    """

    key: str
    relative_path: str
    offset: int
    length: int


class MetadataView:
    """Typed, host-safe description of one DCP store, built from one unpickle.

    Attributes:
        iter_dir: The store directory (the one holding ``.metadata``).
        tensors: Tensor entries by key.
        byte_items: Byte-item entries by key.
        planner_data: ``Metadata.planner_data`` (``None`` for torch_dist stores).
        mcore_data: Legacy Megatron ``mcore_data`` attachment, if present.
        version: DCP format version written by newer torch, if present.
        stubbed: Qualified names of classes replaced by :class:`Stub` while loading.
        metadata_sha256: SHA-256 of the ``.metadata`` file as read.
    """

    def __init__(
        self, iter_dir: Path, metadata: Metadata, *, stubbed: frozenset[str], metadata_sha256: str
    ) -> None:
        self.iter_dir = iter_dir
        self.stubbed = stubbed
        self.metadata_sha256 = metadata_sha256
        self.planner_data = metadata.planner_data
        self.mcore_data = torch_compat.legacy_mcore_data(metadata)
        self.version = torch_compat.metadata_version(metadata)
        self._metadata = metadata
        self._records = _group_records(torch_compat.storage_records(metadata))
        self.tensors, self.byte_items = _build_entries(metadata, self._records, iter_dir)

    @classmethod
    def load(cls, iter_dir: Path) -> MetadataView:
        """Read and unpickle ``iter_dir/.metadata`` once, safely.

        Raises:
            IntegrityError: The file is missing, unreadable, not a DCP metadata
                pickle, references a forbidden class, or describes a chunk grid
                that does not tile its tensor.
        """
        loaded = read_metadata_file(Path(iter_dir))
        return cls(
            Path(iter_dir), loaded.metadata, stubbed=loaded.stubbed, metadata_sha256=loaded.sha256
        )

    def fingerprint(self) -> str:
        """Identity of the store's current content.

        SHA-256 over the ``.metadata`` digest and the name, size and mtime of
        every regular file in the store directory; it changes when any file of
        the store is rewritten, truncated or touched.
        """
        digest = hashlib.sha256(self.metadata_sha256.encode())
        for entry in sorted(os.scandir(self.iter_dir), key=lambda e: e.name):
            if entry.is_file():
                stat = entry.stat()
                digest.update(f"{entry.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        return digest.hexdigest()

    @classmethod
    def of_metadata(cls, iter_dir: Path, metadata: Metadata) -> MetadataView:
        """A view of an already loaded ``Metadata`` (for example a unit's subset).

        The view has no file digest, so :meth:`fingerprint` must not be used.

        Raises:
            IntegrityError: A chunk grid does not tile its tensor.
        """
        return cls(Path(iter_dir), metadata, stubbed=frozenset(), metadata_sha256="")

    def subset(self, keys: Iterable[str], *, tables_only: bool = False) -> Metadata:
        """A ``Metadata`` restricted to ``keys``.

        Args:
            keys: Keys to keep (every chunk of each).
            tables_only: Build a new ``Metadata`` holding only the three
                per-key tables, safe to pickle into a unit file; otherwise every
                other field and attachment of the store's metadata is kept.

        Raises:
            InternalError: A key is not in the store.
        """
        wanted = sorted(set(keys))
        unknown = [k for k in wanted if k not in self._records]
        if unknown:
            raise InternalError(f"subset() of {self.iter_dir}: unknown keys {unknown[:8]}")
        planner_data = self.planner_data
        if isinstance(planner_data, Mapping):
            planner_data = {k: planner_data[k] for k in wanted if k in planner_data}
        state_dict_metadata = {k: self._metadata.state_dict_metadata[k] for k in wanted}
        storage_data = {r.index: r.info for k in wanted for r in self._records[k]}
        if tables_only:
            return torch_compat.tables_only(state_dict_metadata, storage_data, planner_data)
        return torch_compat.with_tables(
            self._metadata,
            state_dict_metadata=state_dict_metadata,
            storage_data=storage_data,
            planner_data=planner_data,
        )


@dataclass(frozen=True)
class LoadedMetadata:
    """A ``.metadata`` file unpickled with :class:`SafeMetadataUnpickler`.

    Attributes:
        metadata: The DCP metadata (saver plan attachments dropped).
        stubbed: Qualified names of the classes replaced by stubs.
        sha256: SHA-256 of the file as read.
    """

    metadata: Metadata
    stubbed: frozenset[str]
    sha256: str


def read_metadata_file(iter_dir: Path) -> LoadedMetadata:
    """Read and safely unpickle ``iter_dir/.metadata``, without validating its tables.

    Raises:
        IntegrityError: The file is missing, unreadable, not a DCP metadata
            pickle, or references a forbidden class.
    """
    path = iter_dir / METADATA_FILE
    try:
        payload = path.read_bytes()
    except OSError as err:
        raise IntegrityError(f"cannot read {path}: {err.strerror or err}") from err
    unpickler = SafeMetadataUnpickler(io.BytesIO(payload))
    try:
        loaded = unpickler.load()
    except Exception as err:  # parsing untrusted bytes can fail in any way: corrupt file
        raise IntegrityError(f"cannot unpickle {path}: {type(err).__name__}: {err}") from err
    metadata = _coerce_metadata(loaded, path)
    torch_compat.drop_saver_plans(metadata)
    return LoadedMetadata(
        metadata, frozenset(unpickler.stubbed), hashlib.sha256(payload).hexdigest()
    )


class CachedMetadataFileSystemReader(FileSystemReader):
    """A ``FileSystemReader`` that serves an already loaded ``Metadata``.

    ``torch.distributed.checkpoint.load`` calls ``read_metadata`` on every call;
    the stock reader unpickles ``.metadata`` each time (and would import every
    class it names). This reader returns the given object instead.
    """

    def __init__(self, path: Path, metadata: Metadata) -> None:
        super().__init__(path)
        self._metadata = metadata

    def read_metadata(self, *args: Any, **kwargs: Any) -> Metadata:
        """Return the cached metadata (rank-local variants are never used)."""
        del args, kwargs
        return self._metadata


def _coerce_metadata(loaded: Any, path: Path) -> Metadata:
    if isinstance(loaded, Metadata):
        return loaded
    state = getattr(loaded, "state", None)
    if isinstance(loaded, Stub) and isinstance(state, dict) and "state_dict_metadata" in state:
        return torch_compat.metadata_from_fields(state)
    raise IntegrityError(f"{path} does not hold DCP Metadata (found {type(loaded).__name__})")


def _group_records(
    records: list[torch_compat.StorageRecord],
) -> dict[str, tuple[torch_compat.StorageRecord, ...]]:
    grouped: dict[str, list[torch_compat.StorageRecord]] = {}
    for record in records:
        grouped.setdefault(record.key, []).append(record)
    return {key: tuple(items) for key, items in grouped.items()}


def _build_entries(
    metadata: Metadata,
    records: Mapping[str, tuple[torch_compat.StorageRecord, ...]],
    iter_dir: Path,
) -> tuple[dict[str, TensorEntry], dict[str, ByteEntry]]:
    tensors: dict[str, TensorEntry] = {}
    byte_items: dict[str, ByteEntry] = {}
    for key, item in metadata.state_dict_metadata.items():
        locations = {r.offsets: r.location for r in records.get(key, ())}
        if isinstance(item, BytesStorageMetadata):
            byte_items[key] = _byte_entry(key, locations, iter_dir)
        elif isinstance(item, TensorStorageMetadata):
            tensors[key] = _tensor_entry(key, item, locations, iter_dir)
        else:
            raise IntegrityError(f"{iter_dir}: key {key} has unknown storage type {item!r}")
    return tensors, byte_items


def _byte_entry(
    key: str, locations: Mapping[Any, torch_compat.StorageLocation], iter_dir: Path
) -> ByteEntry:
    location = locations.get(None)
    if location is None:
        raise IntegrityError(f"{iter_dir}: byte item {key} has no storage entry")
    return ByteEntry(key, location.relative_path, location.offset, location.length)


def _tensor_entry(
    key: str,
    item: TensorStorageMetadata,
    locations: Mapping[Any, torch_compat.StorageLocation],
    iter_dir: Path,
) -> TensorEntry:
    shape = tuple(int(s) for s in item.size)
    dtype = item.properties.dtype
    chunks = []
    for chunk in item.chunks:
        offsets = tuple(int(o) for o in chunk.offsets)
        location = locations.get(offsets)
        if location is None:
            raise IntegrityError(f"{iter_dir}: chunk {offsets} of {key} has no storage entry")
        chunks.append(
            StoredChunk(
                key=key,
                offsets=offsets,
                sizes=tuple(int(s) for s in chunk.sizes),
                dtype=dtype,
                relative_path=location.relative_path,
                offset=location.offset,
                length=location.length,
                transforms=location.transforms,
            )
        )
    chunks.sort(key=lambda c: c.offsets)
    defect = find_tiling_defect([c.region for c in chunks], shape)
    if defect is not None:
        raise IntegrityError(f"{iter_dir}: chunk grid of {key} is invalid: {defect}")
    return TensorEntry(key, shape, dtype, tuple(chunks))
