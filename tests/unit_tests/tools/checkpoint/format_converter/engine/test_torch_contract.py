# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pins of the torch.distributed.checkpoint behaviours the engine relies on.

Each test states the engine assumption it protects. When one fails after a
torch upgrade, fix ``engine/torch_compat.py`` (the only module allowed to touch
these internals) rather than the test.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import importlib
import io
import pickle
from pathlib import Path

import pytest
import torch
import torch.distributed
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import (
    CheckpointException,
    DefaultLoadPlanner,
    DefaultSavePlanner,
    FileSystemReader,
    FileSystemWriter,
)
from torch.distributed.checkpoint import filesystem as dcp_filesystem
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex

from tools.checkpoint.format_converter.engine import torch_compat
from tools.checkpoint.format_converter.engine.metadata import (
    CachedMetadataFileSystemReader,
    MetadataView,
)
from tools.checkpoint.format_converter.engine.reader_dcp import RawBytesLoadPlanner
from tools.checkpoint.format_converter.engine.regions import Region
from tools.checkpoint.format_converter.errors import IntegrityError

from .._harness import REPO_ROOT
from .._synthetic import dim0_grid, torch_save_bytes, write_store

PACKAGE_DIR = REPO_ROOT / "tools" / "checkpoint" / "format_converter"


@pytest.fixture(name="store")
def fixture_store(tmp_path):
    tensors = {"w": torch.arange(64, dtype=torch.float32).reshape(16, 4)}
    write_store(
        tmp_path / "store",
        tensors,
        grids={"w": dim0_grid((16, 4), 8)},
        byte_items={"blob": torch_save_bytes([{"k": 1}])},
    )
    return tmp_path / "store", tensors


@pytest.fixture(name="decoded")
def fixture_decoded(monkeypatch):
    """Numel of every tensor ``torch.load`` decodes."""
    sizes = []
    real_load = torch.load

    def recording_load(*args, **kwargs):
        result = real_load(*args, **kwargs)
        sizes.append(result.numel() if isinstance(result, torch.Tensor) else None)
        return result

    monkeypatch.setattr(torch, "load", recording_load)
    return sizes


def _load(path: Path, state_dict: dict) -> RawBytesLoadPlanner:
    view = MetadataView.load(path)
    planner = RawBytesLoadPlanner()
    reader = CachedMetadataFileSystemReader(path, view.subset(state_dict))
    torch_compat.load_without_process_group(state_dict, reader, planner)
    return planner


def test_every_read_item_decodes_the_whole_stored_item(store, decoded):
    """Protects the read-once planning of reader_dcp.

    A 2-row box of an 8-row stored chunk still ``torch.load``s all 8 rows, and
    two boxes over the same chunk decode it twice. That is why reader_dcp asks
    for exactly one box per touched stored chunk.
    """
    path, tensors = store
    one = torch.empty(2, 4)
    _load(
        path,
        {"w": torch_compat.RegionTarget((16, 4), torch.float32, [(Region((1, 0), (2, 4)), one)])},
    )
    assert decoded == [32] and torch.equal(one, tensors["w"][1:3])
    decoded.clear()
    first, second = torch.empty(1, 4), torch.empty(1, 4)
    parts = [(Region((1, 0), (1, 4)), first), (Region((5, 0), (1, 4)), second)]
    _load(path, {"w": torch_compat.RegionTarget((16, 4), torch.float32, parts)})
    assert decoded == [32, 32]


def test_checkpointable_protocol_is_honoured(store):
    """Protects RegionTarget: DCP must plan reads from ``__create_chunk_list__``
    (one read per box and overlapping stored chunk) and fetch each box's buffer
    through ``__get_tensor_shard__`` with the box offsets."""
    path, tensors = store
    spanning, inside = torch.empty(4, 2), torch.empty(1, 4)
    parts = [(Region((6, 1), (4, 2)), spanning), (Region((12, 0), (1, 4)), inside)]
    planner = _load(path, {"w": torch_compat.RegionTarget((16, 4), torch.float32, parts)})
    assert sorted(planner.tensor_reads) == [("w", (0, 0)), ("w", (8, 0)), ("w", (8, 0))]
    assert torch.equal(spanning, tensors["w"][6:10, 1:3])
    assert torch.equal(inside, tensors["w"][12:13])


def test_plain_tensor_targets_must_have_the_global_shape(store):
    """Protects the whole-tensor fast path and the need for RegionTarget:
    a plain tensor is read as the full tensor and rejected if smaller."""
    path, _ = store
    view = MetadataView.load(path)
    with pytest.raises(CheckpointException, match="Size mismatch"):
        dcp.load(
            {"w": torch.empty(8, 4)},
            storage_reader=CachedMetadataFileSystemReader(path, view.subset(["w"])),
            planner=RawBytesLoadPlanner(),
            no_dist=True,
        )


def test_load_failures_are_wrapped_in_checkpoint_exception(store):
    """Protects load_without_process_group's error mapping: DCP wraps any failure
    of a load in ``CheckpointException``, a ``BaseException`` whose ``failures``
    maps ranks to ``(exception, traceback)``; it becomes an IntegrityError."""
    assert not issubclass(CheckpointException, Exception)
    path, _ = store
    with pytest.raises(IntegrityError, match="failed: ValueError: Size mismatch"):
        _load(path, {"w": torch.empty(8, 4)})


def test_load_bytes_hook_receives_the_raw_payload(store):
    """Protects RawBytesLoadPlanner: byte items reach ``load_bytes`` as a
    ``BytesIO`` of exactly the stored bytes, keyed by the item's fqn."""
    path, _ = store
    planner = _load(path, {"blob": None})
    assert planner.payloads == {"blob": torch_save_bytes([{"k": 1}])}
    assert planner.tensor_reads == []


def test_commit_tensor_runs_once_per_tensor_read(store, decoded):
    """Protects ReadStats: one ``commit_tensor`` per tensor ``ReadItem``, whose
    storage index names the stored chunk that was decoded."""
    path, _ = store
    planner = _load(path, {"w": torch.empty(16, 4)})
    assert sorted(planner.tensor_reads) == [("w", (0, 0)), ("w", (8, 0))]
    assert len(decoded) == len(planner.tensor_reads)


def test_metadata_and_storage_info_fields(store):
    """Protects torch_compat.storage_records / with_tables / metadata_version."""
    path, _ = store
    metadata = FileSystemReader(path).read_metadata()
    fields = {f.name for f in dataclasses.fields(Metadata)}
    assert {"state_dict_metadata", "planner_data", "storage_data", "storage_meta"} <= fields
    assert torch_compat.metadata_version(metadata) == getattr(metadata, "version", None)
    if "version" in fields:
        assert isinstance(torch_compat.metadata_version(metadata), str)
    info_fields = {f.name for f in dataclasses.fields(dcp_filesystem._StorageInfo)}
    assert {"relative_path", "offset", "length"} <= info_fields
    for index, info in metadata.storage_data.items():
        assert isinstance(index, MetadataIndex)
        assert isinstance(info.relative_path, str) and info.length > 0
    records = {(r.key, r.offsets): r.location for r in torch_compat.storage_records(metadata)}
    assert set(records) == {("w", (0, 0)), ("w", (8, 0)), ("blob", None)}
    subset = torch_compat.with_tables(
        metadata, state_dict_metadata={}, storage_data={}, planner_data=None
    )
    assert getattr(subset, "version", None) == getattr(metadata, "version", None)
    assert subset.storage_meta is metadata.storage_meta


def test_stores_written_by_this_torch_unpickle_safely(tmp_path):
    """Protects SafeMetadataUnpickler's allowlist: a store saved by the installed
    torch with its default planner (tensors of several dtypes, a 0-d tensor and
    byte items) must load without refusals or stubs."""
    state = {
        "a": torch.zeros(3, dtype=torch.bfloat16),
        "b": torch.ones(2, 2, dtype=torch.int64),
        "c": torch.tensor(True),
        "obj": {"nested": [1, 2.5, "x"]},
    }
    writer = FileSystemWriter(tmp_path, per_thread_copy_ahead=0)  # no CUDA staging streams
    dcp.save(state, storage_writer=writer, no_dist=True)
    view = MetadataView.load(tmp_path)
    assert view.stubbed == frozenset()
    assert set(view.tensors) == {"a", "b", "c"} and set(view.byte_items) == {"obj.nested"}


def test_saver_plan_attachments_unpickle_safely(tmp_path):
    """Protects the explicit list of DCP classes: the ``all_local_plans`` that the
    nvidia-resiliency-ext async saver attaches (save plans prepared by the
    installed torch's FileSystemWriter) must load without refusals or stubs."""
    state = {"a": torch.zeros(3, dtype=torch.bfloat16), "obj": {"k": 1}}
    writer = FileSystemWriter(tmp_path, per_thread_copy_ahead=0)
    dcp.save(state, storage_writer=writer, no_dist=True)
    planner = DefaultSavePlanner()
    planner.set_up_planner(state, is_coordinator=True)
    local_plan = writer.prepare_local_plan(planner.create_local_plan())
    global_plans, _ = planner.create_global_plan([local_plan])
    metadata = pickle.loads((tmp_path / ".metadata").read_bytes())
    metadata.all_local_plans = [local_plan, *writer.prepare_global_plan(global_plans)]
    (tmp_path / ".metadata").write_bytes(pickle.dumps(metadata))
    view = MetadataView.load(tmp_path)
    assert view.stubbed == frozenset() and set(view.tensors) == {"a"}


@pytest.mark.parametrize("module, name", sorted(torch_compat.PICKLED_DCP_CLASSES))
def test_allowed_dcp_globals_are_dataclasses_or_enums(module, name):
    """Protects SafeMetadataUnpickler: every allow-listed DCP global still exists
    and is a plain data class or enum, never a function or a class with behaviour."""
    resolved = getattr(importlib.import_module(module), name)
    assert isinstance(resolved, type) and resolved.__module__ == module
    assert dataclasses.is_dataclass(resolved) or issubclass(resolved, enum.Enum)


def test_cached_reader_signature_is_accepted(store):
    """Protects CachedMetadataFileSystemReader: ``read_metadata`` takes
    ``*args, **kwargs`` (torch >= 2.10 passes ``rank=`` for rank-local metadata)."""
    path, tensors = store
    view = MetadataView.load(path)
    target = {"w": torch.empty(16, 4)}
    dcp.load(
        target,
        storage_reader=CachedMetadataFileSystemReader(path, view.subset(["w"])),
        planner=DefaultLoadPlanner(),
        no_dist=True,
    )
    assert torch.equal(target["w"], tensors["w"])
    reader = CachedMetadataFileSystemReader(path, view.subset(["w"]))
    assert reader.read_metadata(rank=0) is reader.read_metadata()


def test_no_dist_load_uses_no_collectives_and_no_warning(store, monkeypatch, recwarn):
    """Protects load_without_process_group: with ``no_dist=True`` DCP must not
    touch any process group (one may exist in the CI launcher), and the
    'torch.distributed is disabled' warning is filtered."""

    def forbidden(*args, **kwargs):
        raise AssertionError("dcp.load(no_dist=True) used a collective")

    for name in (
        "all_gather_object",
        "broadcast_object_list",
        "scatter_object_list",
        "gather_object",
        "barrier",
        "all_reduce",
    ):
        monkeypatch.setattr(torch.distributed, name, forbidden)
    path, tensors = store
    target = torch.empty(16, 4)
    _load(path, {"w": target})
    assert torch.equal(target, tensors["w"])
    assert not [w for w in recwarn if "torch.distributed is" in str(w.message)]


_PRIVATE_ATTRIBUTES = frozenset(
    {
        "storage_data",
        "transform_descriptors",
        "all_local_plans",
        "_make_wrapper_subclass",
        "dest_index",
        "storage_index",
    }
)
_PROTOCOL_METHODS = frozenset({"__create_chunk_list__", "__get_tensor_shard__"})


def _private_uses(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("torch"):
            names = [a.name for a in node.names if a.name.startswith("_")]
            if names or any(part.startswith("_") for part in node.module.split(".")):
                found.append(f"line {node.lineno}: from {node.module} import {names}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "torch" and any(p.startswith("_") for p in parts):
                    found.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.Attribute) and node.attr in _PRIVATE_ATTRIBUTES:
            found.append(f"line {node.lineno}: .{node.attr}")
        elif isinstance(node, ast.FunctionDef) and node.name in _PROTOCOL_METHODS:
            found.append(f"line {node.lineno}: def {node.name}")
    return found


def test_private_dcp_access_is_confined_to_torch_compat():
    """Protects the containment rule: only engine/torch_compat.py may touch DCP
    internals, so a torch upgrade breaks exactly one module."""
    compat = PACKAGE_DIR / "engine" / "torch_compat.py"
    offenders = {
        str(path.relative_to(PACKAGE_DIR)): uses
        for path in sorted(PACKAGE_DIR.rglob("*.py"))
        if path != compat and (uses := _private_uses(path))
    }
    assert offenders == {}
    assert _private_uses(compat), "the scan no longer recognises torch_compat's own accesses"


def test_decoded_bytes_match_an_independent_reader(store):
    """Cross-checks the raw payload against a hand-rolled read of the byte range."""
    path, _ = store
    view = MetadataView.load(path)
    entry = view.byte_items["blob"]
    with open(path / entry.relative_path, "rb") as f:
        f.seek(entry.offset)
        raw = f.read(entry.length)
    assert _load(path, {"blob": None}).payloads["blob"] == raw
    assert torch.load(io.BytesIO(raw)) == [{"k": 1}]
