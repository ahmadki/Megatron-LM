# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Host-safe metadata view: typed entries, stubs, refusals, subsets and caching."""

from __future__ import annotations

import importlib
import io
import os
import pickle
import re
import types

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner
from torch.distributed.checkpoint import filesystem as dcp_filesystem

from tools.checkpoint.format_converter.engine import metadata as metadata_module
from tools.checkpoint.format_converter.engine.metadata import (
    CachedMetadataFileSystemReader,
    MetadataView,
    Stub,
)
from tools.checkpoint.format_converter.engine.regions import Region
from tools.checkpoint.format_converter.errors import IntegrityError, InternalError

from .._synthetic import (
    AliasPickler,
    dim0_grid,
    product_grid,
    rewrite_metadata,
    torch_save_bytes,
    write_store,
)

MEGATRON_TORCH_STRATEGY = "megatron.core.dist_checkpointing.strategies.torch"


class FakeSavePlan:
    """Stands in for a Megatron class that only exists where megatron is installed."""

    def __init__(self) -> None:
        self.payload = {"note": "saver side"}


class FakeLegacyMetadata:
    """Stands in for the metadata class old Megatron versions pickled."""


class Trap:
    """Pickles as a call to ``os.system``; unpickling it must never run it."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


GADGETS = [
    ("builtins", "eval"),
    ("builtins", "exec"),
    ("builtins", "getattr"),
    ("builtins", "__import__"),
    ("builtins", "open"),
    ("os", "system"),
    ("subprocess", "Popen"),
    ("torch", "load"),
    ("torch", "save"),
    ("torch.hub", "load"),
    # a dotted name walks out of an allowed module to whatever it imports
    ("torch.distributed.checkpoint.filesystem", "os.system"),
    # DCP functions and classes outside the explicit list of metadata classes
    ("torch.distributed.checkpoint.format_utils", "torch_save_to_dcp"),
    ("torch.distributed.checkpoint.filesystem", "FileSystemWriter"),
]
"""Globals a crafted pickle may name to run code; each must be refused (or stubbed)."""

SENTINEL = "format-converter-gadget-argument"


class Call:
    """Pickles as ``func(*args)``; tests write ``func`` under a chosen global name."""

    def __init__(self, func, *args) -> None:
        self.func = func
        self.args = args

    def __reduce__(self):
        return (self.func, self.args)


class Placeholder:
    """Written under the gadget's global name (a class, so ``AliasPickler`` renames it)."""


@pytest.fixture(name="gadget")
def fixture_gadget(request, monkeypatch, tmp_path):
    """``(module, name, pickle-able call, marker)``; calling the real global creates ``marker``.

    The real global is replaced by a spy that creates ``marker`` when it gets
    the sentinel argument and otherwise behaves as the original, so a pickle
    that manages to import and call it leaves a trace without running anything.
    """
    module, name = request.param
    marker = tmp_path / "gadget-fired"
    owner = importlib.import_module(module)
    *path, attribute = name.split(".")
    for part in path:
        owner = getattr(owner, part)
    real = getattr(owner, attribute)

    def spy(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0] == SENTINEL:
            marker.touch()
            return None
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, attribute, spy)
    return module, name, Call(Placeholder, SENTINEL), marker


def _gadget_pickle(call: Call, module: str, name: str) -> bytes:
    """Protocol 4, where ``pickle`` resolves dotted global names attribute by attribute."""
    buffer = io.BytesIO()
    AliasPickler(buffer, {Placeholder: (module, name)}, protocol=4).dump(call)
    return buffer.getvalue()


def _tensors() -> dict[str, torch.Tensor]:
    return {
        "w": torch.arange(40, dtype=torch.float32).reshape(10, 4),
        "g": torch.randn(6, 4).to(torch.bfloat16),
        "s": torch.tensor(7, dtype=torch.int64),
        "flag": torch.tensor([True, False, True, True, False]),
    }


@pytest.fixture(name="store")
def fixture_store(tmp_path):
    tensors = _tensors()
    write_store(
        tmp_path / "store",
        tensors,
        grids={"w": dim0_grid((10, 4), 4), "g": product_grid((6, 4), [[3], [2]])},
        byte_items={"blob": torch_save_bytes({"answer": 42})},
        planner_data={"w": ("w",), "blob": ("blob",)},
    )
    return tmp_path / "store", tensors


def test_view_describes_every_item(store):
    path, _ = store
    view = MetadataView.load(path)
    assert set(view.tensors) == {"w", "g", "s", "flag"} and set(view.byte_items) == {"blob"}
    w = view.tensors["w"]
    assert (w.shape, w.dtype, w.nbytes) == ((10, 4), torch.float32, 160)
    assert [c.region for c in w.chunks] == list(dim0_grid((10, 4), 4))
    assert len(view.tensors["g"].chunks) == 4 and view.tensors["g"].dtype == torch.bfloat16
    assert view.tensors["s"].shape == () and view.tensors["s"].chunks[0].region == Region((), ())
    assert view.planner_data == {"w": ("w",), "blob": ("blob",)}
    assert view.stubbed == frozenset() and view.mcore_data is None
    assert view.byte_items["blob"].length > 0


def test_storage_locations_point_at_the_stored_chunks(store):
    """Independent oracle: read each item's byte range by hand and torch.load it."""
    path, tensors = store
    view = MetadataView.load(path)
    for entry in view.tensors.values():
        for chunk in entry.chunks:
            with open(path / chunk.relative_path, "rb") as f:
                f.seek(chunk.offset)
                payload = f.read(chunk.length)
            loaded = torch.load(io.BytesIO(payload), weights_only=True)
            assert torch.equal(loaded, tensors[entry.key][chunk.region.slices()])
    blob = view.byte_items["blob"]
    with open(path / blob.relative_path, "rb") as f:
        f.seek(blob.offset)
        assert torch.load(io.BytesIO(f.read(blob.length))) == {"answer": 42}


def test_megatron_classes_are_stubbed_and_saver_plans_dropped(store):
    path, _ = store

    def mutate(md):
        md.all_local_plans = [FakeSavePlan()]
        md.mcore_data = {"w": {"shape": (10, 4), "tags": {"a", "b"}}}

    rewrite_metadata(path, mutate, {FakeSavePlan: (MEGATRON_TORCH_STRATEGY, "MCoreSavePlan")})
    view = MetadataView.load(path)
    assert view.stubbed == {f"{MEGATRON_TORCH_STRATEGY}.MCoreSavePlan"}
    assert view.mcore_data == {"w": {"shape": (10, 4), "tags": {"a", "b"}}}
    assert "all_local_plans" not in vars(view.subset(["w"]))
    assert set(view.tensors) == {"w", "g", "s", "flag"}


def test_stub_records_how_pickle_built_the_object(store):
    path, _ = store
    rewrite_metadata(
        path,
        lambda md: setattr(md, "all_local_plans", [FakeSavePlan()]),
        {FakeSavePlan: ("nvidia_resiliency_ext.checkpointing", "Plan")},
    )
    unpickler = metadata_module.SafeMetadataUnpickler(io.BytesIO((path / ".metadata").read_bytes()))
    stub = unpickler.load().all_local_plans[0]
    assert isinstance(stub, Stub)
    assert stub.qualified_name == "nvidia_resiliency_ext.checkpointing.Plan"
    assert stub.state == {"payload": {"note": "saver side"}}
    assert repr(stub) == "<stub nvidia_resiliency_ext.checkpointing.Plan()>"


def test_legacy_megatron_metadata_root_is_coerced(store):
    path, _ = store

    def mutate(md):
        legacy = FakeLegacyMetadata()
        legacy.__dict__.update(vars(md), mcore_data={"w": {}})
        return legacy

    rewrite_metadata(path, mutate, {FakeLegacyMetadata: (MEGATRON_TORCH_STRATEGY, "MCoreMetadata")})
    view = MetadataView.load(path)
    assert f"{MEGATRON_TORCH_STRATEGY}.MCoreMetadata" in view.stubbed
    assert view.mcore_data == {"w": {}}
    assert [c.region for c in view.tensors["w"].chunks] == list(dim0_grid((10, 4), 4))


def test_forbidden_global_is_refused_and_never_called(store, tmp_path):
    path, _ = store
    marker = tmp_path / "trap-fired"
    rewrite_metadata(
        path,
        lambda md: setattr(md, "planner_data", Trap(str(marker))),
        {os.system: ("os", "system")},
    )
    with pytest.raises(IntegrityError, match="refusing to unpickle global os.system"):
        MetadataView.load(path)
    assert not marker.exists()


@pytest.mark.parametrize("gadget", GADGETS, indirect=True, ids=lambda g: ".".join(g))
def test_code_execution_gadgets_are_refused(gadget, tmp_path):
    module, name, call, marker = gadget
    (tmp_path / ".metadata").write_bytes(_gadget_pickle(call, module, name))
    with pytest.raises(
        IntegrityError, match=re.escape(f"refusing to unpickle global {module}.{name}")
    ):
        MetadataView.load(tmp_path)
    assert not marker.exists()


def test_allowed_non_classes_are_only_rebuild_helpers():
    """Everything allow-listed is a class or value, except these pickle and torch.save helpers."""
    callables = set()
    for module, name in metadata_module._ALLOWED_GLOBALS:  # pylint: disable=protected-access
        try:
            resolved = getattr(importlib.import_module(module), name)
        except ImportError:  # pathlib._local exists on Python 3.13 and newer only
            continue
        if not isinstance(resolved, type):
            callables.add((module, name))
    assert callables == {
        ("_codecs", "encode"),
        ("copyreg", "_reconstructor"),
        ("torch.serialization", "_get_layout"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch._utils", "_rebuild_parameter"),
        ("torch._utils", "_rebuild_parameter_with_state"),
        ("torch._tensor", "_rebuild_from_type_v2"),
    }


def test_subset_restricts_the_tables(store):
    path, _ = store
    view = MetadataView.load(path)
    subset = view.subset(["w", "blob"])
    assert set(subset.state_dict_metadata) == {"w", "blob"}
    assert sorted(index.fqn for index in subset.storage_data) == ["blob", "w", "w", "w"]
    assert subset.planner_data == {"w": ("w",), "blob": ("blob",)}
    assert getattr(subset, "version", None) == view.version
    assert len(view.subset(["g"]).storage_data) == 4
    with pytest.raises(InternalError, match="unknown keys"):
        view.subset(["w", "missing"])


def test_cached_reader_never_unpickles_again(store, monkeypatch):
    path, tensors = store
    calls = []
    real_load = metadata_module.SafeMetadataUnpickler.load

    def counting_load(self):
        calls.append(1)
        return real_load(self)

    monkeypatch.setattr(metadata_module.SafeMetadataUnpickler, "load", counting_load)
    view = MetadataView.load(path)
    assert len(calls) == 1

    class NoPickle:
        @staticmethod
        def load(_file):
            raise AssertionError("DCP unpickled .metadata although it was cached")

    monkeypatch.setattr(dcp_filesystem, "pickle", NoPickle)
    for _ in range(3):
        target = {"w": torch.empty(10, 4)}
        dcp.load(
            target,
            storage_reader=CachedMetadataFileSystemReader(path, view.subset(["w"])),
            planner=DefaultLoadPlanner(),
            no_dist=True,
        )
        assert torch.equal(target["w"], tensors["w"])
    assert len(calls) == 1


def test_fingerprint_tracks_file_size_and_mtime(store):
    path, _ = store
    view = MetadataView.load(path)
    first = view.fingerprint()
    assert view.fingerprint() == first
    data_file = next(p for p in path.iterdir() if p.suffix == ".distcp")
    stat = data_file.stat()
    os.utime(data_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    touched = view.fingerprint()
    assert touched != first
    with open(data_file, "ab") as f:
        f.write(b"\0")
    assert view.fingerprint() not in (first, touched)


def test_chunk_grid_gap_is_an_integrity_error(store):
    path, _ = store

    def mutate(md):
        del md.state_dict_metadata["w"].chunks[-1]

    rewrite_metadata(path, mutate)
    with pytest.raises(IntegrityError, match=r"chunk grid of w is invalid: chunks cover 32 of 40"):
        MetadataView.load(path)


def test_missing_storage_entry_is_an_integrity_error(store):
    path, _ = store

    def mutate(md):
        index = next(i for i in md.storage_data if i.fqn == "g")
        del md.storage_data[index]

    rewrite_metadata(path, mutate)
    with pytest.raises(IntegrityError, match=r"chunk \(0, 0\) of g has no storage entry"):
        MetadataView.load(path)


def test_missing_or_foreign_metadata_files(tmp_path):
    with pytest.raises(IntegrityError, match="cannot read"):
        MetadataView.load(tmp_path)
    (tmp_path / ".metadata").write_bytes(b"not a pickle")
    with pytest.raises(IntegrityError, match="cannot unpickle"):
        MetadataView.load(tmp_path)
    whole = pickle.dumps({"state_dict_metadata": {}})
    (tmp_path / ".metadata").write_bytes(whole[: len(whole) // 2])
    with pytest.raises(IntegrityError, match="cannot unpickle"):
        MetadataView.load(tmp_path)
    (tmp_path / ".metadata").write_bytes(pickle.dumps({"state_dict_metadata": {}}))
    with pytest.raises(IntegrityError, match="does not hold DCP Metadata"):
        MetadataView.load(tmp_path)


def test_decode_payload_stubs_every_unknown_global(tmp_path):
    marker = tmp_path / "trap-fired"
    payload = torch_save_bytes(
        {"trap": Trap(str(marker)), "plan": FakeSavePlan(), "x": torch.arange(3)},
        {os.system: ("os", "system"), FakeSavePlan: (MEGATRON_TORCH_STRATEGY, "MCoreSavePlan")},
    )
    decoded = metadata_module.decode_payload(payload, label="test payload")
    assert not marker.exists()
    trap = decoded.value["trap"]
    assert isinstance(trap, Stub) and trap.qualified_name == "os.system"
    assert trap.args == (f"touch {marker}",)
    assert decoded.value["plan"].state == {"payload": {"note": "saver side"}}
    assert torch.equal(decoded.value["x"], torch.arange(3))
    assert decoded.stubbed == {"os.system", f"{MEGATRON_TORCH_STRATEGY}.MCoreSavePlan"}
    with pytest.raises(IntegrityError, match="cannot decode test payload"):
        metadata_module.decode_payload(b"garbage", label="test payload")


@pytest.mark.parametrize("gadget", GADGETS, indirect=True, ids=lambda g: ".".join(g))
def test_decode_payload_stubs_code_execution_gadgets(gadget):
    module, name, call, marker = gadget

    class GadgetPickler(AliasPickler):
        def __init__(self, file, protocol: int = 4) -> None:
            super().__init__(file, {Placeholder: (module, name)}, protocol=protocol)

    buffer = io.BytesIO()
    pickle_module = types.SimpleNamespace(__name__="gadget_pickle", Pickler=GadgetPickler)
    torch.save({"value": call}, buffer, pickle_module=pickle_module, pickle_protocol=4)
    decoded = metadata_module.decode_payload(buffer.getvalue(), label="gadget")
    value = decoded.value["value"]
    assert isinstance(value, Stub) and value.qualified_name == f"{module}.{name}"
    assert value.args == (SENTINEL,) and decoded.stubbed == {f"{module}.{name}"}
    assert not marker.exists()
