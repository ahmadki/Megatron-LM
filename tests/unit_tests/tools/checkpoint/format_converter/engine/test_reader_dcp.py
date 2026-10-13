# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Read-once DCP region reader on synthetic multi-chunk stores."""

from __future__ import annotations

import os
import warnings

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

from tools.checkpoint.format_converter.engine import reader_dcp
from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.engine.reader import ReadReq, ReadStats, SourceReader
from tools.checkpoint.format_converter.engine.regions import Region
from tools.checkpoint.format_converter.errors import (
    BudgetExceeded,
    ExitCode,
    IntegrityError,
    InternalError,
)

from .._synthetic import (
    alias_dumps,
    dim0_grid,
    product_grid,
    rne_bfloat16,
    torch_save_bytes,
    write_store,
)

W_SHAPE = (10, 4)
G_SHAPE = (6, 8)


class Trap:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


def _tensors() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "w": torch.arange(40, dtype=torch.float32).reshape(W_SHAPE),
        "g": torch.randn(G_SHAPE).to(torch.bfloat16),
        "i": torch.arange(-7, 5, dtype=torch.int64),
        "b": torch.rand(3, 5) > 0.5,
        "s": torch.tensor(-0.0, dtype=torch.float32),
    }


@pytest.fixture(name="store")
def fixture_store(tmp_path):
    tensors = _tensors()
    marker = tmp_path / "trap-fired"
    payloads = {
        "blob": torch_save_bytes({"answer": 42}),
        "trap": alias_dumps(Trap(str(marker)), {os.system: ("os", "system")}),
    }
    write_store(
        tmp_path / "store",
        tensors,
        grids={
            "w": dim0_grid(W_SHAPE, 4),
            "g": product_grid(G_SHAPE, [[2, 4], [4]]),
            "i": dim0_grid((12,), 5),
        },
        byte_items=payloads,
    )
    view = MetadataView.load(tmp_path / "store")
    return view, tensors, payloads, marker


@pytest.fixture(name="load_counter")
def fixture_load_counter(monkeypatch):
    calls = []
    real_load = torch.load

    def counting_load(*args, **kwargs):
        calls.append(1)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)
    return calls


def _req(key: str, region: Region, dtype: torch.dtype) -> ReadReq:
    return ReadReq(key, region, torch.empty(region.sizes, dtype=dtype))


def test_region_spanning_chunks_reads_each_chunk_once(store, load_counter):
    view, tensors, _, _ = store
    req = _req("w", Region((3, 1), (5, 2)), torch.float32)
    stats = SourceReader(view, max_item_bytes=1 << 20).read_into([req])
    assert torch.equal(req.dst, tensors["w"][3:8, 1:3])
    assert stats.items_touched == 2 and stats.torch_load_calls == 2 == len(load_counter)
    assert stats.amplification == 1.0
    chunks = {c.offsets: c for c in view.tensors["w"].chunks}
    assert stats.bytes_read == chunks[(0, 0)].length + chunks[(4, 0)].length


def test_many_requests_on_shared_chunks_read_each_chunk_once(store, load_counter):
    view, tensors, _, _ = store
    regions = [Region((0, 0), (1, 8)), Region((1, 3), (3, 3)), Region((5, 0), (1, 8))]
    regions += [Region((1, 2), (2, 2))]  # overlaps the second request
    reqs = [_req("g", r, torch.bfloat16) for r in regions]
    reqs += [_req("w", Region((9, 0), (1, 4)), torch.float32)]
    stats = SourceReader(view, max_item_bytes=1 << 20).read_into(reqs)
    for req in reqs:
        assert torch.equal(req.dst, tensors[req.key][req.region.slices()]), req.region
    touched_g = {
        c.offsets for c in view.tensors["g"].chunks if any(c.region.intersect(r) for r in regions)
    }
    assert stats.items_touched == len(touched_g) + 1
    assert stats.torch_load_calls == stats.items_touched == len(load_counter)


def test_one_dcp_load_per_batch(store, monkeypatch):
    view, tensors, _, _ = store
    loads = []
    real_dcp_load = dcp.load

    def counting_dcp_load(*args, **kwargs):
        loads.append(kwargs.get("no_dist"))
        return real_dcp_load(*args, **kwargs)

    monkeypatch.setattr(dcp, "load", counting_dcp_load)
    reqs = [_req(key, Region.full(t.shape), t.dtype) for key, t in tensors.items()]
    SourceReader(view, max_item_bytes=1 << 20).read_into(reqs)
    assert loads == [True]
    for req in reqs:
        assert torch.equal(req.dst, tensors[req.key])


def test_whole_tensor_request_loads_into_the_destination_itself(store):
    view, tensors, _, _ = store
    req = _req("g", Region.full(G_SHAPE), torch.bfloat16)
    plan = reader_dcp.plan_region_load(view, [req])
    assert plan.targets["g"] is req.dst and plan.copies == ()
    assert len(plan.touched) == len(view.tensors["g"].chunks)
    reader_dcp.execute_region_load(view, plan)
    assert torch.equal(req.dst, tensors["g"])


def test_chunk_aligned_request_needs_no_staging(store):
    view, tensors, _, _ = store
    dst = torch.empty(6, 4)
    reqs = [
        ReadReq("w", Region((4, 0), (4, 4)), dst[0:4]),
        ReadReq("w", Region((8, 0), (2, 4)), dst[4:6]),
    ]
    plan = reader_dcp.plan_region_load(view, reqs)
    assert plan.copies == ()
    reader_dcp.execute_region_load(view, plan)
    assert torch.equal(dst, tensors["w"][4:10])


def test_scalar_bool_and_int_tensors(store):
    view, tensors, _, _ = store
    reqs = [
        _req("s", Region((), ()), torch.float32),
        _req("b", Region((1, 1), (2, 3)), torch.bool),
        _req("i", Region((4,), (3,)), torch.int64),
    ]
    SourceReader(view, max_item_bytes=1 << 20).read_into(reqs)
    assert torch.equal(reqs[0].dst, tensors["s"]) and torch.signbit(reqs[0].dst)
    assert torch.equal(reqs[1].dst, tensors["b"][1:3, 1:4])
    assert torch.equal(reqs[2].dst, tensors["i"][4:7])


def test_byte_items_are_returned_raw_and_never_unpickled(store, load_counter):
    view, _, payloads, marker = store
    got = SourceReader(view, max_item_bytes=1).read_bytes(["blob", "trap"])
    assert got == payloads
    assert not marker.exists() and not load_counter


def test_budget_is_checked_before_reading(store, load_counter):
    view, _, _, _ = store
    biggest = max(c.nbytes for c in view.tensors["w"].chunks)
    reader = SourceReader(view, max_item_bytes=biggest - 1)
    with pytest.raises(BudgetExceeded, match=r"stored chunk \(0, 0\) of w decodes to 64 B") as err:
        reader.read_into([_req("w", Region.full(W_SHAPE), torch.float32)])
    assert err.value.exit_code == ExitCode.BUDGET_EXCEEDED
    assert f"--max-item-bytes {biggest}" in str(err.value)
    assert not load_counter
    exact = _req("w", Region.full(W_SHAPE), torch.float32)
    SourceReader(view, max_item_bytes=biggest).read_into([exact])  # the budget is inclusive
    assert torch.equal(exact.dst, _tensors()["w"])


@pytest.mark.parametrize(
    "req, message",
    [
        (ReadReq("nope", Region((0,), (1,)), torch.empty(1)), "no tensor"),
        (ReadReq("w", Region((8, 0), (4, 4)), torch.empty(4, 4)), "outside shape"),
        (ReadReq("w", Region((0, 0), (2, 4)), torch.empty(4, 2)), "destination"),
        (
            ReadReq("w", Region((0, 0), (2, 4)), torch.empty(2, 4, dtype=torch.float64)),
            "needs a cast",
        ),
        (ReadReq("w", Region((0, 0), (2, 4)), torch.empty(2, 4), cast=True), "is no cast"),
        (ReadReq("i", Region((0,), (2,)), torch.empty(2), cast=True), "only floating-point"),
        (ReadReq("w", Region((0, 0), (2, 4)), torch.empty(2, 4, device="meta")), "CPU"),
    ],
)
def test_malformed_requests_are_internal_errors(store, req, message):
    view, _, _, _ = store
    with pytest.raises(InternalError, match=message):
        SourceReader(view, max_item_bytes=1 << 20).read_into([req])


def test_read_stats_add_and_empty_amplification():
    total = ReadStats(10, 1, 1) + ReadStats(5, 2, 2)
    assert total == ReadStats(15, 3, 3)
    assert ReadStats().amplification == 1.0
    assert ReadStats(0, 2, 4).amplification == 2.0


def test_truncated_data_file_is_an_integrity_error(store):
    view, _, _, _ = store
    data_file = view.iter_dir / view.tensors["w"].chunks[-1].relative_path
    with open(data_file, "r+b") as f:
        f.truncate(view.tensors["w"].chunks[-1].offset + 10)
    req = _req("w", Region.full(W_SHAPE), torch.float32)
    with pytest.raises(IntegrityError, match="reading .* failed") as err:
        SourceReader(view, max_item_bytes=1 << 20).read_into([req])
    assert err.value.exit_code == ExitCode.INTEGRITY


def test_interrupted_read_raises_keyboard_interrupt(store, monkeypatch):
    """DCP wraps a Ctrl-C during the read in a CheckpointException; it must stay an
    interrupt (exit 130), not become an integrity error (exit 6)."""
    view, _, _, _ = store

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(FileSystemReader, "read_data", interrupted)
    with pytest.raises(KeyboardInterrupt):
        SourceReader(view, max_item_bytes=1 << 20).read_into(
            [_req("w", Region.full(W_SHAPE), torch.float32)]
        )
    with pytest.raises(KeyboardInterrupt):
        SourceReader(view, max_item_bytes=1 << 20).read_bytes(["blob"])


def test_warning_filter_is_added_once(store, monkeypatch):
    """Reads silence DCP's no-process-group warning with one filter, added by the
    first read; later reads leave the process-wide warning filters alone."""
    view, _, _, _ = store
    added = []
    real_filterwarnings = warnings.filterwarnings

    def recording_filterwarnings(*args, **kwargs):
        added.append(kwargs.get("message"))
        real_filterwarnings(*args, **kwargs)

    monkeypatch.setattr(warnings, "filterwarnings", recording_filterwarnings)
    reader = SourceReader(view, max_item_bytes=1 << 20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reader.read_into([_req("w", Region.full(W_SHAPE), torch.float32)])
        reader.read_into([_req("g", Region.full(G_SHAPE), torch.bfloat16)])
        reader.read_bytes(["blob"])
    assert len(added) == 1
    assert not [w for w in caught if "torch.distributed is" in str(w.message)]


def test_a_cast_converts_to_nearest_even_and_reads_once(tmp_path, load_counter):
    """One stored fp32 item read into an fp32 and a bf16 destination: decoded once, staged."""
    values = torch.randn(8, 6, generator=torch.Generator().manual_seed(4))
    values[0, :3] = torch.tensor([1 + 2**-8, 1 + 3 * 2**-8, -(1 + 2**-8)])  # exact ties
    view = MetadataView.load(
        write_store(tmp_path / "s", {"x": values}, grids={"x": dim0_grid((8, 6), 4)})
    )
    exact = _req("x", Region.full((8, 6)), torch.float32)
    rounded = ReadReq("x", Region.full((8, 6)), torch.empty(8, 6, dtype=torch.bfloat16), cast=True)
    plan = reader_dcp.plan_region_load(view, [exact, rounded])
    assert len(plan.copies) == 4  # every chunk is staged once and copied out twice
    stats = SourceReader(view, max_item_bytes=1 << 20).read_into([exact, rounded])
    assert torch.equal(exact.dst, values)
    expected = rne_bfloat16(values)
    assert torch.equal(rounded.dst.view(torch.int16), expected.view(torch.int16))
    assert rounded.dst[0, :3].float().tolist() == [1.0, 1 + 2**-6, -1.0]
    assert stats.torch_load_calls == stats.items_touched == len(load_counter) == 2


def test_a_lone_cast_is_staged(tmp_path):
    view = MetadataView.load(write_store(tmp_path / "s", {"x": torch.randn(4, 2)}))
    req = ReadReq("x", Region.full((4, 2)), torch.empty(4, 2, dtype=torch.bfloat16), cast=True)
    plan = reader_dcp.plan_region_load(view, [req])
    assert plan.targets["x"] is not req.dst and len(plan.copies) == 1
    (chunk,) = view.tensors["x"].chunks
    assert reader_dcp.staging_bytes(chunk, [chunk.region], converts=True) == 4 * 2 * 4
    assert reader_dcp.staging_bytes(chunk, [chunk.region]) == 0
