# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Streaming digests against an independent oracle, bounded reads and budgets."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from tools.checkpoint.format_converter.engine.hashing import (
    DigestJob,
    compute_digests,
    plan_digest_jobs,
    write_digest_file,
)
from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.engine.reader import SourceReader
from tools.checkpoint.format_converter.engine.regions import Region
from tools.checkpoint.format_converter.errors import BudgetExceeded, ExitCode

from .._synthetic import dim0_grid, product_grid, write_store


def _oracle_bytes(tensor: torch.Tensor) -> bytes:
    """Row-major element bytes via numpy, independent of the engine's uint8 views."""
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().tobytes()
    return tensor.numpy().tobytes()


def _oracle(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_oracle_bytes(tensor)).hexdigest()


def _tensors() -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    return {
        "dim0_fp32": torch.randn(12, 5),
        "dim0_bf16": torch.randn(9, 4, 3).to(torch.bfloat16),
        "cols_int64": torch.randint(-(2**40), 2**40, (6, 8), dtype=torch.int64),
        "grid_bool": torch.rand(6, 6) > 0.3,
        "scalar_fp32": torch.tensor(-0.0),
        "scalar_bf16": torch.tensor(1.5, dtype=torch.bfloat16),
        "single_fp32": torch.arange(7, dtype=torch.float32),
    }


GRIDS = {
    "dim0_fp32": dim0_grid((12, 5), 5),
    "dim0_bf16": dim0_grid((9, 4, 3), 2),
    "cols_int64": product_grid((6, 8), [[], [3, 5]]),
    "grid_bool": product_grid((6, 6), [[2], [4]]),
}


@pytest.fixture(name="store")
def fixture_store(tmp_path):
    tensors = _tensors()
    write_store(tmp_path, tensors, grids=GRIDS)
    return MetadataView.load(tmp_path), tensors


@pytest.mark.parametrize("batch_bytes", [1 << 20, 64, 1])
@pytest.mark.parametrize("workers", [1, 4])
def test_digests_match_the_independent_oracle(store, batch_bytes, workers):
    view, tensors = store
    reader = SourceReader(view, max_item_bytes=1 << 20, batch_bytes=batch_bytes)
    digests, stats = compute_digests(reader, sorted(tensors), workers=workers)
    assert list(digests) == sorted(tensors)
    for key, tensor in tensors.items():
        assert digests[key].sha256 == _oracle(tensor), key
        assert digests[key].shape == tuple(tensor.shape)
        assert digests[key].dtype == str(tensor.dtype)
    assert stats.amplification == 1.0
    assert stats.items_touched == sum(len(e.chunks) for e in view.tensors.values())


def test_dim0_grids_stream_in_bounded_slabs(store, monkeypatch):
    view, tensors = store
    requested = []
    real_read_into = SourceReader.read_into

    def recording_read_into(self, reqs):
        requested.append(sum(r.dst.numel() * r.dst.element_size() for r in reqs))
        return real_read_into(self, reqs)

    monkeypatch.setattr(SourceReader, "read_into", recording_read_into)
    reader = SourceReader(view, max_item_bytes=1 << 20, batch_bytes=48)
    digests, _ = compute_digests(reader, ["dim0_fp32"], workers=1)
    assert digests["dim0_fp32"].sha256 == _oracle(tensors["dim0_fp32"])
    chunk_bytes = 5 * 5 * 4
    assert requested and max(requested) <= chunk_bytes
    assert sum(requested) == tensors["dim0_fp32"].numel() * 4


def test_plan_groups_small_tensors_and_streams_large_ones(store):
    view, _ = store
    entries = [view.tensors[k] for k in ("scalar_fp32", "single_fp32", "dim0_fp32", "grid_bool")]
    jobs = plan_digest_jobs(entries, max_item_bytes=1 << 20, batch_bytes=32)
    assert set(jobs) == {
        DigestJob(("scalar_fp32", "single_fp32")),
        DigestJob(("dim0_fp32",), tuple(dim0_grid((12, 5), 5))),
        DigestJob(("grid_bool",), (Region.full((6, 6)),)),
    }


def test_non_dim0_grid_over_budget_is_refused_before_reading(store, monkeypatch):
    view, _ = store
    monkeypatch.setattr(
        SourceReader, "read_into", lambda *a: pytest.fail("read before the budget check")
    )
    reader = SourceReader(view, max_item_bytes=47, batch_bytes=16)
    with pytest.raises(BudgetExceeded, match="hashing cols_int64 needs the whole tensor") as err:
        compute_digests(reader, ["single_fp32", "cols_int64"], workers=1)
    assert err.value.exit_code == ExitCode.BUDGET_EXCEEDED
    assert "--max-item-bytes 384" in str(err.value)


def test_dim0_chunk_over_budget_is_refused(store):
    view, _ = store
    reader = SourceReader(view, max_item_bytes=99, batch_bytes=16)
    with pytest.raises(BudgetExceeded, match=r"stored chunk \(0, 0\) of dim0_fp32"):
        compute_digests(reader, ["dim0_fp32"], workers=1)


def test_digest_file_schema(store, tmp_path):
    view, tensors = store
    reader = SourceReader(view, max_item_bytes=1 << 20)
    digests, _ = compute_digests(reader, ["scalar_bf16", "grid_bool"], workers=2)
    path = tmp_path / "digests.json"
    write_digest_file(path, digests)
    assert json.loads(path.read_text()) == {
        "grid_bool": {
            "shape": [6, 6],
            "dtype": "torch.bool",
            "sha256": _oracle(tensors["grid_bool"]),
        },
        "scalar_bf16": {
            "shape": [],
            "dtype": "torch.bfloat16",
            "sha256": _oracle(tensors["scalar_bf16"]),
        },
    }
    assert not list(tmp_path.glob(".*.tmp"))
