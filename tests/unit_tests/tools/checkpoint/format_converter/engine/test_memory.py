# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The memory model, preflight checks and process memory probes."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.checkpoint.format_converter.engine import memory
from tools.checkpoint.format_converter.engine.memory import (
    MemoryEstimate,
    check_free_disk,
    check_host_memory,
    estimate_memory,
    estimate_output_bytes,
)
from tools.checkpoint.format_converter.engine.pieces import BytePiece, Unit
from tools.checkpoint.format_converter.engine.staging import Staging
from tools.checkpoint.format_converter.errors import BudgetExceeded, ExitCode

from .._programs import convert_store, write_source


def test_estimate_totals():
    in_process = MemoryEstimate(unit_peak=100, worker_baseline=10, workers=0, driver=50)
    assert (in_process.worker_peak, in_process.total) == (100, 150)
    pooled = MemoryEstimate(unit_peak=100, worker_baseline=10, workers=3, driver=50)
    assert (pooled.worker_peak, pooled.total) == (110, 380)
    assert pooled.to_json() == {
        "unit_peak": 100,
        "worker_baseline": 10,
        "workers": 3,
        "driver": 50,
        "worker_peak": 110,
        "total": 380,
    }
    units = [Unit(0, (), est_peak_bytes=7), Unit(1, (), est_peak_bytes=30)]
    estimate = estimate_memory(units, workers=2, driver=5)
    assert estimate.unit_peak == 30 and estimate.worker_baseline == memory.WORKER_BASELINE_BYTES


def test_host_memory_preflight():
    estimate = MemoryEstimate(unit_peak=2**30, worker_baseline=2**29, workers=4, driver=2**30)
    check_host_memory(estimate, estimate.total)
    with pytest.raises(BudgetExceeded, match=r"needs about 7.00 GiB .*4 x 1.50 GiB") as excinfo:
        check_host_memory(estimate, estimate.total - 1)
    assert excinfo.value.exit_code == ExitCode.BUDGET_EXCEEDED
    assert "--max-host-bytes" in excinfo.value.hint


def test_disk_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "free_disk_bytes", lambda path: 1020)
    check_free_disk(tmp_path, 1000)
    with pytest.raises(BudgetExceeded, match="needs about 1021 B .* has 1020 B free"):
        check_free_disk(tmp_path, 1001)


def test_run_refuses_an_output_that_does_not_fit_before_creating_anything(tmp_path, monkeypatch):
    source = write_source(tmp_path / "src")
    checked = []

    def tiny_disk(path):
        checked.append(path)
        return 10

    monkeypatch.setattr(memory, "free_disk_bytes", tiny_disk)
    with pytest.raises(BudgetExceeded, match="the output needs about") as excinfo:
        convert_store(source, tmp_path / "out" / "dst")
    assert excinfo.value.exit_code == ExitCode.BUDGET_EXCEEDED
    assert checked == [tmp_path]  # the nearest existing ancestor of the target
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src"]


def test_output_estimate_bounds_a_real_store(tmp_path):
    result = convert_store(write_source(tmp_path / "src"), tmp_path / "dst", keep_staging=True)
    staging = Staging(result.staging_dir)
    units = [staging.load_unit(record["uid"]).unit for record in result.journal]
    written = sum(p.stat().st_size for p in result.target.iter_dir.iterdir() if p.is_file())
    assert written <= estimate_output_bytes(units)
    blob = Unit(0, (), (BytePiece("b", b"12345"),))
    assert estimate_output_bytes([blob]) == 5


def test_probes_read_this_process():
    rss, peak = memory.process_rss_bytes(), memory.process_peak_rss_bytes()
    assert rss is not None and peak is not None and 0 < rss <= peak
    assert memory.children_peak_rss_bytes() >= 0
    available = memory.available_memory_bytes()
    assert available is None or available > 0
    assert memory.free_disk_bytes(Path("/")) > 0


def test_journal_records_the_worker_peak(tmp_path):
    result = convert_store(write_source(tmp_path / "src"), tmp_path / "dst")
    assert all(record["vmhwm_bytes"] > 0 for record in result.journal)
