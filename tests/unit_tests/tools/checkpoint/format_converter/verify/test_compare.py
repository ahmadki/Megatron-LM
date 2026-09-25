# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bitwise verify: every planted corruption is found, identical stores pass, reads are once."""

from __future__ import annotations

import json
import re

import pytest
import torch
from torch.distributed.checkpoint import FileSystemReader

from tools.checkpoint.format_converter import cli
from tools.checkpoint.format_converter.api import verify_checkpoint
from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.engine.regions import Region
from tools.checkpoint.format_converter.errors import BudgetExceeded, ExitCode, UsageError
from tools.checkpoint.format_converter.options import CompareMode, VerifyOptions
from tools.checkpoint.format_converter.verify.compare import _select_value_keys, overlap_units
from tools.checkpoint.format_converter.verify.report import FindingKind

from .._synthetic import dim0_grid, product_grid, torch_save_bytes, write_store

GRIDS = {"rows": dim0_grid((10, 4), 4), "tiles": product_grid((6, 8), [[3], [4]])}


def _tensors() -> dict[str, torch.Tensor]:
    torch.manual_seed(2)
    return {
        "rows": torch.randn(10, 4),
        "tiles": torch.randn(6, 8).to(torch.bfloat16),
        "ints": torch.arange(-5, 7, dtype=torch.int64),
        "mask": torch.rand(3, 3) > 0.5,
        "scalar": torch.tensor(2.5),
    }


def _common(iteration: int = 7) -> dict[str, bytes]:
    return {"common_state/shard_0_1": torch_save_bytes([{"iteration": iteration}])}


def _pair(tmp_path, mutate=None, *, b_grids=None, a_bytes=None, b_bytes=None):
    tensors = _tensors()
    write_store(
        tmp_path / "a", tensors, grids=GRIDS, byte_items=_common() if a_bytes is None else a_bytes
    )
    b_tensors = {k: v.clone() for k, v in tensors.items()}
    if mutate is not None:
        mutate(b_tensors)
    write_store(
        tmp_path / "b",
        b_tensors,
        grids=GRIDS if b_grids is None else b_grids,
        byte_items=_common() if b_bytes is None else b_bytes,
    )
    return tmp_path / "a", tmp_path / "b"


def _only(report, kind: FindingKind) -> list[str]:
    return [f.key for f in report.findings if f.kind == kind]


def _flip_bit(tensor: torch.Tensor, index: int, bit: int = 0) -> None:
    as_int = tensor.view(torch.int32).view(-1)
    as_int[index] ^= 1 << bit


@pytest.fixture(name="load_counter")
def fixture_load_counter(monkeypatch):
    calls = []
    real_load = torch.load

    def counting_load(*args, **kwargs):
        calls.append(1)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)
    return calls


def test_identical_stores_pass_and_every_chunk_is_read_once(tmp_path, load_counter):
    a, b = _pair(tmp_path)
    report = verify_checkpoint(a, b, VerifyOptions(workers=2))
    assert report.ok, report.findings
    assert (report.keys_compared, report.values_compared, report.byte_items_compared) == (5, 5, 1)
    chunks = sum(len(e.chunks) for e in MetadataView.load(a).tensors.values())
    assert report.read_stats.items_touched == 2 * chunks == len(load_counter)
    assert report.read_stats.amplification == 1.0


def test_one_flipped_bit_is_found(tmp_path):
    a, b = _pair(tmp_path, lambda t: _flip_bit(t["rows"], 21, bit=3))
    report = verify_checkpoint(a, b, VerifyOptions())
    assert not report.ok and _only(report, FindingKind.VALUE) == ["rows"]
    detail = report.findings[0].detail
    assert detail.startswith("1 of 40 elements differ bitwise; first at (5, 1):")


def test_negative_zero_is_not_zero(tmp_path):
    def mutate(t):
        t["scalar"] = torch.tensor(-0.0)

    a, b = _pair(tmp_path, mutate)
    tensors = _tensors()
    tensors["scalar"] = torch.tensor(0.0)
    write_store(tmp_path / "a", tensors, grids=GRIDS, byte_items=_common())
    report = verify_checkpoint(a, b, VerifyOptions())
    assert _only(report, FindingKind.VALUE) == ["scalar"]
    assert "A 0.0 (bytes 00000000) vs B -0.0 (bytes 00000080)" in report.findings[0].detail


def test_nan_payloads_are_compared_bitwise(tmp_path):
    def with_nan(payload: int):
        def mutate(t):
            t["rows"].view(torch.int32)[0, 0] = payload

        return mutate

    same_a, same_b = _pair(tmp_path / "same", with_nan(0x7FC00000))
    tensors = _tensors()
    with_nan(0x7FC00000)(tensors)
    write_store(same_a, tensors, grids=GRIDS, byte_items=_common())
    assert verify_checkpoint(same_a, same_b, VerifyOptions()).ok
    a, b = _pair(tmp_path / "diff", with_nan(0x7FC00001))
    write_store(a, tensors, grids=GRIDS, byte_items=_common())
    report = verify_checkpoint(a, b, VerifyOptions())
    assert _only(report, FindingKind.VALUE) == ["rows"]
    assert "0000c07f" in report.findings[0].detail and "0100c07f" in report.findings[0].detail


def test_dtype_flip_is_found(tmp_path):
    def mutate(t):
        t["ints"] = t["ints"].to(torch.int32)

    report = verify_checkpoint(*_pair(tmp_path, mutate), VerifyOptions())
    assert _only(report, FindingKind.DTYPE) == ["ints"]
    assert report.findings[0].detail == "A torch.int64 vs B torch.int32"


def test_shape_mismatch_is_found(tmp_path):
    def mutate(t):
        t["ints"] = torch.arange(11, dtype=torch.int64)

    report = verify_checkpoint(*_pair(tmp_path, mutate), VerifyOptions())
    assert _only(report, FindingKind.SHAPE) == ["ints"]


def test_extra_and_missing_keys_are_found(tmp_path):
    def mutate(t):
        del t["mask"]
        t["extra"] = torch.zeros(2)

    report = verify_checkpoint(*_pair(tmp_path, mutate), VerifyOptions())
    assert _only(report, FindingKind.ONLY_IN_A) == ["mask"]
    assert _only(report, FindingKind.ONLY_IN_B) == ["extra"]
    ignored = verify_checkpoint(
        *_pair(tmp_path / "x", mutate), VerifyOptions(ignore_keys=(re.compile("^(mask|extra)$"),))
    )
    assert ignored.ok


def test_chunk_grid_is_checked_only_with_layout(tmp_path):
    regridded = {"rows": dim0_grid((10, 4), 3), "tiles": dim0_grid((6, 8), 2)}
    a, b = _pair(tmp_path, b_grids=regridded)
    assert verify_checkpoint(a, b, VerifyOptions()).ok
    report = verify_checkpoint(a, b, VerifyOptions(layout=True))
    assert _only(report, FindingKind.LAYOUT) == ["rows", "tiles"]
    assert report.findings[0].detail == (
        "chunk grids differ: A has 3 chunks, B has 4; "
        "first A-only chunk [0:4, 0:4], first B-only chunk [0:3, 0:4]"
    )


def test_values_are_compared_across_different_grids(tmp_path, load_counter):
    regridded = {"rows": dim0_grid((10, 4), 3), "tiles": dim0_grid((6, 8), 2)}
    a, b = _pair(tmp_path, lambda t: _flip_bit(t["tiles"].view(torch.int16), 0), b_grids=regridded)
    report = verify_checkpoint(a, b, VerifyOptions(common="skip"))
    assert _only(report, FindingKind.VALUE) == ["tiles"]
    a_chunks = sum(len(e.chunks) for e in MetadataView.load(a).tensors.values())
    b_chunks = sum(len(e.chunks) for e in MetadataView.load(b).tensors.values())
    assert len(load_counter) == a_chunks + b_chunks


def test_byte_items_are_compared_only_in_bytes_mode(tmp_path):
    a, b = _pair(tmp_path, b_bytes=_common(iteration=8))
    report = verify_checkpoint(a, b, VerifyOptions())
    assert _only(report, FindingKind.BYTES) == ["common_state/shard_0_1"]
    assert verify_checkpoint(a, b, VerifyOptions(common="skip")).ok
    a2, b2 = _pair(tmp_path / "k", b_bytes={**_common(), "extra/shard_0_1": b"x"})
    assert _only(verify_checkpoint(a2, b2, VerifyOptions()), FindingKind.ONLY_IN_B) == [
        "extra/shard_0_1"
    ]


def test_ignored_byte_items_are_not_compared(tmp_path):
    a, b = _pair(
        tmp_path,
        a_bytes={**_common(), "extra/a_only": b"1", "extra/both": b"x"},
        b_bytes={**_common(), "extra/b_only": b"2", "extra/both": b"y"},
    )
    report = verify_checkpoint(a, b, VerifyOptions())
    assert [(f.kind, f.key) for f in report.findings] == [
        (FindingKind.ONLY_IN_A, "extra/a_only"),
        (FindingKind.ONLY_IN_B, "extra/b_only"),
        (FindingKind.BYTES, "extra/both"),
    ]
    ignored = verify_checkpoint(a, b, VerifyOptions(ignore_keys=(re.compile("^extra/"),)))
    assert ignored.ok and ignored.byte_items_compared == 1


def test_meta_mode_reads_no_values(tmp_path, load_counter):
    a, b = _pair(tmp_path, lambda t: _flip_bit(t["rows"], 0))
    report = verify_checkpoint(a, b, VerifyOptions(mode=CompareMode("meta"), common="skip"))
    assert report.ok and report.values_compared == 0 and not load_counter


def test_sample_mode_is_seeded(tmp_path):
    """sample:N compares the values of the same N keys for a seed, and the seed matters."""
    a, b = _pair(tmp_path, lambda t: _flip_bit(t["rows"], 0))
    keys = sorted(_tensors())
    samples = set()
    for seed in range(8):
        opts = VerifyOptions(mode=CompareMode.parse("sample:2"), seed=seed, common="skip")
        chosen = _select_value_keys(keys, opts)
        assert chosen == _select_value_keys(keys, opts) == sorted(set(chosen))
        assert set(chosen) <= set(keys)
        report = verify_checkpoint(a, b, opts)
        assert report.values_compared == 2
        assert report.ok == ("rows" not in chosen), (seed, chosen)
        samples.add(tuple(chosen))
    assert len(samples) > 1
    everything = VerifyOptions(mode=CompareMode.parse("sample:99"), common="skip")
    assert _only(verify_checkpoint(a, b, everything), FindingKind.VALUE) == ["rows"]


def test_unit_over_budget_is_refused_before_reading(tmp_path, load_counter):
    a, b = _pair(tmp_path)
    with pytest.raises(
        BudgetExceeded, match=r"comparing ints needs \[0:12\] \(96 B per side\)"
    ) as err:
        verify_checkpoint(a, b, VerifyOptions(max_item_bytes=63))
    assert err.value.exit_code == ExitCode.BUDGET_EXCEEDED and not load_counter
    assert verify_checkpoint(a, b, VerifyOptions(max_item_bytes=96, common="skip")).ok  # inclusive


def _entry(tmp_path, name, grid):
    write_store(tmp_path / name, {"t": torch.zeros(4, 4)}, grids={"t": grid})
    return MetadataView.load(tmp_path / name).tensors["t"]


def test_overlap_units_follow_the_chunk_overlap_graph(tmp_path):
    a = _entry(tmp_path, "a", dim0_grid((4, 4), 2))
    b = _entry(
        tmp_path, "b", (Region((0, 0), (1, 4)), Region((1, 0), (1, 4)), Region((2, 0), (2, 4)))
    )
    units = overlap_units(a, b)
    assert [u.region for u in units] == [Region((0, 0), (2, 4)), Region((2, 0), (2, 4))]
    assert [len(u.b_chunks) for u in units] == [2, 1]
    assert all(u.fills_region for u in units)


def test_non_box_units_are_compared_correctly(tmp_path):
    """An L-shaped unit is read into a zero-filled bounding box on both sides."""
    a_grid = (Region((0, 0), (2, 2)), Region((0, 2), (2, 2)), Region((2, 0), (2, 4)))
    b_grid = (Region((0, 0), (2, 2)), Region((0, 2), (4, 2)), Region((2, 0), (2, 2)))
    units = overlap_units(_entry(tmp_path, "ea", a_grid), _entry(tmp_path, "eb", b_grid))
    assert [u.fills_region for u in units] == [True, False]
    tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    write_store(tmp_path / "a", {"t": tensor}, grids={"t": a_grid})
    write_store(tmp_path / "b", {"t": tensor}, grids={"t": b_grid})
    assert verify_checkpoint(tmp_path / "a", tmp_path / "b", VerifyOptions()).ok
    flipped = tensor.clone()
    _flip_bit(flipped, 13)
    write_store(tmp_path / "c", {"t": flipped}, grids={"t": b_grid})
    report = verify_checkpoint(tmp_path / "a", tmp_path / "c", VerifyOptions())
    assert report.findings[0].detail.startswith("1 of 16 elements differ bitwise; first at (3, 1)")


def test_report_jsonl(tmp_path):
    a, b = _pair(tmp_path, lambda t: _flip_bit(t["rows"], 0))
    report = verify_checkpoint(a, b, VerifyOptions())
    path = tmp_path / "report.jsonl"
    report.write_jsonl(path)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["kind"] == "value" and lines[0]["key"] == "rows"
    assert lines[-1]["kind"] == "summary" and lines[-1]["ok"] is False
    assert lines[-1]["values_compared"] == 5


def test_cli_exit_codes_and_output(tmp_path, capsys):
    a, b = _pair(tmp_path / "same")
    assert cli.main(["verify", str(a), str(b)]) == 0
    assert "verify:     OK" in capsys.readouterr().out
    c, d = _pair(tmp_path / "diff", lambda t: _flip_bit(t["rows"], 0))
    report = tmp_path / "r.jsonl"
    args = ["verify", str(c), str(d), "--report", str(report), "--mode", "full", "--workers", "2"]
    assert cli.main(args) == ExitCode.VERIFY_MISMATCH
    out = capsys.readouterr().out
    assert "verify:     MISMATCH" in out and "  value      rows: 1 of 40" in out
    assert report.is_file()
    assert cli.main(["verify", str(c)]) == ExitCode.USAGE
    assert "needs a second store B" in capsys.readouterr().err
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["verify", str(c), str(d), "--mode", "sample:0"])
    assert excinfo.value.code == 2


def test_interrupted_verify_exits_130(tmp_path, monkeypatch, capsys):
    a, b = _pair(tmp_path)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(FileSystemReader, "read_data", interrupted)
    assert cli.main(["verify", str(a), str(b)]) == ExitCode.INTERRUPTED
    assert capsys.readouterr().err == "interrupted\n"


def test_verify_requires_b(tmp_path):
    a, _ = _pair(tmp_path)
    with pytest.raises(UsageError):
        verify_checkpoint(a, None, VerifyOptions())


def test_oversized_byte_item_is_refused_before_reading(tmp_path, load_counter):
    a, b = _pair(tmp_path, b_bytes={"common_state/shard_0_1": b"x" * 4096})
    with pytest.raises(BudgetExceeded, match="byte item common_state/shard_0_1 is 4.00 KiB"):
        verify_checkpoint(a, b, VerifyOptions(max_item_bytes=4000))
    assert not load_counter
