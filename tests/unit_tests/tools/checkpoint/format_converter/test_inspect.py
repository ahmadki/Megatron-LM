# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``inspect`` on synthetic torch_dist and fsdp_dtensor stores (API and CLI)."""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import re

import pytest
import torch

from tools.checkpoint.format_converter import cli
from tools.checkpoint.format_converter.api import inspect_checkpoint
from tools.checkpoint.format_converter.errors import IntegrityError
from tools.checkpoint.format_converter.formats.sections import Section
from tools.checkpoint.format_converter.options import InspectOptions

from ._synthetic import (
    default_args,
    dim0_grid,
    torch_save_bytes,
    write_fsdp_store,
    write_torch_dist_store,
    write_tracker,
)

ENUMS = "megatron.core.transformer.enums"


class FakeAttnBackend(enum.Enum):
    """Pickled as megatron.core.transformer.enums.AttnBackend."""

    flash = 1
    auto = 5


ALIASES = {FakeAttnBackend: (ENUMS, "AttnBackend")}

MODEL = {
    "decoder.layers.0.mlp.linear_fc1.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
    "decoder.layers.0.mlp.linear_fc2.weight": torch.ones(4, 3, dtype=torch.bfloat16),
    "decoder.final_layernorm.weight": torch.ones(4, dtype=torch.bfloat16),
}


def _args(**overrides):
    return default_args(
        attention_backend=FakeAttnBackend.auto, num_experts=None, swiglu=True, **overrides
    )


@pytest.fixture(name="torch_dist_root")
def fixture_torch_dist_root(tmp_path):
    tensors = {**MODEL, "optimizer.state.exp_avg.decoder.final_layernorm.weight": torch.zeros(4)}
    write_torch_dist_store(
        tmp_path / "iter_0000007",
        tensors,
        grids={"decoder.layers.0.mlp.linear_fc1.weight": dim0_grid((6, 4), 2)},
        common={"args": _args(), "iteration": 7, "checkpoint_version": 3.0},
        common_aliases=ALIASES,
        extra_byte_items={
            "decoder.layers.0.self_attention._extra_state/shard_0_1": b"te",
            "rng_state/shard_0_0_1": b"rng",
        },
    )
    write_tracker(tmp_path, 7)
    return tmp_path


@pytest.fixture(name="fsdp_store")
def fixture_fsdp_store(tmp_path):
    return write_fsdp_store(
        tmp_path / "fsdp",
        MODEL,
        args=_args(ckpt_format="fsdp_dtensor", use_megatron_fsdp=True),
        common_aliases=ALIASES,
        extra_byte_items={
            "num_floating_point_operations_so_far": torch_save_bytes(12.5),
            "opt_param_scheduler.num_steps": torch_save_bytes(3),
        },
    )


def test_torch_dist_report(torch_dist_root):
    report = inspect_checkpoint(torch_dist_root, InspectOptions())
    assert report.detection.fmt == "torch_dist" and report.detection.iteration == 7
    census = report.census
    assert (census.tensors, census.byte_items) == (4, 3)
    assert {s: t.keys for s, t in census.sections.items()} == {
        Section.MODEL: 3,
        Section.OPTIMIZER_STATE: 1,
        Section.COMMON: 1,
        Section.RNG: 1,
        Section.EXTRA_STATE: 1,
    }
    assert census.sections[Section.MODEL].tensor_bytes == 96 + 24 + 8
    assert census.dtypes == {"torch.bfloat16": 2, "torch.float32": 2}
    assert census.chunk_counts == {1: 3, 3: 1}
    largest = census.largest_chunk
    assert (largest.key, largest.nbytes) == ("decoder.layers.0.mlp.linear_fc1.weight", 32)
    common = report.common
    assert common.source == "common_state/shard_0_1"
    assert (common.iteration, common.checkpoint_version) == (7, 3.0)
    assert common.keys == ("args", "checkpoint_version", "iteration")
    assert common.args["swiglu"] is True and common.args["num_experts"] is None
    assert common.args["ckpt_format"] == "torch_dist" and "attention_backend" not in common.args
    assert common.stubbed == (f"{ENUMS}.AttnBackend",)
    assert report.keys == () and report.digests is None


def test_legacy_common_file_wins(torch_dist_root):
    common = {"args": _args(), "iteration": 5, "checkpoint_version": 2.0}
    (torch_dist_root / "iter_0000007" / "common.pt").write_bytes(torch_save_bytes(common, ALIASES))
    report = inspect_checkpoint(torch_dist_root, InspectOptions())
    assert (report.common.source, report.common.iteration) == ("common.pt", 5)


def test_common_state_must_be_a_dict(tmp_path):
    write_torch_dist_store(tmp_path, MODEL, common=None)
    (tmp_path / "common.pt").write_bytes(torch_save_bytes([1, 2]))
    with pytest.raises(IntegrityError, match="common.pt does not hold a common-state dict"):
        inspect_checkpoint(tmp_path, InspectOptions())


def test_fsdp_report(fsdp_store):
    report = inspect_checkpoint(fsdp_store, InspectOptions())
    assert (report.detection.fmt, report.detection.dialect) == ("fsdp_dtensor", "mfsdp_v1")
    assert {s: t.keys for s, t in report.census.sections.items()} == {
        Section.MODEL: 3,
        Section.COMMON: 4,
        Section.SCHEDULER: 1,
    }
    assert report.has_planner_data
    common = report.common
    assert common.source.startswith("byte items args, iteration, checkpoint_version")
    assert common.keys == (
        "args",
        "checkpoint_version",
        "iteration",
        "num_floating_point_operations_so_far",
    )
    assert common.args["use_megatron_fsdp"] is True
    assert common.stubbed == (f"{ENUMS}.AttnBackend",)


def test_key_listing_with_chunks_and_digests(torch_dist_root):
    opts = InspectOptions(keys=re.compile(r"linear_fc"), chunks=True, digests=True, workers=2)
    report = inspect_checkpoint(torch_dist_root, opts)
    listed = {listing.key: listing for listing in report.keys}
    assert sorted(listed) == [
        "decoder.layers.0.mlp.linear_fc1.weight",
        "decoder.layers.0.mlp.linear_fc2.weight",
    ]
    fc1 = listed["decoder.layers.0.mlp.linear_fc1.weight"]
    assert fc1.chunks == tuple(dim0_grid((6, 4), 2)) and fc1.num_chunks == 3
    assert sorted(report.digests) == sorted(listed)
    for key, digest in report.digests.items():
        expected = MODEL[key]
        raw = expected.view(torch.int16) if expected.dtype == torch.bfloat16 else expected
        assert digest.sha256 == hashlib.sha256(raw.numpy().tobytes()).hexdigest()
    assert report.read_stats.amplification == 1.0


def test_cli_text_and_json(torch_dist_root, capsys):
    assert cli.main(["inspect", str(torch_dist_root)]) == 0
    text = capsys.readouterr().out
    assert "format:     torch_dist (mcore)" in text
    assert "iteration:  7" in text
    assert "args.swiglu = True" in text
    assert f"stubbed classes: {ENUMS}.AttnBackend" in text
    assert cli.main(["inspect", str(torch_dist_root), "--json", "--keys", "final_layernorm"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["format"] == "torch_dist" and record["iteration"] == 7
    assert record["sections"]["model"] == {"keys": 3, "tensor_bytes": 128}
    assert [k["key"] for k in record["keys"]] == [
        "decoder.final_layernorm.weight",
        "optimizer.state.exp_avg.decoder.final_layernorm.weight",
    ]
    assert record["common"]["args"]["num_layers"] == 2


def test_cli_writes_digest_file(fsdp_store, tmp_path, capsys):
    out = tmp_path / "digests.json"
    assert cli.main(["inspect", str(fsdp_store), "--digests", str(out), "--workers", "3"]) == 0
    assert f"digests:    3 tensors to {out}" in capsys.readouterr().out
    digests = json.loads(out.read_text())
    assert sorted(digests) == sorted(f"model.module.{k}" for k in MODEL)
    fc1 = digests["model.module.decoder.layers.0.mlp.linear_fc1.weight"]
    assert fc1["shape"] == [6, 4] and fc1["dtype"] == "torch.float32"


def test_cli_exit_codes(tmp_path, fsdp_store, capsys):
    v2 = write_fsdp_store(tmp_path / "v2", MODEL, prefix="model.")
    assert cli.main(["inspect", str(v2)]) == 3
    assert "error: FC-R01:" in capsys.readouterr().err
    assert cli.main(["inspect", str(tmp_path / "missing")]) == 2
    assert cli.main(["inspect", str(fsdp_store), "--workers", "0"]) == 2
    digests = str(tmp_path / "d.json")
    assert (
        cli.main(["inspect", str(fsdp_store), "--digests", digests, "--max-item-bytes", "1"]) == 8
    )
    assert "hint: pass --max-item-bytes" in capsys.readouterr().err
    for bad in (["--keys", "("], ["--max-item-bytes", "lots"]):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["inspect", str(fsdp_store), *bad])
        assert excinfo.value.code == 2


def test_log_level_flag_controls_progress_logs(torch_dist_root, caplog):
    package = "tools.checkpoint.format_converter"
    with caplog.at_level(logging.DEBUG, logger=package):
        assert cli.main(["inspect", str(torch_dist_root)]) == 0
        quiet = [r for r in caplog.records if r.name.startswith(package)]
        caplog.clear()
        assert cli.main(["--log-level", "INFO", "inspect", str(torch_dist_root)]) == 0
        verbose = caplog.text
    assert quiet == []
    assert "per-item budget" in verbose and "is torch_dist (mcore)" in verbose


def test_region_listing_json(torch_dist_root):
    report = inspect_checkpoint(torch_dist_root, InspectOptions(chunks=True))
    record = report.to_json()
    fc1 = next(k for k in record["keys"] if k["key"].endswith("linear_fc1.weight"))
    assert fc1["chunks"][1] == {"offsets": [2, 0], "sizes": [2, 4]}
    blob = next(k for k in record["keys"] if k["key"] == "rng_state/shard_0_0_1")
    assert blob == {
        "key": "rng_state/shard_0_0_1",
        "section": "rng",
        "shape": None,
        "dtype": None,
        "nbytes": 3,
        "num_chunks": 0,
    }
