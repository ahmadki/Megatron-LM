# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Plan replay (``verify --map convert``) finds planted engine faults and says it is not independent."""

from __future__ import annotations

import json
import shutil

import pytest
import torch

from tools.checkpoint.format_converter import cli
from tools.checkpoint.format_converter.api import convert_checkpoint, verify_checkpoint
from tools.checkpoint.format_converter.errors import RefusedError, UsageError
from tools.checkpoint.format_converter.options import ConvertOptions, VerifyOptions

from .._synthetic import (
    TORCH_DIST_CONFIG,
    hybrid_args,
    hybrid_tensors,
    torch_save_bytes,
    write_fsdp_store,
    write_store,
)

ITER = "iter_0000007"
MODEL = hybrid_tensors()
QKV = "decoder.layers.0.self_attention.linear_qkv.weight"
COMMON = {"args": hybrid_args(), "iteration": 7, "checkpoint_version": 3.0}
REPLAY = VerifyOptions(map="convert")


@pytest.fixture(name="source")
def fixture_source(tmp_path):
    store = write_store(
        tmp_path / "src" / ITER,
        {**MODEL, "frontend.proj.weight": torch.ones(4)},
        byte_items={"common_state/shard_0_1": torch_save_bytes([COMMON])},
    )
    (store / "metadata.json").write_text(json.dumps(TORCH_DIST_CONFIG))
    return store


def test_a_faithful_conversion_replays_cleanly(tmp_path, source, capsys):
    options = ConvertOptions(to_format="fsdp_dtensor", workers=0, allow_roots=("frontend",))
    convert_checkpoint(source, tmp_path / "out" / ITER, options)
    report = verify_checkpoint(source, tmp_path / "out" / ITER, REPLAY)
    assert report.ok and not report.independent
    assert report.values_compared == len(MODEL) + 1  # the options come from the manifest
    assert "not independent" in report.reference
    assert cli.main(["verify", str(source), str(tmp_path / "out" / ITER), "--map", "convert"]) == 0
    assert "note:       not an independent check" in capsys.readouterr().out


def test_replay_finds_values_keys_and_dtypes_the_engine_got_wrong(tmp_path, source):
    wrong = {**MODEL, "frontend.proj.weight": torch.ones(4)}
    wrong[QKV] = wrong[QKV].clone()
    wrong[QKV][5, 2] += 1
    wrong["decoder.final_norm.weight"] = wrong["decoder.final_norm.weight"].double()
    del wrong["output_layer.weight"]
    broken = write_fsdp_store(tmp_path / "broken" / ITER, wrong, args=hybrid_args())
    convert_checkpoint(
        source,
        tmp_path / "good" / ITER,
        ConvertOptions(to_format="fsdp_dtensor", workers=0, allow_roots=("frontend",)),
    )
    shutil.copy(tmp_path / "good" / ITER / "conversion_manifest.json", broken)  # its options
    report = verify_checkpoint(source, broken, REPLAY)
    assert [(f.kind.value, f.key) for f in report.findings] == [
        ("only_in_a", "model.module.output_layer.weight"),
        ("dtype", "model.module.decoder.final_norm.weight"),
        ("value", f"model.module.{QKV}"),
    ]
    assert "1 of 192 elements differ bitwise; first at (5, 2)" in report.findings[-1].detail


def test_replay_refuses_what_conversion_refuses(tmp_path, source):
    broken = write_fsdp_store(tmp_path / "plain" / ITER, MODEL, args=hybrid_args())
    with pytest.raises(RefusedError) as excinfo:  # no manifest: frontend is an unknown root
        verify_checkpoint(source, broken, REPLAY)
    assert excinfo.value.refusal.code == "FC-R13"


def test_replay_takes_no_contract_or_layout():
    with pytest.raises(UsageError, match="--map convert"):
        VerifyOptions(map="convert", expected_diff="D")
    with pytest.raises(UsageError, match="--map convert"):
        VerifyOptions(map="convert", layout=True)
