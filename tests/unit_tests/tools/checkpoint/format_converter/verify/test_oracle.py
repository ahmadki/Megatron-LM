# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Digest-oracle verification with pluggable key maps.

Oracle files are written by hand here (hashlib over numpy bytes of the original
tensors), never by the code under test.
"""

from __future__ import annotations

import hashlib
import json
import re
import textwrap

import pytest
import torch

from tools.checkpoint.format_converter import cli
from tools.checkpoint.format_converter.api import verify_checkpoint
from tools.checkpoint.format_converter.errors import ExitCode, IntegrityError, UsageError
from tools.checkpoint.format_converter.options import CompareMode, VerifyOptions
from tools.checkpoint.format_converter.verify.report import FindingKind

from .._synthetic import dim0_grid, write_fsdp_store

EXPERTS_PER_FILE = 2
FILES = 2
FC1 = "decoder.layers.0.mlp.experts.linear_fc1.weight"


def _record(tensor: torch.Tensor) -> dict:
    raw = tensor.view(torch.int16) if tensor.dtype == torch.bfloat16 else tensor
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
    }


def _model() -> dict[str, torch.Tensor]:
    torch.manual_seed(3)
    model = {
        "decoder.final_layernorm.weight": torch.randn(8).to(torch.bfloat16),
        "embedding.word_embeddings.weight": torch.randn(16, 8),
    }
    for e in range(EXPERTS_PER_FILE * FILES):
        model[f"{FC1}{e}"] = torch.randn(4, 8).to(torch.bfloat16)
    return model


def _write_oracle(directory, model, *, tamper=None) -> str:
    """Per-rank files with rank-local expert numbers; replicated keys in every file."""
    directory.mkdir(parents=True, exist_ok=True)
    for rank in range(FILES):
        records = {
            "decoder.final_layernorm.weight": _record(model["decoder.final_layernorm.weight"]),
            "embedding.word_embeddings.weight": _record(model["embedding.word_embeddings.weight"]),
        }
        for local in range(EXPERTS_PER_FILE):
            records[f"{FC1}{local}"] = _record(model[f"{FC1}{local + EXPERTS_PER_FILE * rank}"])
        if tamper is not None:
            tamper(rank, records)
        (directory / f"digests_rank{rank}.json").write_text(json.dumps(records))
    return str(directory / "digests_rank*.json")


def _write_key_map(path) -> str:
    path.write_text(textwrap.dedent(f"""
            import re

            EXPERT = re.compile(r"^(?P<stem>.*linear_fc1\\.weight)(?P<local>\\d+)$")

            def to_store_key(oracle_key, file_index):
                match = EXPERT.match(oracle_key)
                if match:
                    index = int(match.group("local")) + {EXPERTS_PER_FILE} * file_index
                    oracle_key = match.group("stem") + str(index)
                return "model.module." + oracle_key
            """))
    return str(path)


@pytest.fixture(name="setup")
def fixture_setup(tmp_path):
    model = _model()
    store = write_fsdp_store(
        tmp_path / "store", model, grids={"embedding.word_embeddings.weight": dim0_grid((16, 8), 5)}
    )
    key_map = _write_key_map(tmp_path / "key_map.py")
    return tmp_path, model, store, key_map


def _verify(store, pattern, key_map, **kwargs):
    return verify_checkpoint(
        store, None, VerifyOptions(oracle=pattern, oracle_key_map=key_map, **kwargs)
    )


def _kinds(report) -> dict[str, list[str]]:
    kinds: dict[str, list[str]] = {}
    for finding in report.findings:
        kinds.setdefault(finding.kind.value, []).append(finding.key)
    return kinds


def test_prefix_and_per_file_expert_remap(setup):
    tmp_path, model, store, key_map = setup
    report = _verify(store, _write_oracle(tmp_path / "oracle", model), key_map, workers=2)
    assert report.ok, report.findings
    assert report.keys_compared == report.values_compared == len(model)
    assert "2 files" in report.reference and report.read_stats.amplification == 1.0


def test_identity_map_reports_every_key_as_unmatched(setup):
    tmp_path, model, store, _ = setup
    report = verify_checkpoint(
        store, None, VerifyOptions(oracle=_write_oracle(tmp_path / "o", model))
    )
    kinds = _kinds(report)
    assert len(kinds["only_in_a"]) == len(model)
    assert "only_in_b" in kinds


def test_disagreeing_duplicates_are_reported(setup):
    tmp_path, model, store, key_map = setup

    def tamper(rank, records):
        if rank == 1:
            records["decoder.final_layernorm.weight"]["sha256"] = "0" * 64

    report = _verify(store, _write_oracle(tmp_path / "oracle", model, tamper=tamper), key_map)
    finding = next(f for f in report.findings if f.kind == FindingKind.ORACLE_INCONSISTENT)
    assert finding.key == "model.module.decoder.final_layernorm.weight"
    assert "digests_rank0.json:" in finding.detail and "digests_rank1.json:" in finding.detail


def test_missing_and_extra_keys(setup):
    tmp_path, model, store, key_map = setup

    def tamper(rank, records):
        if rank == 0:
            del records[f"{FC1}0"]
            records["decoder.bonus.weight"] = _record(torch.zeros(2))

    report = _verify(store, _write_oracle(tmp_path / "oracle", model, tamper=tamper), key_map)
    assert _kinds(report) == {
        "only_in_a": [f"model.module.{FC1}0"],
        "only_in_b": ["model.module.decoder.bonus.weight"],
    }


def test_sha_mismatch_is_found(setup):
    tmp_path, model, _, key_map = setup
    pattern = _write_oracle(tmp_path / "oracle", model)
    corrupted = {k: v.clone() for k, v in model.items()}
    corrupted["embedding.word_embeddings.weight"][7, 3] += 1
    store = write_fsdp_store(
        tmp_path / "corrupted",
        corrupted,
        grids={"embedding.word_embeddings.weight": dim0_grid((16, 8), 5)},
    )
    report = _verify(store, pattern, key_map)
    assert _kinds(report) == {"value": ["model.module.embedding.word_embeddings.weight"]}
    assert "digests_rank0.json:embedding.word_embeddings.weight" in report.findings[0].detail
    assert _verify(store, pattern, key_map, mode=CompareMode("meta")).ok


def test_meta_mode_checks_shape_and_dtype_without_reading(setup, monkeypatch):
    tmp_path, model, store, key_map = setup

    def tamper(rank, records):
        records[f"{FC1}1"]["shape"] = [4, 9]
        records[f"{FC1}0"]["dtype"] = "torch.float32"

    pattern = _write_oracle(tmp_path / "oracle", model, tamper=tamper)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("meta mode read a tensor"))
    report = _verify(store, pattern, key_map, mode=CompareMode("meta"))
    assert _kinds(report) == {
        "shape": [f"model.module.{FC1}1", f"model.module.{FC1}3"],
        "dtype": [f"model.module.{FC1}0", f"model.module.{FC1}2"],
    }
    assert report.values_compared == 0


def test_sample_mode_and_skipped_keys(setup, tmp_path):
    _, model, store, _ = setup
    skip_map = tmp_path / "skip.py"
    skip_map.write_text(
        "def to_store_key(key, file_index):\n"
        "    return None if 'experts' in key or file_index else 'model.module.' + key\n"
    )
    pattern = _write_oracle(tmp_path / "oracle2", model)
    report = verify_checkpoint(
        store,
        None,
        VerifyOptions(
            oracle=pattern,
            oracle_key_map=skip_map,
            mode=CompareMode.parse("sample:1"),
            seed=5,
            ignore_keys=(re.compile("experts"),),
        ),
    )
    assert report.ok and report.keys_compared == 2 and report.values_compared == 1


def test_usage_and_integrity_errors(setup, tmp_path):
    _, model, store, key_map = setup
    pattern = _write_oracle(tmp_path / "oracle3", model)
    with pytest.raises(UsageError, match="matches no file"):
        _verify(store, str(tmp_path / "nothing*.json"), key_map)
    no_function = tmp_path / "empty_map.py"
    no_function.write_text("x = 1\n")
    with pytest.raises(UsageError, match="must define to_store_key"):
        _verify(store, pattern, no_function)
    broken = tmp_path / "broken_map.py"
    broken.write_text("raise RuntimeError('boom')\n")
    with pytest.raises(UsageError, match="failed to load"):
        _verify(store, pattern, broken)
    wrong_type = tmp_path / "int_map.py"
    wrong_type.write_text("def to_store_key(key, file_index):\n    return 3\n")
    with pytest.raises(UsageError, match="expected str or None"):
        _verify(store, pattern, wrong_type)
    raising = tmp_path / "raising_map.py"
    raising.write_text("def to_store_key(key, file_index):\n    raise KeyError(key)\n")
    with pytest.raises(UsageError, match=r"to_store_key\('.*', 0\) raised KeyError") as excinfo:
        _verify(store, pattern, raising)
    assert excinfo.value.exit_code == ExitCode.USAGE
    bad = tmp_path / "bad" / "digests.json"
    bad.parent.mkdir()
    bad.write_text(json.dumps({"k": {"shape": [2], "dtype": "torch.float32", "sha256": "xyz"}}))
    with pytest.raises(IntegrityError, match="record of 'k' must be"):
        _verify(store, str(bad), None)
    with pytest.raises(UsageError, match="either a second store B or --oracle GLOB, not both"):
        verify_checkpoint(store, store, VerifyOptions(oracle=pattern))
    with pytest.raises(UsageError, match="--oracle-key-map needs --oracle"):
        VerifyOptions(oracle_key_map=no_function)
    with pytest.raises(UsageError, match="--layout"):
        VerifyOptions(oracle=pattern, layout=True)


def test_cli_round_trip_with_inspect_digests(setup, tmp_path, capsys):
    """Plumbing check: inspect --digests output is a valid oracle for the same store."""
    _, model, store, _ = setup
    digests = tmp_path / "own.json"
    assert cli.main(["inspect", str(store), "--digests", str(digests)]) == 0
    assert cli.main(["verify", str(store), "--oracle", str(digests)]) == 0
    assert "verify:     OK" in capsys.readouterr().out
    records = json.loads(digests.read_text())
    records[f"model.module.{FC1}2"]["sha256"] = "f" * 64
    digests.write_text(json.dumps(records))
    report = tmp_path / "report.jsonl"
    args = ["verify", str(store), "--oracle", str(digests), "--report", str(report)]
    assert cli.main(args) == ExitCode.VERIFY_MISMATCH
    lines = [json.loads(line) for line in report.read_text().splitlines()]
    assert [line["kind"] for line in lines] == ["value", "summary"]
    assert len(model) == lines[-1]["values_compared"]
