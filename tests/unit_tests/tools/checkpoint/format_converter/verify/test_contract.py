# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Expected-difference contracts: allowed differences are expected, planted ones are findings."""

from __future__ import annotations

import argparse
import re

import pytest
import torch

from tools.checkpoint.format_converter import cli
from tools.checkpoint.format_converter.api import verify_checkpoint
from tools.checkpoint.format_converter.errors import UsageError
from tools.checkpoint.format_converter.options import VerifyOptions
from tools.checkpoint.format_converter.verify.contract import CONTRACTS
from tools.checkpoint.format_converter.verify.report import FindingKind

from .._synthetic import (
    dim0_grid,
    hybrid_args,
    product_grid,
    torch_save_bytes,
    write_fsdp_store,
    write_torch_dist_store,
)

W = "decoder.layers.0.self_attention.linear_qkv.weight"
NORM = "decoder.final_norm.weight"
BF16 = {W: torch.randn(8, 4, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)}
FP32 = {NORM: torch.randn(4, generator=torch.Generator().manual_seed(1))}


def _verify(a, b, contract, **options):
    return verify_checkpoint(a, b, VerifyOptions(expected_diff=contract, **options))


def _kinds(report) -> list[tuple[str, str]]:
    return [(f.kind.value, f.key) for f in report.findings]


def test_contract_ids_match_the_cli():
    assert tuple(CONTRACTS) == cli.EXPECTED_DIFF_CHOICES


def test_a_converted_bf16_store_equals_a_real_fp32_save_after_the_upcast(tmp_path):
    """Contract A: the reference holds fp32 main weights and optimizer, scheduler, rng state."""
    converted = write_fsdp_store(tmp_path / "a", {**BF16, **FP32})
    state = {
        "optimizer.state.module.module.module.decoder.final_norm.weight.exp_avg": torch.ones(4),
        "rng_state.(0, 0).0.random_rng_state": torch.ones(3, dtype=torch.uint8),
    }
    extra = {"opt_param_scheduler.num_steps": torch_save_bytes(3)}
    reference = write_fsdp_store(
        tmp_path / "b",
        {W: BF16[W].float(), **FP32},
        extra_tensors=state,
        extra_byte_items=extra,
        args=hybrid_args(),  # another job's arguments: A skips the common state
    )
    report = _verify(converted, reference, "A")
    assert report.ok, report.findings
    assert {f.key for f in report.expected} == set(state)
    assert report.contract == "A"
    plain = verify_checkpoint(converted, reference, VerifyOptions())
    assert ("dtype", f"model.module.{W}") in _kinds(plain)


def test_contract_a_finds_value_and_key_differences(tmp_path):
    converted = write_fsdp_store(tmp_path / "a", {**BF16, **FP32})
    nudged = BF16[W].float().clone()
    nudged[3, 1] = torch.nextafter(nudged[3, 1], torch.tensor(float("inf")))
    reference = write_fsdp_store(tmp_path / "b", {W: nudged, "decoder.extra.weight": torch.ones(2)})
    assert _kinds(_verify(converted, reference, "A")) == [
        ("only_in_a", f"model.module.{NORM}"),
        ("only_in_b", "model.module.decoder.extra.weight"),
        ("value", f"model.module.{W}"),
    ]


def test_contract_b_rounds_fp32_to_the_reference_bf16(tmp_path):
    converted = write_torch_dist_store(tmp_path / "a", {W: BF16[W].float() + 2**-12})
    rounded = (BF16[W].float() + 2**-12).to(torch.bfloat16)
    extra = "decoder.layers.0.self_attention.core_attention._extra_state"
    reference = write_torch_dist_store(
        tmp_path / "b", {W: rounded, extra: torch.arange(4, dtype=torch.uint8)}
    )
    report = _verify(converted, reference, "B")
    assert report.ok, report.findings
    assert [f.key for f in report.expected] == [extra]
    assert _kinds(_verify(converted, reference, "A")) == [("value", W)]  # an upcast does not round


def test_contract_c_allows_only_the_sanitized_arguments(tmp_path):
    args = hybrid_args(ckpt_format="fsdp_dtensor", use_megatron_fsdp=True)
    real = write_fsdp_store(tmp_path / "real", {W: BF16[W].float()}, args=args)
    sanitized = argparse.Namespace(
        **{**vars(args), "no_save_optim": True, "hybrid_layer_pattern": "*-*-"}
    )
    converted = write_fsdp_store(tmp_path / "conv", BF16, args=sanitized)
    report = _verify(real, converted, "C")
    assert report.ok, report.findings
    assert sorted(f.key for f in report.expected) == [
        "args.hybrid_layer_pattern",
        "args.no_save_optim",
    ]
    other = argparse.Namespace(**{**vars(sanitized), "lr": 1.0})
    tuned = write_fsdp_store(tmp_path / "tuned", BF16, args=other)
    assert _kinds(_verify(real, tuned, "C")) == [("common", "args.lr")]


def _td(path, grid, *, common=None, dtype=torch.bfloat16):
    tensor = torch.arange(48, dtype=torch.float32).reshape(12, 4).to(dtype)
    return write_torch_dist_store(path, {W: tensor}, grids={W: grid}, common=common)


def test_contract_d_accepts_merged_tensor_parallel_splits_only(tmp_path):
    source = _td(tmp_path / "source", dim0_grid((12, 4), 6))
    merged = _td(tmp_path / "merged", dim0_grid((12, 4), 12))
    assert _verify(merged, source, "D").ok
    misaligned = _td(tmp_path / "misaligned", dim0_grid((12, 4), 4))
    (finding,) = _verify(misaligned, source, "D").findings
    assert finding.kind is FindingKind.LAYOUT and "cuts B's chunk" in finding.detail
    wider = _td(tmp_path / "wider", dim0_grid((12, 4), 12), dtype=torch.float32)
    assert _kinds(_verify(wider, source, "D")) == [("dtype", W)]


def test_contract_d_keeps_one_chunk_per_layer_of_a_stacked_tensor(tmp_path):
    """Megatron saves each layer of a tensor stacked over layers as its own chunk (split by
    tensor parallelism): a round trip may merge the splits of a layer, never layers."""
    stacked = "decoder.layers.self_attention.linear_qkv.weight"
    tensor = torch.arange(96, dtype=torch.float32).reshape(3, 8, 4).to(torch.bfloat16)

    def store(name, grid):
        return write_torch_dist_store(tmp_path / name, {stacked: tensor}, grids={stacked: grid})

    source = store("source", product_grid((3, 8, 4), [[1, 2], [4], []]))
    assert _verify(store("per_layer", dim0_grid((3, 8, 4), 1)), source, "D").ok
    (finding,) = _verify(store("whole", dim0_grid((3, 8, 4), 3)), source, "D").findings
    assert finding.kind is FindingKind.LAYOUT and finding.key == stacked
    assert "merges B's chunk" in finding.detail and "with other layers" in finding.detail
    pairs = store("pairs", product_grid((3, 8, 4), [[2], [], []]))
    assert _kinds(_verify(pairs, source, "D")) == [("layout", stacked)]
    whole_source = store("whole_source", dim0_grid((3, 8, 4), 3))  # a single chunk stays one
    assert _verify(store("whole_again", dim0_grid((3, 8, 4), 3)), whole_source, "D").ok


def test_contract_d_allows_only_the_weights_only_flags(tmp_path):
    common = {"args": hybrid_args(), "iteration": 7, "checkpoint_version": 3.0}
    source = _td(tmp_path / "s", dim0_grid((12, 4), 12), common=common)
    weights_only = argparse.Namespace(**{**vars(hybrid_args()), "no_save_optim": True})
    back = _td(tmp_path / "b", dim0_grid((12, 4), 12), common={**common, "args": weights_only})
    report = _verify(back, source, "D")
    assert report.ok and [f.key for f in report.expected] == ["args.no_save_optim"]
    unrestored = argparse.Namespace(**{**vars(weights_only), "pipeline_model_parallel_size": 1})
    content = {**common, "args": unrestored, "content_metadata": {"x": 1}}
    lost = _td(tmp_path / "lost", dim0_grid((12, 4), 12), common=content)
    assert _kinds(_verify(lost, source, "D")) == [
        ("common", "args.pipeline_model_parallel_size"),
        ("common", "content_metadata"),
    ]


RERUN_ITEMS = {
    f"rerun_state_machine_state/shard_{rank}_2": torch_save_bytes({"rerun_requested": False})
    for rank in range(2)
}
"""The rerun items of a model-only save of two data-parallel ranks (``--rerun-mode
validate_results``, Megatron's default)."""


def test_contract_d_allows_rerun_state_only_in_the_source(tmp_path):
    """A model-only save holds rerun state (items and a common-state key); a round trip drops it."""
    common = {"args": hybrid_args(), "iteration": 7, "checkpoint_version": 3.0}
    tensor = torch.arange(48, dtype=torch.float32).reshape(12, 4).to(torch.bfloat16)
    grid = {W: dim0_grid((12, 4), 12)}
    source = write_torch_dist_store(
        tmp_path / "s",
        {W: tensor},
        grids=grid,
        common={**common, "rerun_state_machine": {"mode": "validate_results"}},
        extra_byte_items=RERUN_ITEMS,
    )
    back = write_torch_dist_store(tmp_path / "b", {W: tensor}, grids=grid, common=common)
    report = _verify(back, source, "D")
    assert report.ok, report.findings
    assert [f.key for f in report.expected] == [*sorted(RERUN_ITEMS), "rerun_state_machine"]
    assert _kinds(verify_checkpoint(back, source, VerifyOptions(common="semantic"))) == [
        *(("only_in_b", key) for key in sorted(RERUN_ITEMS)),
        ("common", "rerun_state_machine"),
    ]
    assert _kinds(_verify(source, back, "D")) == [  # never in A only
        *(("only_in_a", key) for key in sorted(RERUN_ITEMS)),
        ("common", "rerun_state_machine"),
    ]


def test_contracts_set_the_common_state_comparison(tmp_path):
    """A re-encoded common state is not byte-identical to its source even when every field is
    equal: C, D and E compare it field by field, and A and B skip it, whatever --common says."""
    common = {"args": hybrid_args(), "iteration": 7, "checkpoint_version": 3.0}
    source = _td(tmp_path / "s", dim0_grid((12, 4), 12), common=common)
    reordered = {key: common[key] for key in reversed(common)}  # the same fields, other bytes
    back = _td(tmp_path / "b", dim0_grid((12, 4), 12), common=reordered)
    plain = verify_checkpoint(back, source, VerifyOptions(common="bytes"))
    assert _kinds(plain) == [("bytes", "common_state/shard_0_1")]
    for contract in ("C", "D"):
        report = _verify(back, source, contract, common="bytes")
        assert report.ok and report.common_fields_compared > 0, (contract, report.findings)
    for contract in ("A", "B"):
        report = _verify(back, source, contract, common="bytes")
        assert report.ok and report.common_fields_compared == report.byte_items_compared == 0
    argv = ["verify", str(back), str(source), "--common", "bytes"]
    assert cli.main([*argv, "--expected-diff", "D"]) == 0
    assert cli.main(argv) == 5


def test_contract_e_allows_dropped_state_and_sanitized_arguments(tmp_path):
    args = hybrid_args(
        "*-*-", ckpt_format="fsdp_dtensor", use_megatron_fsdp=True, pipeline_model_parallel_size=1
    )
    tensors = {W: BF16[W].float()}
    source = write_fsdp_store(
        tmp_path / "s",
        tensors,
        args=args,
        extra_byte_items={"optimizer.param_to_group_meta.module.module.w.lr": b"x"},
    )
    back_args = argparse.Namespace(**{**vars(args), "no_save_optim": True, "no_save_rng": True})
    back = write_fsdp_store(
        tmp_path / "b", tensors, grids={W: dim0_grid((8, 4), 2)}, args=back_args
    )
    report = _verify(back, source, "E")
    assert report.ok, report.findings
    assert {f.key for f in report.expected} == {
        "optimizer.param_to_group_meta.module.module.w.lr",
        "args.no_save_optim",
        "args.no_save_rng",
    }


def test_contracts_compare_two_stores_only():
    with pytest.raises(UsageError, match="--expected-diff must be one of"):
        VerifyOptions(expected_diff="Z")
    with pytest.raises(UsageError, match="not an oracle"):
        VerifyOptions(expected_diff="A", oracle="d*.json")


def test_cli_reports_expected_differences(tmp_path, capsys):
    source = _td(tmp_path / "source", dim0_grid((12, 4), 6))
    merged = _td(tmp_path / "merged", dim0_grid((12, 4), 12))
    assert cli.main(["verify", str(merged), str(source), "--expected-diff", "D"]) == 0
    assert "expected:   0 differences allowed by contract D" in capsys.readouterr().out
    misaligned = _td(tmp_path / "misaligned", dim0_grid((12, 4), 5))
    assert cli.main(["verify", str(misaligned), str(source), "--expected-diff", "D"]) == 5
    assert re.search(r"layout +decoder", capsys.readouterr().out)
