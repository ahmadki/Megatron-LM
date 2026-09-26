# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parity pins against Megatron's own objects (the training container; skipped elsewhere)."""

from __future__ import annotations

import pytest

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.hybrid_pattern import parse_hybrid_pattern
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, TORCH_DIST
from tools.checkpoint.format_converter.core.refusals import (
    GDP_SPEC_NAMES,
    HYBRID_SPEC_MODULES,
    Facts,
    grouped_tensor_parallel,
)
from tools.checkpoint.format_converter.core.registry import REGISTRY, Allowlist
from tools.checkpoint.format_converter.mcore_state.common_state import megatron_available

if not megatron_available():
    pytest.skip("needs Megatron (the training container)", allow_module_level=True)

# pylint: disable=wrong-import-position
import argparse
import importlib

from megatron.core.model_parallel_config import resolve_tensor_parallel_weight_shards
from megatron.core.models.hybrid import hybrid_layer_allocation
from megatron.training.arguments import add_megatron_arguments


@pytest.mark.parametrize(
    "pattern", ["*-", "M*M*/MM/MM", "M-M-|M-M*-/MM/MM", "*-|*-|", "M*/", "*E*E/*E"]
)
def test_patterns_parse_like_megatron(pattern):
    theirs = hybrid_layer_allocation.parse_hybrid_pattern(pattern)
    ours = parse_hybrid_pattern(pattern)
    assert ours.main == (theirs.main_pattern or "").replace("|", "")
    assert ours.mtp == (theirs.mtp_pattern,) * theirs.mtp_num_depths


def test_the_verified_specs_are_megatrons_default_hybrid_stack_spec():
    default = importlib.import_module(HYBRID_SPEC_MODULES[0]).hybrid_stack_spec
    for name in REGISTRY.allowed(Allowlist.STACK_SPECS):
        module, _, attribute = name.rpartition(".")
        assert getattr(importlib.import_module(module), attribute) is default, name


def test_the_gated_deltaproduct_spec_names_exist():
    specs = importlib.import_module(HYBRID_SPEC_MODULES[0])
    (spec,) = {id(getattr(specs, name)) for name in GDP_SPEC_NAMES}
    assert spec != id(specs.hybrid_stack_spec)


def _validated_args(argv: list[str]) -> argparse.Namespace:
    """Megatron's arguments for ``argv`` with the weight-shard fields ``validate_args`` sets."""
    parser = add_megatron_arguments(argparse.ArgumentParser())
    args = parser.parse_known_args(argv)[0]
    if args.expert_tensor_parallel_size is None:  # validate_args defaults it first
        args.expert_tensor_parallel_size = args.tensor_model_parallel_size
    args.tensor_parallel_num_weight_shards, args.gtp_weight_remat_size = (
        resolve_tensor_parallel_weight_shards(
            args.tensor_model_parallel_size,
            args.tensor_parallel_num_weight_shards,
            getattr(args, "gtp_weight_remat_size", 1),
        )
    )
    args.expert_tensor_parallel_num_weight_shards, args.expert_gtp_weight_remat_size = (
        resolve_tensor_parallel_weight_shards(
            args.expert_tensor_parallel_size,
            args.expert_tensor_parallel_num_weight_shards,
            getattr(args, "expert_gtp_weight_remat_size", 1),
        )
    )
    return args


def _gtp_codes(args: argparse.Namespace) -> list[str]:
    facts = Facts(TORCH_DIST, FSDP_DTENSOR, ArgsView(args), {})
    return [refusal.code for refusal in grouped_tensor_parallel(facts)]


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_the_weight_shards_validate_args_stores_for_plain_tensor_parallelism_pass(tp):
    """Every job saves ``tensor_parallel_num_weight_shards``, as many as tensor-parallel ranks
    without grouped tensor parallelism (and the expert pair likewise)."""
    args = _validated_args(["--tensor-model-parallel-size", str(tp)])
    assert args.tensor_parallel_num_weight_shards == tp
    assert args.expert_tensor_parallel_num_weight_shards == tp
    assert _gtp_codes(args) == []
    expert = _validated_args(
        ["--tensor-model-parallel-size", str(tp), "--expert-tensor-parallel-size", "1"]
    )
    assert expert.expert_tensor_parallel_num_weight_shards == 1 and _gtp_codes(expert) == []


@pytest.mark.parametrize(
    "flag", ["--tensor-parallel-num-weight-shards", "--expert-tensor-parallel-num-weight-shards"]
)
def test_the_weight_shard_flags_of_grouped_tensor_parallelism_are_refused(flag):
    args = _validated_args(["--tensor-model-parallel-size", "2", flag, "4"])
    assert _gtp_codes(args) == [errors.FC_R06]
