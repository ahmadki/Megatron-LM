# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The logical pool claims every slice exactly once; the registry orders rules and sentinels."""

from __future__ import annotations

import json
import re
import subprocess
import sys

import pytest
import torch

from tools.checkpoint.format_converter.core.logical import (
    LogicalCopy,
    LogicalId,
    LogicalOutput,
    LogicalSlice,
)
from tools.checkpoint.format_converter.core.registry import RuleRegistry, Sentinel
from tools.checkpoint.format_converter.core.rule import FamilyInstance, LeafRule, LogicalPool, Stage
from tools.checkpoint.format_converter.engine.regions import Region
from tools.checkpoint.format_converter.errors import InternalError

from .._harness import REPO_ROOT


def _slice(layer: int, leaf: str = "weight") -> LogicalSlice:
    lid = LogicalId("decoder", layer, "mlp.linear_fc1", leaf)
    key = f"decoder.layers.{layer}.mlp.linear_fc1.{leaf}"
    return LogicalSlice(lid, key, key, Region((0,), (4,)), torch.float32)


def _copy_of(piece: LogicalSlice) -> FamilyInstance:
    full = Region.full(piece.shape)
    output = LogicalOutput(
        piece.lid, piece.shape, piece.dtype, (LogicalCopy(piece.lid, full, full),)
    )
    return FamilyInstance("copy", (piece,), (output,))


def test_take_claims_each_slice_once_in_pool_order():
    slices = [_slice(i) for i in range(4)]
    pool = LogicalPool(slices)
    assert pool.take("copy", lambda s: s.lid.layer % 2 == 1) == [slices[1], slices[3]]
    assert pool.take("other", lambda s: True) == [slices[0], slices[2]]
    assert pool.take("late", lambda s: True) == []
    assert pool.remaining() == []
    assert pool.claims() == {"copy": 2, "other": 2}


def test_consuming_requires_the_claiming_rule_and_happens_once():
    first, second = _slice(0), _slice(1)
    pool = LogicalPool([first, second])
    pool.take("copy", lambda s: s is first)
    with pytest.raises(InternalError, match="nobody claimed"):
        pool.consume(_copy_of(second))
    pool.take("other", lambda s: s is second)
    with pytest.raises(InternalError, match="rule other claimed"):
        pool.consume(_copy_of(second))
    pool.consume(_copy_of(first))
    with pytest.raises(InternalError, match="twice"):
        pool.consume(_copy_of(first))
    assert pool.unconsumed_claims() == [second]


def test_declined_slices_keep_their_reasons():
    piece = _slice(0)
    pool = LogicalPool([piece])
    pool.decline(piece, "reserved for fused projections")
    assert pool.remaining() == [piece]
    assert pool.declined(piece.lid) == ("reserved for fused projections",)
    assert pool.declined(_slice(1).lid) == ()


def test_duplicate_logical_ids_are_internal_errors():
    piece = _slice(0)
    twin = LogicalSlice(piece.lid, "elsewhere", "elsewhere", piece.region, piece.dtype)
    with pytest.raises(InternalError, match="held by both"):
        LogicalPool([piece, twin])


def test_instances_copy_only_from_their_inputs():
    first, second = _slice(0), _slice(1)
    full = Region((0,), (4,))
    foreign = LogicalOutput(first.lid, (4,), torch.float32, (LogicalCopy(second.lid, full, full),))
    with pytest.raises(InternalError, match="not an input"):
        FamilyInstance("copy", (first,), (foreign,))
    with pytest.raises(InternalError, match="input twice"):
        FamilyInstance("copy", (first, first), ())


def _rule(rule_name: str, rule_stage: Stage, *, active: bool = True) -> type[LeafRule]:
    class _Rule(LeafRule):
        name = rule_name
        stage = rule_stage

        def enabled(self, ctx):
            return active

        def claim(self, pool, ctx):
            return ()

    return _Rule


def test_rules_run_in_stage_then_name_order_when_enabled():
    registry = RuleRegistry()
    for name, stage, active in [
        ("identity", Stage.IDENTITY, True),
        ("b_fused", Stage.FUSED, True),
        ("a_fused", Stage.FUSED, True),
        ("glu", Stage.GLU, False),
    ]:
        registry.rule(_rule(name, stage, active=active))
    assert [rule.name for rule in registry.rules_for(ctx=None)] == [
        "a_fused",
        "b_fused",
        "identity",
    ]
    assert set(registry.rules) == {"identity", "a_fused", "b_fused", "glu"}
    with pytest.raises(InternalError, match="registered twice"):
        registry.rule(_rule("glu", Stage.GLU))


def test_sentinels_are_named_owned_and_per_format():
    registry = RuleRegistry()
    parts = Sentinel(
        "parts", "torch_dist", re.compile(r"\.weight\.[a-z]"), ("FC-P09",), "parameter parts"
    )
    registry.sentinel(parts)
    registry.sentinel(
        Sentinel("halves", "fsdp_dtensor", re.compile(r"_[wv]$"), ("glu",), "GLU halves")
    )
    assert registry.sentinels_for("torch_dist") == (parts,)
    assert parts.matches("decoder.layers.0.mixer.in_proj.weight.z")
    assert not parts.matches("decoder.layers.0.mixer.in_proj.weight")
    with pytest.raises(InternalError, match="registered twice"):
        registry.sentinel(parts)
    with pytest.raises(InternalError, match="no owner"):
        registry.sentinel(Sentinel("orphan", "torch_dist", re.compile("x"), (), "none"))


@pytest.mark.usefixtures("local_rank_zero")
def test_importing_the_rules_alone_registers_the_sentinels(scrubbed_env):
    """Fail-closed must not depend on another module having imported the refusal fence."""
    script = (
        "import json\n"
        "from tools.checkpoint.format_converter import rules\n"
        "from tools.checkpoint.format_converter.core.registry import REGISTRY\n"
        "print(json.dumps(sorted({sentinel.fmt for sentinel in REGISTRY.sentinels})))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=scrubbed_env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["fsdp_dtensor", "torch_dist"]
