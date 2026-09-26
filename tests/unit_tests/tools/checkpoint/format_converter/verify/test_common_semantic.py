# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``verify --common semantic``: common states compared field by field, wherever they live."""

from __future__ import annotations

import enum
import json
import re

import pytest
import torch

from tools.checkpoint.format_converter import cli
from tools.checkpoint.format_converter.api import verify_checkpoint
from tools.checkpoint.format_converter.errors import UsageError
from tools.checkpoint.format_converter.options import CompareMode, VerifyOptions
from tools.checkpoint.format_converter.verify.common import canonical, semantic_findings
from tools.checkpoint.format_converter.verify.report import FindingKind

from .._synthetic import (
    TORCH_DIST_CONFIG,
    default_args,
    torch_save_bytes,
    write_fsdp_store,
    write_store,
    write_torch_dist_store,
)

ENUMS = "megatron.core.transformer.enums"
TENSORS = {"decoder.final_layernorm.weight": torch.ones(4)}


class FakeAttnBackend(enum.Enum):
    """Pickled under Megatron's name ``AttnBackend``."""

    flash = 1
    auto = 5


ALIASES = {FakeAttnBackend: (ENUMS, "AttnBackend")}


def _common(**args_overrides):
    return {
        "args": default_args(**{"attention_backend": FakeAttnBackend.auto, **args_overrides}),
        "iteration": 7,
        "checkpoint_version": 3.0,
    }


def _torch_dist(path, common):
    return write_torch_dist_store(path, TENSORS, common=common, common_aliases=ALIASES)


def _verify(a, b, **options):
    return verify_checkpoint(a, b, VerifyOptions(common="semantic", **options))


def _findings(report):
    return {(f.kind, f.key, f.detail) for f in report.findings}


def test_equal_common_states_with_stubbed_classes_match(tmp_path):
    a = _torch_dist(tmp_path / "a", _common())
    b = _torch_dist(tmp_path / "b", _common())
    report = _verify(a, b)
    assert report.ok and report.common_fields_compared == len(vars(_common()["args"])) + 2


def test_args_are_compared_field_by_field(tmp_path):
    a = _torch_dist(tmp_path / "a", _common(lr=3e-4, attention_backend=FakeAttnBackend.flash))
    b = _torch_dist(tmp_path / "b", {**_common(lr=1e-4), "iteration": 8, "optimizer": {}})
    report = _verify(a, b)
    assert _findings(report) == {
        (FindingKind.COMMON, "args.lr", "A 0.0003 vs B 0.0001"),
        (
            FindingKind.COMMON,
            "args.attention_backend",
            f"A <stub {ENUMS}.AttnBackend(1)> vs B <stub {ENUMS}.AttnBackend(5)>",
        ),
        (FindingKind.COMMON, "iteration", "A 7 vs B 8"),
        (FindingKind.COMMON, "optimizer", "only in B"),
    }
    patterns = (r"^args\.", "^iteration$", "^optimizer$")
    assert _verify(a, b, ignore_keys=tuple(re.compile(p) for p in patterns)).ok


def test_legacy_common_pt_matches_the_dcp_item(tmp_path):
    a = _torch_dist(tmp_path / "a", _common())
    legacy = write_store(tmp_path / "legacy", TENSORS)
    (legacy / "metadata.json").write_text(json.dumps(TORCH_DIST_CONFIG))
    (legacy / "common.pt").write_bytes(torch_save_bytes(_common(), ALIASES))
    assert _verify(a, legacy).ok
    bytes_mode = verify_checkpoint(a, legacy, VerifyOptions(common="bytes"))
    assert _findings(bytes_mode) == {
        (FindingKind.ONLY_IN_A, "common_state/shard_0_1", "missing in B")
    }


def test_fsdp_and_torch_dist_common_states_compare(tmp_path):
    common = _common()
    fsdp = write_fsdp_store(tmp_path / "fsdp", TENSORS, args=common["args"], common_aliases=ALIASES)
    torch_dist = _torch_dist(tmp_path / "td", common)
    tensors = re.compile(r"^(model\.module\.)?decoder\.")
    report = _verify(fsdp, torch_dist, mode=CompareMode("meta"), ignore_keys=(tensors,))
    assert report.ok, report.findings
    assert report.common_fields_compared == len(vars(common["args"])) + 2


def test_other_byte_items_are_still_compared_bytewise(tmp_path):
    extra = "decoder.layers.self_attention._extra_state/shard_0_1"
    a = write_torch_dist_store(tmp_path / "a", TENSORS, extra_byte_items={extra: b"one"})
    b = write_torch_dist_store(tmp_path / "b", TENSORS, extra_byte_items={extra: b"two"})
    report = _verify(a, b)
    assert [(f.kind, f.key) for f in report.findings] == [(FindingKind.BYTES, extra)]


def test_canonical_forms():
    assert canonical(float("nan")) == canonical(float("nan"))
    assert canonical(torch.tensor([1.0, -0.0])) != canonical(torch.tensor([1.0, 0.0]))
    assert canonical({"a": [1, (2, 3)]}) == canonical({"a": [1, (2, 3)]})
    assert canonical([1]) != canonical((1,))
    assert canonical({1, 2}) == canonical({2, 1})
    findings, compared = semantic_findings({"iteration": 1}, None, lambda name: False)
    assert compared == 1 and [(f.key, f.detail) for f in findings] == [("iteration", "only in A")]


def test_cli_semantic_mode(tmp_path, capsys):
    a = _torch_dist(tmp_path / "a", _common(lr=1.0))
    b = _torch_dist(tmp_path / "b", _common(lr=5.0))
    assert cli.main(["verify", str(a), str(b), "--common", "semantic"]) == 5
    out = capsys.readouterr().out
    assert "  common     args.lr: A 1.0 vs B 5.0" in out
    assert f"{len(vars(_common()['args'])) + 3} common-state fields" in out
    assert cli.main(["verify", str(a), str(a), "--common", "semantic"]) == 0
    with pytest.raises(UsageError, match="--common semantic compares two stores"):
        VerifyOptions(common="semantic", oracle="x*.json")
