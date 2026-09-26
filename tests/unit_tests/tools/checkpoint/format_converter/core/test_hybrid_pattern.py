# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hybrid layer patterns are parsed like Megatron's ``parse_hybrid_pattern``."""

from __future__ import annotations

import argparse

import pytest

from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.hybrid_pattern import (
    HybridPattern,
    hybrid_pattern_of,
    parse_hybrid_pattern,
)


@pytest.mark.parametrize(
    ("text", "main", "mtp"),
    [
        ("M*M*", "M*M*", ()),
        ("M*M*/MM/MM", "M*M*", ("MM", "MM")),
        ("M-M-|M-M*-/MM/MM", "M-M-M-M*-", ("MM", "MM")),
        ("*-|*-|", "*-*-", ()),
        ("M*/", "M*", ()),  # empty MTP parts mean no MTP
        ("/*E", "", ("*E",)),
    ],
)
def test_parse(text, main, mtp):
    parsed = parse_hybrid_pattern(text)
    assert parsed == HybridPattern(text, main, mtp)
    assert parsed.symbols == frozenset(main + "".join(mtp))


def _args(**fields):
    return ArgsView(argparse.Namespace(**fields))


def test_pattern_arguments_and_their_legacy_forms():
    assert hybrid_pattern_of(_args(hybrid_layer_pattern=None)) is None
    assert hybrid_pattern_of(_args()) is None
    assert hybrid_pattern_of(_args(hybrid_layer_pattern="*-")).main == "*-"
    assert hybrid_pattern_of(_args(hybrid_override_pattern="M*")).main == "M*"
    legacy = _args(
        hybrid_layer_pattern=None,
        hybrid_override_pattern="M*",
        mtp_hybrid_override_pattern="*E",
        mtp_num_layers=2,
    )
    assert hybrid_pattern_of(legacy) == HybridPattern("M*/*E/*E", "M*", ("*E", "*E"))
    unified = _args(
        hybrid_layer_pattern="M*/*-", mtp_hybrid_override_pattern="*E", mtp_num_layers=1
    )
    assert hybrid_pattern_of(unified).mtp == ("*-",)
    no_depths = _args(hybrid_layer_pattern=None, mtp_hybrid_override_pattern="*E", mtp_num_layers=0)
    assert hybrid_pattern_of(no_depths) is None
