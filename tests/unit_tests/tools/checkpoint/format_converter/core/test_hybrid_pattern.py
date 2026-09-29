# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hybrid layer patterns are parsed like Megatron's ``parse_hybrid_pattern``, and their layers
grouped into modules like ``group_layers_into_shortcut_blocks`` groups them (hand-built tables;
``parity/test_parity_shortcut.py`` runs Megatron's own grouping on the same patterns)."""

from __future__ import annotations

import argparse

import pytest

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.hybrid_pattern import (
    HybridPattern,
    group_modules,
    hybrid_pattern_of,
    parse_hybrid_pattern,
)
from tools.checkpoint.format_converter.errors import RefusedError


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


@pytest.mark.parametrize(
    ("text", "starts"),
    [("*-*-", set()), ("*-|*-", {2}), ("*E|*E|-/*E", {2, 4}), ("*-|*-|", {2, 4}), ("|*-", {0})],
)
def test_stage_starts_are_the_positions_after_each_pipe(text, starts):
    assert parse_hybrid_pattern(text).stage_starts == starts


def _grouped(text: str, shortcut: bool = True) -> list[tuple[int, str]]:
    pattern = parse_hybrid_pattern(text)
    return list(group_modules(pattern.main, shortcut=shortcut, stage_starts=pattern.stage_starts))


@pytest.mark.parametrize(
    ("text", "modules"),
    [
        ("E*E", [(0, "E"), (1, "*E")]),  # a leading E stays a module of its own
        ("*EE", [(0, "*E"), (2, "E")]),  # and so does an E after a pair
        ("*EE*", [(0, "*E"), (2, "E"), (3, "*")]),
        ("M-*E", [(0, "M"), (1, "-"), (2, "*E")]),
        ("*E|*E", [(0, "*E"), (2, "*E")]),
        ("*E-*E-", [(0, "*E"), (2, "-"), (3, "*E"), (5, "-")]),
        (
            "ME*EGE+EDECEHEWE",  # every layer that may precede a shortcut MoE layer
            [
                (0, "ME"),
                (2, "*E"),
                (4, "GE"),
                (6, "+E"),
                (8, "DE"),
                (10, "CE"),
                (12, "HE"),
                (14, "WE"),
            ],
        ),
        ("E", [(0, "E")]),
        ("*", [(0, "*")]),
        ("", []),
    ],
)
def test_shortcut_moe_layers_join_the_layer_before_them(text, modules):
    assert _grouped(text) == modules


@pytest.mark.parametrize("text", ["E*E", "*EE"])
def test_the_dense_indices_are_not_the_positions_of_the_other_layers(text):
    """Counting the positions of the layers that are not ``E`` gives the wrong map wherever an
    ``E`` stays a module of its own."""
    dense_to_td = [position for position, _ in _grouped(text)]
    assert dense_to_td != [position for position, symbol in enumerate(text) if symbol != "E"]


def test_without_shortcut_every_layer_is_a_module():
    assert _grouped("E*E|-", shortcut=False) == [(0, "E"), (1, "*"), (2, "E"), (3, "-")]
    assert _grouped("-E", shortcut=False) == [(0, "-"), (1, "E")]


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("-E", "layer 1 ('E') follows layer 0 ('-'), which cannot precede"),
        ("EE", "layer 1 ('E') follows layer 0 ('E'), which cannot precede"),
        ("*-E", "layer 2 ('E') follows layer 1 ('-'), which cannot precede"),
        ("*|E*E", "layer 1 ('E') begins a pipeline stage"),
        ("*E|E", "layer 2 ('E') begins a pipeline stage"),
        ("*-|E*E", "layer 2 ('E') begins a pipeline stage"),
    ],
)
def test_groupings_megatron_cannot_build_are_refused(text, reason):
    with pytest.raises(RefusedError) as excinfo:
        _grouped(text)
    refusal = excinfo.value.refusal
    assert refusal.code == errors.FC_R11
    assert refusal.message.startswith(
        "with args.moe_shortcut_connection=True, Megatron cannot group the hybrid layers "
        f"{parse_hybrid_pattern(text).main!r}: {reason}"
    )
