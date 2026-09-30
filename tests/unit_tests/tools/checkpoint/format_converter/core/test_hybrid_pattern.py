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
    family_mismatch,
    group_modules,
    hybrid_pattern_of,
    parse_hybrid_pattern,
    shortcut_warnings,
)
from tools.checkpoint.format_converter.errors import InternalError, RefusedError


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


QKV = ("self_attention.linear_qkv.weight", "self_attention.linear_qkv.layer_norm_weight")
MLP = ("mlp.linear_fc1.weight", "mlp.linear_fc1.layer_norm_weight", "mlp.linear_fc2.weight")
MOE = (
    "pre_mlp_layernorm.weight",
    "mlp.router.weight",
    "mlp.router.expert_bias",
    "mlp.experts.linear_fc1.weight3",
    "mlp.shared_experts.linear_fc1.weight",
)


def _under(child: str, names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{child}.{name}" for name in names)


@pytest.mark.parametrize(
    ("symbols", "names"),
    [
        ("*", QKV + ("self_attention.linear_proj.weight",)),
        ("-", MLP),
        ("E", MOE),  # shared experts are no dense MLP
        ("M", ("mixer.in_proj.weight.z", "mixer.conv1d.weight.V0", "mixer.A_log")),  # parts
        ("M", ("mixer.in_proj.weight", "mixer.conv1d_weight", "mixer.norm.weight")),  # fused
        ("G", ("self_attention.in_proj.weight", "self_attention.conv1d.weight")),
        (
            "+",
            ("self_attention.linear_q_down_proj.weight", "self_attention.linear_kv_up_proj.weight"),
        ),
        ("*E", _under("attn_layer", QKV) + _under("moe_layer", MOE)),
        (
            "*E",  # child names are not assumed; the pair may hold tensors of its own
            _under("first", QKV)
            + _under("second", MOE)
            + ("shortcut_pre_mlp_layernorm.weight", "shortcut_post_norm.weight", "gate.scale"),
        ),
        ("ME", ("x.mixer.in_proj.weight", "y.mlp.router.weight")),
    ],
)
def test_a_module_holding_its_layers_families_fits(symbols, names):
    assert family_mismatch(symbols, names) is None


@pytest.mark.parametrize(
    ("symbols", "names", "reason"),
    [
        (
            "-",
            MOE,
            "its tensors are those of 'E' (mlp.router.weight), but '-' means '-' (mlp.linear_fc1)",
        ),
        (
            "*",
            QKV + MLP,  # a GPT-style layer: attention and MLP in one module
            "its tensors are those of '*' (self_attention.linear_qkv), '-' (mlp.linear_fc1), but "
            "'*' means '*' (self_attention.linear_qkv)",
        ),
        ("*", (), "its tensors are those of no layer type, but '*' means"),
        ("*E", QKV, "its tensors are those of '*' (self_attention.linear_qkv), but '*E' means"),
        (
            "*E",
            _under("a", QKV) + _under("b", MOE) + _under("c", MLP),  # a second family
            "its tensors are those of '*' (self_attention.linear_qkv), '-' (mlp.linear_fc1), 'E'",
        ),
        (
            "*E",
            QKV + _under("moe_layer", MOE),
            "a shortcut pair holds each layer in a child module of its own, but it has its '*' "
            "tensors in the pair itself and its 'E' tensors in moe_layer",
        ),
        (
            "*E",
            _under("layer", QKV + MOE),
            "but it has its '*' tensors in layer and its 'E' tensors in layer",
        ),
        (
            "*E",
            _under("a", QKV) + _under("b", QKV) + _under("c", MOE),
            "but it has its '*' tensors in a, b and its 'E' tensors in c",
        ),
    ],
)
def test_a_module_holding_other_families_does_not_fit(symbols, names, reason):
    assert reason in family_mismatch(symbols, names)


def test_symbols_without_a_family_signature_are_a_bug():
    with pytest.raises(InternalError, match="no family signature"):
        family_mismatch("DE", ())


def test_only_an_fsdp_dtensor_target_of_a_shortcut_model_is_warned_about():
    """Megatron refuses ``--moe-shortcut-connection`` with Megatron-FSDP."""
    shortcut = _args(moe_shortcut_connection=True)
    assert shortcut_warnings(shortcut, "fsdp_dtensor") == [
        "args.moe_shortcut_connection=True: Megatron refuses --moe-shortcut-connection with "
        "Megatron-FSDP, so no current Megatron job loads this fsdp_dtensor store (it is written "
        "in the layout a Megatron-FSDP v1 job uses)"
    ]
    assert shortcut_warnings(shortcut, "torch_dist") == []
    assert shortcut_warnings(_args(moe_shortcut_connection=False), "fsdp_dtensor") == []
    assert shortcut_warnings(_args(), "fsdp_dtensor") == []
