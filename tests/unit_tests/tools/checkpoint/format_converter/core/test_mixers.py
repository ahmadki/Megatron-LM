# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The family of every state-space mixer: by its names, the spec and a unique width match.

Part names, part sizes and fused widths are written out by hand from the
mixers' definitions (``GatedDeltaProductMixer``: ``in_proj`` is
``z, V0..V(M-1), K0..K(M-1), Q, b0..b(M-1), a`` of sizes ``d, d x M,
GN x M, GN, H x M, H``; ``conv1d`` is ``V*, K*, Q``), never from the code under
test.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.blocks import BlockLayout
from tools.checkpoint.format_converter.core.logical import LogicalId
from tools.checkpoint.format_converter.core.mixers import (
    GDN,
    GDN2,
    GDP,
    MAMBA2,
    FusedProjection,
    built_family,
    check_tp_bound,
    family_layouts,
    gdn2_parts,
    gdn_parts,
    gdp_parts,
    is_gdp_spec,
    linear_attention_layers,
    mamba2_parts,
    megatron_fsdp_warnings,
    mixer_layout,
)
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, TORCH_DIST
from tools.checkpoint.format_converter.core.part_layout import PartLayout
from tools.checkpoint.format_converter.engine.metadata import TensorEntry
from tools.checkpoint.format_converter.errors import RefusedError

HIDDEN = 8
GDP_SPEC = ["megatron.core.models.hybrid.hybrid_layer_specs", "gated_delta_product_stack_spec"]
DIMS = {"mamba_num_heads": 4, "mamba_head_dim": 2, "mamba_num_groups": 2, "mamba_state_dim": 3}
"""d = 4 x 2 = 8, GN = 2 x 3 = 6, H = 4."""
IN_PROJ = {
    "z": 8,
    "V0": 8,
    "V1": 8,
    "V2": 8,
    "K0": 6,
    "K1": 6,
    "K2": 6,
    "Q": 6,
    "b0": 4,
    "b1": 4,
    "b2": 4,
    "a": 4,
}
"""The in_proj parts of a Gated DeltaProduct mixer of :data:`DIMS` with 3 Householder steps (72
rows)."""
CONV = {"V0": 8, "V1": 8, "V2": 8, "K0": 6, "K1": 6, "K2": 6, "Q": 6}
"""Its conv1d parts (48 rows)."""
DECODER = BlockLayout("decoder", "hybrid_stack", 2, "per_layer", (0, 1), ("M", "*"), "")
MIXER = "decoder.layers.0.mixer"


def mixer_shapes(
    parts: bool, *, mixer: str = MIXER, in_proj=None, conv=None, bias: bool = False
) -> dict[str, tuple[int, ...]]:
    """A mixer's fused projections: one tensor per part (torch_dist) or fused (fsdp_dtensor)."""
    layouts = {"in_proj": IN_PROJ if in_proj is None else in_proj, "conv1d": CONV}
    if conv is not None:
        layouts["conv1d"] = conv
    rest = {"in_proj": (HIDDEN,), "conv1d": (1, 4)}
    shapes = {}
    for name, layout in layouts.items():
        for leaf in ("weight", "bias") if bias else ("weight",):
            trailing = rest[name] if leaf == "weight" else ()
            if parts:
                for part, rows in layout.items():
                    shapes[f"{mixer}.{name}.{leaf}.{part}"] = (rows, *trailing)
            else:
                shapes[f"{mixer}.{name}.{leaf}"] = (sum(layout.values()), *trailing)
    shapes[f"{mixer}.in_proj.layer_norm_weight"] = (HIDDEN,)
    shapes[f"{mixer}.A_log"] = (4,)
    shapes[f"{mixer}.out_proj.weight"] = (HIDDEN, 8)
    return shapes


def derive(shapes, *, source=TORCH_DIST, blocks=(DECODER,), **fields):
    args = {"spec": GDP_SPEC, **DIMS, "gdp_num_householder": 3, "hidden_size": HIDDEN, **fields}
    model = {key: TensorEntry(key, shape, torch.bfloat16, ()) for key, shape in shapes.items()}
    return mixer_layout(ArgsView(argparse.Namespace(**args)), model, blocks, source)


def refusal(shapes, **kwargs):
    with pytest.raises(RefusedError) as caught:
        derive(shapes, **kwargs)
    assert caught.value.refusal.code == errors.FC_R10
    return caught.value.refusal


def test_gated_deltaproduct_parts_are_listed_component_by_component():
    in_proj, conv = gdp_parts(8, 6, 4, 2)
    assert in_proj == PartLayout(
        ("z", "V0", "V1", "K0", "K1", "Q", "b0", "b1", "a"), (8, 8, 8, 6, 6, 6, 4, 4, 4)
    )
    assert conv == PartLayout(("V0", "V1", "K0", "K1", "Q"), (8, 8, 6, 6, 6))
    in_proj, conv = gdp_parts(8, 6, 4, 1)
    assert in_proj.parts == ("z", "V0", "K0", "Q", "b0", "a")
    assert conv.parts == ("V0", "K0", "Q")
    # a large mixer: d = 128 x 64, GN = 8 x 128, H = 128, M = 3
    in_proj, conv = gdp_parts(8192, 1024, 128, 3)
    assert (in_proj.fused_rows, conv.fused_rows) == (4 * (8192 + 1024 + 128), 3 * 8192 + 4 * 1024)


def test_the_other_families_parts():
    assert mamba2_parts(8, 6, 4) == (
        PartLayout(("z", "x", "B", "C", "dt"), (8, 8, 6, 6, 4)),
        PartLayout(("x", "B", "C"), (8, 6, 6)),
    )
    assert gdn_parts(4, 8, 2) == (
        PartLayout(("query", "key", "value", "z", "beta", "alpha"), (4, 4, 8, 8, 2, 2)),
        PartLayout(("query", "key", "value"), (4, 4, 8)),
    )
    assert gdn2_parts(4, 8) == (
        PartLayout(("query", "key", "value", "z", "f", "b", "w"), (4, 4, 8, 8, 4, 4, 8)),
        PartLayout(("query", "key", "value"), (4, 4, 8)),
    )


def test_the_arguments_give_every_familys_widths_or_why_not():
    fields = {
        **DIMS,
        "mamba_num_heads": None,
        "hidden_size": 12,
        "linear_key_head_dim": 4,
        "linear_num_key_heads": 1,
        "linear_value_head_dim": 4,
        "linear_num_value_heads": 2,
    }
    layouts = family_layouts(ArgsView(argparse.Namespace(**fields)))
    # Mamba-2's heads default to 2 x hidden_size / mamba_head_dim = 12
    assert layouts[MAMBA2].in_proj.sizes == (24, 24, 6, 6, 12)
    assert (
        layouts[MAMBA2].dims == "d=24, GN=6, H=12 (args.hidden_size=12 x 2 / args.mamba_head_dim)"
    )
    assert layouts[GDP] == "args.mamba_num_heads=None: sizes must be positive integers"
    assert layouts[GDN].in_proj.fused_rows == 28 and layouts[GDN].conv1d.fused_rows == 16
    assert layouts[GDN2].in_proj.fused_rows == 4 * 4 + 3 * 8
    empty = family_layouts(ArgsView(argparse.Namespace()))
    assert empty[MAMBA2] == (
        "args.mamba_head_dim=None, args.mamba_num_groups=None, args.mamba_state_dim=None: sizes "
        "must be positive integers"
    )
    assert empty[GDN] == empty[GDN2]


@pytest.mark.parametrize(
    ("spec", "gdp"),
    [
        (GDP_SPEC, True),
        (["megatron.core.models.mamba.mamba_layer_specs", "gdp_stack_spec"], True),
        ("megatron.core.models.hybrid.hybrid_layer_specs.gdp_stack_spec", True),
        (["megatron.core.models.hybrid.hybrid_layer_specs", "hybrid_stack_spec"], False),
        (["my_package.specs", "gdp_stack_spec"], False),
        (None, False),
    ],
)
def test_the_gated_deltaproduct_spec_is_a_hybrid_spec_modules(spec, gdp):
    assert is_gdp_spec(ArgsView(argparse.Namespace(spec=spec))) is gdp


def expected_projections(mixer: str, parts: bool, *, block="decoder", layer=0, bias=False):
    """The fused projections of a mixer of :data:`DIMS` at ``mixer`` (its module path)."""
    stored = mixer_shapes(parts, mixer=f"{block}.layers.{layer}.{mixer}", bias=bias)
    projections = {}
    for name, layout in (("in_proj", IN_PROJ), ("conv1d", CONV)):
        for leaf in ("weight", "bias") if bias else ("weight",):
            lid = LogicalId(block, layer, f"{mixer}.{name}", leaf)
            prefix = f"{block}.layers.{layer}.{mixer}.{name}.{leaf}"
            keys = tuple(sorted(k for k in stored if k == prefix or k.startswith(prefix + ".")))
            parts_layout = PartLayout(tuple(layout), tuple(layout.values()))
            projections[lid] = FusedProjection(lid, GDP, parts_layout, keys)
    return projections


@pytest.mark.parametrize("source", [TORCH_DIST, FSDP_DTENSOR])
@pytest.mark.parametrize("bias", [False, True])
def test_a_gated_deltaproduct_mixer_in_either_format(source, bias):
    layout, evidence = derive(mixer_shapes(source == TORCH_DIST, bias=bias), source=source)
    assert layout.families_json() == {"decoder": {GDP: 1}}
    assert dict(layout.projections) == expected_projections(
        "mixer", source == TORCH_DIST, bias=bias
    )
    stored = "as parts" if source == TORCH_DIST else "whole"
    assert evidence == (
        f"args.spec={'.'.join(GDP_SPEC)!r} builds Gated DeltaProduct mixers; 1 mixers stored "
        f"{stored}; d=8 (args.mamba_num_heads=4 x args.mamba_head_dim=2), GN=6 "
        "(args.mamba_num_groups=2 x args.mamba_state_dim=3), H=4, M=3 "
        "(args.gdp_num_householder=3): in_proj 72 rows, conv1d 48 rows, the only family whose "
        "widths the stored ones match (Mamba-2: in_proj 32 rows, conv1d 20 rows; Gated DeltaNet: "
        "not derivable, args.linear_key_head_dim=None, args.linear_num_key_heads=None, "
        "args.linear_value_head_dim=None, args.linear_num_value_heads=None: sizes must be "
        "positive integers; Gated DeltaNet 2: not derivable, args.linear_key_head_dim=None, "
        "args.linear_num_key_heads=None, args.linear_value_head_dim=None, "
        "args.linear_num_value_heads=None: sizes must be positive integers)"
    )
    in_proj = layout.of(LogicalId("decoder", 0, "mixer.in_proj", "weight", None, "V1"))
    assert in_proj is not None and in_proj.kind == "Gated DeltaProduct in_proj/conv1d"
    assert layout.of(LogicalId("decoder", 0, "mixer.in_proj", "layer_norm_weight")) is None


def test_mixers_of_shortcut_pairs_and_multi_token_prediction_layers():
    """A pair's child container is not assumed; an MTP inner layer list is a block of its own."""
    inner = BlockLayout(
        "mtp.layers.0.mtp_model_layer", "hybrid_mtp_inner", 1, "per_layer", (0,), ("M",), ""
    )
    shapes = {
        **mixer_shapes(True, mixer="decoder.layers.0.pair_child.mixer"),
        **mixer_shapes(True, mixer="mtp.layers.0.mtp_model_layer.layers.0.mixer"),
    }
    layout, _ = derive(shapes, blocks=(DECODER, inner))
    assert layout.families_json() == {"decoder": {GDP: 1}, "mtp.layers.0.mtp_model_layer": {GDP: 1}}
    assert dict(layout.projections) == {
        **expected_projections("pair_child.mixer", True),
        **expected_projections("mixer", True, block="mtp.layers.0.mtp_model_layer"),
    }


def test_no_mixer_no_layout():
    layout, evidence = derive({"decoder.layers.1.self_attention.linear_qkv.weight": (24, 8)})
    assert not layout.families and not layout.projections
    assert evidence == "no mixer (in_proj or conv1d of a mixer or a Gated DeltaNet)"


def test_torch_dist_parts_must_be_the_specs_family():
    mamba = {"z": 8, "x": 8, "B": 6, "C": 6, "dt": 4}
    found = refusal(mixer_shapes(True, in_proj=mamba, conv={"x": 8, "B": 6, "C": 6}))
    assert found.message == (
        "decoder[0].mixer is a Mamba-2 mixer by its in_proj parts z, x, B, C, dt, but args.spec="
        f"{'.'.join(GDP_SPEC)!r} builds Gated DeltaProduct mixers"
    )
    assert found.hint == "check args.spec against the model that saved the store"
    found = refusal(mixer_shapes(True), spec=None)
    assert found.message.startswith(
        "decoder[0].mixer is a Gated DeltaProduct mixer by its in_proj parts z, V0, V1, V2, K0, "
        "K1, K2, Q, b0, b1, b2, a, but args.spec=None builds Mamba-2 mixers"
    )
    assert found.keys[0] == f"{MIXER}.conv1d.weight.K0"


def test_mamba_2s_fsdp_dtensor_convolution_is_no_gated_deltaproduct_one():
    shapes = mixer_shapes(False)
    shapes[f"{MIXER}.conv1d_weight"] = shapes.pop(f"{MIXER}.conv1d.weight")
    found = refusal(shapes, source=FSDP_DTENSOR)
    assert found.message.startswith(
        "decoder[0].mixer is a Mamba-2 mixer by its convolution parameter conv1d_weight, but "
        "args.spec="
    )


@pytest.mark.parametrize("householders", [3, None])
def test_the_householder_steps_must_be_the_arguments(householders):
    """Megatron back-fills 3 for a store saved before the argument existed."""
    two = {p: r for p, r in IN_PROJ.items() if p not in ("V2", "K2", "b2")}
    conv = {p: r for p, r in CONV.items() if p not in ("V2", "K2")}
    found = refusal(mixer_shapes(True, in_proj=two, conv=conv), gdp_num_householder=householders)
    evidence = "args.gdp_num_householder=3" if householders else "args.gdp_num_householder unset: 3"
    assert found.message == (
        f"decoder[0].mixer: its in_proj holds 2 Householder steps (V0..V1), but {evidence}"
    )
    layout, _ = derive(mixer_shapes(True), gdp_num_householder=householders)
    assert layout.families_json() == {"decoder": {GDP: 1}}
    layout, _ = derive(mixer_shapes(True, in_proj=two, conv=conv), gdp_num_householder=2)
    assert layout.families_json() == {"decoder": {GDP: 1}}


def test_gated_deltaproduct_needs_the_number_of_heads():
    found = refusal(mixer_shapes(False), source=FSDP_DTENSOR, mamba_num_heads=None)
    assert found.message == (
        "decoder[0].mixer is a Gated DeltaProduct mixer, but its dimensions cannot be derived from "
        "the arguments: args.mamba_num_heads=None: sizes must be positive integers"
    )


COLLISION = {
    "mamba_num_heads": 4,
    "mamba_head_dim": 4,
    "mamba_num_groups": 2,
    "mamba_state_dim": 16,
}
"""A Gated DeltaProduct mixer (d = 16, GN = 32, H = 4, M = 3) whose in_proj width,
4 x (16 + 32 + 4) = 208 rows, is also the Mamba-2 formula's for 16 heads of dimension 4:
2 x 64 + 2 x 32 + 16 = 208 rows (its conv1d, 3 x 16 + 4 x 32 = 176 rows, is not: 64 + 64)."""


def collision_store() -> dict[str, tuple[int, ...]]:
    return {
        f"{MIXER}.in_proj.weight": (4 * (16 + 32 + 4), HIDDEN),
        f"{MIXER}.conv1d.weight": (3 * 16 + 4 * 32, 1, 4),
    }


def test_a_gated_deltaproduct_width_that_solves_mambas_formula_is_never_mamba():
    """Regression: a converter that derives Mamba-2's heads from the in_proj width splits this
    mixer as a Mamba-2 one with 16 heads. Without the Gated DeltaProduct spec the store is refused,
    whether the arguments give Mamba-2 heads of their own or none."""
    assert 4 * (16 + 32 + 4) == 2 * 16 * 4 + 2 * 32 + 16
    found = refusal(collision_store(), source=FSDP_DTENSOR, spec=None, **COLLISION)
    assert found.message == (
        "decoder[0].mixer: its widths (conv1d.weight 176 rows, in_proj.weight 208 rows) are not "
        "what the arguments give Mamba-2 mixers (d=16, GN=32, H=4 (args.mamba_num_heads=4): "
        "in_proj 100 rows, conv1d 80 rows), which args.spec=None builds Mamba-2 mixers; they are "
        "what the arguments give Gated DeltaProduct mixers"
    )
    unset = {**COLLISION, "mamba_num_heads": None, "hidden_size": 32}  # 2 x 32 / 4 = 16 heads
    found = refusal(collision_store(), source=FSDP_DTENSOR, spec=None, **unset)
    assert found.message == (
        "decoder[0].mixer: its widths (conv1d.weight 176 rows, in_proj.weight 208 rows) are not "
        "what the arguments give Mamba-2 mixers (d=64, GN=32, H=16 (args.hidden_size=32 x 2 / "
        "args.mamba_head_dim): in_proj 208 rows, conv1d 128 rows), which args.spec=None builds "
        "Mamba-2 mixers"
    )
    layout, _ = derive(collision_store(), source=FSDP_DTENSOR, **COLLISION)  # the spec says GDP
    assert layout.families_json() == {"decoder": {GDP: 1}}


def test_a_family_the_names_exclude_does_not_count_by_its_widths():
    """A Gated DeltaProduct mixer (d = 8, GN = 4, H = 2, M = 1: in_proj 2 x (8 + 4 + 2) = 28 rows,
    conv1d 8 + 2 x 4 = 16 rows) has the widths the arguments give a Gated DeltaNet (qk = 4 x 1,
    v = 4 x 2, 2 value heads: 2 x 4 + 2 x 8 + 2 x 2 = 28 rows, 2 x 4 + 8 = 16 rows), but no
    Gated DeltaNet is stored under ``mixer.in_proj``: the names leave Gated DeltaProduct and
    Mamba-2, and the spec says Gated DeltaProduct."""
    dims = {"mamba_num_heads": 2, "mamba_head_dim": 4, "mamba_num_groups": 1, "mamba_state_dim": 4}
    gdn = {
        "linear_key_head_dim": 4,
        "linear_num_key_heads": 1,
        "linear_value_head_dim": 4,
        "linear_num_value_heads": 2,
    }
    store = {f"{MIXER}.in_proj.weight": (28, HIDDEN), f"{MIXER}.conv1d.weight": (16, 1, 4)}
    layout, evidence = derive(store, source=FSDP_DTENSOR, gdp_num_householder=1, **dims, **gdn)
    assert layout.families_json() == {"decoder": {GDP: 1}}
    assert (
        "in_proj 28 rows, conv1d 16 rows, the family the stored names give (another family "
        "has these widths too)" in evidence
    )


MEGATRON_LINEAR_DEFAULTS = {
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_value_head_dim": 128,
    "linear_num_value_heads": 32,
}
"""TransformerConfig's Gated DeltaNet defaults, in every Megatron argument namespace: in_proj
2 x 2048 + 2 x 4096 + 2 x 32 = 12352 rows, conv1d 2 x 2048 + 4096 = 8192 rows."""


@pytest.mark.parametrize("source", [TORCH_DIST, FSDP_DTENSOR])
def test_a_mamba_2_mixer_with_gated_deltanets_default_widths(source):
    """Regression: a Mamba-2 model of hidden size 2048 (64 heads of 64: d = 4096) with 16 groups
    of state 128 (GN = 2048) has in_proj 2 x 4096 + 2 x 2048 + 64 = 12352 and conv1d 4096 +
    2 x 2048 = 8192 rows, Gated DeltaNet's widths under Megatron's default arguments. Its parts
    (torch_dist) or conv1d_weight (fsdp_dtensor) name the family, so it converts."""
    in_proj = {"z": 4096, "x": 4096, "B": 2048, "C": 2048, "dt": 64}
    conv = {"x": 4096, "B": 2048, "C": 2048}
    assert sum(in_proj.values()) == 12352 and sum(conv.values()) == 8192
    shapes: dict[str, tuple[int, ...]] = {}
    for name, parts, rest in (
        ("in_proj.weight", in_proj, (2048,)),
        ("conv1d.weight", conv, (1, 4)),
        ("conv1d.bias", conv, ()),
    ):
        if source == TORCH_DIST:
            shapes.update({f"{MIXER}.{name}.{part}": (rows, *rest) for part, rows in parts.items()})
        else:
            shapes[f"{MIXER}.{name.replace('conv1d.', 'conv1d_')}"] = (sum(parts.values()), *rest)
    fields = {"spec": None, "hidden_size": 2048, "mamba_num_heads": None, "mamba_head_dim": 64}
    fields.update(mamba_num_groups=16, mamba_state_dim=128, **MEGATRON_LINEAR_DEFAULTS)
    layout, evidence = derive(shapes, source=source, **fields)
    assert layout.families_json() == {"decoder": {MAMBA2: 1}}
    assert "in_proj 12352 rows, conv1d 8192 rows, the family the stored names give" in evidence


def test_widths_the_arguments_do_not_give_are_refused():
    store = mixer_shapes(False)
    store[f"{MIXER}.in_proj.weight"] = (80, HIDDEN)
    found = refusal(store, source=FSDP_DTENSOR)
    assert found.message.startswith(
        "decoder[0].mixer: its widths (conv1d.weight 48 rows, in_proj.weight 80 rows) are not what "
        "the arguments give Gated DeltaProduct mixers ("
    )
    assert found.keys == (f"{MIXER}.conv1d.weight", f"{MIXER}.in_proj.weight")


def test_parts_must_have_their_familys_sizes():
    """The widths match, but two parts swapped their sizes."""
    swapped = {**IN_PROJ, "z": 6, "Q": 8}
    found = refusal(mixer_shapes(True, in_proj=swapped))
    assert found.message.startswith(
        "decoder[0].mixer: in_proj.weight is stored as parts z:6, V0:8, V1:8, V2:8, K0:6, K1:6, "
        "K2:6, Q:8, b0:4, b1:4, b2:4, a:4, but a Gated DeltaProduct in_proj has z:8, V0:8, "
    )


def test_torch_dist_stores_fused_projections_as_parts():
    """Megatron's split factory stores every part of a Gated DeltaProduct mixer's in_proj and
    conv1d weights and biases, so a whole torch_dist projection is refused at any pack size (only
    a Mamba-2 in_proj.bias is stored whole, below)."""
    shapes = mixer_shapes(True)
    shapes[f"{MIXER}.conv1d.weight"] = (48, 1, 4)
    for part in CONV:
        del shapes[f"{MIXER}.conv1d.weight.{part}"]
    found = refusal(shapes)
    assert found.message == (
        "decoder[0].mixer: torch_dist stores a mixer's in_proj and conv1d as one tensor per part, "
        "but the store holds them whole"
    )
    assert found.keys == (f"{MIXER}.conv1d.weight",)


@pytest.mark.parametrize("source", [TORCH_DIST, FSDP_DTENSOR])
def test_a_gdp_mixers_per_head_and_per_channel_parameters_are_whole(source):
    """A Megatron-FSDP job above TP 1 holds and saves one rank's shard of a mixer's A_log,
    dt_bias (H = 4 rows here) and norm.weight (d = 8): no conversion completes such a store."""
    shapes = mixer_shapes(source == TORCH_DIST)
    shapes[f"{MIXER}.dt_bias"] = (4,)
    shapes[f"{MIXER}.norm.weight"] = (8,)
    layout, _ = derive(shapes, source=source)
    assert len(layout.projections) == 2
    shapes[f"{MIXER}.A_log"] = (2,)
    shapes[f"{MIXER}.norm.weight"] = (4,)
    found = refusal(shapes, source=source)
    assert found.message.startswith(
        "decoder[0].mixer: A_log (2,), norm.weight (4,), but a Gated DeltaProduct mixer's A_log "
        "and dt_bias have 4 rows and its norm.weight 8 (d=8 "
    )
    assert found.hint.startswith("a Megatron-FSDP job above TP 1 saves one tensor-parallel rank's")
    assert [key.rpartition("mixer.")[2] for key in found.keys] == ["A_log", "norm.weight"]


def test_gdp_stores_packed_above_tp_1_are_warned_about():
    """Current Megatron-FSDP jobs above TP 1 cannot hold a Gated DeltaProduct mixer whole."""
    layout, _ = derive(mixer_shapes(True))
    assert megatron_fsdp_warnings(layout, FSDP_DTENSOR, 1) == []
    assert megatron_fsdp_warnings(layout, TORCH_DIST, 2) == []
    (warning,) = megatron_fsdp_warnings(layout, FSDP_DTENSOR, 2)
    assert warning.startswith(
        "Gated DeltaProduct mixers packed for --fsdp-tp 2: current Megatron-FSDP jobs above TP 1 "
        "hold one tensor-parallel rank's shard of a mixer's A_log, dt_bias and norm.weight"
    )
    none, _ = derive({"decoder.layers.1.self_attention.linear_qkv.weight": (24, 8)})
    assert megatron_fsdp_warnings(none, FSDP_DTENSOR, 2) == []


def test_parts_that_are_no_familys_are_refused():
    found = refusal(mixer_shapes(True, in_proj={"z": 8, "V0": 8, "X": 4}))
    assert found.message == "decoder[0].mixer: its in_proj parts V0, X, z are no mixer family's"


def test_a_mixer_needs_its_in_proj_and_conv1d_weights():
    shapes = {k: v for k, v in mixer_shapes(False).items() if ".conv1d." not in k}
    found = refusal(shapes, source=FSDP_DTENSOR)
    assert found.message == "decoder[0].mixer has no conv1d.weight"


def test_parts_of_one_tensor_share_their_trailing_dimensions():
    shapes = mixer_shapes(True)
    shapes[f"{MIXER}.in_proj.weight.a"] = (4, HIDDEN + 1)
    found = refusal(shapes)
    assert found.message == (
        "decoder[0].mixer: the parts of in_proj.weight differ in dtype or in their dimensions "
        "after the first"
    )


MAMBA_DIMS = {"mamba_head_dim": 2, "mamba_num_groups": 2, "mamba_state_dim": 3}
"""With 4 heads: d = 8, GN = 6, H = 4."""
MAMBA_IN_PROJ = {"z": 8, "x": 8, "B": 6, "C": 6, "dt": 4}
"""The in_proj parts of a Mamba-2 mixer of :data:`MAMBA_DIMS` with 4 heads (32 rows)."""
MAMBA_CONV = {"x": 8, "B": 6, "C": 6}
"""Its conv1d parts (20 rows)."""


def mamba_shapes(parts: bool, *, in_proj_bias: str | None = None, conv: str = "conv1d_weight"):
    """A Mamba-2 mixer: torch_dist parts (``parts``) or fsdp_dtensor whole tensors, whose
    convolution is ``conv`` (``conv1d_weight`` and ``conv1d_bias``, or the legacy
    ``conv1d.weight`` and ``conv1d.bias``); with ``in_proj_bias`` ``"whole"`` or ``"parts"``."""
    shapes: dict[str, tuple[int, ...]] = {}
    conv_names = {"weight": f"{MIXER}.conv1d.weight", "bias": f"{MIXER}.conv1d.bias"}
    if not parts and conv == "conv1d_weight":
        conv_names = {leaf: f"{MIXER}.conv1d_{leaf}" for leaf in conv_names}
    tensors = [(f"{MIXER}.in_proj.weight", MAMBA_IN_PROJ, (HIDDEN,), parts)]
    if in_proj_bias is not None:
        tensors.append((f"{MIXER}.in_proj.bias", MAMBA_IN_PROJ, (), in_proj_bias == "parts"))
    tensors.append((conv_names["weight"], MAMBA_CONV, (1, 4), parts))
    tensors.append((conv_names["bias"], MAMBA_CONV, (), parts))
    for name, layout, rest, split in tensors:
        if split:
            shapes.update({f"{name}.{part}": (rows, *rest) for part, rows in layout.items()})
        else:
            shapes[name] = (sum(layout.values()), *rest)
    shapes.update({f"{MIXER}.{name}": (4,) for name in ("A_log", "dt_bias", "D")})
    shapes[f"{MIXER}.norm.weight"] = (8,)
    shapes[f"{MIXER}.out_proj.weight"] = (HIDDEN, 8)
    return shapes


def derive_mamba(shapes, *, source=TORCH_DIST, **fields):
    return derive(
        shapes, source=source, **{"spec": None, "mamba_num_heads": 4, **MAMBA_DIMS, **fields}
    )


@pytest.mark.parametrize("source", [TORCH_DIST, FSDP_DTENSOR])
def test_a_mamba_2_mixer_in_either_format(source):
    layout, evidence = derive_mamba(mamba_shapes(source == TORCH_DIST), source=source)
    assert layout.families_json() == {"decoder": {MAMBA2: 1}}
    conv = "mixer.conv1d" if source == TORCH_DIST else "mixer"
    leaves = ("weight", "bias") if source == TORCH_DIST else ("conv1d_weight", "conv1d_bias")
    expected = {LogicalId("decoder", 0, "mixer.in_proj", "weight"): MAMBA_IN_PROJ}
    expected.update({LogicalId("decoder", 0, conv, leaf): MAMBA_CONV for leaf in leaves})
    assert {
        lid: dict(zip(fused.parts.parts, fused.parts.sizes))
        for lid, fused in layout.projections.items()
    } == expected
    assert evidence.startswith(
        "args.spec=None builds Mamba-2 mixers; 1 mixers stored "
        + ("as parts" if source == TORCH_DIST else "whole")
        + "; d=8, GN=6, H=4 (args.mamba_num_heads=4): in_proj 32 rows, conv1d 20 rows, the only "
        "family whose widths the stored ones match"
    )


def test_a_mamba_2_convolution_is_named_by_each_format():
    """fsdp_dtensor names it the mixer's own conv1d_weight and conv1d_bias, torch_dist (and an
    fsdp_dtensor store saved before MambaMixer owned it) the conv1d module's weight and bias."""
    for shapes, source in (
        (mamba_shapes(True), TORCH_DIST),
        (mamba_shapes(False), FSDP_DTENSOR),
        (mamba_shapes(False, conv="conv1d.weight"), FSDP_DTENSOR),
    ):
        layout, _ = derive_mamba(shapes, source=source)
        names = {
            (fused.target_lid(FSDP_DTENSOR).name, fused.target_lid(TORCH_DIST).name)
            for fused in layout.projections.values()
        }
        assert names == {
            ("mixer.in_proj.weight", "mixer.in_proj.weight"),
            ("mixer.conv1d_weight", "mixer.conv1d.weight"),
            ("mixer.conv1d_bias", "mixer.conv1d.bias"),
        }, source
    layout, _ = derive(mixer_shapes(True))
    assert {fused.target_lid(FSDP_DTENSOR).name for fused in layout.projections.values()} == {
        "mixer.in_proj.weight",
        "mixer.conv1d.weight",
    }


def test_a_legacy_convolution_name_is_mamba_2_only_by_the_spec():
    """``mixer.conv1d.weight`` in fsdp_dtensor is a Gated DeltaProduct mixer's or a legacy Mamba-2
    one's: with the Gated DeltaProduct spec these widths are refused."""
    legacy = mamba_shapes(False, conv="conv1d.weight")
    layout, _ = derive_mamba(legacy, source=FSDP_DTENSOR)
    assert layout.families_json() == {"decoder": {MAMBA2: 1}}
    found = refusal(legacy, source=FSDP_DTENSOR, spec=GDP_SPEC, mamba_num_heads=4, **MAMBA_DIMS)
    assert found.message.startswith(
        "decoder[0].mixer: its widths (conv1d.bias 20 rows, conv1d.weight 20 rows, in_proj.weight "
        "32 rows) are not what the arguments give Gated DeltaProduct mixers"
    )


def test_a_convolution_under_both_names_is_refused():
    shapes = mamba_shapes(False)
    shapes[f"{MIXER}.conv1d.weight"] = shapes[f"{MIXER}.conv1d_weight"]
    with pytest.raises(RefusedError) as caught:
        derive_mamba(shapes, source=FSDP_DTENSOR)
    assert caught.value.refusal.code == errors.FC_R10
    assert caught.value.refusal.message == (
        f"decoder[0].mixer: conv1d.weight is stored twice, as {MIXER}.conv1d.weight and "
        f"{MIXER}.conv1d_weight"
    )


@pytest.mark.parametrize("source", [TORCH_DIST, FSDP_DTENSOR])
def test_a_mamba_2_in_proj_bias_is_whole_in_both_formats(source):
    """MambaMixer.sharded_state_dict splits in_proj.weight, conv1d.weight and conv1d.bias, never
    an in_proj.bias (of a mixer built otherwise: MambaMixer asserts bias=False): torch_dist stores
    it whole (the ranks' local tensors one after the other)."""
    layout, _ = derive_mamba(
        mamba_shapes(source == TORCH_DIST, in_proj_bias="whole"), source=source
    )
    bias = layout.of(LogicalId("decoder", 0, "mixer.in_proj", "bias"))
    assert bias is not None and bias.torch_dist_whole and bias.keys == (f"{MIXER}.in_proj.bias",)
    assert dict(zip(bias.parts.parts, bias.parts.sizes)) == MAMBA_IN_PROJ
    assert [fused.torch_dist_whole for fused in layout.projections.values()].count(True) == 1


def test_a_mamba_2_in_proj_bias_stored_as_parts_is_refused():
    found = refusal(
        mamba_shapes(True, in_proj_bias="parts"), spec=None, mamba_num_heads=4, **MAMBA_DIMS
    )
    assert found.message == (
        "decoder[0].mixer: torch_dist stores a Mamba-2 mixer's in_proj.bias whole, but the store "
        "holds it as parts"
    )
    assert found.keys == tuple(sorted(f"{MIXER}.in_proj.bias.{part}" for part in MAMBA_IN_PROJ))


def test_only_a_mamba_2_in_proj_bias_may_be_whole_in_torch_dist():
    shapes = mamba_shapes(True, in_proj_bias="whole")
    for part in MAMBA_CONV:
        del shapes[f"{MIXER}.conv1d.bias.{part}"]
    shapes[f"{MIXER}.conv1d.bias"] = (20,)
    found = refusal(shapes, spec=None, mamba_num_heads=4, **MAMBA_DIMS)
    assert found.message.endswith("as one tensor per part, but the store holds them whole")
    assert found.keys == (f"{MIXER}.conv1d.bias",)
    gdp = mixer_shapes(True, bias=True)
    for part in IN_PROJ:
        del gdp[f"{MIXER}.in_proj.bias.{part}"]
    gdp[f"{MIXER}.in_proj.bias"] = (72,)
    assert refusal(gdp).keys == (f"{MIXER}.in_proj.bias",)


@pytest.mark.parametrize(
    ("src_tp", "pack_tp", "refused"),
    [(1, 1, False), (2, 2, False), (2, None, False), (2, 1, True), (2, 4, True), (1, 2, True)],
)
def test_a_tensor_torch_dist_stores_whole_converts_only_at_its_size(src_tp, pack_tp, refused):
    """TP-bound identity: a whole torch_dist in_proj.bias holds the ranks' local biases at the size
    it was saved at, so it is fsdp_dtensor's only when packed at that size."""
    layout, _ = derive_mamba(mamba_shapes(True, in_proj_bias="whole"))
    if not refused:
        check_tp_bound(layout, TORCH_DIST, src_tp, pack_tp)
        return
    with pytest.raises(RefusedError) as caught:
        check_tp_bound(layout, TORCH_DIST, src_tp, pack_tp)
    found = caught.value.refusal
    assert found.code == errors.FC_R10
    assert found.message == (
        "1 Mamba-2 mixer in_proj.bias tensors (such as decoder[0].mixer.in_proj.bias) are stored "
        f"whole, as the {src_tp} tensor-parallel ranks' local tensors one after the other, which "
        f"no fsdp_dtensor store packed at {pack_tp} holds"
    )
    assert found.keys == (f"{MIXER}.in_proj.bias",)
    assert found.hint == f"pass --fsdp-tp {src_tp}, the size the source was saved with"


def test_an_fsdp_dtensor_source_is_unpacked_at_its_own_size():
    """Its torch_dist target keeps the source's tensor-parallel size: nothing to check."""
    layout, _ = derive_mamba(mamba_shapes(False, in_proj_bias="whole"), source=FSDP_DTENSOR)
    check_tp_bound(layout, FSDP_DTENSOR, 1, 2)
    no_bias, _ = derive_mamba(mamba_shapes(True))
    check_tp_bound(no_bias, TORCH_DIST, 1, 2)


@pytest.mark.parametrize("source", [TORCH_DIST, FSDP_DTENSOR])
def test_a_mamba_2_mixers_per_head_and_per_channel_parameters_are_whole(source):
    """A_log, dt_bias and D have H = 4 rows (D: d = 8 with D_has_hdim), norm.weight d = 8."""
    shapes = mamba_shapes(source == TORCH_DIST)
    shapes[f"{MIXER}.D"] = (8,)
    layout, _ = derive_mamba(shapes, source=source)
    assert len(layout.projections) == 3
    shapes[f"{MIXER}.D"] = (2,)
    shapes[f"{MIXER}.norm.weight"] = (4,)
    found = refusal(shapes, source=source, spec=None, mamba_num_heads=4, **MAMBA_DIMS)
    assert found.message == (
        "decoder[0].mixer: D (2,), norm.weight (4,), but a Mamba-2 mixer's A_log and dt_bias have 4 "
        "rows, its D 4 or 8 and its norm.weight 8 (d=8, GN=6, H=4 (args.mamba_num_heads=4))"
    )
    assert found.hint.startswith("a store that holds one tensor-parallel rank's shard")


def test_mamba_2_dimensions_follow_mamba_mixer():
    """``nheads = mamba_num_heads`` and ``d_inner = nheads x headdim``; without it
    ``d_inner = 2 x hidden_size`` and ``nheads = d_inner / headdim``, which must divide."""
    fields = {**MAMBA_DIMS, "hidden_size": 6}
    layouts = family_layouts(ArgsView(argparse.Namespace(**fields, mamba_num_heads=None)))
    assert layouts[MAMBA2].in_proj.sizes == (12, 12, 6, 6, 6)
    fields["mamba_head_dim"] = 5
    layouts = family_layouts(ArgsView(argparse.Namespace(**fields, mamba_num_heads=None)))
    assert layouts[MAMBA2] == "args.mamba_num_heads is unset and args.hidden_size=6 gives no heads"
    layouts = family_layouts(ArgsView(argparse.Namespace(**fields, mamba_num_heads=3)))
    assert layouts[MAMBA2].in_proj.sizes == (15, 15, 6, 6, 3)


def test_torch_dist_names_no_convolution_conv1d_weight():
    """MambaMixer.sharded_state_dict names the convolution conv1d.weight and conv1d.bias and
    splits it: a torch_dist store holding conv1d_weight and conv1d_bias is no Megatron one."""
    shapes = mamba_shapes(True)
    for leaf, rest in (("weight", (1, 4)), ("bias", ())):
        for part in MAMBA_CONV:
            del shapes[f"{MIXER}.conv1d.{leaf}.{part}"]
        shapes[f"{MIXER}.conv1d_{leaf}"] = (20, *rest)
    with pytest.raises(RefusedError) as caught:
        derive_mamba(shapes)
    found = caught.value.refusal
    assert found.code == errors.FC_R10
    assert found.message == (
        "decoder[0].mixer: torch_dist names a Mamba-2 mixer's convolution conv1d.weight and "
        "conv1d.bias (stored as parts), but the store holds conv1d_weight or conv1d_bias"
    )
    assert found.keys == (f"{MIXER}.conv1d_bias", f"{MIXER}.conv1d_weight")


ATTENTION = "decoder.layers.0.self_attention"
GDN_DIMS = {
    "linear_key_head_dim": 3,
    "linear_num_key_heads": 2,
    "linear_value_head_dim": 2,
    "linear_num_value_heads": 4,
}
"""qk = 3 x 2 = 6, v = 2 x 4 = 8, 2 key heads and 4 value heads."""
GDN_IN_PROJ = {
    GDN: {"query": 6, "key": 6, "value": 8, "z": 8, "beta": 4, "alpha": 4},
    GDN2: {"query": 6, "key": 6, "value": 8, "z": 8, "f": 6, "b": 6, "w": 8},
}
"""The in_proj parts of a Gated DeltaNet (36 rows) and a Gated DeltaNet 2 (48 rows) of
:data:`GDN_DIMS`, written out from ``in_proj_split_names`` and ``in_proj_split_sections``."""
GDN_CONV = {"query": 6, "key": 6, "value": 8}
"""Their conv1d parts (20 rows)."""
GDN_VECTORS = {GDN: {"A_log": 4, "dt_bias": 4}, GDN2: {"A_log": 2, "dt_bias": 6}}
"""``A_log`` and ``dt_bias`` rows: a value head each (GDN); a key head and a key channel (GDN2)."""


def gdn_shapes(
    parts: bool, *, family=GDN, module=ATTENTION, in_proj_bias: str | None = None
) -> dict[str, tuple[int, ...]]:
    """A Gated DeltaNet of :data:`GDN_DIMS` under ``module``: torch_dist parts (``parts``) or
    fsdp_dtensor whole tensors, with its convolution's bias and ``in_proj_bias`` ``"whole"`` or
    ``"parts"``."""
    tensors = [(f"{module}.in_proj.weight", GDN_IN_PROJ[family], (HIDDEN,), parts)]
    if in_proj_bias is not None:
        tensors.append((f"{module}.in_proj.bias", GDN_IN_PROJ[family], (), in_proj_bias == "parts"))
    tensors.append((f"{module}.conv1d.weight", GDN_CONV, (1, 4), parts))
    tensors.append((f"{module}.conv1d.bias", GDN_CONV, (), parts))
    shapes: dict[str, tuple[int, ...]] = {}
    for name, layout, rest, split in tensors:
        if split:
            shapes.update({f"{name}.{part}": (rows, *rest) for part, rows in layout.items()})
        else:
            shapes[name] = (sum(layout.values()), *rest)
    shapes.update({f"{module}.{name}": (rows,) for name, rows in GDN_VECTORS[family].items()})
    shapes[f"{module}.in_proj.layer_norm_weight"] = (HIDDEN,)
    shapes[f"{module}.out_norm.weight"] = (2,)
    shapes[f"{module}.out_proj.weight"] = (HIDDEN, 8)
    return shapes


def derive_gdn(shapes, *, source=TORCH_DIST, variant="gdn", **fields):
    fields = {"spec": None, "experimental_attention_variant": variant, **GDN_DIMS, **fields}
    return derive(shapes, source=source, **fields)


def refused_gdn(shapes, **kwargs):
    with pytest.raises(RefusedError) as caught:
        derive_gdn(shapes, **kwargs)
    assert caught.value.refusal.code == errors.FC_R10
    return caught.value.refusal


@pytest.mark.parametrize(
    ("variant", "family"),
    [(None, GDN), ("gdn", GDN), ("gated_delta_net", GDN), ("gdn2", GDN2), ("dsa", GDN)],
)
def test_the_variant_builds_the_gated_deltanet(variant, family):
    """``HybridBlock`` builds its ``gdn2_layer`` iff the variant is ``gdn2`` (else ``gdn_layer``),
    and so does the GPT linear-attention spec; ``mixer`` modules follow the spec."""
    args = ArgsView(argparse.Namespace(experimental_attention_variant=variant, spec=None))
    assert built_family("self_attention", args)[0] == family
    assert built_family("attn_layer.self_attention", args)[0] == family
    assert built_family("mixer", args)[0] == MAMBA2


@pytest.mark.parametrize("source", [TORCH_DIST, FSDP_DTENSOR])
@pytest.mark.parametrize(("variant", "family"), [("gdn", GDN), ("gdn2", GDN2)])
def test_a_gated_deltanet_in_either_format(source, variant, family):
    shapes = gdn_shapes(source == TORCH_DIST, family=family)
    layout, evidence = derive_gdn(shapes, source=source, variant=variant)
    assert layout.families_json() == {"decoder": {family: 1}}
    expected = {
        LogicalId("decoder", 0, f"self_attention.{name}", leaf): parts
        for name, leaf, parts in (
            ("in_proj", "weight", GDN_IN_PROJ[family]),
            ("conv1d", "weight", GDN_CONV),
            ("conv1d", "bias", GDN_CONV),
        )
    }
    assert {
        lid: dict(zip(fused.parts.parts, fused.parts.sizes))
        for lid, fused in layout.projections.items()
    } == expected
    rows = sum(GDN_IN_PROJ[family].values())
    name = "Gated DeltaNet" if family == GDN else "Gated DeltaNet 2"
    assert evidence.startswith(
        f"args.experimental_attention_variant={variant!r} builds {name} layers; 1 layers stored "
        + ("as parts" if source == TORCH_DIST else "whole")
        + f"; qk=6, v=8, key heads=2, value heads=4 (args.linear_*): in_proj {rows} rows, conv1d "
        "20 rows, the only family whose widths the stored ones match"
    )
    fused = layout.of(LogicalId("decoder", 0, "self_attention.in_proj", "weight"))
    assert fused is not None and fused.kind == f"{name} in_proj/conv1d"
    assert layout.of(LogicalId("decoder", 0, "self_attention.in_proj", "layer_norm_weight")) is None


def test_gated_deltanet_parts_must_be_the_variants():
    """torch_dist parts name the Gated DeltaNet: GDN2's ``f, b, w`` under a ``gdn`` variant are a
    contradiction, and so are GDN's ``beta, alpha`` under ``gdn2``."""
    found = refused_gdn(gdn_shapes(True, family=GDN2), variant="gdn")
    assert found.message == (
        "decoder[0].self_attention is a Gated DeltaNet 2 layer by its in_proj parts query, key, "
        "value, z, f, b, w, but args.experimental_attention_variant='gdn' builds Gated DeltaNet "
        "layers"
    )
    assert found.hint == (
        "check args.experimental_attention_variant against the model that saved the store"
    )
    found = refused_gdn(gdn_shapes(True), variant="gdn2")
    assert found.message.startswith("decoder[0].self_attention is a Gated DeltaNet layer by its")


def test_fsdp_dtensor_widths_tell_the_two_gated_deltanets_apart():
    """fsdp_dtensor's ``self_attention.in_proj`` names both Gated DeltaNets: the widths the
    arguments give them (36 and 48 rows) must leave the variant's."""
    layout, _ = derive_gdn(gdn_shapes(False, family=GDN2), source=FSDP_DTENSOR, variant="gdn2")
    assert layout.families_json() == {"decoder": {GDN2: 1}}
    found = refused_gdn(gdn_shapes(False), source=FSDP_DTENSOR, variant="gdn2")
    assert found.message == (
        "decoder[0].self_attention: its widths (conv1d.bias 20 rows, conv1d.weight 20 rows, "
        "in_proj.weight 36 rows) are not what the arguments give Gated DeltaNet 2 layers (qk=6, "
        "v=8, key heads=2, value heads=4 (args.linear_*): in_proj 48 rows, conv1d 20 rows), which "
        "args.experimental_attention_variant='gdn2' builds Gated DeltaNet 2 layers; they are what "
        "the arguments give Gated DeltaNet layers"
    )


def test_gated_deltanet_widths_both_variants_share_are_refused():
    """With 1-row value heads, 2 key heads of 1 and 4 value heads (qk = 2, v = 4), both in_proj are
    2 x 2 + 2 x 4 + 2 x 4 = 4 x 2 + 3 x 4 = 20 rows and both conv1d 8: the widths cannot tell."""
    dims = {
        "linear_key_head_dim": 1,
        "linear_num_key_heads": 2,
        "linear_value_head_dim": 1,
        "linear_num_value_heads": 4,
    }
    store = {f"{ATTENTION}.in_proj.weight": (20, HIDDEN), f"{ATTENTION}.conv1d.weight": (8, 1, 4)}
    found = refused_gdn(store, source=FSDP_DTENSOR, **dims)
    assert found.message.startswith(
        "decoder[0].self_attention: its widths (conv1d.weight 8 rows, in_proj.weight 20 rows) are "
        "what the arguments give Gated DeltaNet layers (qk=2, v=4, key heads=2, value heads=4 "
        "(args.linear_*)) and Gated DeltaNet 2 layers alike, so they cannot tell the family"
    )


def test_each_mixer_has_the_family_megatron_builds_at_its_module():
    """A Gated DeltaProduct stack spec builds its ``G`` layers as Gated DeltaNets: a model may hold
    both families, as a Mamba-2 hybrid may hold Gated DeltaNet 2 layers (here in a shortcut pair,
    as fsdp_dtensor stores it)."""
    shapes = {**mixer_shapes(True), **gdn_shapes(True, module="decoder.layers.1.self_attention")}
    layout, evidence = derive_gdn(shapes, spec=GDP_SPEC)
    assert layout.families_json() == {"decoder": {GDN: 1, GDP: 1}}
    assert layout.families == {("decoder", 0, "mixer"): GDP, ("decoder", 1, "self_attention"): GDN}
    gdn, gdp = evidence.split(" | ")
    assert gdn.startswith("args.experimental_attention_variant='gdn' builds Gated DeltaNet layers")
    assert gdp.startswith(f"args.spec={'.'.join(GDP_SPEC)!r} builds Gated DeltaProduct mixers")
    pair = "decoder.layers.1.attn_layer.self_attention"
    shapes = {**mamba_shapes(False), **gdn_shapes(False, family=GDN2, module=pair)}
    layout, _ = derive_gdn(
        shapes, source=FSDP_DTENSOR, variant="gdn2", mamba_num_heads=4, **MAMBA_DIMS
    )
    assert layout.families == {
        ("decoder", 0, "mixer"): MAMBA2,
        ("decoder", 1, "attn_layer.self_attention"): GDN2,
    }


def test_a_family_under_the_other_module_is_refused():
    """Megatron builds Gated DeltaNets under ``self_attention`` only, and its state-space mixers
    under ``mixer`` only."""
    mamba = {
        key.replace(".mixer.", ".self_attention."): shape
        for key, shape in mamba_shapes(True).items()
    }
    found = refused_gdn(mamba, mamba_num_heads=4, **MAMBA_DIMS)
    assert found.message == (
        "decoder[0].self_attention is a Mamba-2 mixer by its in_proj parts z, x, B, C, dt, but "
        "args.experimental_attention_variant='gdn' builds Gated DeltaNet layers"
    )
    found = refused_gdn(gdn_shapes(True, module=MIXER))
    assert found.message == (
        "decoder[0].mixer is a Gated DeltaNet layer by its in_proj parts query, key, value, z, "
        "beta, alpha, but args.spec=None builds Mamba-2 mixers"
    )


@pytest.mark.parametrize("source", [TORCH_DIST, FSDP_DTENSOR])
def test_a_gated_deltanet_in_proj_bias_is_whole_in_both_formats(source):
    """``_GDNBase.sharded_state_dict`` splits in_proj.weight and the convolution, not an
    in_proj.bias (``bias=True``): torch_dist holds it whole, bound to the size it was saved at."""
    layout, _ = derive_gdn(gdn_shapes(source == TORCH_DIST, in_proj_bias="whole"), source=source)
    bias = layout.of(LogicalId("decoder", 0, "self_attention.in_proj", "bias"))
    assert bias is not None and bias.torch_dist_whole
    assert bias.keys == (f"{ATTENTION}.in_proj.bias",)
    assert dict(zip(bias.parts.parts, bias.parts.sizes)) == GDN_IN_PROJ[GDN]
    if source == TORCH_DIST:
        with pytest.raises(RefusedError) as caught:
            check_tp_bound(layout, TORCH_DIST, 2, 1)
        assert caught.value.refusal.message.startswith(
            "1 Gated DeltaNet layer in_proj.bias tensors (such as "
            "decoder[0].self_attention.in_proj.bias) are stored whole"
        )
    found = refused_gdn(gdn_shapes(True, family=GDN2, in_proj_bias="parts"), variant="gdn2")
    assert found.message == (
        "decoder[0].self_attention: torch_dist stores a Gated DeltaNet 2 layer's in_proj.bias "
        "whole, but the store holds it as parts"
    )


@pytest.mark.parametrize("source", [TORCH_DIST, FSDP_DTENSOR])
@pytest.mark.parametrize(
    ("variant", "family", "rows"),
    [
        ("gdn", GDN, "A_log and dt_bias have 4 rows"),
        ("gdn2", GDN2, "A_log has 2 rows and its dt_bias 6"),
    ],
)
def test_a_gated_deltanets_per_head_parameters_are_whole(source, variant, family, rows):
    """A store that holds one rank's shard of a Gated DeltaNet's A_log or dt_bias cannot be
    completed (Megatron-FSDP shards them along their partition_dim, so its saves hold them
    whole)."""
    shapes = gdn_shapes(source == TORCH_DIST, family=family)
    shapes[f"{ATTENTION}.dt_bias"] = (GDN_VECTORS[family]["dt_bias"] // 2,)
    found = refused_gdn(shapes, source=source, variant=variant)
    name = "Gated DeltaNet" if family == GDN else "Gated DeltaNet 2"
    assert found.message == (
        f"decoder[0].self_attention: dt_bias ({GDN_VECTORS[family]['dt_bias'] // 2},), but a "
        f"{name} layer's {rows} (qk=6, v=8, key heads=2, value heads=4 (args.linear_*))"
    )
    assert found.hint.startswith("a store that holds one tensor-parallel rank's shard")
    assert found.keys == (f"{ATTENTION}.dt_bias",)


def test_an_fsdp_dtensor_gated_deltanet_is_never_taken_for_a_state_space_mixer():
    """fsdp_dtensor's ``self_attention.in_proj`` names only the two Gated DeltaNets: Mamba-2's
    widths do not count, even when the ``mamba_*`` arguments every namespace carries give it
    the same ones (8 heads of 1, GN = 2 x 3: in_proj 2 x 8 + 2 x 6 + 8 = 36 rows, conv1d
    8 + 2 x 6 = 20, a Gated DeltaNet's)."""
    mamba = {"mamba_num_heads": 8, "mamba_head_dim": 1, "mamba_num_groups": 2, "mamba_state_dim": 3}
    layouts = family_layouts(ArgsView(argparse.Namespace(**mamba, **GDN_DIMS)))
    mamba2, gdn = layouts[MAMBA2], layouts[GDN]
    assert mamba2.widths == gdn.widths == "in_proj 36 rows, conv1d 20 rows"
    layout, _ = derive_gdn(gdn_shapes(False), source=FSDP_DTENSOR, **mamba)
    assert layout.families == {("decoder", 0, "self_attention"): GDN}


def test_gated_deltanets_of_shortcut_pairs_and_multi_token_prediction_layers():
    """A ``G`` layer's Gated DeltaNet is found under a pair's child and in an MTP inner layer
    list, a block of its own."""
    inner = BlockLayout(
        "mtp.layers.0.mtp_model_layer", "hybrid_mtp_inner", 1, "per_layer", (0,), ("G",), ""
    )
    shapes = {
        **gdn_shapes(True, module="decoder.layers.1.attn_layer.self_attention"),
        **gdn_shapes(True, module="mtp.layers.0.mtp_model_layer.layers.0.self_attention"),
    }
    layout, _ = derive_gdn(shapes, blocks=(DECODER, inner))
    assert layout.families == {
        ("decoder", 1, "attn_layer.self_attention"): GDN,
        ("mtp.layers.0.mtp_model_layer", 0, "self_attention"): GDN,
    }
    assert {lid for lid in layout.projections} == {
        LogicalId(block, layer, f"{module}.{name}", leaf)
        for block, layer, module in (
            ("decoder", 1, "attn_layer.self_attention"),
            ("mtp.layers.0.mtp_model_layer", 0, "self_attention"),
        )
        for name, leaf in (("in_proj", "weight"), ("conv1d", "weight"), ("conv1d", "bias"))
    }


GPT_DECODER = BlockLayout("decoder", "transformer_block", 3, "per_layer", (0, 1, 2), None, "")
"""A GPT decoder of 3 layers."""


@pytest.mark.parametrize(
    ("variant", "freq", "layer"),
    [("gdn", 3, 0), ("gdn", 3, 1), ("gated_delta_net", 3, 1), ("gdn2", [0, 1, 0], 1)],
)
def test_a_gpt_decoder_holds_gated_deltanets_where_the_arguments_place_them(variant, freq, layer):
    """``get_linear_attention_pattern``: ``linear_attention_freq`` 3 makes layers 0 and 1 linear
    attention and layer 2 standard attention; a list marks the linear-attention layers with 1."""
    family = GDN2 if variant == "gdn2" else GDN
    shapes = gdn_shapes(True, family=family, module=f"decoder.layers.{layer}.self_attention")
    layout, _ = derive_gdn(
        shapes, blocks=(GPT_DECODER,), variant=variant, linear_attention_freq=freq
    )
    assert layout.families == {("decoder", layer, "self_attention"): family}


@pytest.mark.parametrize(("variant", "freq"), [(None, None), (None, 3), ("gdn", None), ("dsa", 3)])
def test_a_gpt_decoder_without_linear_attention_arguments_is_refused(variant, freq):
    """A ``TransformerBlock`` builds Gated DeltaNets only for a Gated DeltaNet variant with
    ``linear_attention_freq`` (a hybrid stack builds one at every ``G`` whatever the variant)."""
    found = refused_gdn(
        gdn_shapes(True), blocks=(GPT_DECODER,), variant=variant, linear_attention_freq=freq
    )
    assert found.message == (
        "decoder[0].self_attention holds a Gated DeltaNet's in_proj and conv1d, but a "
        "TransformerBlock builds Gated DeltaNet layers only for "
        "args.experimental_attention_variant 'gated_delta_net' or 'gdn' or 'gdn2' with "
        "args.linear_attention_freq placing them among its 3 layers, and the arguments give "
        f"experimental_attention_variant={variant!r}, linear_attention_freq={freq!r}"
    )
    assert found.hint == (
        "check args.experimental_attention_variant and args.linear_attention_freq against the "
        "model that saved the store"
    )
    layout, _ = derive_gdn(gdn_shapes(True), variant=variant, linear_attention_freq=freq)
    assert layout.families == {("decoder", 0, "self_attention"): GDN}


@pytest.mark.parametrize(("freq", "layer"), [(3, 2), (1, 0), ([1, 0, 1], 1), ([1, 1], 0)])
def test_a_gated_deltanet_at_a_standard_attention_layer_is_refused(freq, layer):
    """Layer 2 of 3 is standard attention for ``linear_attention_freq`` 3, every layer for 1,
    and a list that is not one entry per layer places nothing."""
    module = f"decoder.layers.{layer}.self_attention"
    found = refused_gdn(
        gdn_shapes(True, module=module), blocks=(GPT_DECODER,), linear_attention_freq=freq
    )
    if isinstance(freq, list) and len(freq) != 3:
        assert "args.linear_attention_freq placing them among its 3 layers" in found.message
    else:
        assert found.message == (
            f"decoder[{layer}].self_attention holds a Gated DeltaNet's in_proj and conv1d, but "
            f"args.linear_attention_freq={freq!r} makes layer {layer} of decoder standard "
            "attention"
        )


def test_the_linear_attention_layers():
    assert linear_attention_layers(4, 8) == (True, True, True, False) * 2
    assert linear_attention_layers(1, 2) == (False, False)
    assert linear_attention_layers([1, 0], 2) == (True, False)
    assert linear_attention_layers([1, 0], 3) is None
    for wrong in (None, 0, True, "3"):
        assert linear_attention_layers(wrong, 2) is None
