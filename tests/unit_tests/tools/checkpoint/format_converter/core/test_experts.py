# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The expert layout: the number of experts and the fsdp_dtensor naming, checked against the store.

Stores are hand-built key sets: torch_dist expert tensors with their expert
axes, fsdp_dtensor expert tensors named grouped or sequential.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.context import build_arch_context
from tools.checkpoint.format_converter.core.experts import ExpertLayout, expert_layout
from tools.checkpoint.format_converter.core.names import (
    FSDP_DTENSOR,
    GROUPED,
    SEQUENTIAL,
    TORCH_DIST,
)
from tools.checkpoint.format_converter.engine.metadata import TensorEntry
from tools.checkpoint.format_converter.errors import AmbiguousError, RefusedError, UsageError

E = 4
STACKED_FC1 = "decoder.layers.mlp.experts.experts.linear_fc1.weight"
STACKED_FC2 = "decoder.layers.mlp.experts.experts.linear_fc2.weight"


def _args(**fields) -> ArgsView:
    return ArgsView(argparse.Namespace(**{"num_layers": 2, "num_experts": E, **fields}))


def _model(shapes: dict[str, tuple[int, ...]]) -> dict[str, TensorEntry]:
    return {key: TensorEntry(key, shape, torch.bfloat16, ()) for key, shape in shapes.items()}


TORCH_DIST_MOE = _model(
    {
        STACKED_FC1: (2, E, 8, 4),
        STACKED_FC2: (2, E, 4, 8),
        "decoder.layers.mlp.router.weight": (2, E, 4),
        "decoder.layers.mlp.router.expert_bias": (2, E),
    }
)
"""A GPT decoder of two MoE layers, stacked."""


def _fsdp_moe(form: str, experts=range(E), layers=range(2)) -> dict[str, TensorEntry]:
    """Per-layer fsdp_dtensor expert tensors named ``form``."""
    shapes = {}
    for layer in layers:
        for expert in experts:
            for linear, shape in (("linear_fc1", (8, 4)), ("linear_fc2", (4, 8))):
                if form == GROUPED:
                    name = f"mlp.experts.{linear}.weight{expert}"
                else:
                    name = f"mlp.experts.local_experts.{expert}.{linear}.weight"
                shapes[f"decoder.layers.{layer}.{name}"] = shape
        shapes[f"decoder.layers.{layer}.mlp.router.weight"] = (E, 4)
    return _model(shapes)


@pytest.mark.parametrize(
    ("fields", "naming", "form", "grouped_gemm", "why"),
    [
        (
            {"moe_grouped_gemm": True},
            "auto",
            GROUPED,
            True,
            "args.moe_grouped_gemm=True builds TEGroupedMLP experts",
        ),
        ({}, "auto", SEQUENTIAL, False, "args.moe_grouped_gemm=False builds SequentialMLP experts"),
        ({"moe_grouped_gemm": True}, SEQUENTIAL, SEQUENTIAL, False, "--expert-naming sequential"),
        (
            {"hybrid_layer_pattern": "EE", "moe_grouped_gemm": False},
            "auto",
            GROUPED,
            None,
            "the hybrid stack spec builds TEGroupedMLP experts",
        ),
        (
            {"hybrid_layer_pattern": "EE", "moe_grouped_gemm": False},
            GROUPED,
            GROUPED,
            None,
            "--expert-naming grouped",
        ),
    ],
)
def test_a_torch_dist_source_gets_the_naming_its_builder_uses(
    fields, naming, form, grouped_gemm, why
):
    layout, evidence = expert_layout(_args(**fields), TORCH_DIST_MOE, TORCH_DIST, naming)
    assert layout == ExpertLayout(E, form, grouped_gemm)
    assert evidence == (
        f"args.num_experts={E}; 2 torch_dist tensors hold {E} experts each; "
        f"fsdp_dtensor names them {form} ({why})"
    )


def test_a_hybrid_target_cannot_name_its_experts_sequential():
    """No argument makes the hybrid stack spec build SequentialMLP experts."""
    args = _args(hybrid_layer_pattern="EE", moe_grouped_gemm=False)
    with pytest.raises(UsageError) as caught:
        expert_layout(args, TORCH_DIST_MOE, TORCH_DIST, SEQUENTIAL)
    assert caught.value.message == (
        "--expert-naming sequential: the hybrid stack spec builds TEGroupedMLP experts whatever "
        "the arguments say, which names them grouped, so no job built from these arguments "
        "loads experts named sequential"
    )
    assert caught.value.hint == "drop --expert-naming (it names the experts of GPT models)"
    # a hybrid source saved with sequential names is still read as the flag says
    layout, _ = expert_layout(args, _fsdp_moe(SEQUENTIAL), FSDP_DTENSOR, SEQUENTIAL)
    assert layout == ExpertLayout(E, SEQUENTIAL, None)


def test_a_model_without_experts_has_no_layout():
    assert expert_layout(_args(num_experts=None), _model({"a.weight": (2,)}), TORCH_DIST) == (
        None,
        "args.num_experts=None and no routed-expert tensors",
    )


@pytest.mark.parametrize(
    ("args", "model", "code", "message"),
    [
        (
            _args(num_experts=None),
            TORCH_DIST_MOE,
            errors.FC_R13,
            "the store holds 2 routed-expert tensors, but args.num_experts=None",
        ),
        (
            _args(num_experts=8),
            TORCH_DIST_MOE,
            errors.FC_R13,
            f"args.num_experts=8, but the expert axis of {STACKED_FC1} holds {E} experts",
        ),
        (
            _args(num_experts=8),
            _model({"decoder.layers.1.mlp.experts.experts.linear_fc2.bias": (E, 4)}),
            errors.FC_R13,
            "args.num_experts=8, but the expert axis of "
            "decoder.layers.1.mlp.experts.experts.linear_fc2.bias holds 4 experts",
        ),
    ],
)
def test_torch_dist_expert_axes_must_hold_num_experts(args, model, code, message):
    with pytest.raises(RefusedError) as caught:
        expert_layout(args, model, TORCH_DIST)
    assert caught.value.codes == (code,) and caught.value.refusal.message == message


@pytest.mark.parametrize(
    ("form", "fields"), [(GROUPED, {"moe_grouped_gemm": True}), (SEQUENTIAL, {})]
)
def test_an_fsdp_dtensor_source_names_experts_as_its_arguments_build_them(form, fields):
    layout, evidence = expert_layout(_args(**fields), _fsdp_moe(form), FSDP_DTENSOR)
    assert layout == ExpertLayout(E, form, form == GROUPED)
    assert evidence.startswith(
        f"args.num_experts={E}; 16 fsdp_dtensor tensors: 4 expert weights and biases for each "
        f"of experts 0-{E - 1}; the store names them {form} (args.moe_grouped_gemm="
    )


def test_an_fsdp_dtensor_source_named_otherwise_needs_expert_naming():
    grouped = _fsdp_moe(GROUPED)
    with pytest.raises(AmbiguousError) as caught:
        expert_layout(_args(moe_grouped_gemm=False), grouped, FSDP_DTENSOR)
    (refusal,) = caught.value.refusals
    assert refusal.code == errors.FC_A04
    assert refusal.message == (
        "the store names its routed experts grouped (TEGroupedMLP), but args.moe_grouped_gemm="
        "False builds SequentialMLP experts, which names them sequential"
    )
    assert refusal.hint == "pass --expert-naming grouped if the store is right"
    layout, evidence = expert_layout(_args(moe_grouped_gemm=False), grouped, FSDP_DTENSOR, GROUPED)
    assert layout.fsdp_form == GROUPED and evidence.endswith(
        "the store names them grouped (--expert-naming grouped)"
    )
    with pytest.raises(AmbiguousError, match="names them sequential"):
        expert_layout(_args(moe_grouped_gemm=True), grouped, FSDP_DTENSOR, SEQUENTIAL)


@pytest.mark.parametrize(
    ("model", "code", "message"),
    [
        (
            _fsdp_moe(GROUPED, experts=range(3)),
            errors.FC_R13,
            "args.num_experts=4, but the store holds decoder.layers.0.mlp.experts.linear_fc1."
            "weight for 3 experts (0-2)",
        ),
        (
            _fsdp_moe(GROUPED, experts=range(8)),
            errors.FC_R13,
            "args.num_experts=4, but the store holds decoder.layers.0.mlp.experts.linear_fc1."
            "weight for 8 experts (0-7)",
        ),
        (
            _fsdp_moe(GROUPED, experts=(0, 1, 2, 5)),
            errors.FC_R12,
            "the store holds decoder.layers.0.mlp.experts.linear_fc1.weight for experts 0-2, 5, "
            "not 0-3",
        ),
        (
            _fsdp_moe(GROUPED, experts=(0, 1, 3)),
            errors.FC_R12,
            "the store holds decoder.layers.0.mlp.experts.linear_fc1.weight for experts 0-1, 3, "
            "not 0-3",
        ),
        (
            _fsdp_moe(GROUPED, experts=range(1, 5)),
            errors.FC_R12,
            "the store holds decoder.layers.0.mlp.experts.linear_fc1.weight for experts 1-4, "
            "not 0-3",
        ),
        (
            {**_fsdp_moe(GROUPED, layers=[0]), **_fsdp_moe(SEQUENTIAL, layers=[1])},
            errors.FC_R13,
            "the store names its routed experts both grouped and sequential",
        ),
    ],
)
def test_fsdp_dtensor_experts_must_be_zero_to_num_experts(model, code, message):
    with pytest.raises(RefusedError) as caught:
        expert_layout(_args(moe_grouped_gemm=True), model, FSDP_DTENSOR)
    assert caught.value.codes == (code,) and caught.value.refusal.message == message
    assert caught.value.refusal.hint == (
        "check that the store is complete and the arguments belong to it"
        if code == errors.FC_R12
        else "check that the arguments belong to this checkpoint"
    )


def test_the_context_records_the_layout():
    ctx = build_arch_context(
        source=TORCH_DIST,
        target=FSDP_DTENSOR,
        args=_args(moe_grouped_gemm=True),
        model=TORCH_DIST_MOE,
        expert_naming=SEQUENTIAL,
    )
    assert ctx.experts == ExpertLayout(E, SEQUENTIAL, False)
    (logged,) = [d for d in ctx.derivation_log() if d["field"] == "experts"]
    assert logged["value"] == {"num_experts": E, "fsdp_form": SEQUENTIAL, "grouped_gemm": False}
    assert logged["evidence"].endswith(
        "fsdp_dtensor names them sequential (--expert-naming sequential)"
    )
