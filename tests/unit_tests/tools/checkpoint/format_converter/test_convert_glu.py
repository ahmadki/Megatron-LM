# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Gated linear units (``linear_fc1`` of SwiGLU and GEGLU MLPs) on synthetic stores."""

from __future__ import annotations

from tools.checkpoint.format_converter.api import plan_conversion
from tools.checkpoint.format_converter.options import PlanOptions

from ._synthetic import gpt_args, gpt_tensors, stack_layers
from .test_convert_blocks import run, write_gpt_source

LAYERS = 3


def test_the_plan_logs_which_first_projections_are_gated(tmp_path):
    src = write_gpt_source(tmp_path / "src", stack_layers(gpt_tensors(LAYERS), "decoder"))
    plan = plan_conversion(src, PlanOptions(to_format="fsdp_dtensor")).to_json()
    (gated,) = [d for d in plan["derivations"] if d["field"] == "gated_fc1"]
    assert gated["value"] == {"decoder": {"plain": LAYERS}}
    assert gated["evidence"].endswith("make the language model's MLPs (in decoder) plain")
    assert plan["common"]["args_consulted"]["swiglu"] is False


def test_a_language_model_gated_otherwise_than_its_arguments_is_refused(tmp_path, capsys):
    """``--quick-geglu`` builds gated MLPs, but the store's are plain."""
    model = stack_layers(gpt_tensors(LAYERS), "decoder")
    src = write_gpt_source(tmp_path / "src", model, args=gpt_args(LAYERS, quick_geglu=True))
    assert run("plan", src, "--to", "fsdp_dtensor") == 3
    err = capsys.readouterr().err
    assert (
        "error: FC-R12: the language model's linear_fc1 weights (in decoder) are plain, but "
        "args.swiglu=False and args.quick_geglu=True make its MLPs gated "
        "[decoder.layers.mlp.linear_fc1.weight]"
    ) in err
