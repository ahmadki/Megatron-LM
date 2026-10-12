# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parameter groups rebuilt from Megatron-FSDP's per-parameter group hyperparameters."""

from __future__ import annotations

import pytest

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.param_groups import (
    ParamGroup,
    identifier,
    rebuild_param_groups,
)
from tools.checkpoint.format_converter.errors import RefusedError

DECAY = {
    "lr": 1e-3,
    "betas": (0.9, 0.95),
    "weight_decay": 0.1,
    "wd_mult": 1.0,
    "lr_mult": 1.0,
    "max_lr": 1e-3,
    "min_lr": 1e-5,
    "is_expert_parallel": False,
    "is_decoupled_lr": False,
    "default_config": True,
    "step": 12,
}
NO_DECAY = {**DECAY, "wd_mult": 0.0, "weight_decay": 0.0}
EXPERT = {**DECAY, "is_expert_parallel": True}


def test_parameters_with_equal_hyperparameters_share_a_group():
    groups = rebuild_param_groups(
        {
            "decoder.layers.1.mlp.linear_fc1.weight": DECAY,
            "decoder.final_layernorm.weight": dict(NO_DECAY),
            "decoder.layers.0.mlp.linear_fc1.weight": dict(DECAY),
            "decoder.layers.0.mlp.experts.linear_fc1.weight3": EXPERT,
            "decoder.layers.0.mlp.linear_fc1.bias": NO_DECAY,
        }
    )
    assert groups == (
        ParamGroup(
            NO_DECAY, ("decoder.final_layernorm.weight", "decoder.layers.0.mlp.linear_fc1.bias")
        ),
        ParamGroup(EXPERT, ("decoder.layers.0.mlp.experts.linear_fc1.weight3",)),
        ParamGroup(
            DECAY,
            ("decoder.layers.0.mlp.linear_fc1.weight", "decoder.layers.1.mlp.linear_fc1.weight"),
        ),
    )


def test_the_identifier_is_what_the_loader_pairs_groups_by():
    assert identifier(DECAY) == (None, 1e-3, 1e-5, None, None, 1.0, 1.0, False, False)
    nemo = {key: value for key, value in DECAY.items() if key != "lr_mult"} | {"pre_lr_mult": 2.0}
    assert identifier(nemo) == (None, 1e-3, 1e-5, None, None, 1.0, 2.0, False, False)


def test_groups_that_share_an_identifier_are_refused():
    """Equal identifiers, other learning rates: the loader would merge the groups."""
    meta = {"a.weight": DECAY, "b.weight": {**DECAY, "lr": 2e-3}, "c.weight": NO_DECAY}
    with pytest.raises(RefusedError) as caught:
        rebuild_param_groups(meta)
    refusal = caught.value.refusal
    assert refusal.code == errors.FC_R13
    assert refusal.keys == ("a.weight", "b.weight")
    assert refusal.message.startswith("parameter groups that differ in lr share the identifier")
    assert "max_lr=0.001" in refusal.message
