# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parameter groups rebuilt from Megatron-FSDP's per-parameter group hyperparameters."""

from __future__ import annotations

import argparse

import pytest

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.param_groups import (
    ParamGroup,
    chained_groups,
    identifier,
    optimizer_common_state,
    rebuild_param_groups,
    uses_expert_process_groups,
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


def _args(**fields) -> ArgsView:
    return ArgsView(argparse.Namespace(**fields))


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, False),
        ({"tensor_model_parallel_size": 2}, False),  # the expert size follows it
        ({"expert_model_parallel_size": 2}, True),
        ({"tensor_model_parallel_size": 2, "expert_tensor_parallel_size": 1}, True),
        ({"tensor_model_parallel_size": 2, "expert_tensor_parallel_size": 2}, False),
        ({"expert_gtp_weight_remat_size": 2}, True),
        ({"gtp_weight_remat_size": 2, "expert_gtp_weight_remat_size": 2}, False),
    ],
)
def test_routed_experts_use_the_expert_process_groups_as_megatron_decides(fields, expected):
    assert uses_expert_process_groups(_args(**fields)) is expected


EXPERT_FC1 = "decoder.layers.0.mlp.experts.linear_fc1.weight3"
SHARED_FC1 = "decoder.layers.0.mlp.shared_experts.linear_fc1.weight"


def test_expert_parallel_groups_get_an_optimizer_of_their_own():
    dense, no_decay = ParamGroup(DECAY, (SHARED_FC1,)), ParamGroup(NO_DECAY, ("b",))
    experts = ParamGroup(EXPERT, (EXPERT_FC1,))
    chained = chained_groups([dense, experts, no_decay], _args(expert_model_parallel_size=2))
    assert chained == ((dense, no_decay), (experts,))
    assert optimizer_common_state(chained) == {
        0: {
            "optimizer": {"param_groups": [DECAY, NO_DECAY]},
            "param_state_sharding_type": "fully_reshardable",
        },
        1: {
            "optimizer": {"param_groups": [EXPERT]},
            "param_state_sharding_type": "fully_reshardable",
        },
    }


def test_without_expert_process_groups_one_optimizer_holds_every_group():
    groups = [ParamGroup(DECAY, (EXPERT_FC1, SHARED_FC1)), ParamGroup(NO_DECAY, ("b",))]
    chained = chained_groups(groups, _args())
    assert chained == (tuple(groups),)
    assert optimizer_common_state(chained) == {
        "optimizer": {"param_groups": [DECAY, NO_DECAY]},
        "param_state_sharding_type": "fully_reshardable",
    }


@pytest.mark.parametrize(
    ("groups", "fields", "wrong"),
    [
        ([ParamGroup(EXPERT, (EXPERT_FC1,))], {}, (EXPERT_FC1,)),
        ([ParamGroup(DECAY, (EXPERT_FC1,))], {"expert_model_parallel_size": 2}, (EXPERT_FC1,)),
        ([ParamGroup(EXPERT, (SHARED_FC1,))], {"expert_model_parallel_size": 2}, (SHARED_FC1,)),
    ],
)
def test_groups_that_contradict_the_arguments_are_refused(groups, fields, wrong):
    with pytest.raises(RefusedError) as caught:
        chained_groups(groups, _args(**fields))
    refusal = caught.value.refusal
    assert (refusal.code, refusal.keys) == (errors.FC_R13, wrong)
    assert "set is_expert_parallel otherwise than a job of the store's arguments" in refusal.message
