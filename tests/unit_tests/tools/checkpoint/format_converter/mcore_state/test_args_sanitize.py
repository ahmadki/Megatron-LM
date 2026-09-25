# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Weights-only args sanitation: only existing fields change, and every change is reported."""

from __future__ import annotations

import argparse

import torch

from tools.checkpoint.format_converter.mcore_state.args_sanitize import sanitize_weights_only


def test_existing_fields_are_set_and_reported():
    args = argparse.Namespace(no_save_optim=False, no_save_rng=None, params_dtype=torch.bfloat16)
    result = sanitize_weights_only(args)
    assert result.changes == {"no_save_optim": (False, True), "no_save_rng": (None, True)}
    assert result.consulted == {"no_save_optim": False, "no_save_rng": None}
    assert vars(result.args) == {
        "no_save_optim": True,
        "no_save_rng": True,
        "params_dtype": torch.bfloat16,
    }
    assert vars(args) == {
        "no_save_optim": False,
        "no_save_rng": None,
        "params_dtype": torch.bfloat16,
    }


def test_missing_fields_are_not_added_and_unchanged_ones_not_reported():
    args = argparse.Namespace(no_save_optim=True, num_layers=2)
    result = sanitize_weights_only(args)
    assert result.changes == {} and result.consulted == {"no_save_optim": True}
    assert vars(result.args) == {"no_save_optim": True, "num_layers": 2}
    assert result.args is not args
