# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""ArgsView: primitive-only reads that are recorded."""

from __future__ import annotations

import argparse
import enum

import pytest
import torch

from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.errors import IntegrityError, InternalError


class Backend(enum.Enum):
    """A non-primitive argument value."""

    auto = 5


ARGS = argparse.Namespace(
    num_layers=4,
    swiglu=True,
    lr=1e-4,
    hybrid_layer_pattern="M*-",
    tensor_model_parallel_size=2,
    params_dtype=torch.bfloat16,
    attention_backend=Backend.auto,
    moe_layer_freq=[1, 0, 1],
    num_experts=None,
)


def test_primitive_fields_are_read_and_recorded():
    view = ArgsView(ARGS)
    assert view.get("num_layers") == 4 and view.get("swiglu") is True
    assert view.get("moe_layer_freq") == [1, 0, 1] and view.get("num_experts") is None
    assert view.get("missing", 3) == 3
    assert view.has("lr") and not view.has("missing")
    assert view.consumed == {
        "moe_layer_freq": [1, 0, 1],
        "num_experts": None,
        "num_layers": 4,
        "swiglu": True,
    }


@pytest.mark.parametrize("name", ["params_dtype", "attention_backend"])
def test_non_primitive_fields_are_refused(name):
    with pytest.raises(InternalError, match=f"args.{name} is a .*, not a primitive value"):
        ArgsView(ARGS).get(name)


def test_missing_fields_need_a_default():
    with pytest.raises(InternalError, match="args has no field 'missing'"):
        ArgsView(ARGS).get("missing")


def test_args_must_be_a_namespace():
    with pytest.raises(IntegrityError, match="args are a dict, not a Namespace"):
        ArgsView({"num_layers": 4})
