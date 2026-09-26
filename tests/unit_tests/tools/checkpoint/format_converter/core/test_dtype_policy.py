# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Weights-only conversions preserve every dtype, integers and fp32 buffers included."""

from __future__ import annotations

import pytest
import torch

from tools.checkpoint.format_converter.core.dtype_policy import DtypePolicy


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64, torch.int64, torch.bool]
)
def test_preserve_keeps_the_source_dtype(dtype):
    assert DtypePolicy.PRESERVE.output_dtype(dtype) is dtype
