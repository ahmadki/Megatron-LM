# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The element type of converted tensors.

A weights-only conversion preserves every tensor's dtype in both directions.
Megatron's loaders cast on load (``copy_`` into the live tensor), so a
bfloat16 weight loads exactly into Megatron-FSDP's fp32 main weights, and the
fp32 main weights of an fsdp_dtensor store round to the bfloat16 parameters of
a torch_dist job as the training step would. Nothing depends on knowing which
tensors must stay fp32 (router biases, other fp32 buffers) or are integers:
they keep their dtype like every other tensor.
"""

from __future__ import annotations

import enum

import torch


class DtypePolicy(enum.Enum):
    """How the dtype of an output tensor is chosen."""

    PRESERVE = "preserve"
    """The output keeps the source tensor's dtype."""

    def output_dtype(self, source: torch.dtype) -> torch.dtype:
        """The dtype of an output tensor copied from a ``source``-typed tensor."""
        return source
