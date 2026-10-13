# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The element type of converted tensors.

A weights-only conversion preserves every tensor's dtype in both directions.
Megatron's loaders cast on load (``copy_`` into the live tensor), so a
bfloat16 weight loads exactly into Megatron-FSDP's fp32 main weights, and the
fp32 main weights of an fsdp_dtensor store round to the bfloat16 parameters of
a torch_dist job as the training step would. Nothing depends on knowing which
tensors must stay fp32 (router biases, other fp32 buffers) or are integers:
they keep their dtype like every other tensor.

A conversion of optimizer state from fsdp_dtensor to torch_dist
(:attr:`DtypePolicy.MASTERS`) writes what a torch_dist save of the training
job holds: the fp32 main copy (*master*) of every trainable parameter as
optimizer state, and as its model weight the master rounded to nearest even
to the parameter dtype, except for the parameters Megatron keeps in fp32 in a
bf16 or fp16 model (:data:`KEEP_FP32`, those ``mark_keep_in_fp32`` marks, so
``Float16Module`` does not cast them). Frozen parameters and buffers have no
optimizer state and keep their dtype; nothing is ever cast to or from an
integer or bool type.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass

import torch


class DtypePolicy(enum.Enum):
    """How the dtype of an output tensor is chosen."""

    PRESERVE = "preserve"
    """The output keeps the source tensor's dtype."""

    MASTERS = "masters"
    """Rules copy every tensor as stored (as :attr:`PRESERVE`); the optimizer plan then gives
    each trainable parameter its fp32 master and a model weight of :func:`model_weight_dtype`."""

    def output_dtype(self, source: torch.dtype) -> torch.dtype:
        """The dtype of an output tensor a rule copies from a ``source``-typed tensor."""
        return source


@dataclass(frozen=True, slots=True)
class KeepFp32:
    """A parameter Megatron keeps in fp32 in a bf16 or fp16 model (``mark_keep_in_fp32``).

    Attributes:
        owner: The Megatron class whose ``__init__`` marks it.
        attribute: Its attribute path in ``owner`` (``mapping_proj.weight``).
        pattern: Searched in a parameter's logical name (its name inside its
            layer, or from the model root outside layers).
    """

    owner: str
    attribute: str
    pattern: re.Pattern[str]


def _registry() -> tuple[KeepFp32, ...]:
    hyper_connection = ("mapping_proj.weight", "alpha_pre", "alpha_post", "alpha_res", "bias")
    hc_head = ("hc_head_fn", "hc_head_base", "hc_head_scale")
    return (
        *(
            KeepFp32(
                "HyperConnectionModule",
                attribute,
                # self_attention_hyper_connection, mlp_hyper_connection, hyper_connection
                re.compile(rf"(?:^|[._])hyper_connection\.{re.escape(attribute)}$"),
            )
            for attribute in hyper_connection
        ),
        *(
            KeepFp32(owner, attribute, re.compile(rf"(?:^|\.){attribute}$"))
            for owner in ("HybridStack", "MultiTokenPredictionLayer")
            for attribute in hc_head
        ),
        KeepFp32("Compressor", "ape", re.compile(r"(?:^|\.)compressor\.ape$")),
        KeepFp32("CompressedSparseAttention", "attn_sink", re.compile(r"(?:^|\.)attn_sink$")),
    )


KEEP_FP32 = _registry()
"""Every parameter ``mark_keep_in_fp32`` marks in Megatron Core (pinned by an AST scan of its
source): manifold hyper-connections' mapping and gates, the hyper-connection heads of hybrid
stacks and MTP layers, and compressed sparse attention's position embedding and sink."""


def keeps_fp32(name: str) -> bool:
    """Whether the parameter with logical name ``name`` is in the keep-fp32 registry."""
    return any(entry.pattern.search(name) is not None for entry in KEEP_FP32)


def model_weight_dtype(name: str, params_dtype: torch.dtype) -> torch.dtype:
    """The dtype of the model weight of the trainable parameter ``name`` beside its master: the
    model's parameter dtype, or fp32 for a parameter Megatron keeps in fp32."""
    return torch.float32 if keeps_fp32(name) else params_dtype
