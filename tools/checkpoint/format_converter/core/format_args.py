# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The training-argument fields that describe what a store holds and how it is laid out.

Megatron's loader reads some saved arguments to decide what to load (for
example ``no_save_optim``), and ``--use-checkpoint-args`` takes the format and
the parallel layout from them. A converted store's arguments must describe the
converted store; ``mcore_state/args_sanitize.py`` applies these values and
``verify --expected-diff`` accepts differences in exactly these fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

WEIGHTS_ONLY_FIELDS: Mapping[str, Any] = {"no_save_optim": True, "no_save_rng": True}
"""Arguments of a store holding model weights only (no optimizer or RNG state)."""

FSDP_DTENSOR_FIELDS: Mapping[str, Any] = {"ckpt_format": "fsdp_dtensor", "use_megatron_fsdp": True}
"""Arguments of a Megatron-FSDP v1 fsdp_dtensor store."""

TORCH_DIST_FIELDS: Mapping[str, Any] = {"ckpt_format": "torch_dist", "use_megatron_fsdp": False}
"""Arguments of a torch_dist store."""

SINGLE_STAGE_FIELDS: Mapping[str, Any] = {
    "pipeline_model_parallel_size": 1,
    "transformer_pipeline_model_parallel_size": 1,
    "virtual_pipeline_model_parallel_size": None,
    "num_layers_per_virtual_pipeline_stage": None,
    "num_virtual_stages_per_pipeline_rank": None,
    "pipeline_model_parallel_layout": None,
    "decoder_first_pipeline_num_layers": None,
    "decoder_last_pipeline_num_layers": None,
}
"""The pipeline layout of a single-stage job (what an fsdp_dtensor store describes)."""

PATTERN_FIELDS = ("hybrid_layer_pattern", "hybrid_override_pattern")
"""Hybrid layer patterns, whose ``|`` separators mark pipeline stages."""

SANITIZED_FIELDS = frozenset(
    {*WEIGHTS_ONLY_FIELDS, *FSDP_DTENSOR_FIELDS, *SINGLE_STAGE_FIELDS, *PATTERN_FIELDS}
)
"""Every argument field a weights-only conversion may change."""
