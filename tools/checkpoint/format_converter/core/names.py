# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Names of the supported checkpoint formats, their dialects, the fsdp_dtensor key envelope
and the torch_dist block layouts.

This module imports nothing, so the command line can validate format flags
without importing torch.
"""

from __future__ import annotations

TORCH_DIST = "torch_dist"
FSDP_DTENSOR = "fsdp_dtensor"
FORMATS = (TORCH_DIST, FSDP_DTENSOR)
DIALECTS = {TORCH_DIST: "mcore", FSDP_DTENSOR: "mfsdp_v1"}
"""The dialect of each format this tool reads: Megatron Core and Megatron-FSDP v1."""

FSDP_MODEL_PREFIX = "model.module."
"""Prefix of model keys in Megatron-FSDP v1 stores (one ``module.`` level for every precision)."""

FSDP_OPTIMIZER_STATE = "optimizer.state."
"""Prefix of optimizer-state keys in Megatron-FSDP v1 stores (``module.`` levels follow)."""

STACKED = "stacked"
"""torch_dist block layout: one key per tensor, the layers along a new first axis."""

PER_LAYER = "per_layer"
"""torch_dist block layout: one key per layer (``<block>.layers.<index>.<name>``)."""

TD_LAYOUTS = (STACKED, PER_LAYER)
"""How torch_dist stores a block's layers."""

NATIVE_LAYOUT = "native"
"""``--torch-dist-layout``: stack each TransformerBlock as Megatron would (the default)."""

PER_LAYER_LAYOUT = "per-layer"
"""``--torch-dist-layout``: every TransformerBlock per layer, marked in ``content_metadata``."""

TORCH_DIST_LAYOUTS = (NATIVE_LAYOUT, PER_LAYER_LAYOUT)
"""Values of ``--torch-dist-layout``."""
