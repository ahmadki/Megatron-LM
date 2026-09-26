# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Names of the supported checkpoint formats, their dialects and the fsdp_dtensor key envelope.

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
