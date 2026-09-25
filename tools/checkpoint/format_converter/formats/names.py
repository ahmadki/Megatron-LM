# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Names of the supported checkpoint formats and their dialects.

This module imports nothing, so the command line can validate format flags
without importing torch.
"""

from __future__ import annotations

TORCH_DIST = "torch_dist"
FSDP_DTENSOR = "fsdp_dtensor"
FORMATS = (TORCH_DIST, FSDP_DTENSOR)
DIALECTS = {TORCH_DIST: "mcore", FSDP_DTENSOR: "mfsdp_v1"}
"""The dialect of each format this tool reads: Megatron Core and Megatron-FSDP v1."""
