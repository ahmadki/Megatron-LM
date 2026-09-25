# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Layer 3: Megatron-dependent state (common state, training arguments).

Megatron is imported lazily, inside functions, and only where a payload needs
its classes; every module here imports on a host without Megatron.
"""

from __future__ import annotations
