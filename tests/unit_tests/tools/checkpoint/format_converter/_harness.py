# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Process-level helpers shared by the format_converter tests (no megatron imports)."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]

_SCRUBBED_NAMES = frozenset(
    {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE"}
)
_SCRUBBED_PREFIXES = ("ROLE_", "MASTER_", "TORCHELASTIC_", "TORCH_NCCL_")


def is_local_rank_zero() -> bool:
    """True unless a launcher marked this process as a non-zero local rank."""
    return os.environ.get("LOCAL_RANK", "0") == "0"


def scrubbed_env() -> dict[str, str]:
    """Environment for child processes: no launcher variables and no visible GPU."""
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in _SCRUBBED_NAMES and not name.startswith(_SCRUBBED_PREFIXES)
    }
    env["CUDA_VISIBLE_DEVICES"] = ""
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + pythonpath if pythonpath else "")
    return env
