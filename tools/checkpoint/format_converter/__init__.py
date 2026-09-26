# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Streaming inspection, verification and conversion of Megatron DCP checkpoint stores.

Run it as ``python -m tools.checkpoint.format_converter <command>`` from the
repository root. The public Python API is re-exported here lazily (PEP 562), so
importing this package does not import torch.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

_LAZY_EXPORTS: dict[str, str] = {
    "convert_checkpoint": ".api",
    "inspect_checkpoint": ".api",
    "plan_conversion": ".api",
    "verify_checkpoint": ".api",
}

__all__ = ["__version__", *sorted(_LAZY_EXPORTS)]


def __getattr__(name: str) -> Any:
    """Resolve a public API name on first access (PEP 562)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name, __name__), name)


def __dir__() -> list[str]:
    """List the module attributes including the lazily exported API."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
