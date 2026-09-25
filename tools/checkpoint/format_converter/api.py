# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Public Python API of format_converter.

The functions below are placeholders until the commands land later in this
series; they exist so that the package exports are stable from the start.
"""

from __future__ import annotations

from pathlib import Path


def inspect_checkpoint(src: Path, opts: object) -> object:
    """Summarize a checkpoint store (not implemented yet)."""
    raise NotImplementedError(f"inspect is not implemented yet (src={src}, opts={opts})")


def verify_checkpoint(a: Path, b: Path | None, opts: object) -> object:
    """Compare two checkpoint stores (not implemented yet)."""
    raise NotImplementedError(f"verify is not implemented yet (a={a}, b={b}, opts={opts})")
