# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Human-readable byte sizes: parsing ``4GiB``-style values and formatting them."""

from __future__ import annotations

import re

_UNITS = {
    "": 1,
    "B": 1,
    "KB": 10**3,
    "MB": 10**6,
    "GB": 10**9,
    "TB": 10**12,
    "KIB": 2**10,
    "MIB": 2**20,
    "GIB": 2**30,
    "TIB": 2**40,
}
_SIZE_PATTERN = re.compile(r"^\s*(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]*)\s*$")


def parse_size(text: str) -> int:
    """Parse ``"4GiB"``, ``"512MiB"``, ``"1.5GB"`` or ``"1048576"`` into bytes.

    Raises:
        ValueError: The text is not a non-negative number with a known unit.
    """
    match = _SIZE_PATTERN.match(text)
    unit = match.group("unit").upper() if match else None
    if match is None or unit not in _UNITS:
        raise ValueError(
            f"invalid size {text!r}; use a number with an optional unit such as 512MiB or 4GiB"
        )
    return int(float(match.group("number")) * _UNITS[unit])


def format_bytes(count: int) -> str:
    """Format a byte count with a binary unit, e.g. ``1.50 GiB``."""
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024:
            return f"{int(value)} B" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"
