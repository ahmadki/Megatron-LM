# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Byte-size parsing and formatting."""

from __future__ import annotations

import pytest

from tools.checkpoint.format_converter.engine.sizes import format_bytes, parse_size


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0", 0),
        ("1048576", 1048576),
        ("512B", 512),
        ("4GiB", 4 * 2**30),
        ("4gib", 4 * 2**30),
        ("256 MiB", 256 * 2**20),
        ("1.5GiB", 3 * 2**29),
        ("2KB", 2000),
        ("3MB", 3 * 10**6),
        ("1TiB", 2**40),
    ],
)
def test_parse_size(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "GiB", "-1GiB", "4 GiBs", "1e9", "four"])
def test_parse_size_rejects_garbage(text):
    with pytest.raises(ValueError, match="invalid size"):
        parse_size(text)


@pytest.mark.parametrize(
    "count, expected",
    [
        (0, "0 B"),
        (1023, "1023 B"),
        (1024, "1.00 KiB"),
        (3 * 2**29, "1.50 GiB"),
        (2**41, "2.00 TiB"),
    ],
)
def test_format_bytes(count, expected):
    assert format_bytes(count) == expected
