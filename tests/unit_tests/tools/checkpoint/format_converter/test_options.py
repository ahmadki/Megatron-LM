# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Option validation and defaults."""

from __future__ import annotations

import re

import pytest

from tools.checkpoint.format_converter import cli, options
from tools.checkpoint.format_converter.errors import UsageError
from tools.checkpoint.format_converter.formats.detect import FORMATS
from tools.checkpoint.format_converter.options import InspectOptions


def test_default_max_item_bytes_is_a_quarter_of_available_memory(monkeypatch):
    monkeypatch.setattr(options, "available_memory_bytes", lambda: 64 * 2**30)
    assert InspectOptions().max_item_bytes == 16 * 2**30
    monkeypatch.setattr(options, "available_memory_bytes", lambda: None)
    assert InspectOptions().max_item_bytes == 4 * 2**30


def test_available_memory_is_read_on_linux():
    available = options.available_memory_bytes()
    assert available is None or available > 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"workers": 0}, "--workers"),
        ({"max_item_bytes": 0}, "--max-item-bytes"),
        ({"iteration": -1}, "--iteration"),
        ({"from_format": "zarr"}, "--from"),
    ],
)
def test_invalid_inspect_options(kwargs, message):
    with pytest.raises(UsageError, match=message):
        InspectOptions(**kwargs)


def test_key_selection():
    assert InspectOptions().selects("anything") and not InspectOptions().lists_keys
    opts = InspectOptions(keys=re.compile(r"\.bias$"))
    assert opts.lists_keys and opts.selects("a.bias") and not opts.selects("a.bias.x")
    assert InspectOptions(chunks=True).lists_keys


def test_cli_format_choices_match_the_detector():
    assert cli.FORMAT_CHOICES == FORMATS
