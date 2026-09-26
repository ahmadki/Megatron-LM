# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Option validation and defaults."""

from __future__ import annotations

import re

import pytest

from tools.checkpoint.format_converter import cli, options
from tools.checkpoint.format_converter.engine.staging import RESUME_MODES
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


def test_cli_choices_match_the_options():
    assert cli.FORMAT_CHOICES == FORMATS
    assert cli.COMMON_CHOICES == options.COMMON_MODES
    assert cli.OPTIMIZER_CHOICES == options.OPTIMIZER_CHOICES
    assert cli.MAP_CHOICES == options.MAP_MODES
    assert cli.RESUME_CHOICES == options.RESUME_CHOICES == RESUME_MODES


def test_conversion_budgets_default_to_fractions_of_available_memory(monkeypatch):
    monkeypatch.setattr(options, "available_memory_bytes", lambda: 100 * 2**30)
    opts = options.PlanOptions(workers=4)
    assert opts.max_unit_bytes_hard == int(0.6 * 100 * 2**30 / 4)
    assert opts.max_host_bytes == int(0.8 * 100 * 2**30)
    assert options.PlanOptions(workers=0).max_unit_bytes_hard == int(0.6 * 100 * 2**30)
    monkeypatch.setattr(options, "available_memory_bytes", lambda: None)
    assert options.PlanOptions(workers=1).max_host_bytes == int(0.8 * 16 * 2**30)
    explicit = options.PlanOptions(max_unit_bytes_hard=5, max_host_bytes=7)
    assert (explicit.max_unit_bytes_hard, explicit.max_host_bytes) == (5, 7)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"to_format": "zarr"}, "--to must be one of"),
        ({"optimizer": "keep"}, "--optimizer"),
        ({"allow_roots": ("vision.", ".")}, "--allow-root"),
        ({"max_output_chunk_bytes": 0}, "--max-output-chunk-bytes"),
        ({"from_format": "zarr"}, "--from"),
        ({"iteration": -1}, "--iteration"),
        ({"workers": -1}, "--workers"),
        ({"writer_threads": 0}, "--writer-threads"),
        ({"unit_bytes": 0}, "--unit-bytes"),
        ({"max_unit_bytes_hard": 0}, "--max-unit-bytes-hard"),
        ({"max_host_bytes": -1}, "--max-host-bytes"),
        ({"resume": "sometimes"}, "--resume"),
    ],
)
def test_invalid_conversion_options(kwargs, message):
    with pytest.raises(UsageError, match=message):
        options.ConvertOptions(**kwargs)


def test_output_fields_cover_what_shapes_the_output():
    opts = options.ConvertOptions(drop_common_keys=frozenset({"b", "a"}), workers=3)
    assert opts.output_fields() == {
        "to_format": "torch_dist",
        "drop_common_keys": ["a", "b"],
        "unit_bytes": 4 * 2**30,
        "writer_threads": 4,
        "optimizer": "drop",
        "allow_roots": [],
        "max_output_chunk_bytes": 2**30,
        "allow_missing_fp8_extra_state": False,
        "ignore_manifest": False,
        "copy_sidecars": True,
    }


def test_allowed_roots_are_normalized():
    opts = options.PlanOptions(allow_roots=("vision.", "audio", "vision"))
    assert opts.allow_roots == ("audio", "vision")
