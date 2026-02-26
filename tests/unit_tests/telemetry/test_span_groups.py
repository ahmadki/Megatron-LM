# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for span group state and control."""

import pytest

from megatron.core.telemetry._state import (
    is_span_group_enabled,
    set_enabled_span_groups,
)
from megatron.core.telemetry.config import SpanGroup


class TestSpanGroupState:
    def setup_method(self):
        """Reset state before each test."""
        set_enabled_span_groups(frozenset())

    def teardown_method(self):
        """Reset state after each test."""
        set_enabled_span_groups(frozenset())

    def test_all_disabled_by_default(self):
        set_enabled_span_groups(frozenset())
        for group in SpanGroup.ALL_GROUPS:
            assert not is_span_group_enabled(group), f"Expected {group} to be disabled"

    def test_set_and_check_single_group(self):
        set_enabled_span_groups(frozenset([SpanGroup.JOB]))
        assert is_span_group_enabled(SpanGroup.JOB) is True
        assert is_span_group_enabled(SpanGroup.STEP) is False

    def test_set_multiple_groups(self):
        groups = frozenset([SpanGroup.JOB, SpanGroup.CHECKPOINT, SpanGroup.EVALUATE])
        set_enabled_span_groups(groups)
        assert is_span_group_enabled(SpanGroup.JOB) is True
        assert is_span_group_enabled(SpanGroup.CHECKPOINT) is True
        assert is_span_group_enabled(SpanGroup.EVALUATE) is True
        assert is_span_group_enabled(SpanGroup.STEP) is False
        assert is_span_group_enabled(SpanGroup.MICROBATCH) is False

    def test_set_overrides_previous(self):
        set_enabled_span_groups(frozenset([SpanGroup.JOB]))
        assert is_span_group_enabled(SpanGroup.JOB) is True

        set_enabled_span_groups(frozenset([SpanGroup.STEP]))
        assert is_span_group_enabled(SpanGroup.JOB) is False
        assert is_span_group_enabled(SpanGroup.STEP) is True

    def test_unknown_group_returns_false(self):
        set_enabled_span_groups(frozenset([SpanGroup.JOB]))
        assert is_span_group_enabled("nonexistent_group") is False

    def test_default_preset_groups(self):
        groups = SpanGroup.resolve('default')
        set_enabled_span_groups(groups)
        assert is_span_group_enabled(SpanGroup.JOB) is True
        assert is_span_group_enabled(SpanGroup.CHECKPOINT) is True
        assert is_span_group_enabled(SpanGroup.EVALUATE) is True
        assert is_span_group_enabled(SpanGroup.INFERENCE) is True
        assert is_span_group_enabled(SpanGroup.STEP) is False
        assert is_span_group_enabled(SpanGroup.FORWARD_BACKWARD) is False
        assert is_span_group_enabled(SpanGroup.MICROBATCH) is False

    def test_per_step_preset_groups(self):
        groups = SpanGroup.resolve('per_step')
        set_enabled_span_groups(groups)
        assert is_span_group_enabled(SpanGroup.JOB) is True
        assert is_span_group_enabled(SpanGroup.STEP) is True
        assert is_span_group_enabled(SpanGroup.FORWARD_BACKWARD) is True
        assert is_span_group_enabled(SpanGroup.OPTIMIZER) is True
        assert is_span_group_enabled(SpanGroup.MODEL_INIT) is True
        assert is_span_group_enabled(SpanGroup.LOAD_CHECKPOINT) is True
        assert is_span_group_enabled(SpanGroup.INFERENCE) is True
        assert is_span_group_enabled(SpanGroup.MICROBATCH) is False

    def test_all_preset_groups(self):
        groups = SpanGroup.resolve('all')
        set_enabled_span_groups(groups)
        for group in SpanGroup.ALL_GROUPS:
            assert is_span_group_enabled(group) is True, f"Expected {group} to be enabled"

    def test_empty_frozenset_disables_all(self):
        set_enabled_span_groups(SpanGroup.resolve('all'))
        for group in SpanGroup.ALL_GROUPS:
            assert is_span_group_enabled(group) is True
        set_enabled_span_groups(frozenset())
        for group in SpanGroup.ALL_GROUPS:
            assert is_span_group_enabled(group) is False


class TestSpanGroupPublicAPI:
    """Ensure SpanGroup and is_span_group_enabled are importable from package root."""

    def test_importable_from_package(self):
        from megatron.core.telemetry import SpanGroup as _SG, is_span_group_enabled as _ise
        assert _SG.JOB == 'job'
        assert callable(_ise)

    def test_set_enabled_span_groups_importable(self):
        from megatron.core.telemetry import set_enabled_span_groups as _seg
        assert callable(_seg)
