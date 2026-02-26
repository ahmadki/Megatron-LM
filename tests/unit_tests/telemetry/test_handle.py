# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for TelemetryHandle and setup_telemetry."""

import pytest

from megatron.core.telemetry.config import TelemetryConfig
from megatron.core.telemetry.handle import setup_telemetry, TelemetryHandle


class TestSetupTelemetryDisabled:
    def test_returns_telemetry_handle(self):
        cfg = TelemetryConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        assert isinstance(handle, TelemetryHandle)

    def test_tracer_is_accessible(self):
        cfg = TelemetryConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        # Should not raise; returns a no-op tracer
        tracer = handle.tracer
        assert tracer is not None

    def test_meter_is_accessible(self):
        cfg = TelemetryConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        meter = handle.meter
        assert meter is not None

    def test_no_op_tracer_creates_span_without_error(self):
        cfg = TelemetryConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        # Using a no-op tracer must not raise
        with handle.tracer.start_as_current_span("test.span") as span:
            assert span is not None

    def test_shutdown_completes_without_error(self):
        cfg = TelemetryConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        handle.shutdown(timeout_ms=100)


class TestSetupTelemetryNonExportRank:
    def test_non_export_rank_gets_noop_tracer(self):
        """Non-export ranks should get no-op providers (spans discarded)."""
        cfg = TelemetryConfig(enabled=True, export_rank=-1)
        # rank=0 is not the export rank when world_size=4 (export_rank resolves to 3)
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        assert isinstance(handle, TelemetryHandle)
        # Span creation must not raise
        with handle.tracer.start_as_current_span("test.span") as span:
            assert span is not None

    def test_export_rank_zero_config(self):
        """export_rank=0 means rank=0 is the exporter."""
        cfg = TelemetryConfig(enabled=True, export_rank=0, exporter='console')
        # Only rank=0 would import SDK; rank=1 should be no-op
        handle = setup_telemetry(cfg, rank=1, world_size=4)
        assert isinstance(handle, TelemetryHandle)

    def test_last_rank_default_resolution(self):
        """export_rank=-1 resolves to world_size-1."""
        cfg = TelemetryConfig(enabled=True, export_rank=-1, exporter='console')
        # rank=3 is the export rank with world_size=4
        handle = setup_telemetry(cfg, rank=3, world_size=4)
        assert isinstance(handle, TelemetryHandle)


class TestTelemetryHandleShutdown:
    def test_shutdown_idempotent(self):
        cfg = TelemetryConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=1)
        # Multiple shutdowns should not raise
        handle.shutdown(timeout_ms=100)
        handle.shutdown(timeout_ms=100)


class TestSetupTelemetrySpanGroups:
    def setup_method(self):
        from megatron.core.telemetry._state import set_enabled_span_groups
        set_enabled_span_groups(frozenset())

    def teardown_method(self):
        from megatron.core.telemetry._state import set_enabled_span_groups
        set_enabled_span_groups(frozenset())

    def test_disabled_clears_all_groups(self):
        from megatron.core.telemetry._state import is_span_group_enabled
        from megatron.core.telemetry.config import SpanGroup
        cfg = TelemetryConfig(enabled=False, span_groups='all')
        setup_telemetry(cfg, rank=0, world_size=1)
        # All groups must be disabled when telemetry is off
        for group in SpanGroup.ALL_GROUPS:
            assert not is_span_group_enabled(group)

    def test_enabled_registers_default_groups(self):
        from megatron.core.telemetry._state import is_span_group_enabled
        from megatron.core.telemetry.config import SpanGroup
        cfg = TelemetryConfig(enabled=True, span_groups='default', exporter='console')
        setup_telemetry(cfg, rank=0, world_size=1)
        assert is_span_group_enabled(SpanGroup.JOB) is True
        assert is_span_group_enabled(SpanGroup.CHECKPOINT) is True
        assert is_span_group_enabled(SpanGroup.INFERENCE) is True
        assert is_span_group_enabled(SpanGroup.STEP) is False

    def test_enabled_registers_per_step_groups(self):
        from megatron.core.telemetry._state import is_span_group_enabled
        from megatron.core.telemetry.config import SpanGroup
        cfg = TelemetryConfig(enabled=True, span_groups='per_step', exporter='console')
        setup_telemetry(cfg, rank=0, world_size=1)
        assert is_span_group_enabled(SpanGroup.STEP) is True
        assert is_span_group_enabled(SpanGroup.FORWARD_BACKWARD) is True
        assert is_span_group_enabled(SpanGroup.MICROBATCH) is False

    def test_non_export_rank_clears_span_groups(self):
        """Non-export ranks must get frozenset() so no spans are created at all."""
        from megatron.core.telemetry._state import is_span_group_enabled
        from megatron.core.telemetry.config import SpanGroup
        # rank=0 is NOT the export rank (export_rank=-1 resolves to world_size-1=3)
        cfg = TelemetryConfig(enabled=True, span_groups='all', exporter='console')
        setup_telemetry(cfg, rank=0, world_size=4)
        for group in SpanGroup.ALL_GROUPS:
            assert not is_span_group_enabled(group), f"group {group!r} should be disabled"


class TestTelemetryHandleIsExporting:
    def setup_method(self):
        from megatron.core.telemetry._state import set_enabled_span_groups
        set_enabled_span_groups(frozenset())

    def teardown_method(self):
        from megatron.core.telemetry._state import set_enabled_span_groups
        set_enabled_span_groups(frozenset())

    def test_disabled_is_not_exporting(self):
        cfg = TelemetryConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=1)
        assert handle.is_exporting is False

    def test_export_rank_is_exporting(self):
        cfg = TelemetryConfig(enabled=True, export_rank=0, exporter='console')
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        assert handle.is_exporting is True

    def test_non_export_rank_is_not_exporting(self):
        # rank=0 is NOT the export rank when export_rank=-1 and world_size=4
        cfg = TelemetryConfig(enabled=True, export_rank=-1, exporter='console')
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        assert handle.is_exporting is False
