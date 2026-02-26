# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-LM OpenTelemetry instrumentation.

Public API
----------

.. code-block:: python

    from megatron.core.telemetry import (
        TelemetryConfig,
        SpanGroup,
        TelemetryHandle,
        setup_telemetry,
        span_cm,
        get_tracer,
        get_meter,
        is_span_group_enabled,
    )

Quick start
-----------

1. Set ``MEGATRON_OTEL_ENABLED=1`` and optionally
   ``OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317``.
2. Start training normally — spans are emitted at framework boundaries.

See ``megatron/core/telemetry/README.md`` for the full configuration
reference and local dev setup guide.
"""

from megatron.core.telemetry.config import SpanGroup, TelemetryConfig
from megatron.core.telemetry.handle import TelemetryHandle, setup_telemetry
from megatron.core.telemetry._state import is_span_group_enabled, set_enabled_span_groups
from megatron.core.telemetry.helpers import (
    DEFAULT_REDACT_KEYS,
    managed_span,
    redact_value,
    safe_set_span_attributes,
    span_cm,
    trace_fn,
)
from megatron.core.telemetry.propagation import extract_context, inject_context

from opentelemetry import metrics as _metrics_mod, trace as _trace_mod


def get_tracer(name: str = 'megatron.core'):
    """Return the globally registered tracer.

    Returns a no-op tracer if OTel has not been initialised.

    Args:
        name: Instrumentation scope name (default: ``"megatron.core"``).
    """
    return _trace_mod.get_tracer(name)


def get_meter(name: str = 'megatron.core'):
    """Return the globally registered meter.

    Returns a no-op meter if OTel has not been initialised.

    Args:
        name: Instrumentation scope name (default: ``"megatron.core"``).
    """
    return _metrics_mod.get_meter(name)


__all__ = [
    'SpanGroup',
    'TelemetryConfig',
    'TelemetryHandle',
    'setup_telemetry',
    'span_cm',
    'managed_span',
    'trace_fn',
    'safe_set_span_attributes',
    'redact_value',
    'DEFAULT_REDACT_KEYS',
    'inject_context',
    'extract_context',
    'get_tracer',
    'get_meter',
    'is_span_group_enabled',
    'set_enabled_span_groups',
    # Internal metric helpers (not part of the public API, but importable)
    # 'record_training_metrics', 'record_inference_metrics' are in _instruments
]
