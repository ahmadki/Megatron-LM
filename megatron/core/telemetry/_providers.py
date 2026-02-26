# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Internal: build TracerProvider + MeterProvider.

This module is *only* imported on the export rank when telemetry is enabled.
All heavy SDK imports live here; ``opentelemetry-api`` (no-op) is the only
dependency for code paths that never reach this module.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.core.telemetry.config import TelemetryConfig


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


def build_providers(config: 'TelemetryConfig', rank: int, world_size: int) -> None:
    """Initialise TracerProvider and MeterProvider for the export rank.

    Imports the OTel SDK.  If the SDK is not installed this raises an
    :class:`ImportError` with a helpful message.

    Registers the providers globally via
    ``opentelemetry.trace.set_tracer_provider`` and
    ``opentelemetry.metrics.set_meter_provider``.

    Args:
        config: Telemetry configuration.
        rank: Current process rank.
        world_size: Total number of ranks.
    """
    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise ImportError(
            "OpenTelemetry SDK is required for telemetry export but is not installed. "
            "Install it with: pip install 'megatron-core[otel]'"
        ) from exc

    # ------------------------------------------------------------------
    # Resource
    # ------------------------------------------------------------------
    try:
        from megatron.core.package_info import __version__ as _megatron_version
    except ImportError:
        _megatron_version = 'unknown'

    resource_attrs = {
        'service.name': config.service_name,
        'service.version': _megatron_version,
        # GenAI semconv: identifies the AI provider/framework.
        # https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
        'gen_ai.provider.name': 'megatron',
        'megatron.rank': rank,
        'megatron.world_size': world_size,
    }
    env_name = os.environ.get('DEPLOYMENT_ENV', os.environ.get('ENVIRONMENT', ''))
    if env_name:
        resource_attrs['deployment.environment'] = env_name

    resource = Resource.create(resource_attrs)

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------
    if config.traces_enabled:
        from opentelemetry import trace

        span_exporter = _build_span_exporter(config)
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    if config.metrics_enabled:
        from opentelemetry import metrics

        metric_exporter = _build_metric_exporter(config)
        # Export every 10s (default 60s is too slow for interactive dev).
        # Override with OTEL_METRIC_EXPORT_INTERVAL env var (milliseconds).
        _export_interval = int(os.environ.get('OTEL_METRIC_EXPORT_INTERVAL', '10000'))
        reader = PeriodicExportingMetricReader(
            metric_exporter, export_interval_millis=_export_interval
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)

    # ------------------------------------------------------------------
    # Propagator (W3C TraceContext + Baggage)
    # ------------------------------------------------------------------
    _set_propagator()


def build_noop_providers() -> None:
    """Register no-op providers for non-export ranks or disabled telemetry.

    This ensures all ranks have a consistent provider installed without
    importing the SDK.  Note: calling ``set_tracer_provider`` /
    ``set_meter_provider`` is a set-once operation — the OTel API will log a
    warning if a provider is replaced after it has already been set.  The
    default global providers are already no-ops, but we call these explicitly
    so that all ranks share a deterministic initialisation path.
    """
    from opentelemetry import metrics, trace
    from opentelemetry.metrics import NoOpMeterProvider
    from opentelemetry.trace import NoOpTracerProvider

    trace.set_tracer_provider(NoOpTracerProvider())
    metrics.set_meter_provider(NoOpMeterProvider())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_VALID_EXPORTERS = ('otlp', 'console')


def _build_span_exporter(config: 'TelemetryConfig'):
    """Return a span exporter for the configured exporter type."""
    if config.exporter not in _VALID_EXPORTERS:
        raise ValueError(
            f"Unknown exporter type: {config.exporter!r}. "
            f"Expected one of: {_VALID_EXPORTERS}"
        )

    if config.exporter == 'console':
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter()

    # Default: OTLP (gRPC or HTTP based on OTEL_EXPORTER_OTLP_PROTOCOL env var)
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter()
    except ImportError:
        pass
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter()
    except ImportError:
        pass
    raise ImportError(
        "No OTLP span exporter found.  Install with: "
        "pip install 'megatron-core[otel]'"
    )


def _build_metric_exporter(config: 'TelemetryConfig'):
    """Return a metric exporter for the configured exporter type."""
    if config.exporter not in _VALID_EXPORTERS:
        raise ValueError(
            f"Unknown exporter type: {config.exporter!r}. "
            f"Expected one of: {_VALID_EXPORTERS}"
        )

    if config.exporter == 'console':
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

        return ConsoleMetricExporter()

    try:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

        return OTLPMetricExporter()
    except ImportError:
        pass
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        return OTLPMetricExporter()
    except ImportError:
        pass
    raise ImportError(
        "No OTLP metric exporter found.  Install with: "
        "pip install 'megatron-core[otel]'"
    )


def _set_propagator() -> None:
    """Set W3C TraceContext + Baggage as the global text map propagator."""
    from opentelemetry import propagate
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    propagate.set_global_textmap(
        CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()])
    )
