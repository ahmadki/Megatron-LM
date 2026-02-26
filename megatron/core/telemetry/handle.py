# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TelemetryHandle: lifecycle wrapper for the OTel tracer and meter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from opentelemetry import metrics, trace

if TYPE_CHECKING:
    from megatron.core.telemetry.config import TelemetryConfig

# Instrument name used to create the tracer and meter.
_INSTRUMENTATION_SCOPE = 'megatron.core'


class TelemetryHandle:
    """Holds an OTel tracer and meter for the current process.

    On non-export ranks (or when telemetry is disabled) :attr:`tracer` and
    :attr:`meter` are no-op objects — calling any of their methods is a
    zero-overhead no-op.

    Obtain an instance via :func:`setup_telemetry`; do **not** construct this
    class directly.
    """

    def __init__(
        self, tracer: trace.Tracer, meter: metrics.Meter, is_exporting: bool = False
    ) -> None:
        self._tracer = tracer
        self._meter = meter
        #: ``True`` only on the rank that runs the OTLP exporter.  Other ranks
        #: hold no-op providers — checking this flag before computing metrics
        #: avoids unnecessary work across the entire job.
        self.is_exporting = is_exporting

    @property
    def tracer(self) -> trace.Tracer:
        """The OTel :class:`~opentelemetry.trace.Tracer` for this process."""
        return self._tracer

    @property
    def meter(self) -> metrics.Meter:
        """The OTel :class:`~opentelemetry.metrics.Meter` for this process."""
        return self._meter

    def shutdown(self, timeout_ms: int = 5000) -> None:
        """Flush pending spans/metrics and shut down the providers.

        Args:
            timeout_ms: Maximum time (milliseconds) to wait for the flush
                before returning.  Defaults to 5 000 ms.
        """
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, 'force_flush'):
            tracer_provider.force_flush(timeout_millis=timeout_ms)
        if hasattr(tracer_provider, 'shutdown'):
            tracer_provider.shutdown()

        meter_provider = metrics.get_meter_provider()
        if hasattr(meter_provider, 'force_flush'):
            meter_provider.force_flush(timeout_millis=timeout_ms)
        if hasattr(meter_provider, 'shutdown'):
            meter_provider.shutdown()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def setup_telemetry(
    config: 'TelemetryConfig',
    rank: int,
    world_size: int,
) -> TelemetryHandle:
    """Initialise OTel providers and return a :class:`TelemetryHandle`.

    This is the single entry point for telemetry initialisation.  Call it
    once per process (typically from
    :func:`megatron.training.global_vars.set_global_variables`).

    Logic:
    - If ``config.enabled`` is ``False``: registers no-op providers and
      returns a handle with no-op tracer/meter (zero overhead, no SDK import).
    - If ``config.enabled`` is ``True`` and this is the export rank: imports
      the OTel SDK, builds real providers with OTLP or console exporters.
    - If ``config.enabled`` is ``True`` and this is a non-export rank:
      registers no-op providers (spans are created in-process but not sent).

    Args:
        config: Megatron-specific telemetry configuration.
        rank: This process's global rank (``args.rank``).
        world_size: Total number of processes (``args.world_size``).

    Returns:
        A :class:`TelemetryHandle` with ``.tracer`` and ``.meter`` ready to
        use on any rank.
    """
    from megatron.core.telemetry._providers import build_noop_providers, build_providers
    from megatron.core.telemetry._state import set_enabled_span_groups

    # Determine which rank runs the exporter.
    resolved_export_rank = config.export_rank if config.export_rank >= 0 else (world_size - 1)
    is_export_rank = rank == resolved_export_rank

    if not config.enabled:
        build_noop_providers()
        # No groups enabled — all is_span_group_enabled() calls return False.
        set_enabled_span_groups(frozenset())
        _is_exporting = False
    elif is_export_rank:
        build_providers(config, rank, world_size)
        set_enabled_span_groups(config.resolved_span_groups)
        _is_exporting = True
    else:
        # Non-export rank: no-op providers; spans and metrics are discarded.
        # Use frozenset() so is_span_group_enabled() returns False — no span
        # objects are created at all, giving true zero overhead on non-export ranks.
        build_noop_providers()
        set_enabled_span_groups(frozenset())
        _is_exporting = False

    tracer = trace.get_tracer(_INSTRUMENTATION_SCOPE)
    meter = metrics.get_meter(_INSTRUMENTATION_SCOPE)
    return TelemetryHandle(tracer=tracer, meter=meter, is_exporting=_is_exporting)
