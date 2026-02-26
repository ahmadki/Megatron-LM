# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""W3C TraceContext / Baggage propagation helpers."""

from opentelemetry import context, propagate


def inject_context(carrier: dict) -> None:
    """Inject the current span context into *carrier* (W3C TraceContext + Baggage).

    Use this to propagate trace context across process boundaries, e.g. into
    gRPC metadata dicts or HTTP header dicts.

    Args:
        carrier: A mutable dict that will receive the ``traceparent`` (and
            optionally ``tracestate``, ``baggage``) headers.

    Example::

        headers = {}
        inject_context(headers)
        # headers == {"traceparent": "00-<trace_id>-<span_id>-01", ...}
    """
    propagate.inject(carrier)


def extract_context(carrier: dict) -> context.Context:
    """Extract span context from *carrier* and return an OTel :class:`Context`.

    Use this to resume a distributed trace from an incoming HTTP request's
    headers or gRPC metadata.

    Args:
        carrier: A dict (or dict-like mapping) containing W3C ``traceparent``
            (and optionally ``tracestate``, ``baggage``) headers.

    Returns:
        An OTel :class:`~opentelemetry.context.Context`.  If no valid trace
        context is present the returned context is empty (non-remote).

    Example::

        ctx = extract_context(request.headers)
        with tracer.start_as_current_span("megatron.inference.request",
                                          context=ctx):
            ...
    """
    return propagate.extract(carrier)
