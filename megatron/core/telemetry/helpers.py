# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Telemetry helper utilities: span_cm, managed_span, trace_fn, safe_set_span_attributes."""

import functools
from contextlib import contextmanager
from typing import Any, Optional

from opentelemetry import trace

# ---------------------------------------------------------------------------
# Attribute key redaction
# ---------------------------------------------------------------------------

#: Attribute keys that contain potentially sensitive data and are redacted by
#: default.  Users can extend this set by passing ``redact_keys`` to
#: :func:`safe_set_span_attributes`.
DEFAULT_REDACT_KEYS: frozenset = frozenset(
    {'prompt', 'input_text', 'output_text', 'text', 'password', 'token', 'secret', 'key'}
)

# OTel only accepts these scalar types as attribute values.
_SCALAR_TYPES = (bool, int, float, str)


def redact_value(key: str, value: str, redact_keys: frozenset = DEFAULT_REDACT_KEYS) -> str:
    """Return ``'[REDACTED]'`` if *key* is in *redact_keys*, else *value*.

    Args:
        key: The attribute key.
        value: The attribute value (as a string).
        redact_keys: Set of key names whose values should be redacted.

    Returns:
        The original *value* or ``'[REDACTED]'``.
    """
    return '[REDACTED]' if key in redact_keys else value


def safe_set_span_attributes(
    span: trace.Span,
    attributes: dict,
    redact_keys: frozenset = DEFAULT_REDACT_KEYS,
) -> None:
    """Set span attributes, silently skipping non-scalar values.

    OTel span attributes must be scalars (bool, int, float, str) or sequences
    of scalars.  This helper silently drops any values that are not one of
    those types, and redacts string values whose keys are in *redact_keys*.

    Args:
        span: The OTel span to annotate.
        attributes: Dict of ``{key: value}`` pairs to set.
        redact_keys: Keys whose string values should be replaced with
            ``'[REDACTED]'``.
    """
    if not span.is_recording():
        return
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, _SCALAR_TYPES):
            if isinstance(value, str):
                value = redact_value(key, value, redact_keys)
            span.set_attribute(key, value)
        elif isinstance(value, (list, tuple)) and all(isinstance(v, _SCALAR_TYPES) for v in value):
            span.set_attribute(key, list(value))
        # else: silently skip non-scalar / mixed-type sequences


@contextmanager
def span_cm(
    name: str,
    tracer: Optional[trace.Tracer] = None,
    record_exception: bool = True,
    **attributes: Any,
):
    """Context manager that creates an OTel span for a code block.

    Safe to use when *tracer* is a no-op tracer — the code block always
    executes normally.  Span attributes are set via :func:`safe_set_span_attributes`.

    Args:
        name: Span name (e.g. ``"megatron.train_step"``).
        tracer: OTel tracer.  Defaults to the global tracer if ``None``.
        record_exception: If ``True`` (default), record exceptions as span
            events and re-raise.
        **attributes: Key/value pairs set as span attributes before the
            block executes.

    Yields:
        The active :class:`opentelemetry.trace.Span`.

    Example::

        with span_cm("megatron.evaluate", tracer=telemetry.tracer,
                     eval_iters=args.eval_iters) as span:
            result = evaluate(...)
            span.set_attribute("megatron.loss", result)
    """
    if tracer is None:
        tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span(name, record_exception=record_exception) as span:
        if attributes:
            safe_set_span_attributes(span, attributes)
        yield span


@contextmanager
def managed_span(
    group: str,
    name: str,
    tracer: Optional[trace.Tracer] = None,
    **attributes: Any,
):
    """Explicit-lifecycle span guarded by a span-group check.

    Unlike :func:`span_cm`, this helper checks whether *group* is enabled
    before creating a span.  When the group is disabled the body executes
    normally and ``None`` is yielded — no span object is created.  When
    enabled, the span is started, its context attached, the body executed,
    and the span always ended in a ``finally`` block — ensuring spans close
    even when exceptions occur.  Exceptions are recorded on the span and its
    status set to ERROR before being re-raised.

    Args:
        group: Span group name (e.g. ``SpanGroup.STEP``).  If the group is
            not in the active set this is a zero-overhead no-op.
        name: Span name (e.g. ``"megatron.train_step"``).
        tracer: OTel tracer.  Defaults to the global tracer if ``None``.
        **attributes: Key/value pairs set as span attributes.

    Yields:
        The active :class:`opentelemetry.trace.Span`, or ``None`` when the
        group is disabled.

    Example::

        with managed_span(SpanGroup.STEP, "megatron.train_step",
                          tracer=telemetry.tracer,
                          **{"megatron.iteration": iteration}) as span:
            loss = train_step(...)
            if span is not None:
                span.set_attribute("megatron.loss", loss)
    """
    from megatron.core.telemetry._state import is_span_group_enabled

    if not is_span_group_enabled(group):
        yield None
        return

    from opentelemetry import context as otel_ctx
    from opentelemetry.trace import StatusCode, set_span_in_context

    if tracer is None:
        tracer = trace.get_tracer(__name__)

    span = tracer.start_span(name)
    if attributes:
        safe_set_span_attributes(span, attributes)
    token = otel_ctx.attach(set_span_in_context(span))
    try:
        yield span
    except Exception as exc:
        span.record_exception(exc)
        span.set_status(StatusCode.ERROR, str(exc))
        raise
    finally:
        otel_ctx.detach(token)
        span.end()


def trace_fn(group: str, name: str, tracer: Optional[trace.Tracer] = None):
    """Decorator that wraps an entire function in a group-gated OTel span.

    The span group is checked at **call time** (not decoration time), so this
    decorator is safe to apply at module load even before telemetry is
    initialized.

    Exceptions (including :class:`SystemExit`) are automatically recorded on
    the span and its status set to ``ERROR`` by the underlying
    :meth:`~opentelemetry.trace.Tracer.start_as_current_span` context manager.

    Args:
        group: Span group name checked via :func:`is_span_group_enabled`.
            If the group is not enabled the function runs with no OTel
            overhead at all.
        name: OTel span name (e.g. ``"megatron.train"``).
        tracer: OTel tracer.  Defaults to the global tracer if ``None``.

    Returns:
        A decorator that wraps the target function.

    Example::

        from megatron.core.telemetry import trace_fn, SpanGroup
        from opentelemetry import trace

        @trace_fn(SpanGroup.JOB, "megatron.train")
        def train(forward_step_func, ...):
            args = get_args()
            # Set dynamic attributes on the active span
            trace.get_current_span().set_attribute(
                "megatron.train_iters", args.train_iters
            )
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from megatron.core.telemetry._state import is_span_group_enabled

            if not is_span_group_enabled(group):
                return func(*args, **kwargs)
            t = tracer if tracer is not None else trace.get_tracer('megatron.core')
            with t.start_as_current_span(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
