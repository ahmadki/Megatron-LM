# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""OTel metric instruments and recording helpers for Megatron training.

Instruments are created lazily (on first call to :func:`record_training_metrics`
or :func:`record_inference_metrics`) using the meter obtained from the
:class:`~megatron.core.telemetry.handle.TelemetryHandle`.

Training metrics use the ``megatron.training.*`` namespace (no OTel standard
covers training loops).  Inference metrics follow the `GenAI semantic
conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/>`_:

* ``gen_ai.server.request.duration`` (Histogram, seconds)
* ``gen_ai.client.token.usage`` (Histogram, ``{token}`` — split by
  ``gen_ai.token.type = "input" | "output"``)

All functions are safe to call when telemetry is disabled or the meter is a
no-op — they silently become no-ops.
"""

from __future__ import annotations

import logging
import weakref
from typing import Optional

from opentelemetry import metrics

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level instrument cache.  WeakKeyDictionary avoids id()-reuse bugs
# that arise when a no-op meter is garbage-collected and a new one gets the
# same memory address — a plain dict keyed by id() would serve stale entries.
# ---------------------------------------------------------------------------

_TRAINING_INSTRUMENTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_INFERENCE_INSTRUMENTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _get_training_instruments(meter: metrics.Meter) -> dict:
    """Return (creating if necessary) the training metric instruments."""
    instruments = _TRAINING_INSTRUMENTS.get(meter)
    if instruments is None:
        instruments = {
            'step_duration_ms': meter.create_histogram(
                name='megatron.training.step_duration_ms',
                unit='ms',
                description='Duration of one training step in milliseconds.',
            ),
            # Gauge (point-in-time): loss/throughput/grad_norm are last-value
            # readings, not distributions — Gauge maps to a Prometheus gauge
            # rather than a histogram, which is semantically correct.
            'loss': meter.create_gauge(
                name='megatron.training.loss',
                description='Training loss value at each log interval.',
            ),
            'throughput_tps': meter.create_gauge(
                name='megatron.training.throughput_tflops',
                description='Training throughput in TFLOP/s/GPU.',
            ),
            'grad_norm': meter.create_gauge(
                name='megatron.training.grad_norm',
                description='Global gradient norm.',
            ),
            'skipped_iters': meter.create_counter(
                name='megatron.training.skipped_iters',
                description='Number of training iterations skipped (e.g. NaN loss).',
            ),
        }
        _TRAINING_INSTRUMENTS[meter] = instruments
    return instruments


def _get_inference_instruments(meter: metrics.Meter) -> dict:
    """Return (creating if necessary) the inference metric instruments.

    Follows GenAI semantic conventions:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/
    """
    instruments = _INFERENCE_INSTRUMENTS.get(meter)
    if instruments is None:
        instruments = {
            # https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/#metric-gen_aiserverrequest_duration
            'server_request_duration': meter.create_histogram(
                name='gen_ai.server.request.duration',
                unit='s',
                description='GenAI server request duration (time to last byte / last output token).',
            ),
            # https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/#metric-gen_aiclienttokenusage
            'token_usage': meter.create_histogram(
                name='gen_ai.client.token.usage',
                unit='{token}',
                description='Number of input and output tokens used.',
            ),
        }
        _INFERENCE_INSTRUMENTS[meter] = instruments
    return instruments


# ---------------------------------------------------------------------------
# Public recording helpers
# ---------------------------------------------------------------------------

# Required metric attributes per GenAI semconv (gen_ai.operation.name,
# gen_ai.provider.name are always set; gen_ai.request.model is added when known).
_PROVIDER_NAME = 'megatron'
_OPERATION_NAME = 'text_completion'


def record_training_metrics(
    meter: metrics.Meter,
    step_duration_ms: Optional[float] = None,
    loss: Optional[float] = None,
    throughput_tps: Optional[float] = None,
    grad_norm: Optional[float] = None,
    skipped_iters: Optional[int] = None,
) -> None:
    """Record training metrics to the OTel meter.

    Called from :func:`megatron.training.training.training_log` at each
    ``--log-interval`` cadence.  All arguments are optional; ``None`` values
    are silently skipped.

    Args:
        meter: The OTel :class:`~opentelemetry.metrics.Meter` instance.
        step_duration_ms: Mean step duration over the log interval (ms).
        loss: Average training loss over the log interval.
        throughput_tps: Throughput in tokens/second.
        grad_norm: Global gradient norm.
        skipped_iters: Number of skipped iterations since last log.
    """
    try:
        instruments = _get_training_instruments(meter)
    except Exception:
        _logger.warning("Failed to create training metric instruments", exc_info=True)
        return

    if step_duration_ms is not None:
        instruments['step_duration_ms'].record(step_duration_ms)
    if loss is not None:
        instruments['loss'].set(loss)
    if throughput_tps is not None:
        instruments['throughput_tps'].set(throughput_tps)
    if grad_norm is not None:
        instruments['grad_norm'].set(float(grad_norm))
    if skipped_iters is not None and skipped_iters > 0:
        instruments['skipped_iters'].add(skipped_iters)


def record_inference_metrics(
    meter: metrics.Meter,
    request_duration_s: Optional[float] = None,
    model: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> None:
    """Record inference metrics to the OTel meter (GenAI semconv).

    Emits:

    * ``gen_ai.server.request.duration`` — total request wall-clock time (seconds).
    * ``gen_ai.client.token.usage`` — per-request token counts, split by
      ``gen_ai.token.type`` (``"input"`` / ``"output"``).

    Metric attributes follow the GenAI semantic conventions:
    ``gen_ai.operation.name``, ``gen_ai.provider.name``, ``gen_ai.request.model``.

    Args:
        meter: The OTel :class:`~opentelemetry.metrics.Meter` instance.
        request_duration_s: Total request duration in **seconds**.
        model: Model identifier (``gen_ai.request.model``).
        input_tokens: Number of input (prompt) tokens processed.
        output_tokens: Number of output (generated) tokens produced.
    """
    try:
        instruments = _get_inference_instruments(meter)
    except Exception:
        _logger.warning("Failed to create inference metric instruments", exc_info=True)
        return

    # Required attributes per GenAI semconv.
    base_attrs: dict = {
        'gen_ai.operation.name': _OPERATION_NAME,
        'gen_ai.provider.name': _PROVIDER_NAME,
    }
    if model:
        base_attrs['gen_ai.request.model'] = str(model)

    if request_duration_s is not None:
        instruments['server_request_duration'].record(request_duration_s, attributes=base_attrs)
    if input_tokens is not None:
        instruments['token_usage'].record(
            input_tokens,
            attributes={**base_attrs, 'gen_ai.token.type': 'input'},
        )
    if output_tokens is not None:
        instruments['token_usage'].record(
            output_tokens,
            attributes={**base_attrs, 'gen_ai.token.type': 'output'},
        )
