# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TelemetryConfig and SpanGroup: Megatron-specific OTel configuration."""

import os
from dataclasses import dataclass
from typing import ClassVar, Final


class SpanGroup:
    """Named constants for span granularity groups.

    Groups are organised from coarsest (included in the ``"default"`` preset)
    to finest (only in ``"full"`` / ``"all"``).  Use :meth:`resolve` to
    convert a human-readable spec string to a :class:`frozenset` of group
    names.

    Preset keywords
    ---------------
    ``"default"``
        Coarse-grained spans only (lowest overhead): job boundaries,
        checkpoint saves, and evaluation runs.
    ``"per_step"``
        All ``"default"`` groups plus per-step boundaries: model
        initialisation, checkpoint loading, training step, forward+backward
        pass, and optimizer step.
    ``"all"``
        Every span group, including per-microbatch forward and backward spans.
        Highest overhead — use with a sampling strategy in production.

    Individual group names
    ----------------------
    Any of the string constants below may be listed individually or mixed with
    preset keywords::

        MEGATRON_OTEL_SPAN_GROUPS="default,microbatch"
    """

    # ------------------------------------------------------------------ #
    # Coarse-grained (included in "default")
    # ------------------------------------------------------------------ #

    JOB = "job"
    """Outermost job spans: ``megatron.pretrain``, ``megatron.train``."""

    CHECKPOINT = "checkpoint"
    """Checkpoint save span: ``megatron.save_checkpoint``."""

    EVALUATE = "evaluate"
    """Evaluation run span: ``megatron.evaluate``."""

    # ------------------------------------------------------------------ #
    # Medium-grained (included in "per_step")
    # ------------------------------------------------------------------ #

    MODEL_INIT = "model_init"
    """Model and optimizer construction span: ``megatron.model_init``."""

    LOAD_CHECKPOINT = "load_checkpoint"
    """Checkpoint restore span: ``megatron.load_checkpoint``."""

    STEP = "step"
    """Per-training-step span: ``megatron.train_step``."""

    FORWARD_BACKWARD = "forward_backward"
    """Forward+backward pass span per step: ``megatron.forward_backward``."""

    OPTIMIZER = "optimizer"
    """Optimizer step span: ``megatron.optimizer_step``."""

    # ------------------------------------------------------------------ #
    # Fine-grained (included in "full" / "all" only)
    # ------------------------------------------------------------------ #

    MICROBATCH = "microbatch"
    """Per-microbatch spans (highest overhead):
    ``megatron.microbatch.forward``, ``megatron.microbatch.backward``."""

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    INFERENCE = "inference"
    """Inference server request spans: ``text_completion {model}``."""

    # ------------------------------------------------------------------ #
    # Internal: set of all group names and preset lookup tables
    # ------------------------------------------------------------------ #

    ALL_GROUPS: Final[frozenset] = frozenset(
        [
            JOB,
            CHECKPOINT,
            EVALUATE,
            MODEL_INIT,
            LOAD_CHECKPOINT,
            STEP,
            FORWARD_BACKWARD,
            OPTIMIZER,
            MICROBATCH,
            INFERENCE,
        ]
    )

    _PRESETS: ClassVar[dict] = {
        "default": frozenset([JOB, CHECKPOINT, EVALUATE, INFERENCE]),
        "per_step": frozenset(
            [
                JOB,
                CHECKPOINT,
                EVALUATE,
                MODEL_INIT,
                LOAD_CHECKPOINT,
                STEP,
                FORWARD_BACKWARD,
                OPTIMIZER,
                INFERENCE,
            ]
        ),
        "all": ALL_GROUPS,
    }

    @classmethod
    def resolve(cls, spec: str) -> frozenset:
        """Resolve a span-group spec string to a :class:`frozenset` of group names.

        The *spec* may be:

        * A preset keyword: ``"default"``, ``"per_step"``, ``"all"``.
        * A comma-separated list of individual group names, e.g.
          ``"job,checkpoint,step"``.
        * A mix of presets and individual names, e.g.
          ``"default,microbatch"``.

        Args:
            spec: Spec string (case-insensitive).

        Returns:
            A :class:`frozenset` of group-name strings.

        Raises:
            ValueError: If an unknown keyword or group name is encountered.
        """
        result: set = set()
        for part in (p.strip().lower() for p in spec.split(',') if p.strip()):
            if part in cls._PRESETS:
                result |= cls._PRESETS[part]
            elif part in cls.ALL_GROUPS:
                result.add(part)
            else:
                valid = sorted(cls.ALL_GROUPS | set(cls._PRESETS))
                raise ValueError(
                    f"Unknown span group or preset: {part!r}. "
                    f"Valid options: {valid}"
                )
        return frozenset(result)


@dataclass
class TelemetryConfig:
    """Configuration for Megatron-LM OpenTelemetry instrumentation.

    Megatron-specific settings only.  All standard OTLP configuration
    (``OTEL_EXPORTER_OTLP_ENDPOINT``, ``OTEL_EXPORTER_OTLP_HEADERS``,
    ``OTEL_TRACES_SAMPLER``, ``OTEL_BSP_*``, ``OTEL_SDK_DISABLED``, …) is
    handled automatically by the OpenTelemetry SDK — no need to replicate them
    here.  See ``megatron/core/telemetry/README.md`` for the full reference.
    """

    # ------------------------------------------------------------------
    # Master toggle
    # ------------------------------------------------------------------

    #: Must be explicitly set to True to prevent accidental activation.
    #: Env var: ``MEGATRON_OTEL_ENABLED=1``
    enabled: bool = False

    # ------------------------------------------------------------------
    # Service identity
    # ------------------------------------------------------------------

    #: Human-readable service name sent to the OTLP backend.
    #: Env var: ``OTEL_SERVICE_NAME`` (standard OTel SDK variable).
    #: ``service.version`` is auto-populated from
    #: ``megatron.core.package_info.__version__``.
    service_name: str = 'megatron-training'

    # ------------------------------------------------------------------
    # Distributed export
    # ------------------------------------------------------------------

    #: Which rank runs the OTLP exporter.
    #: ``-1`` means the last rank (matches wandb/tensorboard convention).
    #: ``0`` means rank 0.
    #: Env var: ``MEGATRON_OTEL_EXPORT_RANK``
    export_rank: int = -1

    # ------------------------------------------------------------------
    # Feature toggles
    # ------------------------------------------------------------------

    #: Enable trace spans.
    #: Env var: ``MEGATRON_OTEL_TRACES_ENABLED``
    traces_enabled: bool = True

    #: Enable metrics instruments.
    #: Env var: ``MEGATRON_OTEL_METRICS_ENABLED``
    metrics_enabled: bool = True

    # ------------------------------------------------------------------
    # Span granularity
    # ------------------------------------------------------------------

    #: Comma-separated span-group spec controlling which instrumentation
    #: boundaries are active.  Accepts preset keywords
    #: (``"default"``, ``"per_step"``, ``"full"``, ``"all"``) or individual
    #: group names from :class:`SpanGroup`, or a mix.
    #:
    #: Examples::
    #:
    #:   MEGATRON_OTEL_SPAN_GROUPS=default          # coarse only (default)
    #:   MEGATRON_OTEL_SPAN_GROUPS=per_step         # include step-level spans
    #:   MEGATRON_OTEL_SPAN_GROUPS=all               # all spans incl. microbatch
    #:   MEGATRON_OTEL_SPAN_GROUPS=default,microbatch  # mix presets + groups
    #:
    #: Env var: ``MEGATRON_OTEL_SPAN_GROUPS``
    span_groups: str = 'default'

    # ------------------------------------------------------------------
    # Exporter type
    # ------------------------------------------------------------------

    #: Exporter backend.  ``"otlp"`` uses the OTLP gRPC/HTTP exporter
    #: (endpoint from ``OTEL_EXPORTER_OTLP_ENDPOINT``).  ``"console"``
    #: prints spans/metrics to stdout — useful for local development.
    #: Env var: ``MEGATRON_OTEL_EXPORTER``
    exporter: str = 'otlp'

    # ------------------------------------------------------------------
    # Derived property
    # ------------------------------------------------------------------

    @property
    def resolved_span_groups(self) -> frozenset:
        """Resolve :attr:`span_groups` string to a :class:`frozenset` of group names."""
        return SpanGroup.resolve(self.span_groups)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> 'TelemetryConfig':
        """Build a :class:`TelemetryConfig` from environment variables.

        Standard OTel SDK env vars are *not* read here — the SDK picks them
        up automatically when the provider is initialised.
        """

        def _bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, '').strip().lower()
            if not val:
                return default
            if val in ('1', 'true', 'yes', 'on'):
                return True
            if val in ('0', 'false', 'no', 'off'):
                return False
            raise ValueError(
                f"Invalid boolean value for {key!r}: {val!r}. "
                "Expected '1'/'0', 'true'/'false', 'yes'/'no', or 'on'/'off'."
            )

        def _int(key: str, default: int) -> int:
            val = os.environ.get(key, '').strip()
            if not val:
                return default
            try:
                return int(val)
            except ValueError:
                raise ValueError(f"Invalid integer value for {key!r}: {val!r}.")

        def _str(key: str, default: str) -> str:
            val = os.environ.get(key, '').strip()
            return val if val else default

        # OTEL_SERVICE_NAME is a standard SDK var; we honour it here so that
        # the dataclass reflects what will actually be used by the provider.
        service_name = _str('OTEL_SERVICE_NAME', cls.__dataclass_fields__['service_name'].default)

        return cls(
            enabled=_bool('MEGATRON_OTEL_ENABLED', False),
            service_name=service_name,
            export_rank=_int('MEGATRON_OTEL_EXPORT_RANK', -1),
            traces_enabled=_bool('MEGATRON_OTEL_TRACES_ENABLED', True),
            metrics_enabled=_bool('MEGATRON_OTEL_METRICS_ENABLED', True),
            span_groups=_str('MEGATRON_OTEL_SPAN_GROUPS', 'default'),
            exporter=_str('MEGATRON_OTEL_EXPORTER', 'otlp'),
        )
