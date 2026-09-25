# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Decoding and encoding a checkpoint's common state (training args, iteration, ...).

The payloads are ``torch.save`` pickles. Three codec modes decide what they may
contain:

* ``megatron``: Megatron's own safe globals are registered and payloads are
  loaded with ``torch.load(weights_only=True)``, exactly as Megatron's loader
  reads them. Needs Megatron (the training container).
* ``plain``: ``torch.load(weights_only=True)`` allowing only
  ``argparse.Namespace`` besides the types torch always allows (containers,
  numbers, strings, torch dtypes). Works on any host for payloads without
  Megatron classes; a payload with them is refused with FC-E01.
* ``stub``: every unknown class is replaced by an inert stub. For display and
  comparison only: stub-decoded state can never be encoded (FC-E01).

:func:`auto_mode` picks ``megatron`` when Megatron can be imported, else ``plain``.
Encoding happens in the driver and self-checks that the payload reloads with
``weights_only=True`` in the same mode, so workers never unpickle anything.
"""

from __future__ import annotations

import argparse
import enum
import functools
import importlib
import io
import pickle
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .. import errors
from ..engine.metadata import decode_payload
from ..errors import IntegrityError, Refusal
from ..formats.base import CommonLayout, RawCommon, unwrap_common, wrap_common

MEGATRON_MIN_PYTHON = (3, 12)
"""Megatron requires Python 3.12 (``requires-python``); older interpreters never import it."""

_PLAIN_GLOBALS = (argparse.Namespace,)
_UNSUPPORTED_GLOBAL = re.compile(r"Unsupported global: GLOBAL (\S+)")


class CodecMode(enum.Enum):
    """What classes a common-state payload may contain."""

    MEGATRON = "megatron"
    PLAIN = "plain"
    STUB = "stub"


@dataclass(frozen=True)
class DecodedCommon:
    """A decoded common state.

    Attributes:
        state: The common dict.
        mode: The codec mode it was decoded with.
        source: Where it was read from.
    """

    state: dict[str, Any]
    mode: CodecMode
    source: str


@functools.lru_cache(maxsize=1)
def megatron_available() -> bool:
    """Whether Megatron's safe globals can be imported (and are now registered)."""
    if sys.version_info < MEGATRON_MIN_PYTHON:
        return False
    try:
        safe_globals = importlib.import_module("megatron.core.safe_globals")
    except Exception:  # pylint: disable=broad-exception-caught  # any failure: not usable
        return False
    safe_globals.register_safe_globals()
    return True


def auto_mode() -> CodecMode:
    """``megatron`` if Megatron can be imported in this process, else ``plain``."""
    return CodecMode.MEGATRON if megatron_available() else CodecMode.PLAIN


def decode(raw: RawCommon, mode: CodecMode) -> DecodedCommon:
    """Decode a raw common state.

    Raises:
        RefusedError: ``plain`` mode and the payload needs other classes (FC-E01).
        IntegrityError: The payload cannot be decoded or has the wrong shape.
    """
    state = unwrap_common(raw, functools.partial(loads, mode=mode, label=raw.source))
    return DecodedCommon(state, mode, raw.source)


def encode(
    common: Mapping[str, Any], layout: CommonLayout, *, mode: CodecMode, label: str
) -> dict[str, bytes]:
    """``torch.save`` ``common`` in ``layout``, checking it reloads with ``weights_only=True``.

    Returns:
        The payloads by payload key (see :func:`~..formats.base.wrap_common`):
        ``{WHOLE: ...}`` for the single-payload layouts, one per key for
        ``PER_KEY``.

    Raises:
        RefusedError: ``mode`` is ``stub`` (FC-E01).
        IntegrityError: The payloads do not reload in ``mode`` or reload
            with other keys.
    """
    if mode is CodecMode.STUB:
        raise Refusal(
            errors.FC_E01,
            f"{label} was decoded with stubs for display and cannot be written",
            hint="run inside the Megatron container",
        ).to_error()
    payloads = {}
    for key, obj in wrap_common(common, layout).items():
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        payloads[key] = buffer.getvalue()
    reloaded = unwrap_common(
        RawCommon(label, layout, payloads), functools.partial(loads, mode=mode, label=label)
    )
    if sorted(reloaded) != sorted(common):
        raise IntegrityError(f"{label} reloads with keys {sorted(reloaded)}, not {sorted(common)}")
    return payloads


def loads(payload: bytes, *, mode: CodecMode, label: str) -> Any:
    """Unpickle one ``torch.save`` payload in ``mode``.

    Raises:
        RefusedError: ``plain`` mode and the payload needs other classes (FC-E01).
        IntegrityError: The payload cannot be decoded.
    """
    if mode is CodecMode.STUB:
        return decode_payload(payload, label=label).value
    if mode is CodecMode.MEGATRON and not megatron_available():
        raise Refusal(
            errors.FC_E01,
            f"decoding {label} needs Megatron, which cannot be imported here",
            hint="run inside the Megatron container",
        ).to_error()
    try:
        with torch.serialization.safe_globals(list(_PLAIN_GLOBALS)):
            return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as err:
        if mode is CodecMode.PLAIN:
            raise _plain_failure(payload, label, err) from err
        raise IntegrityError(
            f"cannot decode {label} with Megatron's safe globals: {_rejected_global(err)}"
        ) from err
    except Exception as err:  # corrupt bytes can fail in any way
        raise IntegrityError(f"cannot decode {label}: {type(err).__name__}: {err}") from err


def _plain_failure(payload: bytes, label: str, err: Exception) -> errors.ConverterError:
    """Why the plain codec could not load ``payload``: foreign classes (FC-E01) or corruption."""
    try:
        stubbed = sorted(decode_payload(payload, label=label).stubbed)
    except IntegrityError as corrupt:
        return corrupt
    found = ", ".join(stubbed) if stubbed else _rejected_global(err)
    return Refusal(
        errors.FC_E01,
        f"{label} references classes only the Megatron codec can load ({found})",
        hint="run inside the Megatron container",
    ).to_error()


def _rejected_global(err: Exception) -> str:
    """The global ``torch.load(weights_only=True)`` refused, as named in its message."""
    match = _UNSUPPORTED_GLOBAL.search(str(err))
    return match.group(1) if match else type(err).__name__
