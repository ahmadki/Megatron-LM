# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Format-neutral descriptions shared by the format adapters.

The common (non-tensor) state of a store is carried around undecoded as a
:class:`RawCommon`: its payloads plus the layout that says how the common dict
is stored in them. Decoding and encoding the payloads is the job of the
common-state codec (``mcore_state.common_state``), which decides which classes
may be unpickled; the layout rules live here.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..errors import IntegrityError, InternalError


class CommonLayout(enum.Enum):
    """How the common dict is stored in payloads."""

    WRAPPED = "wrapped"
    """One payload, ``torch.save([common])``: the in-DCP ``common_state`` item of torch_dist."""

    DICT = "dict"
    """One payload, ``torch.save(common)``: the legacy torch_dist ``common.pt`` file."""

    PER_KEY = "per_key"
    """One payload per key, ``torch.save(common[key])``: the byte items of fsdp_dtensor."""


WHOLE = "*"
"""The payload key of the single-payload layouts (``WRAPPED`` and ``DICT``)."""


@dataclass(frozen=True)
class RawCommon:
    """An undecoded common state.

    Attributes:
        source: Where it was read from (for messages and reports).
        layout: How the common dict is stored in the payloads.
        payloads: The ``torch.save`` bytes by payload key: ``{WHOLE: ...}`` for
            the single-payload layouts, ``{key: ...}`` for ``PER_KEY``.

    Raises:
        InternalError: The payload keys do not fit the layout.
    """

    source: str
    layout: CommonLayout
    payloads: Mapping[str, bytes]

    def __post_init__(self) -> None:
        keys = set(self.payloads)
        single = self.layout is not CommonLayout.PER_KEY
        if (keys != {WHOLE}) if single else (not keys or WHOLE in keys):
            raise InternalError(
                f"{self.source}: payload keys {sorted(keys)} do not fit {self.layout}"
            )

    @property
    def nbytes(self) -> int:
        """Total size of the payloads."""
        return sum(len(payload) for payload in self.payloads.values())


def payload_keys(common_keys: Iterable[str], layout: CommonLayout) -> tuple[str, ...]:
    """The payload keys that store a common dict with ``common_keys`` in ``layout``."""
    return tuple(common_keys) if layout is CommonLayout.PER_KEY else (WHOLE,)


def unwrap_common(raw: RawCommon, loads: Callable[[bytes], Any]) -> dict[str, Any]:
    """The common dict of ``raw``, each payload decoded with ``loads``.

    Raises:
        IntegrityError: A decoded payload does not have ``raw.layout``'s shape.
    """
    if raw.layout is CommonLayout.PER_KEY:
        return {key: loads(payload) for key, payload in raw.payloads.items()}
    value = loads(raw.payloads[WHOLE])
    if raw.layout is CommonLayout.WRAPPED:
        if not isinstance(value, list) or len(value) != 1:
            raise IntegrityError(
                f"{raw.source} must hold a one-element list, found {type(value).__name__}"
            )
        value = value[0]
    if not isinstance(value, dict):
        raise IntegrityError(
            f"{raw.source} does not hold a common-state dict (found {type(value).__name__})"
        )
    return value


def wrap_common(common: Mapping[str, Any], layout: CommonLayout) -> dict[str, Any]:
    """The objects to ``torch.save``, by payload key, for ``common`` in ``layout``.

    The inverse of :func:`unwrap_common`.
    """
    if layout is CommonLayout.PER_KEY:
        return dict(common)
    return {WHOLE: [dict(common)] if layout is CommonLayout.WRAPPED else dict(common)}
