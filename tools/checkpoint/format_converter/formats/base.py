# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Format-neutral types shared by the format adapters.

* :class:`Section`: the role of a key in a store (model, optimizer, common, ...).
* :class:`FormatAdapter`: what the conversion needs to know about a format.
* :class:`RawCommon`: the common (non-tensor) state, carried around undecoded:
  its payloads plus the layout that says how the common dict is stored in them
  (one wrapped or bare dict, or one payload per key). Decoding and encoding the
  payloads is the job of the common-state codec (``mcore_state.common_state``),
  which decides which classes may be unpickled; the layout rules live here, and
  each adapter names the output byte item of each payload.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from ..errors import IntegrityError, InternalError

if TYPE_CHECKING:
    import torch

    from ..core.logical import OutputSlice
    from ..engine.metadata import MetadataView, TensorEntry
    from ..engine.reader import SourceReader
    from ..engine.regions import Region


class Section(enum.Enum):
    """Role of a key in a store."""

    MODEL = "model"
    OPTIMIZER_STATE = "optimizer.state"
    OPTIMIZER_OTHER = "optimizer other"
    SCHEDULER = "scheduler"
    COMMON = "common"
    RNG = "rng"
    RERUN = "rerun"
    EXTRA_STATE = "extra_state"
    OTHER = "other"


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


class FormatAdapter(Protocol):
    """What a conversion needs to know about one checkpoint format and dialect.

    Detection stays in :mod:`.detect`, which chooses the adapter.
    """

    fmt: ClassVar[str]
    dialect: ClassVar[str]
    common_layout: ClassVar[CommonLayout]

    def classify(self, key: str, *, is_bytes: bool) -> Section:
        """The section of a key of this format (``OTHER`` if unknown)."""

    def sections(self, view: MetadataView) -> dict[str, Section]:
        """The section of every key of a store, refusing keys the format does not know."""

    def stored_key(self, relative: str) -> str:
        """The store key of a model-relative key (see ``core/keys.py``)."""

    def relative_key(self, key: str) -> str:
        """The model-relative key of a store's model key."""

    def planner_path(self, key: str) -> tuple[str, ...] | None:
        """``Metadata.planner_data`` entry of an output model key (``None``: not recorded)."""

    def output_grid(
        self, placed: OutputSlice, dtype: torch.dtype, *, max_chunk_bytes: int
    ) -> tuple[Region, ...]:
        """The chunks the box ``placed.region`` of a converted model tensor is written as."""

    def read_common_raw(self, reader: SourceReader) -> RawCommon | None:
        """The undecoded common state of ``reader``'s store, if it has one."""

    def common_item(self, key: str) -> tuple[str, tuple[str, ...] | None]:
        """The output byte item that stores common-state payload ``key``: its key and planner path.

        ``key`` is a payload key of ``common_layout`` (see :func:`payload_keys`).
        """

    def marker_files(self) -> dict[str, bytes]:
        """Files the format requires next to ``.metadata`` in a published store."""

    def validate_published(self, iter_dir: Path) -> None:
        """Check a written store has exactly this format's markers."""


def model_tensors(
    view: MetadataView, sections: Mapping[str, Section], adapter: FormatAdapter
) -> dict[str, TensorEntry]:
    """A store's model tensors by model-relative key (sorted).

    Raises:
        RefusedError: The adapter refuses a model key's envelope.
    """
    return {
        adapter.relative_key(key): view.tensors[key]
        for key in sorted(view.tensors)
        if sections[key] is Section.MODEL
    }
