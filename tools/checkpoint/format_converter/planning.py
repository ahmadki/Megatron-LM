# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Planning a conversion from metadata and the common state: sections, units and budgets.

This version plans the weights-only rewrite of a torch_dist store into
torch_dist: model tensors and extra state are copied chunk for chunk (so the
output has the source's chunk grid), optimizer, RNG and rerun state are
dropped, and the common state is decoded, filtered, its args sanitized and
re-encoded as the in-DCP ``common_state`` item (a legacy ``common.pt`` is
migrated). Every refusal, and the host-memory preflight, happens here, before
any tensor is read.

A dry run (``plan``) reads nothing but the metadata and the common state: byte
items are sized from the metadata, and a common state that only Megatron's
codec can decode (FC-E01 on a host without Megatron) is decoded with stubs, so
the plan is still complete; ``convert`` needs the real codec.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from . import __version__, errors
from .engine.memory import (
    MemoryEstimate,
    check_host_memory,
    driver_bytes,
    estimate_memory,
    estimate_output_bytes,
)
from .engine.pieces import BytePiece, Piece, Unit, copy_pieces
from .engine.reader import SourceReader
from .engine.runner import WriteProgram, choose_checked
from .engine.scheduler import dispatch_order, pack_units
from .engine.staging import PublishTarget
from .errors import IntegrityError, RefusedError, UsageError
from .formats.base import FormatAdapter, RawCommon, Section, payload_keys
from .formats.detect import Detection, Iteration, Store, detect_format, open_store
from .core.names import TORCH_DIST
from .formats.sections import take_census
from .formats.torch_dist import (
    DROPPED_SECTIONS,
    KEPT_SECTIONS,
    TorchDistAdapter,
    weights_only_common,
    weights_only_sections,
)
from .mcore_state.args_sanitize import sanitize_weights_only
from .mcore_state.common_state import CodecMode, auto_mode, decode, encode
from .options import PlanOptions
from .sidecars import SidecarPlan, plan_sidecars

KEEP, DROP, REWRITE = "keep", "drop", "rewrite"


@dataclass(frozen=True, slots=True)
class SectionAction:
    """What a conversion does with one section of the source.

    Attributes:
        section: The section.
        action: ``keep`` (copied), ``drop`` or ``rewrite`` (decoded and re-encoded).
        keys: Keys of the section in the source.
        tensor_bytes: Decoded size of its tensors.
    """

    section: Section
    action: str
    keys: int
    tensor_bytes: int

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {
            "section": self.section.value,
            "action": self.action,
            "keys": self.keys,
            "tensor_bytes": self.tensor_bytes,
        }


@dataclass(frozen=True)
class CommonPlan:
    """What happens to the common state.

    Attributes:
        source: Where it was read from.
        codec: The codec mode it was decoded and encoded with (``stub`` in a
            dry run on a host that cannot decode it for writing: nothing was
            encoded and ``convert`` needs the Megatron container).
        kept: Keys kept.
        dropped: Keys dropped.
        args_changes: ``{field: (old, new)}`` of the sanitized args.
        args_consulted: The args fields the conversion read, with their values.
        iteration: The stored ``iteration``, if it is an integer.
    """

    source: str
    codec: str
    kept: tuple[str, ...]
    dropped: tuple[str, ...]
    args_changes: dict[str, tuple[Any, Any]]
    args_consulted: dict[str, Any]
    iteration: int | None

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {
            "source": self.source,
            "codec": self.codec,
            "kept": list(self.kept),
            "dropped": list(self.dropped),
            "args_changes": {name: list(change) for name, change in self.args_changes.items()},
            "args_consulted": self.args_consulted,
            "iteration": self.iteration,
        }


@dataclass(frozen=True)
class ConversionPlan:
    """A planned conversion (metadata only; nothing has been read or written yet).

    Attributes:
        store: The source store.
        detection: Its detected format.
        adapter: The target format's adapter.
        options: The planning options.
        iteration: The source iteration (from the directory name, else the
            common state), if known.
        sections: The action per source section.
        common: The common-state plan.
        sidecars: The sidecar plan.
        units: The packed units.
        memory: The host-memory estimate (already checked against the budget).
        output_bytes: Estimated size of the output store.
    """

    store: Store
    detection: Detection
    adapter: FormatAdapter
    options: PlanOptions
    iteration: Iteration
    sections: tuple[SectionAction, ...]
    common: CommonPlan
    sidecars: SidecarPlan
    units: tuple[Unit, ...]
    memory: MemoryEstimate
    output_bytes: int

    def fingerprint(self, target: PublishTarget) -> str:
        """Identity of this conversion into ``target`` (staging is resumed only on a match).

        Covers the source store's content (its ``.metadata`` digest and the size
        and mtime of every file), the target, the options that shape the output,
        the tool version and the torch minor version.
        """
        material = {
            "source": self.store.view.fingerprint(),
            "source_dir": str(self.store.iter_dir),
            "target": str(target.iter_dir),
            "tracker": None if target.tracker is None else str(target.tracker),
            "format": [self.adapter.fmt, self.adapter.dialect],
            "options": self.options.output_fields(),
            "tool": __version__,
            "torch": ".".join(torch.__version__.split(".")[:2]),
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()

    def program(self, target: PublishTarget) -> WriteProgram:
        """The write program that produces the planned store at ``target``.

        Raises:
            InternalError: This is a dry-run plan (byte items were not read).
        """
        fingerprint = self.fingerprint(target)
        return WriteProgram(
            source=self.store.view,
            units=self.units,
            dispatch_order=dispatch_order(self.units),
            checked=choose_checked(self.units, seed=fingerprint),
            writer_threads=self.options.writer_threads,
            max_item_bytes=self.options.max_unit_bytes_hard,
            fingerprint=fingerprint,
            info=self.to_json(),
        )

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable summary (the ``plan`` report)."""
        pieces = sum(len(unit.pieces) for unit in self.units)
        return {
            "source": {
                "iter_dir": str(self.detection.iter_dir),
                "format": self.detection.fmt,
                "dialect": self.detection.dialect,
                "iteration": self.iteration,
            },
            "target": {"format": self.adapter.fmt, "dialect": self.adapter.dialect},
            "sections": [action.to_json() for action in self.sections],
            "common": self.common.to_json(),
            "sidecars": self.sidecars.to_json(),
            "units": {
                "count": len(self.units),
                "pieces": pieces,
                "byte_items": sum(len(unit.byte_pieces) for unit in self.units),
                "largest_bytes": max((u.est_peak_bytes for u in self.units), default=0),
                "unit_bytes": self.options.unit_bytes,
                "hard_cap_bytes": self.options.max_unit_bytes_hard,
            },
            "memory": {**self.memory.to_json(), "budget": self.options.max_host_bytes},
            "output_bytes": self.output_bytes,
        }


def plan_weights_only(src: Path, opts: PlanOptions, *, dry_run: bool) -> ConversionPlan:
    """Plan the weights-only torch_dist rewrite of ``src``.

    Args:
        src: A store directory or a checkpoint root.
        opts: Target format and budgets.
        dry_run: Plan for display only: byte items are sized from the
            metadata instead of read, and a common state that needs Megatron's
            codec is decoded with stubs instead of refused. The plan cannot be
            executed.

    Raises:
        UsageError: The source is not a torch_dist store.
        RefusedError: A refusal (FC-R13, FC-R14, FC-R17, FC-R18; FC-E01 unless
            ``dry_run``).
        IntegrityError: The store is unreadable or has no common state.
        BudgetExceeded: A unit or the whole conversion exceeds its memory budget.
    """
    store = open_store(src, iteration=opts.iteration)
    detection = detect_format(store, from_format=opts.from_format)
    if detection.fmt != TORCH_DIST:
        raise UsageError(
            f"{store.iter_dir} is a {detection.fmt} store; this version converts torch_dist "
            "stores to torch_dist only",
            hint="conversion between torch_dist and fsdp_dtensor lands in a later version",
        )
    adapter = TorchDistAdapter()
    view = store.view
    sections = weights_only_sections(view)
    sidecars = plan_sidecars(view)
    reader = SourceReader(view, max_item_bytes=opts.max_unit_bytes_hard)
    common, common_pieces = _plan_common(adapter, reader, opts, dry_run=dry_run)
    extra_state = sorted(k for k in view.byte_items if sections[k] is Section.EXTRA_STATE)
    if dry_run:
        extra_pieces = [BytePiece.sized(key, view.byte_items[key].length) for key in extra_state]
    else:
        payloads = reader.read_bytes(extra_state)
        extra_pieces = [BytePiece(key, payloads[key]) for key in extra_state]
    byte_pieces = (*common_pieces, *extra_pieces)
    pieces = _kept_pieces(store, sections)
    units = pack_units(
        pieces, byte_pieces, unit_bytes=opts.unit_bytes, hard_cap=opts.max_unit_bytes_hard
    )
    memory = estimate_memory(units, workers=opts.workers, driver=driver_bytes(len(pieces)))
    check_host_memory(memory, opts.max_host_bytes)
    return ConversionPlan(
        store=store,
        detection=detection,
        adapter=adapter,
        options=opts,
        iteration=_iteration(detection, common),
        sections=_section_actions(store),
        common=common,
        sidecars=sidecars,
        units=units,
        memory=memory,
        output_bytes=estimate_output_bytes(units),
    )


def _plan_common(
    adapter: FormatAdapter, reader: SourceReader, opts: PlanOptions, *, dry_run: bool
) -> tuple[CommonPlan, tuple[BytePiece, ...]]:
    raw = adapter.read_common_raw(reader)
    if raw is None:
        raise IntegrityError(
            f"{reader.view.iter_dir} has no common state (common_state/shard_0_1 or common.pt)"
        )
    mode = auto_mode()
    try:
        decoded = decode(raw, mode)
    except RefusedError as err:
        if not dry_run or err.refusal.code != errors.FC_E01:
            raise
        mode = CodecMode.STUB  # plan still reports everything; convert will refuse
        decoded = decode(raw, mode)
    rewrite = weights_only_common(decoded.state, drop_keys=opts.drop_common_keys)
    common = dict(rewrite.kept)
    changes: dict[str, tuple[Any, Any]] = {}
    consulted: dict[str, Any] = {}
    if "args" in common:
        sanitized = sanitize_weights_only(common["args"])
        common["args"], changes, consulted = sanitized.args, sanitized.changes, sanitized.consulted
    items = {key: adapter.common_item(key) for key in payload_keys(common, adapter.common_layout)}
    if mode is CodecMode.STUB:
        pieces = tuple(
            BytePiece.sized(item, _estimated_size(raw, key), path)
            for key, (item, path) in items.items()
        )
    else:
        payloads = encode(common, adapter.common_layout, mode=mode, label="the output common state")
        pieces = tuple(BytePiece(item, payloads[key], path) for key, (item, path) in items.items())
    iteration = common.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        iteration = None
    plan = CommonPlan(
        raw.source, mode.value, tuple(common), rewrite.dropped, changes, consulted, iteration
    )
    return plan, pieces


def _estimated_size(raw: RawCommon, key: str) -> int:
    """The size of an output payload that cannot be encoded here: its source payload's size."""
    source = raw.payloads.get(key)
    return raw.nbytes if source is None else len(source)


def _kept_pieces(store: Store, sections: dict[str, Section]) -> list[Piece]:
    tensors = store.view.tensors
    return [
        piece
        for key in sorted(tensors)
        if sections[key] in KEPT_SECTIONS
        for piece in copy_pieces(tensors[key])
    ]


def _section_actions(store: Store) -> tuple[SectionAction, ...]:
    census = take_census(store.view, TORCH_DIST)
    actions = []
    for section, totals in census.sections.items():
        if section in KEPT_SECTIONS:
            action = KEEP
        elif section in DROPPED_SECTIONS:
            action = DROP
        else:
            action = REWRITE
        actions.append(SectionAction(section, action, totals.keys, totals.tensor_bytes))
    return tuple(actions)


def _iteration(detection: Detection, common: CommonPlan) -> Iteration:
    """The directory's iteration (``iter_N`` or ``release``), else the stored one."""
    return common.iteration if detection.iteration is None else detection.iteration
