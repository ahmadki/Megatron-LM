# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Planning a conversion from metadata and the common state: sections, units and budgets.

Two kinds of conversion are planned here, both weights only:

* **torch_dist to torch_dist** (a rewrite): model tensors and extra state are
  copied chunk for chunk (so the output has the source's chunk grid),
  optimizer, RNG and rerun state are dropped, and the common state is decoded,
  filtered, its args sanitized and re-encoded as the in-DCP ``common_state``
  item (a legacy ``common.pt`` is migrated).
* **torch_dist to fsdp_dtensor and back** (cross-format): the refusal fence
  (``core/refusals.py``) runs on the arguments and the model keys, the
  architecture context is derived (``core/context.py``), the rules plan the
  model section (``core/planner.py``), extra, optimizer, scheduler, RNG and
  rerun state are dropped, and the common state is rewritten for the target
  format with its args sanitized (and, converting back, what the source's
  conversion manifest recorded restored). The plan carries the manifest the
  converted store gets.

Every refusal, and the host-memory preflight, happens here, before any tensor
is read. A dry run (``plan``) reads nothing but the metadata and the common
state: byte items are sized from the metadata, and a common state that only
Megatron's codec can decode (FC-E01 on a host without Megatron) is decoded with
stubs, so the plan is still complete; ``convert`` needs the real codec.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from . import rules  # pylint: disable=unused-import  # importing it registers the rules
from . import __version__, errors
from .core.args_view import ArgsView
from .core.context import ArchContext, build_arch_context
from .core.manifest import MANIFEST_FILE, Manifest
from .core.names import AUTO_EXPERT_NAMING, AUTO_SWIGLU, FSDP_DTENSOR, PER_LAYER_LAYOUT, TORCH_DIST
from .core.planner import plan_model
from .core.refusals import Facts, detect_refusals
from .core.registry import REGISTRY
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
from .errors import IntegrityError, Refusal, RefusedError, UsageError, refusal_error
from .formats.base import FormatAdapter, RawCommon, Section, model_tensors, payload_keys
from .formats.detect import (
    FSDP_COMMON_ITEMS,
    Detection,
    Iteration,
    Store,
    detect_format,
    open_store,
)
from .formats.fsdp_mfsdp_v1 import COMMON_ITEMS as FSDP_COMMON_KEYS
from .formats.fsdp_mfsdp_v1 import FsdpMfsdpV1Adapter
from .formats.sections import take_census
from .formats.torch_dist import (
    KEPT_SECTIONS,
    TorchDistAdapter,
    weights_only_common,
    weights_only_sections,
)
from .mcore_state.args_sanitize import (
    SanitizedArgs,
    sanitize_for_fsdp_dtensor,
    sanitize_for_torch_dist,
    sanitize_weights_only,
)
from .mcore_state.common_state import CodecMode, DecodedCommon, auto_mode, decode, encode
from .options import PlanOptions
from .sidecars import SidecarPlan, plan_sidecars

KEEP, DROP, REWRITE, CONVERT = "keep", "drop", "rewrite", "convert"
CONTENT_METADATA = "content_metadata"

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SectionAction:
    """What a conversion does with one section of the source.

    Attributes:
        section: The section.
        action: ``keep`` (copied as stored), ``convert`` (through the rules),
            ``drop`` or ``rewrite`` (decoded and re-encoded).
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
        context: The architecture context (cross-format conversions).
        claims: Source model tensors claimed per rule (cross-format conversions).
        manifest: The manifest the converted store gets (cross-format
            conversions), without the output digest.
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
    context: ArchContext | None = None
    claims: dict[str, int] = field(default_factory=dict)
    manifest: Manifest | None = None

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
        summary = {
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
        if self.context is not None:
            summary["derivations"] = self.context.derivation_log()
            summary["claims"] = dict(self.claims)
        return summary


def build_plan(src: Path, opts: PlanOptions, *, dry_run: bool) -> ConversionPlan:
    """Plan the conversion of ``src`` into ``opts.to_format``.

    Args:
        src: A store directory or a checkpoint root.
        opts: Target format and budgets.
        dry_run: Plan for display only: byte items are sized from the
            metadata instead of read, and a common state that needs Megatron's
            codec is decoded with stubs instead of refused. The plan cannot be
            executed.

    Raises:
        UsageError: The conversion is not supported (fsdp_dtensor to
            fsdp_dtensor, or ``--optimizer convert`` or ``--expert-naming`` in
            a rewrite), or ``--block-layout`` or ``--torch-dist-layout
            per-layer`` is given without an fsdp_dtensor source and a
            torch_dist target.
        RefusedError: A refusal (FC-R, FC-P; FC-E01 unless ``dry_run``),
            including layer lists the layouts cannot explain (FC-R11, FC-R12).
        AmbiguousError: The source holds another store's manifest, or one
            that records another layout of a block (FC-A06), a block's
            torch_dist layout cannot be decided (FC-A02), or an fsdp_dtensor
            source names its experts otherwise than its arguments build them
            (FC-A04).
        IntegrityError: The store is unreadable or has no common state.
        BudgetExceeded: A unit or the whole conversion exceeds its memory budget.
    """
    store = open_store(src, iteration=opts.iteration)
    detection = detect_format(store, from_format=opts.from_format)
    if (detection.fmt, opts.to_format) != (FSDP_DTENSOR, TORCH_DIST):
        _refuse_torch_dist_layouts(opts)
    if (detection.fmt, opts.to_format) != (TORCH_DIST, FSDP_DTENSOR):
        _refuse_fsdp_target_flags(opts)
    if detection.fmt == opts.to_format == TORCH_DIST:
        if opts.optimizer != "drop":
            raise UsageError(
                "a torch_dist rewrite keeps model weights only", hint="drop --optimizer convert"
            )
        if opts.expert_naming != AUTO_EXPERT_NAMING:
            raise UsageError(
                "--expert-naming chooses fsdp_dtensor expert names; a torch_dist rewrite keeps "
                "the source's keys",
                hint="drop --expert-naming",
            )
        return _plan_rewrite(store, detection, opts, dry_run=dry_run)
    if detection.fmt == opts.to_format:
        raise UsageError(
            f"{store.iter_dir} is already a {detection.fmt} store; only torch_dist stores can "
            "be rewritten in their own format",
            hint="convert to torch_dist",
        )
    return _plan_cross_format(store, detection, opts, dry_run=dry_run)


def _refuse_fsdp_target_flags(opts: PlanOptions) -> None:
    """The flags that describe the job loading an fsdp_dtensor target need such a target."""
    flags = [
        flag
        for flag, given in (
            ("--fsdp-tp", opts.fsdp_tp is not None),
            ("--fsdp-etp", opts.fsdp_etp is not None),
            ("--target-swiglu", opts.target_swiglu != AUTO_SWIGLU),
        )
        if given
    ]
    if flags:
        raise UsageError(
            f"only torch_dist to fsdp_dtensor conversions take {' and '.join(flags)} (they "
            "describe the job that loads the fsdp_dtensor target)",
            hint="drop them: an fsdp_dtensor source is read as its own arguments describe it",
        )


def _refuse_torch_dist_layouts(opts: PlanOptions) -> None:
    """The layout flags only apply to torch_dist targets of fsdp_dtensor sources."""
    flags = ["--block-layout"] if opts.block_layouts else []
    if opts.torch_dist_layout == PER_LAYER_LAYOUT:
        flags.append("--torch-dist-layout per-layer")
    if flags:
        raise UsageError(
            f"only fsdp_dtensor to torch_dist conversions take {' and '.join(flags)} (they "
            "decide how torch_dist stores the blocks)",
            hint="drop them: a torch_dist source's layout is kept",
        )


def _plan_rewrite(
    store: Store, detection: Detection, opts: PlanOptions, *, dry_run: bool
) -> ConversionPlan:
    """Plan the weights-only torch_dist rewrite of ``store`` (see the module docstring)."""
    adapter = TorchDistAdapter()
    view = store.view
    sections = weights_only_sections(view)
    sidecars = plan_sidecars(view, writes_manifest=False)
    reader = SourceReader(view, max_item_bytes=opts.max_unit_bytes_hard)
    raw, decoded = _decode_common(adapter, reader, dry_run=dry_run)
    rewrite = weights_only_common(decoded.state, drop_keys=opts.drop_common_keys)
    common, sanitized = _sanitize(rewrite.kept, sanitize_weights_only)
    common_plan, common_pieces = _encode_common(
        common, raw, decoded, adapter, rewrite.dropped, sanitized
    )
    extra_state = sorted(k for k in view.byte_items if sections[k] is Section.EXTRA_STATE)
    if dry_run:
        extra_pieces = [BytePiece.sized(key, view.byte_items[key].length) for key in extra_state]
    else:
        payloads = reader.read_bytes(extra_state)
        extra_pieces = [BytePiece(key, payloads[key]) for key in extra_state]
    pieces = [
        piece
        for key in sorted(view.tensors)
        if sections[key] in KEPT_SECTIONS
        for piece in copy_pieces(view.tensors[key])
    ]
    actions = _section_actions(store, TORCH_DIST, KEPT_SECTIONS, frozenset())
    return _finish(
        store,
        detection,
        adapter,
        opts,
        pieces,
        (*common_pieces, *extra_pieces),
        common_plan,
        sidecars,
        actions,
    )


def _plan_cross_format(
    store: Store, detection: Detection, opts: PlanOptions, *, dry_run: bool
) -> ConversionPlan:
    """Plan a weights-only conversion of ``store`` into the other format."""
    source, target = _adapter(detection.fmt), _adapter(opts.to_format)
    view = store.view
    sections = source.sections(view)
    reader = SourceReader(view, max_item_bytes=opts.max_unit_bytes_hard)
    raw, decoded = _decode_common(source, reader, dry_run=dry_run)
    if "args" not in decoded.state:
        raise IntegrityError(f"the common state of {store.iter_dir} has no args")
    args = ArgsView(decoded.state["args"])
    model = model_tensors(view, sections, source)
    refusals = detect_refusals(
        Facts(
            source=source.fmt,
            target=target.fmt,
            args=args,
            model=model,
            optimizer=opts.optimizer,
            allowed_roots=opts.allow_roots,
            allow_missing_fp8_extra_state=opts.allow_missing_fp8_extra_state,
            content_metadata=decoded.state.get(CONTENT_METADATA),
            expert_naming=opts.expert_naming,
        )
    )
    if refusals:
        raise refusal_error(refusals)
    ctx = build_arch_context(
        source=source.fmt,
        target=target.fmt,
        args=args,
        model=model,
        optimizer_keys=[k for k, s in sections.items() if s is Section.OPTIMIZER_STATE],
        manifest=None if opts.ignore_manifest else _source_manifest(store, source.fmt),
        metadata_sha256=view.metadata_sha256,
        manifest_ignored=opts.ignore_manifest,
        block_layouts=dict(opts.block_layouts),
        drop_content_metadata=CONTENT_METADATA in opts.drop_common_keys,
        per_layer=opts.torch_dist_layout == PER_LAYER_LAYOUT,
        expert_naming=opts.expert_naming,
        fsdp_tp=opts.fsdp_tp,
        fsdp_etp=opts.fsdp_etp,
        target_swiglu=opts.target_swiglu,
    )
    _warn_lost_fp8_state(ctx, sections)
    for warning in ctx.warnings:
        _LOGGER.warning("%s", warning)
    model_plan = plan_model(
        ctx,
        model,
        target,
        rules=REGISTRY.rules_for(ctx),
        max_chunk_bytes=opts.max_output_chunk_bytes,
    )
    common, dropped_common, content_metadata = _cross_format_common(decoded.state, ctx, opts)
    common, sanitized = _sanitize(common, _cross_format_sanitizer(ctx))
    common_plan, common_pieces = _encode_common(
        common, raw, decoded, target, dropped_common, sanitized, consulted=args.consumed
    )
    plan = _finish(
        store,
        detection,
        target,
        opts,
        list(model_plan.pieces),
        common_pieces,
        common_plan,
        plan_sidecars(view, writes_manifest=True),
        _section_actions(store, source.fmt, frozenset(), frozenset({Section.MODEL})),
    )
    manifest = _manifest(plan, ctx, content_metadata)
    return dataclasses.replace(plan, context=ctx, claims=model_plan.claims, manifest=manifest)


def _manifest(plan: ConversionPlan, ctx: ArchContext, content_metadata: dict[str, Any]) -> Manifest:
    """The manifest the converted store of ``plan`` gets (without its output digest)."""
    return Manifest(
        tool_version=__version__,
        torch_version=torch.__version__,
        source={
            "iter_dir": str(plan.store.iter_dir.absolute()),
            "format": plan.detection.fmt,
            "dialect": plan.detection.dialect,
            "iteration": plan.iteration,
            "fingerprint": plan.store.view.fingerprint(),
        },
        target={"format": plan.adapter.fmt, "dialect": plan.adapter.dialect},
        options=plan.options.output_fields(),
        derivations=tuple(ctx.derivation_log()),
        args_sanitized=plan.common.args_changes,
        content_metadata=content_metadata,
        dropped=tuple(a.to_json() for a in plan.sections if a.action == DROP),
    )


def _adapter(fmt: str) -> FormatAdapter:
    return TorchDistAdapter() if fmt == TORCH_DIST else FsdpMfsdpV1Adapter()


def _source_manifest(store: Store, fmt: str) -> Manifest | None:
    """The conversion manifest an fsdp_dtensor source holds, if any (torch_dist ones are unused).

    Raises:
        IntegrityError: The manifest is unreadable or malformed.
    """
    path = store.iter_dir / MANIFEST_FILE
    if fmt != FSDP_DTENSOR or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as err:
        raise IntegrityError(
            f"cannot read {path}: {err}", hint="pass --ignore-manifest to convert without it"
        ) from err
    return Manifest.from_json(data, label=str(path))


def _warn_lost_fp8_state(ctx: ArchContext, sections: Mapping[str, Section]) -> None:
    """A torch_dist source's FP8 delayed-scaling history does not survive in fsdp_dtensor."""
    if not any(section is Section.EXTRA_STATE for section in sections.values()):
        return
    fp8 = ctx.args.get("fp8", None)
    if fp8 and ctx.args.get("fp8_recipe", "delayed") == "delayed":
        _LOGGER.warning(
            "args.fp8=%r uses delayed scaling: its amax history (_extra_state) is dropped and "
            "restarts when training resumes",
            fp8,
        )


def _cross_format_common(
    common: Mapping[str, Any], ctx: ArchContext, opts: PlanOptions
) -> tuple[dict[str, Any], tuple[str, ...], dict[str, Any]]:
    """The target's common dict, the dropped keys and the ``content_metadata`` record.

    torch_dist to fsdp_dtensor keeps what a weights-only rewrite keeps and fsdp
    stores (``content_metadata`` goes to the manifest), and every item the
    fsdp_dtensor loader requires must be there; fsdp_dtensor to torch_dist
    keeps its items and gets the context's ``content_metadata`` (restored
    from the source's manifest). ``--drop-common-key`` applies in both
    directions.

    Raises:
        UsageError: ``--drop-common-key`` names a key the loader requires.
        RefusedError: The source's common state lacks an item the
            fsdp_dtensor loader requires (FC-R13).
    """
    rewrite = weights_only_common(common, drop_keys=opts.drop_common_keys)
    if ctx.target[0] == FSDP_DTENSOR:
        kept = {key: value for key, value in rewrite.kept.items() if key in FSDP_COMMON_KEYS}
        missing = tuple(key for key in FSDP_COMMON_ITEMS if key not in kept)
        if missing:
            raise Refusal(
                errors.FC_R13,
                f"the common state has no {', '.join(missing)}, which the fsdp_dtensor loader "
                "requires",
                keys=missing,
                hint="convert a checkpoint whose common state holds "
                + ", ".join(FSDP_COMMON_ITEMS),
            ).to_error()
        dropped = tuple(sorted(set(common) - set(kept)))
        return kept, dropped, _content_metadata_record(rewrite.kept)
    kept, dropped = dict(rewrite.kept), set(rewrite.dropped)
    if ctx.td_content_metadata is not None:
        kept[CONTENT_METADATA] = dict(ctx.td_content_metadata)
    elif ctx.manifest is not None and ctx.manifest.content_metadata["present"]:
        dropped.add(CONTENT_METADATA)  # --drop-common-key content_metadata
    return kept, tuple(sorted(dropped)), {"present": False}


def _content_metadata_record(common: Mapping[str, Any]) -> dict[str, Any]:
    """The manifest's record of the ``content_metadata`` a conversion keeps, if any.

    Raises:
        RefusedError: The manifest (JSON) cannot hold its value (FC-R13).
    """
    if CONTENT_METADATA not in common:
        return {"present": False}
    value = common[CONTENT_METADATA]
    try:
        json.dumps(value)
    except (TypeError, ValueError) as err:
        raise Refusal(
            errors.FC_R13,
            f"content_metadata holds values the conversion manifest cannot record ({err})",
            keys=(CONTENT_METADATA,),
            hint=f"pass --drop-common-key {CONTENT_METADATA} to convert without it",
        ).to_error() from err
    return {"present": True, "value": value}


def _cross_format_sanitizer(ctx: ArchContext) -> Callable[[argparse.Namespace], SanitizedArgs]:
    """The args sanitation of the target format (restoring the source manifest's originals)."""
    if ctx.target[0] == FSDP_DTENSOR:
        grouped_gemm = None if ctx.experts is None else ctx.experts.grouped_gemm
        return lambda args: sanitize_for_fsdp_dtensor(
            args,
            grouped_gemm=grouped_gemm,
            pack_tp=ctx.pack_tp,
            pack_etp=ctx.pack_etp,
            glu_args=ctx.fsdp_glu_args,
        )
    restore = {} if ctx.manifest is None else ctx.manifest.original_args()
    return lambda args: sanitize_for_torch_dist(args, restore)


def _sanitize(
    common: Mapping[str, Any], sanitizer: Callable[[argparse.Namespace], SanitizedArgs]
) -> tuple[dict[str, Any], SanitizedArgs | None]:
    """``common`` with its args sanitized, and the sanitation (``None`` without args)."""
    common = dict(common)
    if "args" not in common:
        return common, None
    sanitized = sanitizer(common["args"])
    common["args"] = sanitized.args
    return common, sanitized


def _decode_common(
    adapter: FormatAdapter, reader: SourceReader, *, dry_run: bool
) -> tuple[RawCommon, DecodedCommon]:
    """Read and decode the source's common state (stub-decoded in a dry run without Megatron)."""
    raw = adapter.read_common_raw(reader)
    if raw is None:
        raise IntegrityError(f"{reader.view.iter_dir} has no common state")
    try:
        return raw, decode(raw, auto_mode())
    except RefusedError as err:
        if not dry_run or err.refusal.code != errors.FC_E01:
            raise
        return raw, decode(raw, CodecMode.STUB)  # the plan reports everything; convert refuses


def _encode_common(
    common: Mapping[str, Any],
    raw: RawCommon,
    decoded: DecodedCommon,
    target: FormatAdapter,
    dropped: tuple[str, ...],
    sanitized: SanitizedArgs | None,
    *,
    consulted: Mapping[str, Any] | None = None,
) -> tuple[CommonPlan, tuple[BytePiece, ...]]:
    """The target's common-state byte items (sized only when stub-decoded) and their plan.

    ``consulted`` adds the argument fields other decisions read to the ones
    the sanitation read.
    """
    items = {key: target.common_item(key) for key in payload_keys(common, target.common_layout)}
    if decoded.mode is CodecMode.STUB:
        pieces = tuple(
            BytePiece.sized(item, _estimated_size(raw, key), path)
            for key, (item, path) in items.items()
        )
    else:
        payloads = encode(
            common, target.common_layout, mode=decoded.mode, label="the output common state"
        )
        pieces = tuple(BytePiece(item, payloads[key], path) for key, (item, path) in items.items())
    iteration = common.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        iteration = None
    plan = CommonPlan(
        raw.source,
        decoded.mode.value,
        tuple(common),
        dropped,
        {} if sanitized is None else sanitized.changes,
        dict(
            sorted(
                {**(consulted or {}), **({} if sanitized is None else sanitized.consulted)}.items()
            )
        ),
        iteration,
    )
    return plan, pieces


def _estimated_size(raw: RawCommon, key: str) -> int:
    """The size of an output payload that cannot be encoded here: its source payload's size."""
    source = raw.payloads.get(key)
    return raw.nbytes if source is None else len(source)


def _finish(
    store: Store,
    detection: Detection,
    adapter: FormatAdapter,
    opts: PlanOptions,
    pieces: list[Piece],
    byte_pieces: tuple[BytePiece, ...],
    common: CommonPlan,
    sidecars: SidecarPlan,
    actions: tuple[SectionAction, ...],
) -> ConversionPlan:
    """Pack the pieces into units, check the memory budget and assemble the plan."""
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
        sections=actions,
        common=common,
        sidecars=sidecars,
        units=units,
        memory=memory,
        output_bytes=estimate_output_bytes(units),
    )


def _section_actions(
    store: Store, fmt: str, kept: frozenset[Section], converted: frozenset[Section]
) -> tuple[SectionAction, ...]:
    """The action per source section: kept, converted, common rewritten, the rest dropped."""
    census = take_census(store.view, fmt)
    actions = []
    for section, totals in census.sections.items():
        if section in kept:
            action = KEEP
        elif section in converted:
            action = CONVERT
        elif section is Section.COMMON:
            action = REWRITE
        else:
            action = DROP
        actions.append(SectionAction(section, action, totals.keys, totals.tensor_bytes))
    return tuple(actions)


def _iteration(detection: Detection, common: CommonPlan) -> Iteration:
    """The directory's iteration (``iter_N`` or ``release``), else the stored one."""
    return common.iteration if detection.iteration is None else detection.iteration
