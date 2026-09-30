# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The architecture context of a conversion, derived from the arguments and the model keys.

Rules decide from an :class:`ArchContext`, never from raw arguments. Every
field is derived by :func:`build_arch_context` from the checkpoint's training
arguments, cross-checked against what the store holds where the store can
tell, and recorded as a :class:`Derivation` (field, value and evidence) that
``plan`` prints and the conversion manifest keeps. A contradiction the
conversion depends on is a refusal naming the field, the argument's value and
the observed value.

The context is built only after the refusal fence (``core/refusals.py``)
passed, so it describes a supported model.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from .. import errors
from ..engine.metadata import TensorEntry
from ..errors import Refusal
from .args_view import ArgsView
from .blocks import (
    TRANSFORMER_BLOCK,
    BlockFacts,
    BlockLayout,
    classify_blocks,
    language_model_blocks,
)
from .dtype_policy import DtypePolicy
from .experts import ExpertLayout, expert_layout, grouped_bias_warnings
from .format_args import GLU_ARGS
from .glu import Fc1, GluLayout, glu_layout, swiglu_split
from .hybrid_pattern import shortcut_warnings
from .keys import root_of
from .logical import LogicalId
from .manifest import MANIFEST_FILE, Manifest
from .mixers import MixerLayout, megatron_fsdp_warnings, mixer_layout
from .names import (
    AUTO_EXPERT_NAMING,
    AUTO_SWIGLU,
    DIALECTS,
    FSDP_DTENSOR,
    FSDP_MODEL_PREFIX,
    FSDP_OPTIMIZER_STATE,
    PER_LAYER,
    SWIGLU_ON,
    TORCH_DIST,
)
from .part_layout import PartLayout
from .stacking import requested_layout_warnings, transformer_block

TD_TO_FSDP = "td_to_fsdp"
FSDP_TO_TD = "fsdp_to_td"
DIRECTIONS = {(TORCH_DIST, FSDP_DTENSOR): TD_TO_FSDP, (FSDP_DTENSOR, TORCH_DIST): FSDP_TO_TD}
"""The cross-format directions by (source, target) format."""

BLOCK_FIELD = "blocks."
"""Prefix of the derivation fields of the blocks (``blocks.<prefix>``)."""

NON_HOMOGENEOUS = "non_homogeneous_layers"
PER_LAYER_CONTENT_METADATA: Mapping[str, Any] = {
    NON_HOMOGENEOUS: True,
    "singleton_local_shards": False,
    "chained_optim_avoid_prefix": True,
}
"""The ``content_metadata`` of a per-layer torch_dist target without a recorded one: what
Megatron writes (``_build_sharded_state_dict_metadata``) plus ``non_homogeneous_layers``."""

_MODULE_LEVELS = re.compile(r"(?:module\.)+")


@dataclass(frozen=True, slots=True)
class PackKind:
    """A tensor-parallel size fsdp_dtensor packs fused tensors at.

    Attributes:
        field: The context field (and derivation) of the size.
        flag: The flag that gives it for an fsdp_dtensor target.
        arg: The training argument of the size.
        experts: Whether it packs the tensors of routed experts (else every other one).
        rank: What it packs per, for messages.
    """

    field: str
    flag: str
    arg: str
    experts: bool
    rank: str


TENSOR_PARALLEL = PackKind(
    "pack_tp", "--fsdp-tp", "tensor_model_parallel_size", False, "tensor-parallel rank"
)
EXPERT_TENSOR_PARALLEL = PackKind(
    "pack_etp", "--fsdp-etp", "expert_tensor_parallel_size", True, "expert tensor-parallel rank"
)


def pack_kind(lid: LogicalId) -> PackKind:
    """The size a fused tensor of ``lid`` is packed at: the expert size for a routed expert's."""
    return EXPERT_TENSOR_PARALLEL if lid.expert is not None else TENSOR_PARALLEL


class RankMajor(Protocol):
    """A tensor fsdp_dtensor stores packed per tensor-parallel rank (a gated ``linear_fc1`` it
    keeps fused, a mixer's fused projection).

    Attributes:
        lid: Its logical id.
        key: A stored source key that holds it (for messages).
        parts: Its parts, each sharded over the ranks.
        kind: What it is (``gated linear_fc1``), for messages.
        parts_noun: What its parts are called (``gate and value halves``), for messages.
    """

    @property
    def lid(self) -> LogicalId: ...  # pylint: disable=missing-function-docstring

    @property
    def key(self) -> str: ...  # pylint: disable=missing-function-docstring

    @property
    def parts(self) -> PartLayout: ...  # pylint: disable=missing-function-docstring

    @property
    def kind(self) -> str: ...  # pylint: disable=missing-function-docstring

    @property
    def parts_noun(self) -> str: ...  # pylint: disable=missing-function-docstring


@dataclass(frozen=True, slots=True)
class Derivation:
    """How one context field was derived.

    Attributes:
        field: The context field.
        value: Its value (JSON-serializable).
        evidence: The arguments and store facts it was derived from.
    """

    field: str
    value: Any
    evidence: str

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {"field": self.field, "value": self.value, "evidence": self.evidence}


@dataclass(frozen=True)
class ArchContext:
    """Everything the rules may know about a conversion's model and formats.

    Attributes:
        direction: ``td_to_fsdp`` or ``fsdp_to_td``.
        source: Source (format, dialect).
        target: Target (format, dialect).
        args: The checkpoint's training arguments.
        params_dtype: The model's parameter dtype (``--bf16``, ``--fp16`` or fp32).
        src_tp: Tensor-parallel size the source was saved with.
        src_etp: Expert tensor-parallel size the source was saved with.
        ep: Expert-parallel size the source was saved with.
        pp: Pipeline-parallel size the source was saved with.
        vpp: Virtual-pipeline size the source was saved with (``None`` if unused).
        td_content_metadata: The ``content_metadata`` a torch_dist target gets
            (``fsdp_to_td``; ``None`` when it gets none or the target is fsdp_dtensor).
        blocks: The model's layered blocks and how each format numbers and
            stacks their layers.
        experts: How each format stores the model's routed experts (``None``
            without mixture-of-experts layers).
        glu: The model's MLP first projections (``linear_fc1``): which are
            gated linear units, and which fsdp_dtensor stores as SwiGLU halves.
        mixers: The model's state-space mixers: the family of each, and the
            parts of its fused projections.
        pack_tp: The tensor-parallel size the fsdp_dtensor side packs fused
            tensors at (``None``: it packs none).
        pack_etp: The expert tensor-parallel size it packs the fused tensors
            of routed experts at (``None``: it packs none).
        fsdp_glu_args: The gated-MLP arguments an fsdp_dtensor target's job
            has where they differ from the source's (``--target-swiglu``).
        fsdp_model_prefix: Prefix of the fsdp_dtensor side's model keys.
        fsdp_optim_depth: ``module.`` levels before a parameter name in the
            fsdp_dtensor side's optimizer keys (one more for bf16 and fp16).
        dtype_policy: How output dtypes are chosen.
        manifest: The manifest of the conversion that produced the source, if
            it has one that describes it (used to restore what that
            conversion changed).
        derivations: How each field was derived, in derivation order.
        warnings: What the conversion does that a job built from the
            arguments may not load (``--block-layout`` against Megatron's
            layout, grouped expert biases under expert parallelism).
    """

    direction: str
    source: tuple[str, str]
    target: tuple[str, str]
    args: ArgsView
    params_dtype: torch.dtype
    src_tp: int
    src_etp: int
    ep: int
    pp: int
    vpp: int | None
    td_content_metadata: Mapping[str, Any] | None
    blocks: tuple[BlockLayout, ...]
    experts: ExpertLayout | None
    glu: GluLayout
    mixers: MixerLayout
    pack_tp: int | None
    pack_etp: int | None
    fsdp_glu_args: Mapping[str, bool]
    fsdp_model_prefix: str
    fsdp_optim_depth: str
    dtype_policy: DtypePolicy
    manifest: Manifest | None
    derivations: tuple[Derivation, ...]
    warnings: tuple[str, ...] = ()

    def derivation_log(self) -> list[dict[str, Any]]:
        """The derivations as JSON records."""
        return [derivation.to_json() for derivation in self.derivations]

    def pack_size(self, lid: LogicalId) -> int | None:
        """The size fsdp_dtensor packs a fused tensor of ``lid`` at (``None``: it packs none)."""
        return self.pack_etp if pack_kind(lid).experts else self.pack_tp


def build_arch_context(
    *,
    source: str,
    target: str,
    args: ArgsView,
    model: Mapping[str, TensorEntry],
    optimizer_keys: Iterable[str] = (),
    manifest: Manifest | None = None,
    metadata_sha256: str = "",
    manifest_ignored: bool = False,
    block_layouts: Mapping[str, str] | None = None,
    drop_content_metadata: bool = False,
    per_layer: bool = False,
    expert_naming: str = AUTO_EXPERT_NAMING,
    fsdp_tp: int | None = None,
    fsdp_etp: int | None = None,
    target_swiglu: str = AUTO_SWIGLU,
) -> ArchContext:
    """Derive the context of converting a ``source`` store into ``target``.

    Args:
        source: Source format.
        target: Target format.
        args: The checkpoint's training arguments.
        model: The source's model tensors by model-relative key.
        optimizer_keys: The source's optimizer-state keys (evidence only).
        manifest: The manifest found in the source store, if any.
        metadata_sha256: SHA-256 of the source's ``.metadata`` (a manifest
            must name it to describe the source).
        manifest_ignored: The source's manifest was not read (``--ignore-manifest``).
        block_layouts: How a torch_dist target stores a block, by prefix
            (``--block-layout``).
        drop_content_metadata: A torch_dist target gets no ``content_metadata``
            (``--drop-common-key content_metadata``).
        per_layer: A torch_dist target stores every TransformerBlock per layer
            and marks it in its ``content_metadata`` (``--torch-dist-layout
            per-layer``).
        expert_naming: How fsdp_dtensor names routed experts
            (``--expert-naming``; ``auto``: as the model's builder does).
        fsdp_tp: The tensor-parallel size an fsdp_dtensor target packs fused
            tensors at (``--fsdp-tp``; ``None``: the source's, if it is 1).
        fsdp_etp: The expert tensor-parallel size it packs routed experts'
            fused tensors at (``--fsdp-etp``; likewise).
        target_swiglu: Whether the job that loads an fsdp_dtensor target sets
            ``--swiglu`` (``--target-swiglu``: ``auto``, ``on`` or ``off``).

    Raises:
        UsageError: ``source`` to ``target`` is not a cross-format direction,
            ``block_layouts`` names a block that is not a TransformerBlock, or
            ``target_swiglu`` differs from the source's ``args.swiglu`` for a
            language model with MLPs (a job with another activation).
        RefusedError: A derivation is contradictory (FC-R13 for both ``bf16``
            and ``fp16``, or routed experts that disagree with
            ``args.num_experts``; FC-R11 for layer indices the blocks do not
            explain; FC-R12 for inconsistently stacked layers, layers that
            cannot be stacked as asked, expert indices other than
            ``0..E-1``, or gated linear units that contradict one another or
            the arguments; FC-R10 for packed tensors that do not divide by
            their pack size, or pack sizes the manifest records otherwise).
        AmbiguousError: The source holds a manifest of another store, or one
            that records another torch_dist layout of a block (FC-A06); a
            block's torch_dist layout cannot be decided (FC-A02); an
            fsdp_dtensor source names its experts otherwise than its
            arguments build them (FC-A04); an fsdp_dtensor target packs
            tensors per tensor-parallel rank, the source was saved at a
            tensor-parallel size above 1 and no pack size is given (FC-A01).
        IntegrityError: The manifest's ``content_metadata`` is not a mapping.
    """
    direction = DIRECTIONS.get((source, target))
    if direction is None:
        raise errors.UsageError(f"{source} to {target} is not a cross-format conversion")
    if manifest_ignored:
        manifest = None
    manifest_log = _check_manifest(manifest, source, metadata_sha256, ignored=manifest_ignored)
    log: list[Derivation] = [
        Derivation("direction", direction, f"source {source}, target {target}")
    ]
    params_dtype = _params_dtype(args, model, log)
    tp = _parallel_size(args, "tensor_model_parallel_size", 1, log, field="src_tp")
    etp = _parallel_size(args, "expert_tensor_parallel_size", tp, log, field="src_etp")
    ep = _parallel_size(args, "expert_model_parallel_size", 1, log, field="ep")
    pp = _parallel_size(args, "pipeline_model_parallel_size", 1, log, field="pp")
    vpp = args.get("virtual_pipeline_model_parallel_size", None)
    log.append(Derivation("vpp", vpp, f"args.virtual_pipeline_model_parallel_size={vpp}"))
    content_metadata = _td_content_metadata(
        direction, manifest, log, dropped=drop_content_metadata, per_layer=per_layer
    )
    requested = dict(block_layouts or {})
    facts = BlockFacts(args, source, model, content_metadata, requested, per_layer)
    log.append(_language_model(facts))
    blocks = classify_blocks(facts, default=transformer_block)
    log.extend(
        Derivation(f"{BLOCK_FIELD}{block.prefix}", _block_summary(block), block.evidence)
        for block in blocks
    )
    if direction == FSDP_TO_TD:
        _check_recorded_layouts(blocks, manifest, requested, per_layer=per_layer)
    experts, evidence = expert_layout(args, model, source, expert_naming)
    log.append(Derivation("experts", None if experts is None else experts.to_json(), evidence))
    language_model = language_model_blocks(blocks, facts.lm_decoder)
    glu, evidence = glu_layout(args, model, blocks, source, language_model)
    log.append(Derivation("gated_fc1", glu.gated_json(), evidence))
    fsdp_swiglu, fsdp_glu_args = _fsdp_swiglu(
        args, direction, target_swiglu, glu.language_model(language_model), log
    )
    glu, evidence = swiglu_split(glu, fsdp_swiglu, blocks, source, experts)
    log.append(Derivation("swiglu_split", glu.split_json(), evidence))
    mixers, evidence = mixer_layout(args, model, blocks, source)
    log.append(Derivation("mixers", mixers.families_json(), evidence))
    warnings = (
        requested_layout_warnings(facts, blocks)
        + grouped_bias_warnings(experts, model, source)
        + shortcut_warnings(args, target)
    )
    rank_major = [fc1 for fc1 in glu.fc1s.values() if fc1.packed]
    rank_major += list(mixers.projections.values())
    packs = _Packs(
        direction, rank_major, manifest, log, warnings, args=args, src_tp=tp, fsdp_tp=fsdp_tp
    )
    pack_tp = packs.size(TENSOR_PARALLEL, saved=tp, flag=fsdp_tp)
    pack_etp = packs.size(EXPERT_TENSOR_PARALLEL, saved=etp, flag=fsdp_etp)
    warnings += megatron_fsdp_warnings(mixers, target, pack_tp)
    log.append(
        Derivation("fsdp_model_prefix", FSDP_MODEL_PREFIX, f"dialect {DIALECTS[FSDP_DTENSOR]}")
    )
    depth = _fsdp_optim_depth(params_dtype, source, optimizer_keys, log)
    log.append(Derivation("dtype_policy", DtypePolicy.PRESERVE.value, "weights-only conversion"))
    log.append(manifest_log)
    return ArchContext(
        direction=direction,
        source=(source, DIALECTS[source]),
        target=(target, DIALECTS[target]),
        args=args,
        params_dtype=params_dtype,
        src_tp=tp,
        src_etp=etp,
        ep=ep,
        pp=pp,
        vpp=vpp,
        td_content_metadata=content_metadata,
        blocks=blocks,
        experts=experts,
        glu=glu,
        mixers=mixers,
        pack_tp=pack_tp,
        pack_etp=pack_etp,
        fsdp_glu_args=fsdp_glu_args,
        fsdp_model_prefix=FSDP_MODEL_PREFIX,
        fsdp_optim_depth=depth,
        dtype_policy=DtypePolicy.PRESERVE,
        manifest=manifest,
        derivations=tuple(log),
        warnings=tuple(warnings),
    )


def _check_manifest(
    manifest: Manifest | None, source: str, metadata_sha256: str, *, ignored: bool
) -> Derivation:
    """A source's manifest must describe the source store itself; its derivation record."""
    if ignored:
        return Derivation("manifest", None, f"{MANIFEST_FILE} ignored (--ignore-manifest)")
    if manifest is None:
        return Derivation("manifest", None, f"the source has no {MANIFEST_FILE}")
    written = manifest.target.get("format")
    if written != source or manifest.output_metadata_sha256 != metadata_sha256:
        raise Refusal(
            errors.FC_A06,
            f"the source's {MANIFEST_FILE} describes another store (a {written} store whose "
            f".metadata digest is {manifest.output_metadata_sha256}; this {source} store's is "
            f"{metadata_sha256})",
            keys=(MANIFEST_FILE,),
            hint="pass --ignore-manifest to convert without restoring what it records",
        ).to_error()
    return Derivation(
        "manifest",
        manifest.source.get("format"),
        f"{MANIFEST_FILE} of the conversion from {manifest.source.get('iter_dir')} "
        "(restores its sanitized arguments and content_metadata)",
    )


def _language_model(facts: BlockFacts) -> Derivation:
    """Where the language model's decoder is: the model's roots say single or vision-language.

    Raises:
        RefusedError: The keys mix single-root and vision-language roots (FC-R13).
    """
    roots = sorted({root_of(key) for key in facts.model})
    return Derivation(
        "lm_decoder", facts.lm_decoder, f"model keys under {', '.join(roots) or 'no module'}"
    )


def _td_content_metadata(
    direction: str,
    manifest: Manifest | None,
    log: list[Derivation],
    *,
    dropped: bool,
    per_layer: bool,
) -> Mapping[str, Any] | None:
    """The ``content_metadata`` a torch_dist target gets: what the source's manifest records.

    ``--torch-dist-layout per-layer`` sets ``non_homogeneous_layers`` in it
    (starting from :data:`PER_LAYER_CONTENT_METADATA` when nothing is recorded).

    Raises:
        IntegrityError: The recorded value is not a mapping.
    """
    if direction != FSDP_TO_TD:
        return None
    value = None
    if manifest is None or not manifest.content_metadata["present"]:
        evidence = f"none recorded (no {MANIFEST_FILE}, or it records none)"
    elif dropped:
        evidence = "--drop-common-key content_metadata"
    else:
        value = manifest.source_content_metadata
        evidence = f"restored from the source's {MANIFEST_FILE}"
        if not isinstance(value, Mapping):
            raise errors.IntegrityError(
                f"the source's {MANIFEST_FILE} records a content_metadata that is not a mapping",
                hint="pass --ignore-manifest to convert without restoring what it records",
            )
    if per_layer:
        value = {**(PER_LAYER_CONTENT_METADATA if value is None else value), NON_HOMOGENEOUS: True}
        evidence += f"; {NON_HOMOGENEOUS} set by --torch-dist-layout per-layer"
    log.append(Derivation("td_content_metadata", value, evidence))
    return value


def _check_recorded_layouts(
    blocks: tuple[BlockLayout, ...],
    manifest: Manifest | None,
    requested: Mapping[str, str],
    *,
    per_layer: bool,
) -> None:
    """A torch_dist target gets the block layouts the source's manifest records (FC-A06).

    Blocks whose layout was asked for (``--block-layout``, or every
    TransformerBlock with ``--torch-dist-layout per-layer``) are exempt.
    """
    if manifest is None:
        return
    recorded = _recorded_layouts(manifest)
    for block in blocks:
        layout = recorded.get(block.prefix)
        asked = block.prefix in requested or (per_layer and block.kind == TRANSFORMER_BLOCK)
        if layout is None or layout == block.td_layout or asked:
            continue
        raise Refusal(
            errors.FC_A06,
            f"the source's {MANIFEST_FILE} records that torch_dist stored {block.prefix} "
            f"{layout}, but this conversion would store it {block.td_layout} ({block.evidence})",
            keys=(MANIFEST_FILE,),
            hint=_restore_hint(block, layout),
        ).to_error()


def _restore_hint(block: BlockLayout, layout: str) -> str:
    """How to convert a block whose recorded torch_dist layout is ``layout``."""
    ignore = "or --ignore-manifest to store it as the arguments build it"
    if block.kind != TRANSFORMER_BLOCK:
        return "pass --ignore-manifest to store it as the arguments build it"
    if layout == PER_LAYER:
        return (
            "keep the content_metadata the manifest records or pass --torch-dist-layout "
            "per-layer (neither with --drop-common-key content_metadata) so that Megatron loads "
            f"{block.prefix} per layer, {ignore}"
        )
    return (
        f"pass --block-layout {block.prefix}={layout} if the job that loads the store builds "
        f"{block.prefix} {layout}, {ignore}"
    )


def _recorded_layouts(manifest: Manifest) -> dict[str, str]:
    """The torch_dist layout of each block, by prefix, as the manifest's derivations record."""
    layouts = {}
    for derivation in manifest.derivations:
        name, value = derivation.get("field"), derivation.get("value")
        if isinstance(name, str) and name.startswith(BLOCK_FIELD) and isinstance(value, Mapping):
            if isinstance(value.get("td_layout"), str):
                layouts[name[len(BLOCK_FIELD) :]] = value["td_layout"]
    return layouts


def _block_summary(block: BlockLayout) -> dict[str, Any]:
    summary = block.to_json()
    del summary["prefix"], summary["evidence"]
    return summary


class _Packs:
    """Derives the sizes fsdp_dtensor packs fused tensors at, checked against the manifest, the
    part sizes and the padded vocabulary.

    Args:
        direction: The conversion's direction.
        rank_major: The tensors fsdp_dtensor packs per rank.
        manifest: The source's conversion manifest, if any.
        log: The derivations, appended to.
        warnings: The context's warnings, appended to.
        args: The checkpoint's training arguments.
        src_tp: The tensor-parallel size the source was saved with.
        fsdp_tp: ``--fsdp-tp``, if given.
    """

    def __init__(
        self,
        direction: str,
        rank_major: Sequence[RankMajor],
        manifest: Manifest | None,
        log: list[Derivation],
        warnings: list[str],
        *,
        args: ArgsView,
        src_tp: int,
        fsdp_tp: int | None,
    ) -> None:
        self._direction = direction
        self._packed = list(rank_major)
        self._recorded = {
            record.get("field"): record.get("value")
            for record in (() if manifest is None else manifest.derivations)
        }
        self._log = log
        self._warnings = warnings
        self._args = args
        self._src_tp = src_tp
        self._fsdp_tp = fsdp_tp

    def size(self, kind: PackKind, *, saved: int, flag: int | None) -> int | None:
        """The pack size of ``kind`` (``None`` when nothing is packed at it); logged.

        Args:
            kind: Which size.
            saved: The size the source was saved with.
            flag: The size given for an fsdp_dtensor target (``kind.flag``).

        Raises:
            AmbiguousError: A torch_dist source saved at a size above 1 is
                converted with packed tensors and no ``flag``, or with packed
                routed experts, no ``--fsdp-etp`` and a ``--fsdp-tp`` other
                than its tensor-parallel size (FC-A01).
            RefusedError: A packed tensor does not divide by the size, an
                fsdp_dtensor source's manifest records another size, or the
                padded vocabulary does not divide by ``--fsdp-tp`` (FC-R10).
        """
        packed = [tensor for tensor in self._packed if pack_kind(tensor.lid) is kind]
        if not packed:
            unused = "" if flag is None else f"; {kind.flag} {flag} has no effect"
            if flag is not None:
                self._warnings.append(self._unused_flag_warning(kind, flag))
            self._log.append(
                Derivation(kind.field, None, f"no fused tensor to pack per rank{unused}")
            )
            return None
        keys = tuple(sorted({tensor.key for tensor in packed}))
        if self._direction == TD_TO_FSDP:
            size, evidence = self._target_size(kind, saved, flag, _counted(packed), keys)
            _check_divisible(
                packed, size, f"{kind.flag} {size}", f"pass a {kind.flag} that divides them"
            )
            if kind is TENSOR_PARALLEL and size != saved:
                evidence += self._padded_vocab(size, saved)
        else:
            size, evidence = saved, f"args.{kind.arg}={saved} of the job that saved the source"
            self._check_recorded(kind, size, keys)
            _check_divisible(
                packed, size, f"the store's args.{kind.arg}={size}", "check the store's arguments"
            )
        self._log.append(
            Derivation(
                kind.field,
                size,
                f"{evidence}; {_counted(packed)} that fsdp_dtensor keeps fused, packed per rank",
            )
        )
        return size

    def _unused_flag_warning(self, kind: PackKind, flag: int) -> str:
        """What a pack-size flag with nothing to pack at it means for the loading job."""
        unused = (
            f"{kind.flag} {flag} has no effect: the fsdp_dtensor store packs no tensor per "
            f"{kind.rank}"
        )
        experts = [tensor for tensor in self._packed if pack_kind(tensor.lid).experts]
        if kind.experts or not experts:
            return f"{unused}, so it loads at any {kind.arg}"
        etp = EXPERT_TENSOR_PARALLEL
        return (
            f"{unused}, but it packs {_counted(experts)} of routed experts per "
            f"{etp.rank}: they load right only at the {etp.arg} the store's arguments record "
            f"({etp.field}), which the loading job must set with --expert-tensor-parallel-size "
            "(Megatron does not take it from a checkpoint)"
        )

    def _target_size(
        self, kind: PackKind, saved: int, flag: int | None, counted: str, keys: tuple[str, ...]
    ) -> tuple[int, str]:
        if flag is not None:
            return flag, f"{kind.flag} {flag}"
        if kind.experts and self._fsdp_tp is not None and self._fsdp_tp != self._src_tp:
            raise Refusal(
                errors.FC_A01,
                f"fsdp_dtensor packs {counted} of routed experts per "
                f"{kind.rank} of the job that loads it, and {TENSOR_PARALLEL.flag} "
                f"{self._fsdp_tp} is not the size the source was saved with "
                f"(args.{TENSOR_PARALLEL.arg}={self._src_tp}): a job at tensor-parallel size "
                f"{self._fsdp_tp} runs its routed experts at expert tensor-parallel size "
                f"{self._fsdp_tp} unless it sets --expert-tensor-parallel-size",
                keys=keys,
                hint=f"pass {kind.flag} N with the {kind.arg} of that job (the store's arguments "
                "then record it)",
            ).to_error()
        if saved == 1:
            return 1, f"args.{kind.arg}=1 of the source"
        raise Refusal(
            errors.FC_A01,
            f"fsdp_dtensor packs {counted} per rank of the job that loads it, and the source "
            f"was saved with args.{kind.arg}={saved}: the store loads right only at the size it "
            "is packed at",
            keys=keys,
            hint=f"pass {kind.flag} N with the {kind.arg} of that job (the store's arguments then "
            "record it)",
        ).to_error()

    def _padded_vocab(self, size: int, saved: int) -> str:
        """The store's padded vocabulary under the tensor-parallel size ``size`` its arguments get
        (the source's: ``saved``); the evidence.

        A job at that size needs it to divide by ``size`` (FC-R10 otherwise). A
        job started with ``--use-checkpoint-args`` takes ``padded_vocab_size``
        from the store; one that pads its vocabulary itself pads it to a
        multiple of ``make_vocab_size_divisible_by * size``, which is warned
        about when it differs, or may (a smaller size with ``args.vocab_size``
        unset).
        """
        padded = self._args.get("padded_vocab_size", None)
        divisible_by = self._args.get("make_vocab_size_divisible_by", None)
        if padded is None or divisible_by is None:
            return (
                "; the padded vocabulary is not checked (args.padded_vocab_size or "
                "args.make_vocab_size_divisible_by is unset)"
            )
        if padded % size:
            raise Refusal(
                errors.FC_R10,
                f"args.padded_vocab_size={padded} does not divide by {TENSOR_PARALLEL.flag} "
                f"{size}: a job at that tensor-parallel size cannot shard the embedding and the "
                "output layer",
                hint=f"pass a {TENSOR_PARALLEL.flag} that divides it",
            ).to_error()
        multiple = divisible_by * size
        padding = f"make_vocab_size_divisible_by x {size} = {multiple}"
        vocab = self._args.get("vocab_size", None)
        if vocab is not None:
            repadded = math.ceil(vocab / multiple) * multiple
            if repadded == padded:
                return (
                    f"; args.padded_vocab_size={padded} is args.vocab_size={vocab} padded to a "
                    f"multiple of {padding}, as a job at that size pads it"
                )
            differs = f"pads args.vocab_size={vocab} to {repadded} ({padding}), not to"
        elif padded % multiple:
            differs = f"pads it to a multiple of {padding}, not to"
        elif size >= saved:
            return (
                f"; args.padded_vocab_size={padded} is a multiple of {padding}, as a job at that "
                "size pads it"
            )
        else:
            differs = (
                f"pads it to a multiple of {padding}, which may be fewer rows (args.vocab_size is "
                "unset) than"
            )
        self._warnings.append(
            f"{TENSOR_PARALLEL.flag} {size}: a job at that tensor-parallel size that pads its "
            f"vocabulary itself {differs} the store's args.padded_vocab_size={padded}: load the "
            "store with --use-checkpoint-args, which takes padded_vocab_size from it"
        )
        return f"; a job that pads its vocabulary itself {differs} args.padded_vocab_size={padded}"

    def _check_recorded(self, kind: PackKind, size: int, keys: tuple[str, ...]) -> None:
        recorded = self._recorded.get(kind.field)
        if recorded is not None and recorded != size:
            raise Refusal(
                errors.FC_R10,
                f"the source's {MANIFEST_FILE} records that its fused tensors were packed at "
                f"{kind.field}={recorded}, but its args.{kind.arg}={size}",
                keys=keys,
                hint="check the store's arguments; --ignore-manifest unpacks at the arguments' "
                "size",
            ).to_error()


def _check_divisible(packed: Sequence[RankMajor], size: int, given: str, hint: str) -> None:
    """Every packed tensor's parts divide by its pack size ``size``, ``given`` as said (FC-R10)."""
    uneven = [tensor for tensor in packed if any(part % size for part in tensor.parts.sizes)]
    if uneven:
        described = " and ".join(
            f"the {tensor.parts_noun} of {count} stored {tensor.kind} tensors"
            for tensor, count in _kinds(uneven)
        )
        raise Refusal(
            errors.FC_R10,
            f"{described} do not divide by {given}",
            keys=tuple(sorted({tensor.key for tensor in uneven})),
            hint=hint,
        ).to_error()


def _kinds(tensors: Sequence[RankMajor]) -> list[tuple[RankMajor, int]]:
    """One tensor of each kind of ``tensors`` and the number of that kind, in first-seen order."""
    first: dict[str, RankMajor] = {}
    counts: Counter[str] = Counter()
    for tensor in tensors:
        first.setdefault(tensor.kind, tensor)
        counts[tensor.kind] += 1
    return [(tensor, counts[kind]) for kind, tensor in first.items()]


def _counted(tensors: Sequence[RankMajor]) -> str:
    """``3 gated linear_fc1 and 8 Gated DeltaProduct in_proj/conv1d``."""
    return " and ".join(f"{count} {tensor.kind}" for tensor, count in _kinds(tensors))


def _fsdp_swiglu(
    args: ArgsView,
    direction: str,
    target_swiglu: str,
    language_model: list[Fc1],
    log: list[Derivation],
) -> tuple[bool, dict[str, bool]]:
    """Whether the fsdp_dtensor side's job splits SwiGLU ``linear_fc1`` (its ``args.swiglu``), and
    the gated-MLP arguments an fsdp_dtensor target's job has where they differ from the source's.

    An fsdp_dtensor source was saved by a job with its own arguments; a target
    is written for a job with the source's, unless ``--target-swiglu`` says
    otherwise: ``on`` is a job with ``--swiglu``, ``off`` one without it.
    ``--swiglu`` also makes the MLPs SiLU-gated, and a job without it gates
    them only with ``--quick-geglu`` (quick-GELU), so a job whose
    ``args.swiglu`` differs from the source's builds another activation for a
    language model with MLPs: ``--target-swiglu`` may differ from the source
    only for a language model without MLPs (whose arguments then record it).

    Raises:
        UsageError: ``--target-swiglu`` differs from the source's
            ``args.swiglu`` for a language model with MLPs (``on`` for plain
            or GEGLU MLPs, ``off`` for SwiGLU ones).
    """
    swiglu = bool(args.get("swiglu", False))
    if direction != TD_TO_FSDP or target_swiglu == AUTO_SWIGLU:
        job = "loads the fsdp_dtensor target" if direction == TD_TO_FSDP else "saved the source"
        log.append(Derivation("fsdp_swiglu", swiglu, f"args.swiglu={swiglu} of the job that {job}"))
        return swiglu, {}
    split = target_swiglu == SWIGLU_ON
    gated = any(fc1.gated for fc1 in language_model)
    if language_model and split != swiglu:
        raise errors.UsageError(_activation_change(args, split, gated), hint="drop --target-swiglu")
    swiglu_arg, geglu_arg = GLU_ARGS
    job_args = {swiglu_arg: split, geglu_arg: gated and not split}
    log.append(
        Derivation(
            "fsdp_swiglu",
            split,
            f"--target-swiglu {target_swiglu} (args.swiglu={swiglu} of the source): the job that "
            "loads the fsdp_dtensor target has "
            + ", ".join(f"{name}={value}" for name, value in job_args.items()),
        )
    )
    return split, job_args


def _activation_change(args: ArgsView, split: bool, gated: bool) -> str:
    """Why a job with (``split``) or without ``--swiglu`` does not build the language model's
    MLPs (``gated`` or not) with the source's activation."""
    if not split:
        return (
            "--target-swiglu off: a job without --swiglu gates MLPs only with --quick-geglu "
            "(quick-GELU), but the language model's MLPs are SwiGLU (args.swiglu=True), so it "
            "builds another model; a job of this model sets --swiglu and loads the halves"
        )
    if not gated:
        return (
            "--target-swiglu on: a job with --swiglu builds gated MLPs, but the language model's "
            "linear_fc1 are not gated"
        )
    return (
        "--target-swiglu on: a job with --swiglu builds SiLU-gated MLPs, but the language "
        f"model's are gated otherwise (args.swiglu=False, args.quick_geglu="
        f"{args.get('quick_geglu', False)}), so it builds another model; a job of this model "
        "keeps them fused"
    )


def _params_dtype(
    args: ArgsView, model: Mapping[str, TensorEntry], log: list[Derivation]
) -> torch.dtype:
    bf16 = bool(args.get("bf16", False))
    fp16 = bool(args.get("fp16", False))
    if bf16 and fp16:
        raise Refusal(
            errors.FC_R13,
            "args.bf16=True and args.fp16=True: the parameter dtype is contradictory",
            hint="check that the source checkpoint's arguments are intact",
        ).to_error()
    dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32
    stored = Counter(str(entry.dtype).removeprefix("torch.") for entry in model.values())
    histogram = ", ".join(f"{name} x{count}" for name, count in sorted(stored.items()))
    log.append(
        Derivation(
            "params_dtype",
            str(dtype),
            f"args.bf16={bf16}, args.fp16={fp16}; stored model dtypes: {histogram or 'none'}",
        )
    )
    return dtype


def _parallel_size(
    args: ArgsView, name: str, default: int, log: list[Derivation], *, field: str
) -> int:
    value = args.get(name, None)
    size = default if value is None else int(value)
    note = "" if value is not None else f" (unset: {default})"
    log.append(Derivation(field, size, f"args.{name}={value}{note}"))
    return size


def _fsdp_optim_depth(
    params_dtype: torch.dtype, source: str, optimizer_keys: Iterable[str], log: list[Derivation]
) -> str:
    """``module.`` levels of fsdp_dtensor optimizer keys: Float16Module adds one for bf16/fp16."""
    levels = 3 if params_dtype in (torch.bfloat16, torch.float16) else 2
    depth = "module." * levels
    evidence = f"params_dtype {str(params_dtype).removeprefix('torch.')}"
    if source == FSDP_DTENSOR:
        observed = sorted(_observed_depths(optimizer_keys))
        if observed:
            agrees = observed == [depth]
            evidence += f"; source optimizer keys use {' and '.join(observed)}" + (
                "" if agrees else " (disagrees; that state is not converted)"
            )
    log.append(Derivation("fsdp_optim_depth", depth, evidence))
    return depth


def _observed_depths(optimizer_keys: Iterable[str]) -> set[str]:
    depths = set()
    for key in optimizer_keys:
        if key.startswith(FSDP_OPTIMIZER_STATE):
            match = _MODULE_LEVELS.match(key, len(FSDP_OPTIMIZER_STATE))
            depths.add(match.group(0) if match else "")
    return depths
