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

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .. import errors
from ..engine.metadata import TensorEntry
from ..errors import Refusal
from .args_view import ArgsView
from .blocks import TRANSFORMER_BLOCK, BlockFacts, BlockLayout, classify_blocks
from .dtype_policy import DtypePolicy
from .experts import ExpertLayout, expert_layout, grouped_bias_warnings
from .glu import GluLayout, glu_layout
from .keys import root_of
from .manifest import MANIFEST_FILE, Manifest
from .names import (
    AUTO_EXPERT_NAMING,
    DIALECTS,
    FSDP_DTENSOR,
    FSDP_MODEL_PREFIX,
    FSDP_OPTIMIZER_STATE,
    PER_LAYER,
    TORCH_DIST,
)
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
        glu: The model's MLP first projections (``linear_fc1``) and which
            are gated linear units.
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
    fsdp_model_prefix: str
    fsdp_optim_depth: str
    dtype_policy: DtypePolicy
    manifest: Manifest | None
    derivations: tuple[Derivation, ...]
    warnings: tuple[str, ...] = ()

    def derivation_log(self) -> list[dict[str, Any]]:
        """The derivations as JSON records."""
        return [derivation.to_json() for derivation in self.derivations]


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

    Raises:
        UsageError: ``source`` to ``target`` is not a cross-format direction,
            or ``block_layouts`` names a block that is not a TransformerBlock.
        RefusedError: A derivation is contradictory (FC-R13 for both ``bf16``
            and ``fp16``, or routed experts that disagree with
            ``args.num_experts``; FC-R11 for layer indices the blocks do not
            explain; FC-R12 for inconsistently stacked layers, layers that
            cannot be stacked as asked, expert indices other than
            ``0..E-1``, or gated linear units that contradict one another or
            the arguments).
        AmbiguousError: The source holds a manifest of another store, or one
            that records another torch_dist layout of a block (FC-A06); a
            block's torch_dist layout cannot be decided (FC-A02); an
            fsdp_dtensor source names its experts otherwise than its
            arguments build them (FC-A04).
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
    glu, evidence = glu_layout(args, model, blocks, source, facts.lm_decoder)
    log.append(Derivation("gated_fc1", glu.to_json(), evidence))
    warnings = requested_layout_warnings(facts, blocks) + grouped_bias_warnings(
        experts, model, source
    )
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
