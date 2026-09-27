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
from .blocks import BlockFacts, BlockLayout, classify_blocks
from .dtype_policy import DtypePolicy
from .manifest import MANIFEST_FILE, Manifest
from .names import DIALECTS, FSDP_DTENSOR, FSDP_MODEL_PREFIX, FSDP_OPTIMIZER_STATE, TORCH_DIST
from .stacking import transformer_block

TD_TO_FSDP = "td_to_fsdp"
FSDP_TO_TD = "fsdp_to_td"
DIRECTIONS = {(TORCH_DIST, FSDP_DTENSOR): TD_TO_FSDP, (FSDP_DTENSOR, TORCH_DIST): FSDP_TO_TD}
"""The cross-format directions by (source, target) format."""

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
        blocks: The model's layered blocks and how each format numbers their layers.
        fsdp_model_prefix: Prefix of the fsdp_dtensor side's model keys.
        fsdp_optim_depth: ``module.`` levels before a parameter name in the
            fsdp_dtensor side's optimizer keys (one more for bf16 and fp16).
        dtype_policy: How output dtypes are chosen.
        manifest: The manifest of the conversion that produced the source, if
            it has one that describes it (used to restore what that
            conversion changed).
        derivations: How each field was derived, in derivation order.
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
    blocks: tuple[BlockLayout, ...]
    fsdp_model_prefix: str
    fsdp_optim_depth: str
    dtype_policy: DtypePolicy
    manifest: Manifest | None
    derivations: tuple[Derivation, ...]

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

    Raises:
        UsageError: ``source`` to ``target`` is not a cross-format direction.
        RefusedError: A derivation is contradictory (FC-R13 for both ``bf16``
            and ``fp16``; FC-R11 for layer indices the blocks do not explain;
            FC-R12 for inconsistently stacked layers).
        AmbiguousError: The source holds a manifest of another store (FC-A06).
    """
    direction = DIRECTIONS.get((source, target))
    if direction is None:
        raise errors.UsageError(f"{source} to {target} is not a cross-format conversion")
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
    blocks = classify_blocks(BlockFacts(args, source, model), default=transformer_block)
    log.extend(
        Derivation(f"blocks.{block.prefix}", _block_summary(block), block.evidence)
        for block in blocks
    )
    log.append(
        Derivation("fsdp_model_prefix", FSDP_MODEL_PREFIX, f"dialect {DIALECTS[FSDP_DTENSOR]}")
    )
    depth = _fsdp_optim_depth(params_dtype, source, optimizer_keys, log)
    log.append(Derivation("dtype_policy", DtypePolicy.PRESERVE.value, "weights-only conversion"))
    if manifest_ignored:
        manifest = None
    _check_manifest(manifest, source, metadata_sha256, log, ignored=manifest_ignored)
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
        blocks=blocks,
        fsdp_model_prefix=FSDP_MODEL_PREFIX,
        fsdp_optim_depth=depth,
        dtype_policy=DtypePolicy.PRESERVE,
        manifest=manifest,
        derivations=tuple(log),
    )


def _check_manifest(
    manifest: Manifest | None,
    source: str,
    metadata_sha256: str,
    log: list[Derivation],
    *,
    ignored: bool,
) -> None:
    """A source's manifest must describe the source store itself."""
    if ignored:
        log.append(Derivation("manifest", None, f"{MANIFEST_FILE} ignored (--ignore-manifest)"))
        return
    if manifest is None:
        log.append(Derivation("manifest", None, f"the source has no {MANIFEST_FILE}"))
        return
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
    log.append(
        Derivation(
            "manifest",
            manifest.source.get("format"),
            f"{MANIFEST_FILE} of the conversion from {manifest.source.get('iter_dir')} "
            "(restores its sanitized arguments and content_metadata)",
        )
    )


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
