# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gated linear units: which MLP first projections are gated, and how fsdp_dtensor stores them.

A gated MLP (``gated_linear_unit``: SwiGLU with ``--swiglu``, GEGLU with
``--quick-geglu``) computes ``act(x W^T) * (x V^T)``: its first projection
``linear_fc1`` holds the gate ``W`` and the value ``V`` projections along its
first axis, so its weight has twice the rows its ``linear_fc2`` weight has
inputs. A plain MLP's first projection has as many rows as the second has
inputs. :func:`glu_layout` classifies every first projection of the model this
way: the ``linear_fc1`` of a dense MLP, of shared experts, of each routed
expert (per expert, as each has its own tensor in fsdp_dtensor) and of MLPs
outside the layered blocks (a vision-language model's projection). The global
shapes are compared, so the tensor-parallel size does not matter.

The shapes are cross-checked against the configuration:

* the MLPs of one layer share its configuration, so the first projections of
  a layer (its dense MLP, or its shared and routed experts) are all gated or
  all plain;
* the language model's configuration is gated iff ``args.swiglu`` or
  ``args.quick_geglu`` is set (Megatron's ``core_transformer_config_from_args``),
  so the first projections of its decoder and of its multi-token prediction
  layers must be gated exactly then. Other blocks (a vision encoder) and MLPs
  outside blocks have configurations of their own.

A contradiction is refused (FC-R12), as is a first projection without its
second, whose gatedness cannot be told. A ``linear_fc1`` outside every layer
with no ``linear_fc2`` beside it, under a name Megatron-FSDP never splits, is
no MLP's first projection but a plain linear layer (a vision encoder's patch
merger holds one): it is left out, and converted unchanged.

torch_dist stores a gated ``linear_fc1`` whole, ``[W; V]`` (Megatron's SwiGLU
factory saves each half as chunks of its own). Megatron-FSDP splits it
into SwiGLU halves ``<name>_w`` (``W``) and ``<name>_v`` (``V``), each
TP-canonical, exactly when ``handle_swiglu_in_state_dict`` does
(:func:`swiglu_split`): the job's ``args.swiglu`` is set, the name is one of
its SwiGLU names (:data:`~.keys.SWIGLU_KEYS`: dense, shared-expert and
expert MLPs, not a vision-language projection's ``encoder.linear_fc1``), and
the layer that owns it is gated (a first projection outside every layer has
no owner and is split, as Megatron assumes). An fsdp_dtensor source must store
every first projection as the predicate says, both halves of a split tensor
(FC-R20 otherwise) and its weight and bias alike (FC-R12 otherwise).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from .. import errors
from ..engine.metadata import TensorEntry
from ..errors import Refusal, RefusalError
from .args_view import ArgsView
from .blocks import BlockLayout
from .experts import ExpertLayout
from .format_args import GLU_ARGS
from .keys import GLU_PARTS, LINEAR_FC, KeyGrammarError, fc2_path_of, is_fc1_path, is_swiglu_key
from .logical import LogicalId
from .names import FSDP_DTENSOR
from .part_layout import PartLayout
from .structure import compose, decompose

WEIGHT, BIAS = "weight", "bias"

GATED, PLAIN, SPLIT, FUSED = "gated", "plain", "split", "fused"
_ARGUMENTS_HINT = "check that the arguments belong to this checkpoint"
_Stored = Mapping[LogicalId, tuple[str, tuple[int, ...]]]
"""``{logical id: (stored key, shape)}`` of the first and second projections' tensors."""


@dataclass(frozen=True, slots=True)
class Fc1:
    """One MLP first projection: of a dense MLP, of shared experts, or of one routed expert.

    Attributes:
        lid: The logical id of its weight (the whole tensor).
        key: The stored source key that holds its weight (its gate half, if split).
        rows: Rows of its weight (both projections of a gated unit).
        gated: Whether it holds a gate and a value projection.
        halves: Whether the source stores it as SwiGLU halves.
        split: Whether fsdp_dtensor stores it as SwiGLU halves (see :func:`swiglu_split`).
    """

    lid: LogicalId
    key: str
    rows: int
    gated: bool
    halves: bool
    split: bool = False

    kind: ClassVar[str] = "gated linear_fc1"
    """What it is, for messages about the tensors fsdp_dtensor packs per rank."""

    parts_noun: ClassVar[str] = "gate and value halves"
    """What its parts are called in those messages."""

    @property
    def group(self) -> str:
        """Where it is, for summaries: its block, or the module holding it outside blocks."""
        return self.lid.block if self.lid.block is not None else self.lid.path.rpartition(".")[0]

    @property
    def packed(self) -> bool:
        """Whether fsdp_dtensor keeps it fused though it is gated: packed per tensor-parallel rank
        (per expert tensor-parallel rank for a routed expert)."""
        return self.gated and not self.split

    @property
    def parts(self) -> PartLayout:
        """Its gate and value halves, of its weight's (and bias's) rows."""
        return PartLayout(GLU_PARTS, (self.rows // len(GLU_PARTS),) * len(GLU_PARTS))


@dataclass(frozen=True)
class GluLayout:
    """The model's MLP first projections, by the logical id of their weight.

    Attributes:
        fc1s: Every first projection.
    """

    fc1s: Mapping[LogicalId, Fc1]

    def of(self, lid: LogicalId) -> Fc1 | None:
        """The first projection whose weight or bias (or a half of it) ``lid`` is, if any."""
        if lid.leaf not in (WEIGHT, BIAS) or not is_fc1_path(lid.path):
            return None
        return self.fc1s.get(LogicalId(lid.block, lid.layer, lid.path, WEIGHT, lid.expert))

    def language_model(self, blocks: Collection[str]) -> list[Fc1]:
        """The first projections of the language model's ``blocks`` (decoder, multi-token
        prediction)."""
        return [fc1 for fc1 in self.fc1s.values() if fc1.lid.block in blocks]

    def gated_json(self) -> dict[str, dict[str, int]]:
        """The number of gated and plain first projections of each block (or module)."""
        return _counts((fc1.group, GATED if fc1.gated else PLAIN) for fc1 in self.fc1s.values())

    def split_json(self) -> dict[str, dict[str, int]]:
        """The number of gated (or split) first projections fsdp_dtensor splits or keeps fused."""
        return _counts(
            (fc1.group, SPLIT if fc1.split else FUSED)
            for fc1 in self.fc1s.values()
            if fc1.gated or fc1.split
        )


def _counts(items: Iterable[tuple[str, str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for group, kind in items:
        kinds = counts.setdefault(group, {})
        kinds[kind] = kinds.get(kind, 0) + 1
    return {group: dict(sorted(kinds.items())) for group, kinds in sorted(counts.items())}


def glu_layout(
    args: ArgsView,
    model: Mapping[str, TensorEntry],
    blocks: Sequence[BlockLayout],
    source: str,
    language_model: Sequence[str],
) -> tuple[GluLayout, str]:
    """The model's first projections and whether each is gated, and the evidence.

    Args:
        args: The checkpoint's training arguments.
        model: The source's model tensors by model-relative key.
        blocks: The model's blocks.
        source: The source format.
        language_model: The blocks built with the language model's configuration (its
            decoder and multi-token prediction blocks).

    Raises:
        RefusedError: A first projection has no second projection beside it
            (unless it is a plain linear layer, see above), the first
            projections of a layer are not all gated or all plain,
            the language model's blocks are gated otherwise than its
            arguments say, or a first projection's weight and bias are not
            both split into SwiGLU halves or both whole (FC-R12); a SwiGLU
            half has no pair of its shape, or is stored beside its whole
            tensor (FC-R20).
    """
    stored = _projections(model, blocks, source)
    halves = _stored_halves(stored)
    fc1s, plain_linears = {}, []
    for lid, (key, shape) in sorted(stored.items(), key=lambda item: str(item[0])):
        if lid.leaf != WEIGHT or lid.part == GLU_PARTS[1] or not is_fc1_path(lid.path):
            continue
        whole = lid.whole
        fc2 = stored.get(LogicalId(lid.block, lid.layer, fc2_path_of(lid.path), WEIGHT, lid.expert))
        if fc2 is None:
            if _is_plain_linear(whole):
                plain_linears.append(key)
                continue
            raise _refusal(
                f"{whole} has no {fc2_path_of(lid.path)} weight beside it, so whether it is a "
                "gated linear unit cannot be told",
                (key,),
            )
        rows = shape[0] * (len(GLU_PARTS) if lid.part is not None else 1)
        fc1s[whole] = Fc1(whole, key, rows, rows == 2 * fc2[1][-1], halves[whole])
    _check_layers(fc1s.values())
    lm_gated = _check_language_model(args, fc1s.values(), language_model)
    flags = ", ".join(f"args.{name}={args.get(name, False)}" for name in GLU_ARGS)
    evidence = (
        f"{len(fc1s)} linear_fc1 weights against their linear_fc2 inputs (gated: twice as many "
        f"rows); {flags} make the language model's MLPs (in {', '.join(language_model)}) "
        + ("gated" if lm_gated else "plain")
        + _plain_linears_note(plain_linears)
    )
    return GluLayout(fc1s), evidence


def swiglu_split(
    layout: GluLayout,
    fsdp_swiglu: bool,
    blocks: Sequence[BlockLayout],
    source: str,
    experts: ExpertLayout | None,
) -> tuple[GluLayout, str]:
    """Which first projections fsdp_dtensor stores as SwiGLU halves, and the evidence.

    Mirrors ``handle_swiglu_in_state_dict``: a first projection is split iff
    ``fsdp_swiglu`` (the fsdp_dtensor job's ``args.swiglu``), its fsdp_dtensor
    name is a SwiGLU name and its owning layer is gated (no owning layer:
    split).

    Raises:
        RefusedError: A first projection to split has an odd number of rows,
            or an fsdp_dtensor source stores a first projection otherwise
            than the predicate says (FC-R12).
    """
    decided = {}
    for lid, fc1 in layout.fc1s.items():
        name = compose(lid, (fc1.rows,), blocks, FSDP_DTENSOR, experts).key
        split = fsdp_swiglu and is_swiglu_key(name) and (fc1.gated or lid.block is None)
        if split and fc1.rows % len(GLU_PARTS):
            raise _refusal(
                f"Megatron-FSDP splits {lid} into SwiGLU halves, but it has {fc1.rows} rows",
                (fc1.key,),
            )
        decided[lid] = dataclasses.replace(fc1, split=split)
    result = GluLayout(decided)
    if source == FSDP_DTENSOR:
        _check_stored_split(result.fc1s.values(), fsdp_swiglu)
    counts = [fc1.split for fc1 in decided.values() if fc1.gated or fc1.split]
    evidence = (
        f"Megatron-FSDP splits a linear_fc1 into SwiGLU halves iff its job's args.swiglu "
        f"(here {fsdp_swiglu}) is set, its name is a SwiGLU name of handle_swiglu_in_state_dict "
        f"and its layer is gated: {sum(counts)} split, {len(counts) - sum(counts)} gated kept fused"
    )
    return result, evidence


def _is_plain_linear(lid: LogicalId) -> bool:
    """Whether a first projection with no second projection beside it is a plain linear layer:
    one outside every layer, which is no MLP's, under a name Megatron-FSDP never splits (it would
    split one under a SwiGLU name, gated or not)."""
    return lid.block is None and not is_swiglu_key(lid.name)


def _plain_linears_note(keys: Sequence[str]) -> str:
    """The evidence about the plain linear layers named ``linear_fc1`` (none: empty)."""
    if not keys:
        return ""
    return (
        f"; {len(keys)} linear_fc1 outside every layer with no linear_fc2 beside it (plain "
        f"linear layers, converted unchanged): {', '.join(sorted(keys))}"
    )


def _projections(
    model: Mapping[str, TensorEntry], blocks: Sequence[BlockLayout], source: str
) -> dict[LogicalId, tuple[str, tuple[int, ...]]]:
    """The first projections' weights and biases (whole or halves) and the second projections'
    weights, by logical id.

    Malformed keys are left out (the planner refuses them with the grammar error).
    """
    stored = {}
    for relative, entry in model.items():
        if LINEAR_FC.search(relative) is None:
            continue
        try:
            slices = decompose(relative, entry, blocks, source)
        except KeyGrammarError:
            continue
        for piece in slices:
            if piece.shape:
                stored[piece.lid] = (piece.key, piece.shape)
    return stored


def _stored_halves(stored: _Stored) -> dict[LogicalId, bool]:
    """Whether each first-projection weight is stored as SwiGLU halves (and its bias alike).

    Raises:
        RefusedError: A half has no pair of its shape, or is stored beside
            its whole tensor (FC-R20); a weight and its bias are not both
            halves or both whole (FC-R12).
    """
    parts: dict[LogicalId, dict[str | None, tuple[str, tuple[int, ...]]]] = {}
    for lid, found in stored.items():
        if is_fc1_path(lid.path) and lid.leaf in (WEIGHT, BIAS):
            parts.setdefault(lid.whole, {})[lid.part] = found
    split = {}
    for whole, found in sorted(parts.items(), key=lambda item: str(item[0])):
        shapes = {shape for _, shape in found.values()}
        if set(found) not in ({None}, set(GLU_PARTS)) or len(shapes) > 1:
            stored_as = ", ".join(
                f"{'the whole tensor' if part is None else f'half {part}'} {found[part][1]}"
                for part in sorted(found, key=str)
            )
            raise Refusal(
                errors.FC_R20,
                f"{whole} is stored as {stored_as}: SwiGLU halves come in pairs (w and v) of one "
                "shape that replace the whole tensor",
                keys=tuple(sorted(key for key, _ in found.values())),
                hint="check that the store is complete",
            ).to_error()
        split[whole] = None not in found
    for whole, halves in split.items():
        weight = LogicalId(whole.block, whole.layer, whole.path, WEIGHT, whole.expert)
        if whole.leaf == BIAS and weight in split and split[weight] != halves:
            raise _refusal(
                f"{weight} and its bias are not both stored as SwiGLU halves",
                tuple(sorted(key for key, _ in parts[weight].values()))
                + tuple(sorted(key for key, _ in parts[whole].values())),
            )
    return {whole: halves for whole, halves in split.items() if whole.leaf == WEIGHT}


def _check_stored_split(fc1s: Iterable[Fc1], fsdp_swiglu: bool) -> None:
    """An fsdp_dtensor source stores each first projection as the predicate says (FC-R12)."""
    unexpected = sorted(fc1.key for fc1 in fc1s if fc1.halves and not fc1.split)
    if unexpected:
        raise _refusal(
            "the store holds SwiGLU halves that Megatron-FSDP does not write: it splits a "
            f"linear_fc1 only if the job's args.swiglu (here {fsdp_swiglu}) is set, its name is "
            "a SwiGLU name and its layer is gated",
            tuple(unexpected),
        )
    missing = sorted(fc1.key for fc1 in fc1s if fc1.split and not fc1.halves)
    if missing:
        raise _refusal(
            "the store holds gated linear_fc1 whole that Megatron-FSDP splits into SwiGLU halves "
            f"(args.swiglu={fsdp_swiglu})",
            tuple(missing),
        )


def _check_layers(fc1s: Iterable[Fc1]) -> None:
    """The first projections of a layer are all gated or all plain (FC-R12)."""
    layers: dict[tuple[str, int], dict[bool, list[str]]] = {}
    for fc1 in fc1s:
        if fc1.lid.block is not None and fc1.lid.layer is not None:
            kinds = layers.setdefault((fc1.lid.block, fc1.lid.layer), {})
            kinds.setdefault(fc1.gated, []).append(fc1.key)
    for (block, layer), kinds in sorted(layers.items()):
        if len(kinds) > 1:
            raise _refusal(
                f"{block}[{layer}] has gated and plain linear_fc1 weights, but the MLPs of a "
                "layer share its configuration",
                tuple(sorted({key for keys in kinds.values() for key in keys})),
            )


def _check_language_model(
    args: ArgsView, fc1s: Iterable[Fc1], language_model: Sequence[str]
) -> bool:
    """The language model's blocks are gated iff its arguments say so (FC-R12); whether they are."""
    gated = any(bool(args.get(name, False)) for name in GLU_ARGS)
    wrong = sorted(
        {fc1.key for fc1 in fc1s if fc1.lid.block in language_model and fc1.gated != gated}
    )
    if wrong:
        flags = " and ".join(f"args.{name}={args.get(name, False)}" for name in GLU_ARGS)
        observed = "plain" if gated else "gated (twice the rows linear_fc2 has inputs)"
        raise _refusal(
            f"the language model's linear_fc1 weights (in {', '.join(language_model)}) are "
            f"{observed}, but {flags} make its MLPs {'gated' if gated else 'plain'}",
            tuple(wrong),
        )
    return gated


def _refusal(message: str, keys: tuple[str, ...]) -> RefusalError:
    return Refusal(errors.FC_R12, message, keys=keys, hint=_ARGUMENTS_HINT).to_error()
