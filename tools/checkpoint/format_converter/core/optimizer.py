# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The optimizer state of a cross-format conversion, planned through the model's own rules.

Every optimizer state of a parameter (Adam's moments ``exp_avg`` and
``exp_avg_sq``) is a tensor of the parameter's shape that both formats lay out
as the parameter itself: Megatron builds a torch_dist optimizer-state tensor
by replacing the data of the parameter's sharded tensor (stacked layers,
expert axes and split-factory parts carry over), and Megatron-FSDP's save
handlers rename and split the optimizer state as they do the model (global
expert indices, SwiGLU halves, the GPT MTP inner layer, unfused multi-latent
attention). A moment's tensors therefore decompose to the parameters' logical
ids, go through the rules that convert the model section (stacking, routed
experts, gated linear units, shortcut-MoE renumbering, packed mixer
projections; no second mapping), and are composed in the target's names under
the target's key of that state.

The fp32 main copy of a parameter (its *master*) is the model section of a
Megatron-FSDP store and the optimizer state ``param`` of a fully reshardable
torch_dist store, whose model weight is the master rounded to the parameter
dtype (``core/dtype_policy.py``). The masters and the model weights of the
trainable parameters (those with moments) are both built from the one plan the
rules make of the model section: their pieces read the same stored chunks,
which the scheduler packs into one unit, so each is read once.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from .. import errors
from ..engine.metadata import TensorEntry
from ..engine.pieces import Piece
from ..errors import Refusal
from .context import ArchContext
from .dtype_policy import model_weight_dtype
from .logical import LogicalId, LogicalOutput
from .names import MAIN_PARAM
from .planner import RuledSection, TargetLayout, apply_rules, compile_outputs, plan_model
from .rule import LeafRule
from .structure import compose


class StateTarget(TargetLayout, Protocol):
    """What planning optimizer state needs from the target format's adapter: its model layout,
    and the layout of each optimizer state."""

    def state_layout(self, state: str) -> TargetLayout:
        """The layout of every parameter's optimizer state ``state``."""


@dataclass(frozen=True)
class StatePlan:
    """The planned optimizer state of a conversion.

    Attributes:
        pieces: The output pieces of every state, by (key, offsets).
        claims: Source tensors claimed per rule, over every state.
    """

    pieces: tuple[Piece, ...]
    claims: dict[str, int]


def plan_moments(
    ctx: ArchContext,
    model: Mapping[str, TensorEntry],
    moments: Mapping[str, Mapping[str, TensorEntry]],
    target: StateTarget,
    *,
    rules: Sequence[LeafRule],
    max_chunk_bytes: int,
) -> StatePlan:
    """Plan every moment of every parameter as the target stores it.

    Args:
        ctx: The architecture context.
        model: The source's model tensors by model-relative key.
        moments: Each moment's tensors by the model-relative key of their parameter.
        target: The target format's adapter.
        rules: The enabled rules, in stage order (the model section's).
        max_chunk_bytes: Chunk-size limit passed to the target's grid.

    Raises:
        RefusedError: A parameter lacks a moment another parameter has, or a
            moment's shape is not its parameter's (FC-R13); and as
            :func:`~.planner.plan_model` for each moment.
    """
    _check_moments(model, moments)
    pieces: list[Piece] = []
    claims: Counter[str] = Counter()
    for moment in sorted(moments):
        plan = plan_model(
            ctx,
            moments[moment],
            target.state_layout(moment),
            rules=rules,
            max_chunk_bytes=max_chunk_bytes,
        )
        pieces.extend(plan.pieces)
        claims.update(plan.claims)
    return StatePlan(tuple(pieces), dict(sorted(claims.items())))


def plan_masters(
    ctx: ArchContext,
    model: Mapping[str, TensorEntry],
    trainable: Collection[str],
    target: StateTarget,
    *,
    rules: Sequence[LeafRule],
    max_chunk_bytes: int,
    allow_bf16_masters: bool = False,
) -> StatePlan:
    """Plan the model section and the master of every trainable parameter.

    The source's model section holds the masters of the trainable
    parameters. The target gets each master in fp32 as the optimizer state
    ``param``, and as the parameter's model weight the master rounded to the
    model's parameter dtype (to nearest even), or kept in fp32 for a parameter
    Megatron keeps in fp32; every other model tensor keeps its dtype.

    Args:
        ctx: The architecture context.
        model: The source's model tensors by model-relative key.
        trainable: Model-relative keys of the tensors that have optimizer state.
        target: The target format's adapter.
        rules: The enabled rules, in stage order.
        max_chunk_bytes: Chunk-size limit passed to the target's grid.
        allow_bf16_masters: Take trainable tensors stored in a narrower type
            than fp32 as masters (upcast exactly).

    Raises:
        RefusedError: A trainable tensor is not fp32 without
            ``allow_bf16_masters`` (FC-R21), or an output tensor is built from
            tensors with and without optimizer state (FC-R13); and as
            :func:`~.planner.apply_rules`.
    """
    trainable = frozenset(trainable)
    ruled = apply_rules(ctx, model, rules)
    tensors: dict[str, list[LogicalOutput]] = {}
    for output in ruled.outputs:
        key = compose(output.lid, output.shape, ctx.blocks, target.fmt, ctx.experts).key
        tensors.setdefault(key, []).append(output)
    masters = _masters(tensors, ruled, trainable, allow_bf16_masters=allow_bf16_masters)
    weights = tuple(
        (
            dataclasses.replace(output, dtype=model_weight_dtype(output.lid.name, ctx.params_dtype))
            if output.lid in masters
            else output
        )
        for output in ruled.outputs
    )
    master_slices = {
        lid: piece for lid, piece in ruled.slices.items() if piece.relative in trainable
    }
    pieces = compile_outputs(
        ctx, weights, ruled.slices, model, target, max_chunk_bytes=max_chunk_bytes, casts=True
    ) + compile_outputs(
        ctx,
        tuple(masters.values()),
        master_slices,
        model,
        target.state_layout(MAIN_PARAM),
        max_chunk_bytes=max_chunk_bytes,
        casts=True,
    )
    return StatePlan(pieces, dict(ruled.claims))


def _masters(
    tensors: Mapping[str, Sequence[LogicalOutput]],
    ruled: RuledSection,
    trainable: frozenset[str],
    *,
    allow_bf16_masters: bool,
) -> dict[LogicalId, LogicalOutput]:
    """The fp32 master of every output of a target tensor built from trainable tensors, by its
    logical id.

    Args:
        tensors: The outputs of each target tensor, by its model-relative key.
        ruled: What the rules made of the model section.
        trainable: Model-relative keys of the source tensors with optimizer state.
        allow_bf16_masters: Take trainable tensors narrower than fp32 as masters.

    Raises:
        RefusedError: A target tensor is built from tensors with and without
            optimizer state (FC-R13), or a trainable tensor is narrower than
            fp32 without ``allow_bf16_masters`` (FC-R21).
    """
    masters: dict[LogicalId, LogicalOutput] = {}
    mixed: set[str] = set()
    narrow: set[str] = set()
    for outputs in tensors.values():
        sources = [ruled.slices[copy.src] for output in outputs for copy in output.copies]
        held = [piece.relative in trainable for piece in sources]
        if not any(held):
            continue
        if not all(held):
            mixed.update(piece.key for piece in sources)
            continue
        narrow.update(piece.key for piece in sources if piece.dtype != torch.float32)
        masters.update(
            (output.lid, dataclasses.replace(output, dtype=torch.float32)) for output in outputs
        )
    refusals = []
    if mixed:
        refusals.append(
            Refusal(
                errors.FC_R13,
                "tensors that make up one target tensor, of which only some have optimizer state",
                keys=tuple(sorted(mixed)),
                hint="check that the store holds one complete save",
            )
        )
    if narrow and not allow_bf16_masters:
        refusals.append(
            Refusal(
                errors.FC_R21,
                "parameters with optimizer state are stored in a narrower type than fp32, where "
                "Megatron-FSDP keeps their fp32 main copies (the masters a torch_dist store needs)",
                keys=tuple(sorted(narrow)),
                hint="pass --allow-bf16-masters to take them as masters, upcast exactly",
            )
        )
    if refusals:
        raise errors.refusal_error(refusals)
    return masters


def _check_moments(
    model: Mapping[str, TensorEntry], moments: Mapping[str, Mapping[str, TensorEntry]]
) -> None:
    """Every parameter with a moment has every moment, each of the parameter's shape (FC-R13)."""
    holders = {name for tensors in moments.values() for name in tensors}
    refusals = []
    for moment, tensors in sorted(moments.items()):
        missing = sorted(holders - set(tensors))
        if missing:
            refusals.append(
                Refusal(
                    errors.FC_R13,
                    f"parameters with optimizer state but no {moment}",
                    keys=tuple(missing),
                    hint="check that the store holds one complete save",
                )
            )
        reshaped = sorted(
            entry.key
            for name, entry in tensors.items()
            if name in model and entry.shape != model[name].shape
        )
        if reshaped:
            refusals.append(
                Refusal(
                    errors.FC_R13,
                    f"{moment} tensors whose shape is not their parameter's",
                    keys=tuple(reshaped),
                    hint="check that the store holds one complete save",
                )
            )
    if refusals:
        raise errors.refusal_error(refusals)
