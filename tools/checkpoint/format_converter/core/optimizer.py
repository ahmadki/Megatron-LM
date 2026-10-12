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
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol

from .. import errors
from ..engine.metadata import TensorEntry
from ..engine.pieces import Piece
from ..errors import Refusal
from .context import ArchContext
from .planner import TargetLayout, plan_model
from .rule import LeafRule


class StateTarget(Protocol):
    """What planning optimizer state needs from the target format's adapter."""

    fmt: ClassVar[str]

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
