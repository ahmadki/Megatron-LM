# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Planning the optimizer state of an fsdp_dtensor to torch_dist conversion (``--optimizer
convert``).

A Megatron-FSDP save keeps the trainable parameters' fp32 masters as its model
section, Adam's moments and each parameter's group hyperparameters per
parameter (``formats/fsdp_optimizer.py``) and the learning-rate scheduler as
``opt_param_scheduler.<field>`` byte items. The fully reshardable torch_dist
store a classic job saves holds, per parameter, its master
(``optimizer.state.param``), its model weight (the master rounded to the
parameter dtype) and its moments under its model key, and in its common state
the parameter groups of each optimizer the job chains (``optimizer``), the
scheduler (``opt_param_scheduler``) and the ``content_metadata`` that makes
Megatron load them as fully reshardable state. Everything is planned from the
metadata and the byte items, which the driver reads; no tensor is read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import errors
from .core.context import ArchContext
from .core.names import FSDP_MODEL_PREFIX, MAIN_PARAM, MOMENTS
from .core.optimizer import StatePlan, StateTarget, plan_masters, plan_moments
from .core.param_groups import (
    FULLY_RESHARDABLE,
    chained_groups,
    optimizer_common_state,
    rebuild_param_groups,
)
from .core.rule import LeafRule
from .engine.metadata import MetadataView, TensorEntry
from .engine.pieces import Piece
from .engine.reader import SourceReader
from .errors import InternalError, Refusal
from .formats.base import Section
from .formats.fsdp_optimizer import STEP, FsdpOptimizerState, live_names, parse_optimizer_state
from .formats.torch_dist import OPTIMIZER_STATE_PREFIX
from .mcore_state.common_state import CodecMode, loads

OPTIMIZER_SECTIONS = frozenset({Section.OPTIMIZER_STATE, Section.OPTIMIZER_OTHER})
"""The sections of an fsdp_dtensor store that hold optimizer state."""

SCHEDULER = "opt_param_scheduler"
"""The common-state key (and fsdp_dtensor key prefix) of the learning-rate scheduler's state."""

OPTIMIZER = "optimizer"
"""The common-state key of the optimizer's parameter groups."""

CONTENT_METADATA_FLAGS: Mapping[str, Any] = {
    "distrib_optim_sharding_type": FULLY_RESHARDABLE,
    "distrib_optim_fully_reshardable_mem_efficient": False,
    "singleton_local_shards": False,
    "chained_optim_avoid_prefix": True,
}
"""The ``content_metadata`` of a fully reshardable torch_dist save
(``_build_sharded_state_dict_metadata`` with ``--dist-ckpt-optim-fully-reshardable``): Megatron
loads the optimizer state as fully reshardable only when it says so."""

ADAM = "adam"
"""``args.optimizer`` of the optimizer whose state is converted."""

PARAM_GROUP_MAP = "param_to_param_group_map.json"
"""The sidecar mapping every parameter (``model.module.<live name>``) to the index of its group
in the chained optimizers' groups, as ``print-torch-dcp-in-json`` renders the map
``--dump-param-to-param-group-map`` saves."""


@dataclass(frozen=True)
class OptimizerPlan:
    """The planned optimizer state of a conversion.

    Attributes:
        pieces: The model weights, masters and moments, by (key, offsets).
        claims: Model tensors claimed per rule.
        common: The common-state entries it adds (``optimizer``,
            ``opt_param_scheduler``).
        group_map: The group index of every parameter (the sidecar).
        summary: What was converted and dropped (for the report).
    """

    pieces: tuple[Piece, ...]
    claims: dict[str, int]
    common: dict[str, Any]
    group_map: dict[str, int]
    summary: dict[str, Any]


def plan_optimizer(
    view: MetadataView,
    sections: Mapping[str, Section],
    model: Mapping[str, TensorEntry],
    ctx: ArchContext,
    target: StateTarget,
    *,
    reader: SourceReader,
    mode: CodecMode,
    rules: Sequence[LeafRule],
    max_chunk_bytes: int,
    allow_bf16_masters: bool,
) -> OptimizerPlan:
    """Plan the model section and the optimizer state of an fsdp_dtensor store for torch_dist.

    Args:
        view: The source store.
        sections: The section of every key of the store.
        model: Its model tensors by model-relative key.
        ctx: The architecture context (with the masters dtype policy).
        target: The torch_dist adapter.
        reader: Reads the store's byte items.
        mode: The codec mode the common state was decoded with.
        rules: The enabled rules, in stage order.
        max_chunk_bytes: Chunk-size limit passed to the target's grid.
        allow_bf16_masters: Take trainable tensors narrower than fp32 as masters.

    Raises:
        RefusedError: The store holds no optimizer or scheduler state, the
            parameters with optimizer state and with parameter groups differ
            (FC-R13), or the optimizer is not Adam or keeps steps its groups
            do not (FC-R16); and as the parsing and planning steps.
    """
    keys = sorted(key for key, section in sections.items() if section in OPTIMIZER_SECTIONS)
    if not keys:
        raise Refusal(
            errors.FC_R13,
            f"{view.iter_dir} holds no optimizer state to convert",
            hint="drop --optimizer convert to convert the weights only",
        ).to_error()
    _check_adam(ctx)
    live = live_names(model, ctx.args)
    state = parse_optimizer_state(view, keys, depth=ctx.fsdp_optim_depth, live=live)
    payloads = reader.read_bytes([k for items in state.group_meta.values() for k in items.values()])
    meta = {
        name: {hparam: _decode(payloads, key, mode) for hparam, key in items.items()}
        for name, items in state.group_meta.items()
    }
    _check_groups(state, live, meta)
    groups = rebuild_param_groups(meta)
    chained = chained_groups(groups, ctx.args)
    masters = plan_masters(
        ctx,
        model,
        state.trainable,
        target,
        rules=rules,
        max_chunk_bytes=max_chunk_bytes,
        allow_bf16_masters=allow_bf16_masters,
    )
    moments = plan_moments(
        ctx, model, state.moments, target, rules=rules, max_chunk_bytes=max_chunk_bytes
    )
    _check_states(masters, moments)
    flat = [group for optimizer in chained for group in optimizer]
    return OptimizerPlan(
        pieces=masters.pieces + moments.pieces,
        claims=masters.claims,
        common={
            OPTIMIZER: optimizer_common_state(chained),
            SCHEDULER: _scheduler(view, sections, reader, mode),
        },
        group_map={
            f"{FSDP_MODEL_PREFIX}{name}": index
            for index, group in enumerate(flat)
            for name in group.params
        },
        summary={
            "parameters": len(state.trainable),
            "param_groups": [len(optimizer) for optimizer in chained],
            "dropped_steps": len(state.steps),
        },
    )


def _check_adam(ctx: ArchContext) -> None:
    """The arguments' optimizer is Adam (FC-R16)."""
    optimizer = ctx.args.get("optimizer", ADAM)
    if optimizer != ADAM:
        raise Refusal(
            errors.FC_R16,
            f"args.optimizer={optimizer!r}: only Adam's optimizer state is converted",
            hint="convert the weights only (--optimizer drop)",
        ).to_error()


def _decode(payloads: Mapping[str, bytes], key: str, mode: CodecMode) -> Any:
    return loads(payloads[key], mode=mode, label=key)


def _check_groups(
    state: FsdpOptimizerState, live: Mapping[str, str], meta: Mapping[str, Mapping[str, Any]]
) -> None:
    """The parameters with moments are those with group hyperparameters (FC-R13); per-parameter
    steps (``torch.optim.Adam``) are dropped only where the groups keep the step (FC-R16)."""
    trained = {live[name] for name in state.trainable}
    refusals = []
    for names, message in (
        (sorted(trained - set(meta)), "parameters with optimizer state but no parameter group"),
        (sorted(set(meta) - trained), "parameters with a parameter group but no optimizer state"),
    ):
        if names:
            refusals.append(
                Refusal(
                    errors.FC_R13,
                    message,
                    keys=tuple(names),
                    hint="check that the store holds one complete save",
                )
            )
    stepless = sorted(live[name] for name in state.steps if STEP not in meta.get(live[name], {}))
    if stepless:
        refusals.append(
            Refusal(
                errors.FC_R16,
                "per-parameter step counts (torch.optim.Adam) of parameters whose groups keep no "
                "step, which the torch_dist loader reads from the groups",
                keys=tuple(stepless),
                hint="convert a checkpoint of a job with Transformer Engine's or Apex's fused "
                "Adam, or convert the weights only (--optimizer drop)",
            )
        )
    if refusals:
        raise errors.refusal_error(refusals)


def _check_states(masters: StatePlan, moments: StatePlan) -> None:
    """Every parameter with a master has every moment, under the same model key."""

    def holders(plan: StatePlan, state: str) -> set[str]:
        prefix = f"{OPTIMIZER_STATE_PREFIX}{state}."
        return {piece.key[len(prefix) :] for piece in plan.pieces if piece.key.startswith(prefix)}

    expected = holders(masters, MAIN_PARAM)
    for moment in MOMENTS:
        if holders(moments, moment) != expected:
            raise InternalError(f"the {moment} keys of the plan are not its masters' keys")


def _scheduler(
    view: MetadataView, sections: Mapping[str, Section], reader: SourceReader, mode: CodecMode
) -> dict[str, Any]:
    """The scheduler state from its ``opt_param_scheduler.<field>`` byte items.

    Raises:
        RefusedError: The store has none (FC-R13).
    """
    keys = sorted(key for key, section in sections.items() if section is Section.SCHEDULER)
    if not keys:
        raise Refusal(
            errors.FC_R13,
            f"{view.iter_dir} holds optimizer state but no learning-rate scheduler state "
            f"({SCHEDULER}.*), which a torch_dist resume loads with it",
            hint="convert a complete training save, or the weights only (--optimizer drop)",
        ).to_error()
    payloads = reader.read_bytes(keys)
    return {key[len(SCHEDULER) + 1 :]: _decode(payloads, key, mode) for key in keys}
