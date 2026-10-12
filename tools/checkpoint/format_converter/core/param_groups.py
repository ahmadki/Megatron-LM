# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optimizer parameter groups, rebuilt from per-parameter group hyperparameters.

Megatron-FSDP saves each parameter's group as its own hyperparameters
(``param_to_group_meta``: the group dict without ``params``, one copy per
parameter of the group); a torch_dist store keeps the groups themselves
(``param_groups`` without ``params``). Megatron-FSDP's loader puts every
parameter back into the live group whose hyperparameters equal its own
(``DistributedOptimizer._param2group_meta_to_param_groups``), so the groups are
the classes of parameters with equal hyperparameters. The classic loader
pairs each saved group with a live one by its identifier tuple, the values of
:data:`IDENTIFIER_KEYS` (``load_state_dict``'s ``make_needed_groups``), so the
rebuilt groups' identifiers must be pairwise distinct: two saved groups with
one identifier would load into one live group.

A classic job keeps the groups of routed experts' parameters that use the
expert process groups (``is_expert_parallel``) in an optimizer of their own
(``get_megatron_optimizer``): its ``ChainedOptimizer`` then saves one
optimizer state per optimizer, ``{0: dense, 1: experts}``, and a single one
otherwise. Which parameters use those groups is decided by the linear layers'
``use_expert_pgs`` (:func:`uses_expert_process_groups`); the groups a store
records are checked against it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .. import errors
from ..errors import Refusal
from .args_view import ArgsView
from .keys import ROUTED_EXPERTS

IDENTIFIER_KEYS = (
    "end_wd",
    "max_lr",
    "min_lr",
    "optimizer",
    "start_wd",
    "wd_mult",
    "lr_mult",
    "is_expert_parallel",
    "is_decoupled_lr",
)
"""The hyperparameters that identify a parameter group when a checkpoint is loaded
(``param_group_identifier_keys`` of ``megatron/core/optimizer/optimizer.py``: the fields of
``ParamGroupOverride``, sorted, then Megatron's structural flags; pinned by a test)."""

ALIAS_PREFIX = "pre_"
"""NeMo's alias of an identifier hyperparameter (``pre_lr_mult``), which the loader also reads."""

EXPERT_PARALLEL = "is_expert_parallel"
"""The hyperparameter of the groups of parameters that use the expert process groups."""

FULLY_RESHARDABLE = "fully_reshardable"
"""The distributed optimizer's model-space torch_dist format (``distrib_optim_sharding_type``)."""


@dataclass(frozen=True)
class ParamGroup:
    """One optimizer parameter group.

    Attributes:
        hparams: Every hyperparameter of the group (``params`` excluded).
        params: The live names of its parameters, sorted.
    """

    hparams: Mapping[str, Any]
    params: tuple[str, ...]

    @property
    def identifier(self) -> tuple[Any, ...]:
        """The group's identifier tuple (see :func:`identifier`)."""
        return identifier(self.hparams)


def identifier(hparams: Mapping[str, Any]) -> tuple[Any, ...]:
    """The values of :data:`IDENTIFIER_KEYS` in ``hparams``, as Megatron's loader reads them: a
    key, else its ``pre_`` alias, else ``None``."""
    return tuple(hparams.get(key, hparams.get(f"{ALIAS_PREFIX}{key}")) for key in IDENTIFIER_KEYS)


def rebuild_param_groups(group_meta: Mapping[str, Mapping[str, Any]]) -> tuple[ParamGroup, ...]:
    """The parameter groups of parameters with equal hyperparameters, in first-parameter order.

    Args:
        group_meta: Every parameter's group hyperparameters, by its live name.

    Raises:
        RefusedError: Two groups have one identifier tuple, so the torch_dist
            loader could not tell them apart (FC-R13).
    """
    classes: list[tuple[Mapping[str, Any], list[str]]] = []
    for name in sorted(group_meta):
        hparams = group_meta[name]
        group = next((params for known, params in classes if known == hparams), None)
        if group is None:
            classes.append((dict(hparams), [name]))
        else:
            group.append(name)
    groups = tuple(ParamGroup(hparams, tuple(params)) for hparams, params in classes)
    _check_identifiers(groups)
    return groups


def _check_identifiers(groups: tuple[ParamGroup, ...]) -> None:
    """Every group has its own identifier tuple (FC-R13)."""
    seen: dict[tuple[Any, ...], ParamGroup] = {}
    clashes = []
    for group in groups:
        other = seen.setdefault(group.identifier, group)
        if other is not group:
            clashes.append((other, group))
    if not clashes:
        return
    first, second = clashes[0]
    differ = sorted(
        key
        for key in {*first.hparams, *second.hparams}
        if first.hparams.get(key) != second.hparams.get(key)
    )
    shared = ", ".join(f"{k}={v!r}" for k, v in zip(IDENTIFIER_KEYS, first.identifier))
    raise Refusal(
        errors.FC_R13,
        f"parameter groups that differ in {', '.join(differ)} share the identifier ({shared}): "
        "the torch_dist loader pairs saved and live groups by their identifier, so it would "
        "load them into one group",
        keys=tuple(dict.fromkeys(group.params[0] for pair in clashes for group in pair)),
        hint="convert a checkpoint whose parameter groups differ in an identifying "
        "hyperparameter (see param_group_identifier_keys)",
    ).to_error()


def uses_expert_process_groups(args: ArgsView) -> bool:
    """Whether a job of ``args`` puts its routed experts' parameters in the expert process groups.

    The mirror of the ``use_expert_pgs`` of Megatron's linear layers (pinned by
    a test): expert parallelism, or an expert tensor-parallel size or grouped
    tensor-parallel rematerialization size other than the dense one.
    """
    tp = args.get("tensor_model_parallel_size", None) or 1
    etp = args.get("expert_tensor_parallel_size", None) or tp
    ep = args.get("expert_model_parallel_size", None) or 1
    gtp = args.get("gtp_weight_remat_size", None) or 1
    expert_gtp = args.get("expert_gtp_weight_remat_size", None) or 1
    return ep > 1 or etp != tp or expert_gtp != gtp


def chained_groups(
    groups: Sequence[ParamGroup], args: ArgsView
) -> tuple[tuple[ParamGroup, ...], ...]:
    """The groups of each optimizer a classic job of ``args`` chains: the dense groups, then (only
    if any) those of the parameters that use the expert process groups.

    Raises:
        RefusedError: A group marks parameters ``is_expert_parallel`` other than
            the arguments put them in the expert process groups: a routed
            expert's parameter otherwise than :func:`uses_expert_process_groups`
            says, or any other parameter at all (FC-R13).
    """
    expected = uses_expert_process_groups(args)
    wrong = [
        name
        for group in groups
        for name in group.params
        if bool(group.hparams.get(EXPERT_PARALLEL, False))
        != (expected and ROUTED_EXPERTS.search(name) is not None)
    ]
    if wrong:
        sizes = ", ".join(
            f"args.{name}={args.get(name, None)}"
            for name in (
                "expert_model_parallel_size",
                "expert_tensor_parallel_size",
                "tensor_model_parallel_size",
            )
        )
        what = "its routed experts' parameters" if expected else "no parameter"
        raise Refusal(
            errors.FC_R13,
            f"the parameter groups set {EXPERT_PARALLEL} otherwise than a job of the store's "
            f"arguments, which puts {what} in the expert process groups ({sizes})",
            keys=tuple(wrong),
            hint="check that the arguments belong to this checkpoint",
        ).to_error()
    dense = tuple(group for group in groups if not group.hparams.get(EXPERT_PARALLEL, False))
    experts = tuple(group for group in groups if group.hparams.get(EXPERT_PARALLEL, False))
    return (dense, experts) if experts else (dense,)


def optimizer_common_state(chained: Sequence[Sequence[ParamGroup]]) -> dict[Any, Any]:
    """The ``optimizer`` entry of a fully reshardable torch_dist store's common state.

    Each optimizer's ``{'optimizer': {'param_groups': [...]}, 'param_state_sharding_type':
    'fully_reshardable'}`` (``DistributedOptimizer.state_dict`` without the
    groups' ``params``), keyed by its index in the chain when there are
    several (``ChainedOptimizer.sharded_state_dict``).
    """
    states = [
        {
            "optimizer": {"param_groups": [dict(group.hparams) for group in groups]},
            "param_state_sharding_type": FULLY_RESHARDABLE,
        }
        for groups in chained
    ]
    return states[0] if len(states) == 1 else dict(enumerate(states))
