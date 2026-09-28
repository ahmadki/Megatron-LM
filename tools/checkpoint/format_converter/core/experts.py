# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The routed experts of mixture-of-experts layers and how each format stores them.

A mixture-of-experts layer's experts module (``<...>.mlp.experts``) holds one
``linear_fc1`` and one ``linear_fc2`` weight (and bias) per routed expert
(``core/keys.py`` has the key grammar):

* torch_dist stores each as one tensor over all experts,
  ``<...>.mlp.experts.experts.linear_fc<n>.<leaf>`` of shape ``[E, ...]`` (or
  ``[L, E, ...]`` in a block stacked over its layers), expert ``e`` at the
  global index ``e`` of its expert axis and saved as a chunk of its own
  (``TEGroupedMLP`` and ``SequentialMLP`` write the same keys,
  ``moe/experts.py``);
* fsdp_dtensor stores one tensor per expert under the global index, named as
  the model's experts module names its parameters: *grouped* for
  ``TEGroupedMLP``, *sequential* for ``SequentialMLP``.

:func:`expert_layout` derives the layout from the arguments and checks it
against the store:

* the number of experts is ``args.num_experts``; every torch_dist expert axis
  has that length (FC-R13 otherwise), and every fsdp_dtensor expert tensor is
  stored for exactly the experts ``0..E-1`` (FC-R12 for indices that are not
  contiguous from 0, FC-R13 for another number of experts);
* fsdp_dtensor names the experts as the model's builder builds them:
  Megatron's GPT builders build ``TEGroupedMLP`` with ``--moe-grouped-gemm``
  and ``SequentialMLP`` without, and the hybrid stack spec always builds
  ``TEGroupedMLP``. ``--expert-naming`` chooses the naming instead, of a
  source or of a GPT target (whose ``moe_grouped_gemm`` the conversion then
  sets to match); a hybrid target cannot be named otherwise, since no
  argument changes the experts module its stack spec builds. An fsdp_dtensor
  source that names its experts otherwise is ambiguous (FC-A04): its
  arguments would build other expert modules than the ones that saved it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .. import errors
from ..engine.metadata import TensorEntry
from ..errors import InternalError, Refusal, RefusalError, UsageError
from .args_view import ArgsView
from .blocks import index_ranges
from .hybrid_pattern import hybrid_pattern_of
from .keys import ExpertName, KeyGrammarError, innermost_layer_key, split_expert_name
from .names import AUTO_EXPERT_NAMING, EXPERT_FORMS, GROUPED, SEQUENTIAL, TORCH_DIST

EXPERT_MODULES = {GROUPED: "TEGroupedMLP", SEQUENTIAL: "SequentialMLP"}
"""The experts module that names its parameters in each fsdp_dtensor expert naming."""

_ARGUMENTS_HINT = "check that the arguments belong to this checkpoint"
_COMPLETE_HINT = "check that the store is complete and the arguments belong to it"


@dataclass(frozen=True, slots=True)
class ExpertLayout:
    """How the routed experts of the model's mixture-of-experts layers are stored.

    Attributes:
        num_experts: Routed experts per layer: the length of torch_dist's expert axis.
        fsdp_form: How fsdp_dtensor names them: ``grouped`` or ``sequential``.
        grouped_gemm: The ``moe_grouped_gemm`` argument under which Megatron's
            builder builds the experts module of ``fsdp_form`` (``None``: the
            model's builder does not read it).

    Raises:
        InternalError: ``num_experts`` is not positive, or ``fsdp_form`` is unknown.
    """

    num_experts: int
    fsdp_form: str
    grouped_gemm: bool | None = None

    def __post_init__(self) -> None:
        if self.num_experts < 1:
            raise InternalError(f"an expert layout needs experts, not {self.num_experts}")
        if self.fsdp_form not in EXPERT_FORMS:
            raise InternalError(f"unknown fsdp_dtensor expert naming {self.fsdp_form!r}")

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {
            "num_experts": self.num_experts,
            "fsdp_form": self.fsdp_form,
            "grouped_gemm": self.grouped_gemm,
        }


def expert_layout(
    args: ArgsView, model: Mapping[str, TensorEntry], source: str, naming: str = AUTO_EXPERT_NAMING
) -> tuple[ExpertLayout | None, str]:
    """The model's expert layout, and the evidence for it (``None``: no routed experts).

    Args:
        args: The checkpoint's training arguments.
        model: The source's model tensors by model-relative key.
        source: The source format.
        naming: ``--expert-naming``: ``auto``, ``grouped`` or ``sequential``.

    Raises:
        RefusedError: The store holds routed-expert tensors of another number
            of experts than ``args.num_experts`` (FC-R13), or fsdp_dtensor
            tensors of experts whose indices are not contiguous from 0
            (FC-R12), or names them both ways (FC-R13).
        AmbiguousError: An fsdp_dtensor source names its experts otherwise
            than the arguments (or ``naming``) build them (FC-A04).
        UsageError: ``naming`` asks for fsdp_dtensor names of a hybrid
            model's experts other than the ones its stack spec builds.
    """
    tensors = _expert_tensors(model, source)
    num_experts = args.get("num_experts", None)
    if not num_experts:
        if not tensors:
            return None, f"args.num_experts={num_experts!r} and no routed-expert tensors"
        keys = tuple(sorted(tensors))
        raise _refusal(
            errors.FC_R13,
            f"the store holds {len(keys)} routed-expert tensors, but "
            f"args.num_experts={num_experts!r}",
            keys,
        )
    if source == TORCH_DIST:
        stored = _check_expert_axes(model, tensors, num_experts)
    else:
        stored = _check_expert_indices(tensors, num_experts)
    form, why = _fsdp_form(args, tensors, source, naming)
    grouped_gemm = None if hybrid_pattern_of(args) is not None else form == GROUPED
    layout = ExpertLayout(num_experts, form, grouped_gemm)
    return layout, f"args.num_experts={num_experts}; {stored}; {why}"


def built_expert_form(args: ArgsView) -> tuple[str, str]:
    """The expert naming of the experts module Megatron's builder builds, and why.

    The hybrid stack spec builds ``TEGroupedMLP`` whatever the arguments say
    (``hybrid_layer_specs.py``); the GPT builders build ``TEGroupedMLP`` iff
    ``moe_grouped_gemm`` (``gpt_layer_specs.py``).
    """
    if hybrid_pattern_of(args) is not None:
        return GROUPED, f"the hybrid stack spec builds {EXPERT_MODULES[GROUPED]} experts"
    grouped = bool(args.get("moe_grouped_gemm", False))
    form = GROUPED if grouped else SEQUENTIAL
    return form, f"args.moe_grouped_gemm={grouped} builds {EXPERT_MODULES[form]} experts"


def _expert_tensors(model: Mapping[str, TensorEntry], source: str) -> dict[str, ExpertName]:
    """The model's routed-expert tensors by model-relative key.

    Malformed keys are left out: the planner refuses them with the grammar error.
    """
    tensors = {}
    for key in model:
        try:
            name = split_expert_name(key, source)
        except KeyGrammarError:
            continue
        if name is not None:
            tensors[key] = name
    return tensors


def _check_expert_axes(
    model: Mapping[str, TensorEntry], tensors: Mapping[str, ExpertName], num_experts: int
) -> str:
    """Every torch_dist expert axis holds ``num_experts`` experts (FC-R13); the evidence."""
    lengths = {key: _expert_axis_length(key, model[key]) for key in tensors}
    wrong = sorted(key for key, length in lengths.items() if length not in (None, num_experts))
    if wrong:
        raise _refusal(
            errors.FC_R13,
            f"args.num_experts={num_experts}, but the expert axis of {wrong[0]} holds "
            f"{lengths[wrong[0]]} experts",
            tuple(wrong),
        )
    return f"{len(tensors)} torch_dist tensors hold {num_experts} experts each"


def _expert_axis_length(key: str, entry: TensorEntry) -> int | None:
    """The length of a torch_dist expert tensor's expert axis (after a stacked layer axis).

    ``None`` for a malformed key or a tensor without the axis (the planner refuses both).
    """
    try:
        layer_key = innermost_layer_key(key)
    except KeyGrammarError:
        return None
    axis = int(layer_key is not None and layer_key.stacked)
    return entry.shape[axis] if len(entry.shape) > axis else None


def _check_expert_indices(tensors: Mapping[str, ExpertName], num_experts: int) -> str:
    """Every fsdp_dtensor expert tensor is stored for experts ``0..E-1`` exactly; the evidence.

    Raises:
        RefusedError: A tensor is stored for experts whose indices are not
            contiguous from 0 (FC-R12), or for another number of experts
            (FC-R13).
    """
    groups: dict[str, dict[int, str]] = {}
    for key, name in tensors.items():
        assert name.expert is not None  # fsdp_dtensor names one expert per tensor
        groups.setdefault(f"{name.path}.{name.leaf}", {})[name.expert] = key
    for label, experts in sorted(groups.items()):
        indices = sorted(experts)
        keys = tuple(experts[index] for index in indices)
        if indices != list(range(len(indices))):
            raise _refusal(
                errors.FC_R12,
                f"the store holds {label} for experts {index_ranges(indices)}, not "
                f"0-{num_experts - 1}",
                keys,
                hint=_COMPLETE_HINT,
            )
        if len(indices) != num_experts:
            raise _refusal(
                errors.FC_R13,
                f"args.num_experts={num_experts}, but the store holds {label} for "
                f"{len(indices)} experts ({index_ranges(indices)})",
                keys,
            )
    return (
        f"{len(tensors)} fsdp_dtensor tensors: {len(groups)} expert weights and biases for each "
        f"of experts 0-{num_experts - 1}"
    )


def _fsdp_form(
    args: ArgsView, tensors: Mapping[str, ExpertName], source: str, naming: str
) -> tuple[str, str]:
    """The fsdp_dtensor expert naming, checked against an fsdp_dtensor source; and why.

    Raises:
        RefusedError: The source names its experts both ways (FC-R13).
        AmbiguousError: The source names them otherwise than expected (FC-A04).
        UsageError: ``naming`` asks for other names of a hybrid model's
            experts than its stack spec builds, converting to fsdp_dtensor.
    """
    built, built_why = built_expert_form(args)
    if naming == AUTO_EXPERT_NAMING:
        expected, why = built, built_why
    else:
        expected, why = naming, f"--expert-naming {naming}"
    if source == TORCH_DIST:
        if expected != built and hybrid_pattern_of(args) is not None:
            raise UsageError(
                f"--expert-naming {naming}: {built_why} whatever the arguments say, which "
                f"names them {built}, so no job built from these arguments loads experts "
                f"named {naming}",
                hint="drop --expert-naming (it names the experts of GPT models)",
            )
        return expected, f"fsdp_dtensor names them {expected} ({why})"
    by_form: dict[str, list[str]] = {}
    for key, name in sorted(tensors.items()):
        by_form.setdefault(name.form or "", []).append(key)
    if len(by_form) > 1:
        raise _refusal(
            errors.FC_R13,
            "the store names its routed experts both grouped and sequential",
            tuple(keys[0] for keys in by_form.values()),
        )
    observed = next(iter(by_form), expected)
    if observed != expected:
        raise Refusal(
            errors.FC_A04,
            f"the store names its routed experts {observed} ({EXPERT_MODULES[observed]}), but "
            f"{why}, which names them {expected}",
            keys=tuple(by_form[observed][:1]),
            hint=f"pass --expert-naming {observed} if the store is right",
        ).to_error()
    return observed, f"the store names them {observed} ({why})"


def _refusal(
    code: str, message: str, keys: tuple[str, ...], *, hint: str = _ARGUMENTS_HINT
) -> RefusalError:
    return Refusal(code, message, keys=keys, hint=hint).to_error()
