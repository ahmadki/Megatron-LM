# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gated linear units: which MLP first projections are gated, derived from the stored shapes.

A gated MLP (``gated_linear_unit``: SwiGLU with ``--swiglu``, GEGLU with
``--quick-geglu``) computes ``act(x W^T) * (x V^T)``: its first projection
``linear_fc1`` holds the gate ``W`` and the value ``V`` projections along its
first axis, so its weight has twice the rows its ``linear_fc2`` weight has
inputs. A plain MLP's first projection has as many rows as the second has
inputs. Every first projection of the model is classified this way: the
``linear_fc1`` of a dense MLP, of shared experts, of each routed expert
(per expert, as each has its own tensor in fsdp_dtensor) and of MLPs outside
the layered blocks (a vision-language model's projection). The global shapes
are compared, so the tensor-parallel size does not matter.

The shapes are cross-checked against the configuration:

* the MLPs of one layer share its configuration, so the first projections of
  a layer (its dense MLP, or its shared and routed experts) are all gated or
  all plain;
* the language model's configuration is gated iff ``args.swiglu`` or
  ``args.quick_geglu`` is set (Megatron's ``core_transformer_config_from_args``),
  so its decoder's first projections must be gated exactly then. Other
  blocks (a vision encoder) and MLPs outside blocks have configurations of
  their own.

A contradiction is refused (FC-R12), as is a first projection of a layer
without its second, whose gatedness cannot be told. A ``linear_fc1`` outside
every layer with no ``linear_fc2`` beside it is no MLP's first projection but
a plain linear layer (a vision encoder's patch merger holds one): it is left
out, and converted unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .. import errors
from ..engine.metadata import TensorEntry
from ..errors import Refusal, RefusalError
from .args_view import ArgsView
from .blocks import BlockLayout
from .keys import LINEAR_FC_WEIGHT, KeyGrammarError, fc2_path_of, is_fc1_path
from .logical import LogicalId
from .structure import decompose

WEIGHT, BIAS = "weight", "bias"
GLU_ARGS = ("swiglu", "quick_geglu")
"""Arguments that make Megatron build the language model's MLPs gated."""

GATED, PLAIN = "gated", "plain"
_ARGUMENTS_HINT = "check that the arguments belong to this checkpoint"


@dataclass(frozen=True, slots=True)
class Fc1:
    """One MLP first projection: of a dense MLP, of shared experts, or of one routed expert.

    Attributes:
        lid: The logical id of its weight.
        key: The stored source key that holds its weight.
        rows: Rows of its weight (both projections of a gated unit).
        gated: Whether it holds a gate and a value projection.
    """

    lid: LogicalId
    key: str
    rows: int
    gated: bool

    @property
    def group(self) -> str:
        """Where it is, for summaries: its block, or the module holding it outside blocks."""
        return self.lid.block if self.lid.block is not None else self.lid.path.rpartition(".")[0]


@dataclass(frozen=True)
class GluLayout:
    """The model's MLP first projections, by the logical id of their weight.

    Attributes:
        fc1s: Every first projection.
    """

    fc1s: Mapping[LogicalId, Fc1]

    def of(self, lid: LogicalId) -> Fc1 | None:
        """The first projection whose weight or bias ``lid`` is (``None`` for other tensors)."""
        if lid.leaf not in (WEIGHT, BIAS) or not is_fc1_path(lid.path):
            return None
        return self.fc1s.get(LogicalId(lid.block, lid.layer, lid.path, WEIGHT, lid.expert))

    def to_json(self) -> dict[str, dict[str, int]]:
        """The number of gated and plain first projections of each block (or module), sorted."""
        counts: dict[str, dict[str, int]] = {}
        for fc1 in self.fc1s.values():
            kinds = counts.setdefault(fc1.group, {})
            kind = GATED if fc1.gated else PLAIN
            kinds[kind] = kinds.get(kind, 0) + 1
        return {group: dict(sorted(kinds.items())) for group, kinds in sorted(counts.items())}


def glu_layout(
    args: ArgsView,
    model: Mapping[str, TensorEntry],
    blocks: Sequence[BlockLayout],
    source: str,
    lm_decoder: str,
) -> tuple[GluLayout, str]:
    """The model's first projections and whether each is gated, and the evidence.

    Args:
        args: The checkpoint's training arguments.
        model: The source's model tensors by model-relative key.
        blocks: The model's blocks.
        source: The source format.
        lm_decoder: The language model's decoder block.

    Raises:
        RefusedError: A first projection of a layer has no second projection
            beside it, the first projections of a layer are not all gated or
            all plain, or the language model's decoder is gated otherwise than
            its arguments say (FC-R12).
    """
    weights = _fc_weights(model, blocks, source)
    fc1s, plain_linears = {}, []
    for lid, (key, shape) in weights.items():
        if not is_fc1_path(lid.path):
            continue
        fc2 = weights.get(
            LogicalId(lid.block, lid.layer, fc2_path_of(lid.path), WEIGHT, lid.expert)
        )
        if fc2 is None:
            if _is_plain_linear(lid):
                plain_linears.append(key)
                continue
            raise _refusal(
                f"{lid} has no {fc2_path_of(lid.path)} weight beside it, so whether it is a gated "
                "linear unit cannot be told",
                (key,),
            )
        fc1s[lid] = Fc1(lid, key, shape[0], shape[0] == 2 * fc2[1][-1])
    _check_layers(fc1s.values())
    lm_gated = _check_language_model(args, fc1s.values(), lm_decoder)
    layout = GluLayout(dict(sorted(fc1s.items(), key=lambda item: str(item[0]))))
    flags = ", ".join(f"args.{name}={args.get(name, False)}" for name in GLU_ARGS)
    evidence = (
        f"{len(fc1s)} linear_fc1 weights against their linear_fc2 inputs (gated: twice as many "
        f"rows); {flags} make the language model's MLPs (in {lm_decoder}) "
        + ("gated" if lm_gated else "plain")
        + _plain_linears_note(plain_linears)
    )
    return layout, evidence


def _is_plain_linear(lid: LogicalId) -> bool:
    """Whether a first projection with no second projection beside it is a plain linear layer:
    one outside every layer, which is no MLP's."""
    return lid.block is None


def _plain_linears_note(keys: Sequence[str]) -> str:
    """The evidence about the plain linear layers named ``linear_fc1`` (none: empty)."""
    if not keys:
        return ""
    return (
        f"; {len(keys)} linear_fc1 outside every layer with no linear_fc2 beside it (plain "
        f"linear layers, converted unchanged): {', '.join(sorted(keys))}"
    )


def _fc_weights(
    model: Mapping[str, TensorEntry], blocks: Sequence[BlockLayout], source: str
) -> dict[LogicalId, tuple[str, tuple[int, ...]]]:
    """``{logical id: (stored key, shape)}`` of every first and second projection weight.

    Malformed keys are left out (the planner refuses them with the grammar error).
    """
    weights = {}
    for relative, entry in model.items():
        if LINEAR_FC_WEIGHT.search(relative) is None:
            continue
        try:
            slices = decompose(relative, entry, blocks, source)
        except KeyGrammarError:
            continue
        for piece in slices:
            if piece.shape:
                weights[piece.lid] = (piece.key, piece.shape)
    return weights


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


def _check_language_model(args: ArgsView, fc1s: Iterable[Fc1], lm_decoder: str) -> bool:
    """The language model's decoder is gated iff its arguments say so (FC-R12); whether it is."""
    gated = any(bool(args.get(name, False)) for name in GLU_ARGS)
    wrong = sorted({fc1.key for fc1 in fc1s if fc1.lid.block == lm_decoder and fc1.gated != gated})
    if wrong:
        flags = " and ".join(f"args.{name}={args.get(name, False)}" for name in GLU_ARGS)
        observed = "plain" if gated else "gated (twice the rows linear_fc2 has inputs)"
        raise _refusal(
            f"the language model's linear_fc1 weights (in {lm_decoder}) are {observed}, but "
            f"{flags} make its MLPs {'gated' if gated else 'plain'}",
            tuple(wrong),
        )
    return gated


def _refusal(message: str, keys: tuple[str, ...]) -> RefusalError:
    return Refusal(errors.FC_R12, message, keys=keys, hint=_ARGUMENTS_HINT).to_error()
