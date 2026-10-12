# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The optimizer state of a Megatron-FSDP v1 fsdp_dtensor store, parsed from its keys.

A Megatron-FSDP job saves its distributed optimizer's state
(``DistributedOptimizer.sharded_param_state_fsdp_dtensor``) as

* ``optimizer.state.<depth><name>.<leaf>``: one tensor per parameter and
  state: Adam's moments ``exp_avg`` and ``exp_avg_sq`` (and a ``step`` per
  parameter when the optimizer is ``torch.optim.Adam``). ``<name>`` is the
  parameter's model key without ``model.module.``: the save handlers rename the
  optimizer state as they rename the model (SwiGLU halves, global expert
  indices, the GPT MTP inner layer, unfused multi-latent attention);
* ``optimizer.param_to_group_meta.<depth><live>.<hparam>``: one byte item per
  parameter and hyperparameter of its parameter group (every key of the group
  but ``params``). ``<live>`` is the parameter's *live* name
  (``named_parameters`` with global expert indices,
  ``DistributedOptimizer._param_name``), which the handlers leave alone
  (:func:`live_names`).

``<depth>`` is one ``module.`` per wrapper around the model:
``module.module.module.`` for bf16 and fp16 jobs (``Float16Module``) and
``module.module.`` for fp32 ones (the architecture context derives it from the
arguments).

Names hold dots, so a key is split with its ``Metadata.planner_data`` path
``('optimizer', 'state', '<depth><name>', '<leaf>')``. A key without such a
path (a store whose paths are the keys themselves) is split strictly: its
leaf must be a known one and the rest a name the model section gives; a key
is never split at its last dot on trust.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field

from .. import errors
from ..core.args_view import ArgsView
from ..core.blocks import MTP_BLOCKS, fused_mla_live_name, gpt_mtp_live_name, is_gpt_mtp
from ..core.hybrid_pattern import hybrid_pattern_of
from ..core.keys import (
    INPUT_LAYERNORM,
    MLA_DOWN_PROJ,
    KeyGrammarError,
    join_layer_key,
    split_glu_half,
    split_layer_key,
)
from ..core.names import FSDP_GROUP_META, FSDP_OPTIMIZER_STATE, MOMENTS
from ..core.param_groups import IDENTIFIER_KEYS
from ..engine.metadata import MetadataView, TensorEntry
from ..errors import Refusal, refusal_error

OPTIMIZER = "optimizer"
"""First element of the planner path of every optimizer key."""

STATE, GROUP_META = "state", "param_to_group_meta"
"""Second element of the planner path: per-parameter state tensors, and group hyperparameters."""

STEP = "step"
"""``torch.optim.Adam``'s per-parameter step count (a 0-d tensor)."""

STATE_LEAVES = (*MOMENTS, STEP)
"""Every per-parameter state of Adam."""

KNOWN_HPARAMS = frozenset(
    {
        "amsgrad",
        "betas",
        "bias_correction",
        "capturable",
        "default_config",
        "differentiable",
        "eps",
        "foreach",
        "fused",
        "lr",
        "maximize",
        STEP,
        "weight_decay",
        *IDENTIFIER_KEYS,
    }
)
"""Hyperparameters a key without a planner path may end with: Adam's (``torch.optim.Adam`` and
Transformer Engine's ``FusedAdam``) and Megatron's (``_get_param_groups``, the identifier keys)."""

_MODULE_LEVELS = re.compile(r"(?:module\.)*")


@dataclass(frozen=True)
class FsdpOptimizerState:
    """The optimizer state an fsdp_dtensor store holds.

    Attributes:
        depth: The ``module.`` levels before every parameter name.
        moments: The moment tensors by moment (:data:`MOMENTS`), each by the
            parameter's model key without ``model.module.``.
        steps: ``torch.optim.Adam``'s per-parameter step tensors, by the same names.
        group_meta: The byte item of each hyperparameter, by the parameter's
            live name and the hyperparameter.
    """

    depth: str
    moments: Mapping[str, Mapping[str, TensorEntry]]
    steps: Mapping[str, TensorEntry] = field(default_factory=dict)
    group_meta: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    @property
    def trainable(self) -> frozenset[str]:
        """Model-relative names of the parameters that have optimizer moments."""
        return frozenset(name for names in self.moments.values() for name in names)


def live_names(model: Collection[str], args: ArgsView) -> dict[str, str]:
    """The live name of the parameter behind each model-relative fsdp_dtensor model key.

    The live name is what ``param_to_group_meta`` calls the parameter:

    * a SwiGLU half ``<name>_w`` or ``<name>_v`` is a half of the live ``<name>``;
    * the inner ``transformer_layer`` of a GPT MTP layer is ``mtp_model_layer``;
    * with ``--mla-down-proj-fusion``, multi-latent attention's q and kv
      down-projections are ``linear_qkv_down_proj``, which in a GPT model also
      holds the input layer norm of its layer as ``layer_norm_<leaf>``.

    Every other key is its own live name (the model section and the live
    model name experts by their global indices alike).

    Args:
        model: The model-relative keys of the store's model section.
        args: The checkpoint's training arguments.
    """
    gpt_mla = hybrid_pattern_of(args) is None
    fused_mla = bool(args.get("mla_down_proj_fusion", False))
    mla_layers = frozenset(
        match.group("attention").rpartition(".")[0]
        for key in model
        if (match := MLA_DOWN_PROJ.fullmatch(key)) is not None
    )
    renamed = {}
    for key in model:
        name, _ = split_glu_half(key)
        if fused_mla:
            norm = INPUT_LAYERNORM.fullmatch(name)
            absorbs = gpt_mla and norm is not None and norm.group("layer") in mla_layers
            name = fused_mla_live_name(name, absorbs_norm=absorbs)
        if is_gpt_mtp(args):
            name = _gpt_mtp_live_name(name)
        renamed[key] = name
    return renamed


def _gpt_mtp_live_name(key: str) -> str:
    """``key`` with the inner layer of its GPT MTP layer under its live name, if it is in one."""
    for block in MTP_BLOCKS:
        try:
            layer_key = split_layer_key(key, block)
        except KeyGrammarError:
            continue
        if layer_key is not None and layer_key.index is not None:
            return join_layer_key(block, layer_key.index, gpt_mtp_live_name(layer_key.inner))
    return key


def parse_optimizer_state(
    view: MetadataView, keys: Iterable[str], *, depth: str, live: Mapping[str, str]
) -> FsdpOptimizerState:
    """Split every optimizer key of an fsdp_dtensor store into its parameter and leaf.

    Args:
        view: The store.
        keys: Its optimizer keys (``optimizer.state.*`` tensors and the other
            ``optimizer.*`` items).
        depth: The ``module.`` levels the arguments give parameter names.
        live: The live name of each model-relative model key (:func:`live_names`).

    Raises:
        RefusedError: A key is neither a state tensor nor a hyperparameter byte
            item of a parameter of the model section, or names it at other
            ``module.`` levels than ``depth`` (FC-R13); a state is neither an
            Adam moment nor a step (FC-R16).
    """
    parser = _Parser(view, depth, live)
    for key in sorted(keys):
        parser.add(key)
    return parser.result()


@dataclass
class _Parser:
    """Accumulates the parsed keys and the refusals of :func:`parse_optimizer_state`."""

    view: MetadataView
    depth: str
    live: Mapping[str, str]
    moments: dict[str, dict[str, TensorEntry]] = field(default_factory=dict)
    steps: dict[str, TensorEntry] = field(default_factory=dict)
    group_meta: dict[str, dict[str, str]] = field(default_factory=dict)
    refused: dict[tuple[str, str, str], list[str]] = field(default_factory=dict)

    def add(self, key: str) -> None:
        """Parse one optimizer key (or record why it cannot be)."""
        path = self._path(key)
        if path is None:
            return
        _, section, prefixed, leaf = path
        levels = _MODULE_LEVELS.match(prefixed).group(0)  # type: ignore[union-attr]
        if levels != self.depth:
            self._refuse(
                errors.FC_R13,
                f"optimizer keys name parameters under {levels or 'no'} module. levels, but "
                f"the arguments' parameter dtype gives {self.depth}",
                "check that the arguments belong to this checkpoint",
                key,
            )
            return
        name = prefixed[len(levels) :]
        if section == STATE:
            self._add_state(key, name, leaf)
        else:
            self._add_meta(key, name, leaf)

    def result(self) -> FsdpOptimizerState:
        """The parsed state.

        Raises:
            RefusedError: A key was refused.
        """
        if self.refused:
            raise refusal_error(
                [
                    Refusal(code, message, keys=tuple(keys), hint=hint)
                    for (code, message, hint), keys in self.refused.items()
                ]
            )
        return FsdpOptimizerState(
            self.depth,
            {moment: dict(sorted(self.moments.get(moment, {}).items())) for moment in MOMENTS},
            dict(sorted(self.steps.items())),
            {name: dict(sorted(items.items())) for name, items in sorted(self.group_meta.items())},
        )

    def _path(self, key: str) -> tuple[str, str, str, str] | None:
        """The planner path of ``key``, or its strict split without one (``None``: refused)."""
        recorded = (self.view.planner_data or {}).get(key)
        if isinstance(recorded, tuple) and len(recorded) == 4:
            if recorded[:2] in ((OPTIMIZER, STATE), (OPTIMIZER, GROUP_META)) and (
                ".".join(recorded) == key
            ):
                return recorded
            self._refuse(
                errors.FC_R13,
                "optimizer keys whose planner paths are not ('optimizer', 'state' or "
                "'param_to_group_meta', <parameter>, <leaf>) spelling the key",
                "the store was not written by Megatron-FSDP v1",
                key,
            )
            return None
        path = self._split_strictly(key)
        if path is None:
            self._refuse(
                errors.FC_R13,
                "optimizer keys that are neither optimizer.state.<depth><parameter>.<moment> nor "
                "optimizer.param_to_group_meta.<depth><parameter>.<hyperparameter> of a parameter "
                "of the model section (they have no planner paths to split them by)",
                "the store was not written by Megatron-FSDP v1, or by a version whose optimizer "
                "keys this tool does not know",
                key,
            )
        return path

    def _split_strictly(self, key: str) -> tuple[str, str, str, str] | None:
        """``key`` split as ``(optimizer, section, <levels><name>, leaf)`` with a known leaf and a
        name of the model section, if exactly one split does that."""
        for prefix, section, names, leaves in (
            (FSDP_OPTIMIZER_STATE, STATE, self.live.keys(), STATE_LEAVES),
            (FSDP_GROUP_META, GROUP_META, self.live.values(), KNOWN_HPARAMS),
        ):
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix) :]
            levels = _MODULE_LEVELS.match(rest).group(0)  # type: ignore[union-attr]
            known = frozenset(names)
            splits = [
                (rest[: -len(leaf) - 1], leaf)
                for leaf in leaves
                if rest.endswith(f".{leaf}") and rest[len(levels) : -len(leaf) - 1] in known
            ]
            if len(splits) == 1:
                ((prefixed, leaf),) = splits
                return (OPTIMIZER, section, prefixed, leaf)
        return None

    def _add_state(self, key: str, name: str, leaf: str) -> None:
        if name not in self.live:
            self._unknown_parameter(key, "state")
        elif leaf not in STATE_LEAVES:
            self._refuse(
                errors.FC_R16,
                f"optimizer state other than Adam's ({', '.join(STATE_LEAVES)})",
                "convert a checkpoint of an Adam optimizer, or convert the weights only "
                "(--optimizer drop)",
                key,
            )
        elif key not in self.view.tensors:
            self._wrong_kind(key)
        elif leaf == STEP:
            self.steps[name] = self.view.tensors[key]
        else:
            self.moments.setdefault(leaf, {})[name] = self.view.tensors[key]

    def _add_meta(self, key: str, name: str, leaf: str) -> None:
        if name not in self.live.values():
            self._unknown_parameter(key, "parameter-group")
        elif key not in self.view.byte_items:
            self._wrong_kind(key)
        else:
            self.group_meta.setdefault(name, {})[leaf] = key

    def _unknown_parameter(self, key: str, what: str) -> None:
        self._refuse(
            errors.FC_R13,
            f"optimizer {what} keys of parameters the model section does not hold",
            "check that the store holds one complete save",
            key,
        )

    def _wrong_kind(self, key: str) -> None:
        self._refuse(
            errors.FC_R13,
            "optimizer state stored as byte items, or parameter-group values stored as tensors",
            "the store was not written by Megatron-FSDP v1",
            key,
        )

    def _refuse(self, code: str, message: str, hint: str, key: str) -> None:
        self.refused.setdefault((code, message, hint), []).append(key)
