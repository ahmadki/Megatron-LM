# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The key grammar of model tensors: blocks, layer indices, paths and leaves.

The grammar and every pattern over model keys that the conversion layers use
(block classifiers, refusal detectors, sentinels and rules) live in this
module, so that they agree on one grammar; each format adapter owns its
envelope and its section rules.

A *model-relative key* is a model tensor's key without its format envelope
(torch_dist keys are bare; fsdp_dtensor keys lose their ``model.module.``
prefix). Inside a layered block it reads ``<block>.layers.<index>.<path>.<leaf>``
(one layer), or ``<block>.layers.<path>.<leaf>`` for a tensor stacked over all
the layers of the block (torch_dist only); elsewhere it reads ``<path>.<leaf>``:

* ``block``: the dotted prefix of the block's layer list, e.g. ``decoder``;
* ``index``: the layer index as the format stores it (the two formats may
  number the layers of a block differently, see ``core/blocks.py``), written
  in canonical decimal (no sign, no leading zero);
* ``path``: the module path inside the layer, or from the model root outside
  blocks, e.g. ``self_attention.linear_qkv``; it may be empty;
* ``leaf``: the parameter or buffer name, e.g. ``weight`` or ``layer_norm_weight``.

Blocks nest: a layer may hold a layer list of its own (a multi-token
prediction depth holds the layers of its inner model, e.g.
``mtp.layers.0.mtp_model_layer.layers.2.<name>``). A key belongs to the
*innermost* block whose layer list holds it: the longest block prefix, and
only that block's layer segment is interpreted; the segments of the blocks
around it are part of the prefix and kept as they are. Blocks of different
model roots never share keys (``vision_model.decoder`` and
``language_model.decoder`` are different prefixes).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

LAYERS = "layers"
"""Name of the layer list of a block (``nn.ModuleList`` attribute)."""

LAYER_SEGMENT = f".{LAYERS}."
"""The segment that separates a block prefix from what its layer list holds."""

_INDEX_AND_INNER = re.compile(r"(?P<index>0|[1-9][0-9]*)\.(?P<inner>.+)")
_NAME_START = re.compile(r"[A-Za-z_]")


class KeyGrammarError(ValueError):
    """A key does not follow the grammar it is parsed with."""


@dataclass(frozen=True, slots=True)
class LayerKey:
    """A model-relative key inside a block's layer list.

    Attributes:
        block: The block prefix, e.g. ``decoder``.
        index: The stored layer index, or ``None`` for a tensor stacked over
            every layer of the block (``<block>.layers.<inner>``).
        inner: The rest of the key inside the layer, ``<path>.<leaf>``.
    """

    block: str
    index: int | None
    inner: str

    @property
    def stacked(self) -> bool:
        """Whether the key holds every layer of the block (no layer index)."""
        return self.index is None

    def render(self) -> str:
        """The key: ``<block>.layers.<index>.<inner>`` (``<block>.layers.<inner>`` if stacked)."""
        return join_layer_key(self.block, self.index, self.inner)


def split_layer_key(key: str, block: str) -> LayerKey | None:
    """Parse ``key`` as a key inside the layer list of ``block``.

    Returns:
        The parsed key, or ``None`` when ``key`` is not under ``<block>.layers.``.

    Raises:
        KeyGrammarError: ``key`` is under ``<block>.layers.`` but the segment
            after it is neither a canonical layer index followed by a name nor
            a name (a stacked tensor).
    """
    head = block + LAYER_SEGMENT
    if not key.startswith(head):
        return None
    return _parse_layers(key, block, key[len(head) :])


def innermost_layer_key(key: str) -> LayerKey | None:
    """Parse ``key`` at its last layer segment: the innermost layer list that holds it.

    This finds the blocks a store holds without knowing them in advance.

    Returns:
        The parsed key, or ``None`` when ``key`` is in no layer list.

    Raises:
        KeyGrammarError: The segment after the last ``.layers.`` is malformed
            (see :func:`split_layer_key`).
    """
    at = key.rfind(LAYER_SEGMENT)
    if at <= 0:
        return None
    return _parse_layers(key, key[:at], key[at + len(LAYER_SEGMENT) :])


def attribute_layer_key(key: str, blocks: Iterable[str]) -> LayerKey | None:
    """Parse ``key`` inside the innermost of ``blocks`` whose layer list holds it.

    The innermost block is the one with the longest prefix; the layer segments
    of the blocks around it stay part of that prefix.

    Returns:
        The parsed key, or ``None`` when no block's layer list holds ``key``.

    Raises:
        KeyGrammarError: The innermost block's layer segment is malformed.
    """
    holders = [block for block in blocks if key.startswith(block + LAYER_SEGMENT)]
    if not holders:
        return None
    return split_layer_key(key, max(holders, key=len))


def _parse_layers(key: str, block: str, rest: str) -> LayerKey:
    match = _INDEX_AND_INNER.fullmatch(rest)
    if match is not None:
        return LayerKey(block, int(match.group("index")), match.group("inner"))
    if _NAME_START.match(rest) is None:
        raise KeyGrammarError(
            f"{key}: expected {block}{LAYER_SEGMENT}<index>.<name> with a canonical index, or "
            f"{block}{LAYER_SEGMENT}<name> for a tensor stacked over the layers"
        )
    return LayerKey(block, None, rest)


def join_layer_key(block: str, index: int | None, inner: str) -> str:
    """The model-relative key ``<block>.layers.<index>.<inner>`` (``index`` ``None``: stacked)."""
    if index is None:
        return f"{block}{LAYER_SEGMENT}{inner}"
    if index < 0:
        raise KeyGrammarError(f"layer index {index} of {block} is negative")
    return f"{block}{LAYER_SEGMENT}{index}.{inner}"


def split_leaf(name: str) -> tuple[str, str]:
    """Split ``<path>.<leaf>`` into ``(path, leaf)``; the path is empty for a bare leaf.

    Raises:
        KeyGrammarError: The name is empty or ends with a dot.
    """
    path, _, leaf = name.rpartition(".")
    if not leaf:
        raise KeyGrammarError(f"{name!r} has no leaf name")
    return path, leaf


def join_leaf(path: str, leaf: str) -> str:
    """``<path>.<leaf>``, or ``leaf`` alone when the path is empty."""
    return f"{path}.{leaf}" if path else leaf


def root_of(key: str) -> str:
    """The first dotted segment of a model-relative key (``embedding``, ``decoder``, ...)."""
    return key.partition(".")[0]


FACTORY_PART = re.compile(r"\.(?:weight|bias)\.[A-Za-z_][A-Za-z0-9_]*$")
"""A torch_dist factory sub-key: one part of a parameter stored as parts (``in_proj.weight.z``)."""

GLU_HALF = re.compile(r"\.(?:weight|bias)[0-9]*_[wv]$")
"""An fsdp_dtensor SwiGLU half of ``linear_fc1`` (``weight_w``, ``weight3_v``, ``bias_w``)."""

FUSED_MIXER = re.compile(
    r"\.(?:mixer|self_attention)\."
    r"(?:in_proj\.(?:weight|bias)|conv1d(?:_weight|_bias|\.weight|\.bias))$"
)
"""A fused state-space or linear-attention projection as fsdp_dtensor stores it."""

EXPERT_PARAM = re.compile(r"(?:^|\.)(?:experts|shared_experts|router)\.")
"""A parameter or buffer of a mixture-of-experts layer."""

MIXER_PARAM = re.compile(r"(?:^|\.)mixer\.")
"""A parameter of a state-space mixer (Mamba-2 or Gated DeltaProduct)."""

GDP_PART = re.compile(r"\.mixer\.(?:in_proj|conv1d)\.(?:weight|bias)\.(?:[VKb][0-9]+|Q|a)$")
"""A torch_dist part of a Gated DeltaProduct ``in_proj`` or ``conv1d``."""

GDN_PARAM = re.compile(r"\.self_attention\.(?:in_proj|conv1d)\.")
"""A projection of a Gated DeltaNet attention (standard attention has ``linear_qkv``)."""

MLA_PARAM = re.compile(
    r"\.self_attention\.(?:linear_q_proj|linear_(?:q|kv|qkv)_(?:down|up)_proj)\."
)
"""A projection of multi-latent attention."""

TE_OP_FUSER_MLP = re.compile(r"\.mlp\.[0-9]+\.")
"""An MLP built with Transformer Engine's operation fuser (``mlp.<i>.basic_ops...``)."""

EXTRA_STATE = re.compile(r"\._extra_state\d*(/|$)")
"""A Transformer Engine ``_extra_state`` key (``_extra_state<i>`` when grouped; torch_dist
stores it as a tensor or a ``/shard_*`` byte item)."""

FC1_WEIGHT = re.compile(r"(?:^|\.)linear_fc1\.weight$")
"""The weight of an MLP's first projection (a candidate gated linear unit)."""


def fc2_weight_of(fc1_key: str) -> str:
    """The key of the ``linear_fc2.weight`` beside a ``linear_fc1.weight`` key."""
    return fc1_key[: -len("fc1.weight")] + "fc2.weight"
