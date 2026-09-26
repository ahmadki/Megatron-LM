# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The key grammar of model tensors: blocks, layer indices, paths and leaves.

The grammar and every pattern over model keys that the conversion layers use
(block classifiers, refusal detectors, sentinels and rules) live in this
module, so that they agree on one grammar; each format adapter owns its
envelope and its section rules.

A *model-relative key* is a model tensor's key without its format envelope
(torch_dist keys are bare; fsdp_dtensor keys lose their ``model.module.``
prefix). Inside a layered block it reads ``<block>.layers.<index>.<path>.<leaf>``,
elsewhere ``<path>.<leaf>``:

* ``block``: the dotted prefix of the block's layer list, e.g. ``decoder``;
* ``index``: the layer index as the format stores it (the two formats may
  number the layers of a block differently, see ``core/blocks.py``), written
  in canonical decimal (no sign, no leading zero);
* ``path``: the module path inside the layer, or from the model root outside
  blocks, e.g. ``self_attention.linear_qkv``; it may be empty;
* ``leaf``: the parameter or buffer name, e.g. ``weight`` or ``layer_norm_weight``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LAYERS = "layers"
"""Name of the layer list of a block (``nn.ModuleList`` attribute)."""

_INDEX_AND_INNER = re.compile(r"(?P<index>0|[1-9][0-9]*)\.(?P<inner>.+)")


class KeyGrammarError(ValueError):
    """A key does not follow the grammar it is parsed with."""


@dataclass(frozen=True, slots=True)
class LayerKey:
    """A model-relative key inside a block's layer list.

    Attributes:
        block: The block prefix, e.g. ``decoder``.
        index: The stored layer index.
        inner: The rest of the key inside the layer, ``<path>.<leaf>``.
    """

    block: str
    index: int
    inner: str

    def render(self) -> str:
        """The key: ``<block>.layers.<index>.<inner>``."""
        return join_layer_key(self.block, self.index, self.inner)


def split_layer_key(key: str, block: str) -> LayerKey | None:
    """Parse ``key`` as a key inside the layer list of ``block``.

    Returns:
        The parsed key, or ``None`` when ``key`` is not under ``<block>.layers.``.

    Raises:
        KeyGrammarError: ``key`` is under ``<block>.layers.`` but the segment
            after it is not a canonical layer index followed by a name.
    """
    head = f"{block}.{LAYERS}."
    if not key.startswith(head):
        return None
    match = _INDEX_AND_INNER.fullmatch(key[len(head) :])
    if match is None:
        raise KeyGrammarError(f"{key}: expected {head}<index>.<name> with a canonical index")
    return LayerKey(block, int(match.group("index")), match.group("inner"))


def join_layer_key(block: str, index: int, inner: str) -> str:
    """The model-relative key ``<block>.layers.<index>.<inner>``."""
    if index < 0:
        raise KeyGrammarError(f"layer index {index} of {block} is negative")
    return f"{block}.{LAYERS}.{index}.{inner}"


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


STACKED_LAYERS = re.compile(r"(?:^|\.)layers\.(?![0-9]+\.)")
"""A ``layers.`` segment not followed by a layer index: a tensor stacked over layers."""

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

FC1_WEIGHT = re.compile(r"(?:^|\.)linear_fc1\.weight$")
"""The weight of an MLP's first projection (a candidate gated linear unit)."""


def fc2_weight_of(fc1_key: str) -> str:
    """The key of the ``linear_fc2.weight`` beside a ``linear_fc1.weight`` key."""
    return fc1_key[: -len("fc1.weight")] + "fc2.weight"
