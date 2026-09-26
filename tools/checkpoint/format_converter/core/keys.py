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
