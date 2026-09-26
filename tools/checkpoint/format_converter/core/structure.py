# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-relative keys to logical ids and back, through the blocks' layer numbering.

:func:`decompose` parses a source model key with the key grammar, attributes
it to the block whose layer list it is in (the longest matching block prefix)
and maps the stored layer index to the dense index; keys outside every block
keep their path. :func:`compose` is the inverse for the target format.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..errors import InternalError
from .blocks import BlockLayout
from .keys import join_layer_key, split_layer_key, split_leaf
from .logical import LogicalId


def decompose(relative: str, blocks: Sequence[BlockLayout], fmt: str) -> LogicalId:
    """The logical id of the model-relative key ``relative`` of a store of format ``fmt``.

    Raises:
        KeyGrammarError: The key's layer segment is malformed, its layer index
            is not one the format stores for the block, or it has no leaf.
    """
    for block in sorted(blocks, key=lambda b: len(b.prefix), reverse=True):
        layer_key = split_layer_key(relative, block.prefix)
        if layer_key is not None:
            path, leaf = split_leaf(layer_key.inner)
            return LogicalId(block.prefix, block.dense_index(layer_key.index, fmt), path, leaf)
    path, leaf = split_leaf(relative)
    return LogicalId(None, None, path, leaf)


def compose(lid: LogicalId, blocks: Sequence[BlockLayout], fmt: str) -> str:
    """The model-relative key of ``lid`` in a store of format ``fmt``.

    Raises:
        InternalError: ``lid`` names a block that is not in ``blocks``.
    """
    if lid.block is None or lid.layer is None:
        return lid.name
    by_prefix = {block.prefix: block for block in blocks}
    if lid.block not in by_prefix:
        raise InternalError(f"{lid} names an unknown block")
    block = by_prefix[lid.block]
    return join_layer_key(block.prefix, block.stored_index(lid.layer, fmt), lid.name)
