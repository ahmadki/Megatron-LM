# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-relative keys to logical ids and back, through the blocks' layer numbering.

:func:`decompose` parses a source model key with the key grammar, attributes
it to the innermost block whose layer list holds it (the longest matching
block prefix; the layer segments of enclosing blocks stay in that prefix) and
maps the stored layer index to the dense index; a torch_dist key stacked over
the layers of a block becomes one slice per layer (``[l:l+1, ...]``, whose
logical tensor has the layer's shape). Keys outside every block keep their
path. :func:`compose` is the inverse for the target format: it places each
layer of a block that torch_dist stores stacked at ``[l:l+1, ...]`` of the
stacked tensor ``<block>.layers.<name>``.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..engine.metadata import TensorEntry
from ..engine.regions import Region
from ..errors import InternalError
from .blocks import BlockLayout
from .keys import KeyGrammarError, attribute_layer_key, join_layer_key, split_leaf
from .logical import LogicalId, LogicalSlice, OutputSlice
from .names import TORCH_DIST

STACKED_AXES = 1
"""Leading axes a tensor stacked over layers has beyond one layer's shape."""


def decompose(
    relative: str, entry: TensorEntry, blocks: Sequence[BlockLayout], fmt: str
) -> list[LogicalSlice]:
    """The logical slices of the stored tensor ``entry`` of a store of format ``fmt``.

    Args:
        relative: The model-relative key of ``entry``.
        entry: The stored tensor.
        blocks: The model's blocks.
        fmt: The store's format.

    Returns:
        One slice holding the whole tensor, or one per layer for a tensor
        stacked over the layers of a block.

    Raises:
        KeyGrammarError: The key's layer segment is malformed, its form
            (stacked or per layer) is not the one the format stores the block
            in, its layer index is not one the format stores for the block,
            a stacked tensor does not hold every layer, or the key has no leaf.
    """
    by_prefix = {block.prefix: block for block in blocks}
    layer_key = attribute_layer_key(relative, by_prefix)
    whole = Region.full(entry.shape)
    if layer_key is None:
        path, leaf = split_leaf(relative)
        return [LogicalSlice(LogicalId(None, None, path, leaf), entry.key, whole, entry.dtype)]
    block = by_prefix[layer_key.block]
    path, leaf = split_leaf(layer_key.inner)
    if layer_key.stacked != (fmt == TORCH_DIST and block.stacked):
        form = "stacked over the layers" if layer_key.stacked else "one layer"
        raise KeyGrammarError(
            f"{relative} holds {form} of {block.prefix}, which {fmt} stores "
            + ("stacked" if fmt == TORCH_DIST and block.stacked else "one layer per key")
        )
    if layer_key.index is not None:
        lid = LogicalId(block.prefix, block.dense_index(layer_key.index, fmt), path, leaf)
        return [LogicalSlice(lid, entry.key, whole, entry.dtype)]
    if entry.shape[:STACKED_AXES] != (block.num_modules,):
        raise KeyGrammarError(
            f"{relative} of shape {entry.shape} does not hold the {block.num_modules} layers "
            f"of {block.prefix}"
        )
    rest = entry.shape[STACKED_AXES:]
    return [
        LogicalSlice(
            LogicalId(block.prefix, block.dense_index(stored, fmt), path, leaf),
            entry.key,
            Region((stored,) + (0,) * len(rest), (1, *rest)),
            entry.dtype,
            squeezed=STACKED_AXES,
        )
        for stored in range(block.num_modules)
    ]


def compose(
    lid: LogicalId, shape: tuple[int, ...], blocks: Sequence[BlockLayout], fmt: str
) -> OutputSlice:
    """Where the logical tensor ``lid`` of shape ``shape`` goes in a store of format ``fmt``.

    Returns:
        The slice with the model-relative key: the whole tensor, or its layer
        ``[l:l+1, ...]`` of a tensor stacked over the layers of a block that
        torch_dist stores stacked (``l`` its torch_dist index).

    Raises:
        InternalError: ``lid`` names a block that is not in ``blocks``.
    """
    whole = Region.full(shape)
    if lid.block is None or lid.layer is None:
        return OutputSlice(lid.name, shape, whole)
    by_prefix = {block.prefix: block for block in blocks}
    if lid.block not in by_prefix:
        raise InternalError(f"{lid} names an unknown block")
    block = by_prefix[lid.block]
    stored = block.stored_index(lid.layer, fmt)
    if not (fmt == TORCH_DIST and block.stacked):
        return OutputSlice(join_layer_key(block.prefix, stored, lid.name), shape, whole)
    return OutputSlice(
        join_layer_key(block.prefix, None, lid.name),
        (block.num_modules, *shape),
        Region((stored,) + whole.offsets, (1, *shape)),
        squeezed=STACKED_AXES,
    )
