# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``TransformerBlock`` layer lists and how torch_dist stacks them.

A ``TransformerBlock`` (a GPT decoder, a vision encoder) is stored by
torch_dist in one of two layouts (``transformer_block.py``,
``sharded_state_dict``):

* *stacked*: one key per tensor, ``<block>.layers.<path>.<leaf>`` of shape
  ``[L, ...]``, layer ``l`` at ``[l:l+1, ...]`` (each layer is its own chunk);
* *per layer*: ``<block>.layers.<l>.<path>.<leaf>``, ``l`` the global layer
  index (``layer_number - 1``) at any pipeline-parallel size.

fsdp_dtensor stores one key per layer under the ``nn.ModuleList`` index,
which is the global index (an fsdp_dtensor store has one pipeline stage), so
both formats number the layers identically.

Converting a torch_dist store, the layout is observed from the keys: every
tensor of a stacked block must have the same number of layers along its first
axis, a per-layer block must hold layers ``0..L-1``, a block holds either form
but not both, and the language model's decoder must have ``args.num_layers``
layers; anything else is refused (FC-R12).
"""

from __future__ import annotations

from .. import errors
from ..errors import InternalError, Refusal, RefusalError
from .blocks import (
    LM_DECODER,
    PER_LAYER,
    STACKED,
    TRANSFORMER_BLOCK,
    BlockCensus,
    BlockFacts,
    BlockLayout,
    index_ranges,
)
from .keys import join_layer_key
from .names import TORCH_DIST


def transformer_block(facts: BlockFacts, census: BlockCensus) -> BlockLayout:
    """The layout of a ``TransformerBlock`` the source holds.

    Raises:
        RefusedError: The torch_dist source stores the block inconsistently (FC-R12).
        InternalError: The source is not torch_dist (the refusal fence lets no
            ``TransformerBlock`` of an fsdp_dtensor source through).
    """
    if facts.source != TORCH_DIST:
        raise InternalError(f"{census.prefix}: the torch_dist layout of this block is not derived")
    return _observed_layout(facts, census)


def _observed_layout(facts: BlockFacts, census: BlockCensus) -> BlockLayout:
    """The layout a torch_dist source stores the block in."""
    if census.stacked and census.layers:
        per_layer = sorted(
            join_layer_key(census.prefix, index, inner)
            for index, inners in census.layers.items()
            for inner in inners
        )
        raise _refusal(
            f"{census.prefix} holds both tensors stacked over its layers and tensors of single "
            f"layers ({index_ranges(census.indices)})",
            census.stacked_keys() + tuple(per_layer),
        )
    if census.stacked:
        count = _stacked_count(census)
        layout = STACKED
        evidence = f"torch_dist stores {len(census.stacked)} tensors stacked over {count} layers"
    else:
        count = _contiguous_count(census)
        layout = PER_LAYER
        evidence = f"torch_dist stores {count} layers one per key"
    _check_decoder_length(facts, census, count)
    return BlockLayout(
        prefix=census.prefix,
        kind=TRANSFORMER_BLOCK,
        num_modules=count,
        td_layout=layout,
        dense_to_td=tuple(range(count)),
        symbols=None,
        evidence=evidence,
    )


def _stacked_count(census: BlockCensus) -> int:
    """The common length of the stacked tensors' first axis (the number of layers)."""
    lengths = {
        inner: entry.shape[0] if entry.shape else 0 for inner, entry in census.stacked.items()
    }
    distinct = sorted(set(lengths.values()))
    if len(distinct) != 1 or distinct[0] < 1:
        found = ", ".join(f"{inner} {length}" for inner, length in sorted(lengths.items()))
        raise _refusal(
            f"the tensors stacked over the layers of {census.prefix} disagree on the number of "
            f"layers ({found})",
            census.stacked_keys(),
        )
    return distinct[0]


def _contiguous_count(census: BlockCensus) -> int:
    """The number of layers of a per-layer block, whose indices must be ``0..L-1``."""
    indices = census.indices
    if indices != tuple(range(len(indices))):
        raise _refusal(
            f"{census.prefix} stores layers {index_ranges(indices)}, not 0-{len(indices) - 1}", ()
        )
    return len(indices)


def _check_decoder_length(facts: BlockFacts, census: BlockCensus, count: int) -> None:
    """The language model's decoder has ``args.num_layers`` layers."""
    if census.prefix != LM_DECODER:
        return
    num_layers = facts.args.get("num_layers", None)
    if num_layers is not None and num_layers != count:
        raise _refusal(
            f"args.num_layers={num_layers} but the store holds {count} layers of {census.prefix}",
            census.stacked_keys(),
        )


def _refusal(message: str, keys: tuple[str, ...]) -> RefusalError:
    return Refusal(
        errors.FC_R12,
        message,
        keys=keys,
        hint="check that the store is complete and the arguments belong to it",
    ).to_error()
