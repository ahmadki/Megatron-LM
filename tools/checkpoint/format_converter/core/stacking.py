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

Converting an fsdp_dtensor store, the layout is decided the way the loading
job will ask for it, block by block:

* ``--block-layout PREFIX=stacked|per_layer`` decides for its block (stacking
  needs layers that are alike). It does not change the ``content_metadata``,
  so a layout other than the one Megatron builds from the arguments and the
  ``content_metadata`` is only loadable by a job whose configuration differs
  (:func:`requested_layout_warnings`).
* The language model's decoder is per layer iff Megatron builds it so: the
  predicate of ``TransformerBlock.sharded_state_dict``
  (:func:`per_layer_reasons`: the torch_dist target's ``content_metadata``
  flags and the arguments), else stacked. Arguments that make it stacked
  while its layers differ (names, shapes or dtypes) are refused (FC-R12), and
  the layout is never switched silently.
* A block of an unknown kind needs ``--block-layout`` (FC-A02).

Megatron also builds a block per layer for two configuration fields that no
argument sets: ``heterogeneous_block_specs`` (a heterogeneous layers
configuration) and ``hetereogenous_dist_checkpoint`` (``dsv4_hybrid``
attention). The arguments that set them are refused before any block is laid
out (FC-R03, FC-R20), so :func:`per_layer_reasons` does not read them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .. import errors
from ..errors import Refusal, RefusalError
from .args_view import ArgsView
from .blocks import (
    LM_DECODER,
    TRANSFORMER_BLOCK,
    BlockCensus,
    BlockFacts,
    BlockLayout,
    index_ranges,
)
from .keys import join_layer_key
from .names import PER_LAYER, STACKED, TORCH_DIST

PER_LAYER_METADATA = ("non_homogeneous_layers", "singleton_local_shards")
"""``content_metadata`` flags that make Megatron build every TransformerBlock per layer."""

LAYER_FREQUENCIES = ("moe_layer_freq", "linear_attention_freq")
"""Layer-type frequencies that make a TransformerBlock per layer as an int above 1 or a list."""


def transformer_block(facts: BlockFacts, census: BlockCensus) -> BlockLayout:
    """The layout of a ``TransformerBlock``: observed in a torch_dist source, else decided.

    Raises:
        RefusedError: The block is stored inconsistently, or cannot be
            stacked as the arguments or ``--block-layout`` ask (FC-R12).
        AmbiguousError: The block's kind is unknown and ``--block-layout``
            does not name it (FC-A02).
    """
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
    if facts.source == TORCH_DIST:
        return _observed_layout(facts, census)
    return _decided_layout(facts, census)


def per_layer_reasons(args: ArgsView, content_metadata: Mapping[str, Any] | None) -> list[str]:
    """Why Megatron builds a ``TransformerBlock`` of this configuration per layer (none: stacked).

    Mirrors ``TransformerBlock.sharded_state_dict`` (``transformer_block.py``),
    with the model's configuration taken from its training arguments (the
    configuration-only flags it also reads come from refused arguments; see
    the module docstring).
    """
    reasons = metadata_reasons(content_metadata)
    for name in LAYER_FREQUENCIES:
        value = args.get(name, None)
        if isinstance(value, list) or (isinstance(value, int) and value > 1):
            reasons.append(f"args.{name}={value!r}")
    return reasons


def requested_layout_warnings(facts: BlockFacts, blocks: Iterable[BlockLayout]) -> list[str]:
    """The blocks ``--block-layout`` lays out otherwise than Megatron builds them.

    A job built from the arguments asks for the layout of
    ``TransformerBlock.sharded_state_dict`` given the store's
    ``content_metadata``, which ``--block-layout`` does not change: such a job
    cannot load these blocks strictly.
    """
    reasons = per_layer_reasons(facts.args, facts.content_metadata)
    built = PER_LAYER if reasons else STACKED
    why = ", ".join(reasons) or "no per-layer setting"
    return [
        f"--block-layout {block.prefix}={block.td_layout}: a job built from these arguments and "
        f"this content_metadata builds {block.prefix} {built} ({why}), so it cannot load this "
        "store strictly"
        for block in blocks
        if block.prefix in facts.layouts and block.prefix == LM_DECODER and block.td_layout != built
    ]


def metadata_reasons(content_metadata: Mapping[str, Any] | None) -> list[str]:
    """The ``content_metadata`` flags that make every TransformerBlock per layer."""
    flags = content_metadata or {}
    return [
        f"content_metadata.{flag}={flags[flag]!r}" for flag in PER_LAYER_METADATA if flags.get(flag)
    ]


def _observed_layout(facts: BlockFacts, census: BlockCensus) -> BlockLayout:
    """The layout a torch_dist source stores the block in."""
    if census.stacked:
        count = _stacked_count(census)
        layout = STACKED
        evidence = f"torch_dist stores {len(census.stacked)} tensors stacked over {count} layers"
    else:
        count = _contiguous_count(census)
        layout = PER_LAYER
        evidence = f"torch_dist stores {count} layers one per key"
    _check_decoder_length(facts, census, count)
    return _layout(census, count, layout, evidence)


def _decided_layout(facts: BlockFacts, census: BlockCensus) -> BlockLayout:
    """The layout a torch_dist target gets for a block of an fsdp_dtensor source."""
    if census.stacked:
        raise _refusal(
            f"{census.prefix}: fsdp_dtensor stores one key per layer, but the store holds "
            "tensors stacked over its layers",
            census.stacked_keys(),
        )
    count = _contiguous_count(census)
    _check_decoder_length(facts, census, count)
    layout, evidence = _torch_dist_layout(facts, census)
    return _layout(census, count, layout, evidence)


def _torch_dist_layout(facts: BlockFacts, census: BlockCensus) -> tuple[str, str]:
    """The torch_dist layout of a block of an fsdp_dtensor source, and why."""
    prefix = census.prefix
    difference = _layer_difference(census)
    requested = facts.layouts.get(prefix)
    if requested is not None:
        if requested == STACKED and difference is not None:
            raise _refusal(
                f"--block-layout {prefix}={STACKED} cannot stack layers that differ: {difference}",
                (),
                hint=f"pass --block-layout {prefix}={PER_LAYER}",
            )
        return requested, f"--block-layout {prefix}={requested}"
    if prefix != LM_DECODER:
        raise Refusal(
            errors.FC_A02,
            f"{prefix} is a layer list of an unknown kind: whether torch_dist stacks it depends "
            "on its module",
            hint=f"pass --block-layout {prefix}={STACKED} or --block-layout {prefix}={PER_LAYER}",
        ).to_error()
    reasons = per_layer_reasons(facts.args, facts.content_metadata)
    if reasons:
        return PER_LAYER, f"Megatron builds it per layer ({', '.join(reasons)})"
    if difference is not None:
        raise _refusal(
            f"the arguments make Megatron stack {prefix}, but its layers differ: {difference}",
            (),
            hint="check that the arguments belong to this checkpoint",
        )
    return STACKED, "Megatron stacks it (no per-layer setting) and its layers are alike"


def _layer_difference(census: BlockCensus) -> str | None:
    """How a layer differs from the first in tensor names, shapes or dtypes (``None``: alike)."""
    first, *others = census.indices
    reference = census.layers[first]
    for index in others:
        layer = census.layers[index]
        unmatched = sorted(set(reference) ^ set(layer))
        if unmatched:
            has, lacks = (index, first) if unmatched[0] in layer else (first, index)
            return f"layer {has} has {unmatched[0]}, layer {lacks} does not"
        for name in sorted(reference):
            this, that = reference[name], layer[name]
            if (this.shape, this.dtype) != (that.shape, that.dtype):
                return (
                    f"{name} is {list(this.shape)} {this.dtype} in layer {first} and "
                    f"{list(that.shape)} {that.dtype} in layer {index}"
                )
    return None


def _layout(census: BlockCensus, count: int, layout: str, evidence: str) -> BlockLayout:
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


def _refusal(
    message: str,
    keys: tuple[str, ...],
    *,
    hint: str = "check that the store is complete and the arguments belong to it",
) -> RefusalError:
    return Refusal(errors.FC_R12, message, keys=keys, hint=hint).to_error()
