# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Layered blocks of a model and how each format numbers their layers.

A block is a module with a layer list (``<prefix>.layers.<index>``). Its
:class:`BlockLayout` records the kind of block, its number of modules (the
``nn.ModuleList`` length, i.e. dense indices ``0..n-1``) and the torch_dist
index of every module; fsdp_dtensor always stores the dense index. Block
classifiers, registered with ``@REGISTRY.classifier``, derive the layouts from
the arguments and the stored keys, and every layout is cross-checked against
the layer indices the source actually stores (FC-R11 on a mismatch).

This version knows one kind of block: the ``HybridStack`` decoder of a hybrid
model, stored per layer in both formats and, without shortcut-MoE grouping,
numbered identically (torch_dist uses ``layer_number - 1``, which is the
pattern position; ``hybrid_block.py``).
"""

from __future__ import annotations

import functools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from .. import errors
from ..errors import InternalError, Refusal
from .args_view import ArgsView
from .hybrid_pattern import hybrid_pattern_of
from .keys import KeyGrammarError, split_layer_key
from .names import FSDP_DTENSOR, TORCH_DIST
from .registry import REGISTRY

HYBRID_STACK = "hybrid_stack"
PER_LAYER = "per_layer"
HYBRID_DECODER = "decoder"
"""Prefix of a single-root hybrid model's decoder."""


@dataclass(frozen=True)
class BlockLayout:
    """How a block's layers are stored.

    Attributes:
        prefix: The block prefix, e.g. ``decoder``.
        kind: The kind of block (``hybrid_stack``).
        num_modules: Number of modules in the block's layer list.
        td_layout: How torch_dist stores the layers (``per_layer``).
        dense_to_td: The torch_dist layer index of each module (dense order).
        symbols: The hybrid pattern symbol(s) of each module, if known.
        evidence: What the layout was derived from.

    Raises:
        InternalError: ``dense_to_td`` does not have one distinct index per module.
    """

    prefix: str
    kind: str
    num_modules: int
    td_layout: str
    dense_to_td: tuple[int, ...]
    symbols: tuple[str, ...] | None
    evidence: str

    def __post_init__(self) -> None:
        if len(self.dense_to_td) != self.num_modules or len(set(self.dense_to_td)) != len(
            self.dense_to_td
        ):
            raise InternalError(f"block {self.prefix}: bad torch_dist index map {self.dense_to_td}")

    def stored_indices(self, fmt: str) -> tuple[int, ...]:
        """The layer indices a store of format ``fmt`` holds, in dense order."""
        if fmt == TORCH_DIST:
            return self.dense_to_td
        if fmt == FSDP_DTENSOR:
            return tuple(range(self.num_modules))
        raise InternalError(f"unknown format {fmt}")

    def dense_index(self, stored: int, fmt: str) -> int:
        """The dense index of a layer stored under index ``stored`` in format ``fmt``.

        Raises:
            KeyGrammarError: The format never stores that index for this block.
        """
        dense = self._dense_by_stored[fmt].get(stored)
        if dense is None:
            raise KeyGrammarError(f"{self.prefix} has no layer stored as {stored} in {fmt}")
        return dense

    @functools.cached_property
    def _dense_by_stored(self) -> dict[str, dict[int, int]]:
        return {
            fmt: {stored: dense for dense, stored in enumerate(self.stored_indices(fmt))}
            for fmt in (TORCH_DIST, FSDP_DTENSOR)
        }

    def stored_index(self, dense: int, fmt: str) -> int:
        """The index a store of format ``fmt`` holds the dense layer ``dense`` under."""
        return self.stored_indices(fmt)[dense]

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {
            "prefix": self.prefix,
            "kind": self.kind,
            "num_modules": self.num_modules,
            "td_layout": self.td_layout,
            "dense_to_td": list(self.dense_to_td),
            "symbols": None if self.symbols is None else list(self.symbols),
            "evidence": self.evidence,
        }


@REGISTRY.classifier
def hybrid_stack(args: ArgsView) -> Iterator[BlockLayout]:
    """The ``HybridStack`` decoder of a hybrid model: per layer, numbered by pattern position."""
    pattern = hybrid_pattern_of(args)
    if pattern is None:
        return
    count = len(pattern.main)
    yield BlockLayout(
        prefix=HYBRID_DECODER,
        kind=HYBRID_STACK,
        num_modules=count,
        td_layout=PER_LAYER,
        dense_to_td=tuple(range(count)),
        symbols=tuple(pattern.main),
        evidence=f"hybrid layer pattern {pattern.text!r} ({count} layers)",
    )


def classify_blocks(
    args: ArgsView, model: Mapping[str, Any], source: str
) -> tuple[BlockLayout, ...]:
    """The blocks of the model, each checked against the layer indices the source stores.

    Args:
        args: The checkpoint's training arguments.
        model: The source's model tensors by model-relative key.
        source: The source format.

    Raises:
        RefusedError: A block's stored layer indices or ``args.num_layers``
            disagree with its layout (FC-R11).
    """
    blocks = tuple(
        block for classifier in REGISTRY.classifiers.values() for block in classifier(args)
    )
    for block in blocks:
        _check_num_layers(block, args)
        _check_stored_indices(block, model, source)
    return blocks


def _check_num_layers(block: BlockLayout, args: ArgsView) -> None:
    if block.kind != HYBRID_STACK:
        return
    num_layers = args.get("num_layers", None)
    if num_layers is not None and num_layers != block.num_modules:
        raise Refusal(
            errors.FC_R11,
            f"args.num_layers={num_layers} but {block.evidence}",
            hint="check that the source checkpoint's arguments are intact",
        ).to_error()


def _check_stored_indices(block: BlockLayout, model: Mapping[str, Any], source: str) -> None:
    """The layer indices under the block's own layer segment (nested blocks' keys included)."""
    stored: set[int] = set()
    for key in model:
        try:
            layer_key = split_layer_key(key, block.prefix)
        except KeyGrammarError:
            continue  # refused with the key by the planner
        if layer_key is not None and layer_key.index is not None:
            stored.add(layer_key.index)
    expected = set(block.stored_indices(source))
    if stored != expected:
        missing, unexpected = sorted(expected - stored), sorted(stored - expected)
        raise Refusal(
            errors.FC_R11,
            f"{block.prefix}: {block.evidence} means {source} layer indices "
            f"{_ranges(sorted(expected))}, but the store has {_ranges(sorted(stored))} "
            f"(missing {missing or 'none'}, unexpected {unexpected or 'none'})",
            hint="check that the arguments belong to this checkpoint",
        ).to_error()


def _ranges(indices: list[int]) -> str:
    """``0-3, 5, 7-9`` for ``[0, 1, 2, 3, 5, 7, 8, 9]`` (sorted input)."""
    if not indices:
        return "none"
    runs = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            runs.append((start, previous))
            start = index
        previous = index
    runs.append((start, previous))
    return ", ".join(str(first) if first == last else f"{first}-{last}" for first, last in runs)
