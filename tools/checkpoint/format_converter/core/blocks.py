# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Layered blocks of a model and how each format numbers and stacks their layers.

A block is a module with a layer list (``<prefix>.layers``). Its
:class:`BlockLayout` records the kind of block, its number of modules (the
``nn.ModuleList`` length, i.e. dense indices ``0..n-1``), how torch_dist stores
the layers (one key per layer, or one key per tensor *stacked* over the layers
along a new first axis) and the torch_dist index of every module;
fsdp_dtensor always stores one key per layer under the dense index.

The blocks a store holds are found from its keys: :func:`take_census` gives
each key's innermost layer list. Block classifiers, registered with
``@REGISTRY.classifier``, recognize the blocks of their kind from the
arguments and the census; :func:`classify_blocks` hands every other block of
the census to a default classifier (``TransformerBlock``, see
``core/stacking.py``) and cross-checks every layout against the layer indices
the source stores (FC-R11 on a mismatch).

Kinds of blocks:

* ``hybrid_stack``: the ``HybridStack`` decoder of a hybrid model, stored per
  layer in both formats and, without shortcut-MoE grouping, numbered
  identically (torch_dist uses ``layer_number - 1``, which is the pattern
  position; ``hybrid_block.py``); a tensor stacked over its layers is refused
  (FC-R12).
* ``transformer_block``: a ``TransformerBlock`` (a GPT decoder, a vision
  encoder), numbered identically in both formats and stored stacked or per
  layer in torch_dist (``core/stacking.py``).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from .. import errors
from ..engine.metadata import TensorEntry
from ..errors import InternalError, Refusal
from .args_view import ArgsView
from .hybrid_pattern import hybrid_pattern_of
from .keys import KeyGrammarError, innermost_layer_key, join_layer_key, split_layer_key
from .names import FSDP_DTENSOR, TORCH_DIST
from .registry import REGISTRY

HYBRID_STACK = "hybrid_stack"
TRANSFORMER_BLOCK = "transformer_block"
PER_LAYER = "per_layer"
STACKED = "stacked"
TD_LAYOUTS = (STACKED, PER_LAYER)
"""How torch_dist stores a block's layers."""

LM_DECODER = "decoder"
"""Prefix of a single-root model's language-model decoder."""


@dataclass(frozen=True)
class BlockLayout:
    """How a block's layers are stored.

    Attributes:
        prefix: The block prefix, e.g. ``decoder``.
        kind: The kind of block (``hybrid_stack`` or ``transformer_block``).
        num_modules: Number of modules in the block's layer list.
        td_layout: How torch_dist stores the layers: ``per_layer``, or
            ``stacked`` (one key per tensor, the layers along its first axis
            at their torch_dist index).
        dense_to_td: The torch_dist layer index of each module (dense order).
        symbols: The hybrid pattern symbol(s) of each module, if known.
        evidence: What the layout was derived from.

    Raises:
        InternalError: ``dense_to_td`` does not have one distinct index per
            module, or ``td_layout`` is unknown.
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
        if self.td_layout not in TD_LAYOUTS:
            raise InternalError(f"block {self.prefix}: unknown torch_dist layout {self.td_layout}")

    @property
    def stacked(self) -> bool:
        """Whether torch_dist stores the layers stacked (one key per tensor)."""
        return self.td_layout == STACKED

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


@dataclass(frozen=True)
class BlockCensus:
    """What a store holds in one block's layer list (each key in its innermost block).

    Attributes:
        prefix: The block prefix.
        layers: Tensors stored one layer per key, ``{stored index: {inner name: entry}}``.
        stacked: Tensors stacked over the layers, ``{inner name: entry}``.
    """

    prefix: str
    layers: Mapping[int, Mapping[str, TensorEntry]]
    stacked: Mapping[str, TensorEntry]

    @property
    def indices(self) -> tuple[int, ...]:
        """The stored layer indices, sorted."""
        return tuple(sorted(self.layers))

    def stacked_keys(self) -> tuple[str, ...]:
        """The model-relative keys of the stacked tensors, sorted."""
        return tuple(join_layer_key(self.prefix, None, inner) for inner in sorted(self.stacked))


def take_census(model: Mapping[str, TensorEntry]) -> dict[str, BlockCensus]:
    """The blocks ``model`` (model-relative keys) holds, by prefix (sorted).

    Keys whose layer segment is malformed are left out (the planner refuses
    them with the grammar error).
    """
    layers: dict[str, dict[int, dict[str, TensorEntry]]] = {}
    stacked: dict[str, dict[str, TensorEntry]] = {}
    for relative, entry in model.items():
        try:
            layer_key = innermost_layer_key(relative)
        except KeyGrammarError:
            continue
        if layer_key is None:
            continue
        layers.setdefault(layer_key.block, {})
        if layer_key.index is None:
            stacked.setdefault(layer_key.block, {})[layer_key.inner] = entry
        else:
            layers[layer_key.block].setdefault(layer_key.index, {})[layer_key.inner] = entry
    return {
        prefix: BlockCensus(prefix, layers[prefix], stacked.get(prefix, {}))
        for prefix in sorted(layers)
    }


@dataclass(frozen=True)
class BlockFacts:
    """What block classifiers decide from.

    Attributes:
        args: The checkpoint's training arguments.
        source: The source format.
        model: The source's model tensors by model-relative key.
    """

    args: ArgsView
    source: str
    model: Mapping[str, TensorEntry]

    @functools.cached_property
    def census(self) -> dict[str, BlockCensus]:
        """The blocks the source holds, by prefix."""
        return take_census(self.model)


DefaultClassifier = Callable[[BlockFacts, BlockCensus], BlockLayout]
"""Classifies a block of the census that no registered classifier recognized."""


@REGISTRY.classifier
def hybrid_stack(facts: BlockFacts) -> Iterator[BlockLayout]:
    """The ``HybridStack`` decoder of a hybrid model: per layer, numbered by pattern position.

    Raises:
        RefusedError: The source stores tensors stacked over its layers (FC-R12).
    """
    pattern = hybrid_pattern_of(facts.args)
    if pattern is None:
        return
    census = facts.census.get(LM_DECODER)
    if census is not None and census.stacked:
        raise Refusal(
            errors.FC_R12,
            f"{LM_DECODER} is a hybrid decoder ({pattern.text!r}), whose layers are stored one "
            "per key, but the store holds tensors stacked over its layers",
            keys=census.stacked_keys(),
            hint="check that the arguments belong to this checkpoint",
        ).to_error()
    count = len(pattern.main)
    yield BlockLayout(
        prefix=LM_DECODER,
        kind=HYBRID_STACK,
        num_modules=count,
        td_layout=PER_LAYER,
        dense_to_td=tuple(range(count)),
        symbols=tuple(pattern.main),
        evidence=f"hybrid layer pattern {pattern.text!r} ({count} layers)",
    )


def classify_blocks(facts: BlockFacts, *, default: DefaultClassifier) -> tuple[BlockLayout, ...]:
    """The blocks of the model, each checked against the layer indices the source stores.

    Args:
        facts: What the classifiers decide from.
        default: Classifies every block of the census that no registered
            classifier recognized.

    Returns:
        The layouts, sorted by prefix.

    Raises:
        RefusedError: A block's stored layer indices or ``args.num_layers``
            disagree with its layout (FC-R11), or its stacking is
            inconsistent (FC-R12).
        InternalError: Two classifiers claim one block.
    """
    blocks = [block for classifier in REGISTRY.classifiers.values() for block in classifier(facts)]
    known = {block.prefix for block in blocks}
    if len(known) != len(blocks):
        raise InternalError(f"classifiers claim a block twice: {sorted(b.prefix for b in blocks)}")
    blocks.extend(
        default(facts, census) for prefix, census in facts.census.items() if prefix not in known
    )
    for block in blocks:
        _check_num_layers(block, facts.args)
        if facts.source == FSDP_DTENSOR or not block.stacked:
            _check_stored_indices(block, facts.model, facts.source)
    return tuple(sorted(blocks, key=lambda block: block.prefix))


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
            f"{index_ranges(sorted(expected))}, but the store has {index_ranges(sorted(stored))} "
            f"(missing {missing or 'none'}, unexpected {unexpected or 'none'})",
            hint="check that the arguments belong to this checkpoint",
        ).to_error()


def index_ranges(indices: list[int] | tuple[int, ...]) -> str:
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
