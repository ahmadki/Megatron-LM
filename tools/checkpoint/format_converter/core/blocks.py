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
  layer in both formats (a tensor stacked over its layers is refused,
  FC-R12). torch_dist numbers a module by its first layer's
  ``layer_number - 1``, which is the pattern position (``hybrid_block.py``),
  and fsdp_dtensor by its layer-list index: without shortcut-MoE grouping
  every layer is a module and both formats number it identically; with it,
  each ``E`` joins the layer before it in one module
  (``core/hybrid_pattern.py``), so the torch_dist indices skip the ``E``
  positions of the pairs while the fsdp_dtensor indices stay dense.
* ``transformer_block``: a ``TransformerBlock`` (a GPT decoder, a vision
  encoder), numbered identically in both formats and stored stacked or per
  layer in torch_dist (``core/stacking.py``).

A model has a single root (``embedding``, ``decoder``, ``output_layer``,
``mtp``), or is a vision-language model in the LLaVA layout: its language
model under ``language_model.`` (``language_model.decoder`` and so on), a
vision encoder under ``vision_model.`` (``vision_model.decoder`` is a
``TransformerBlock``) and a projection under ``vision_projection.``
(:func:`language_model_prefix`).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .. import errors
from ..engine.metadata import TensorEntry
from ..errors import InternalError, Refusal, UsageError
from .args_view import ArgsView
from .hybrid_pattern import group_modules, hybrid_pattern_of
from .keys import KeyGrammarError, innermost_layer_key, join_layer_key, root_of, split_layer_key
from .names import FSDP_DTENSOR, PER_LAYER, STACKED, TD_LAYOUTS, TORCH_DIST
from .registry import REGISTRY

HYBRID_STACK = "hybrid_stack"
TRANSFORMER_BLOCK = "transformer_block"

DECODER = "decoder"
"""The language model's decoder, under the language model's prefix."""

LANGUAGE_MODEL = "language_model"
SINGLE_ROOTS = frozenset({"embedding", DECODER, "output_layer", "mtp"})
"""Top-level modules of a single-root (GPT or hybrid) model."""

VLM_ROOTS = frozenset({LANGUAGE_MODEL, "vision_model", "vision_projection"})
"""Top-level modules of a vision-language (LLaVA-style) model."""

VISION_DECODER = "vision_model.decoder"
"""The vision encoder's ``TransformerBlock`` in a vision-language model."""


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

    def index_table(self) -> str:
        """Every module's dense index, torch_dist index and symbols: ``0: 0 E, 1: 1 *E, 2: 3 -``."""
        return ", ".join(
            f"{dense}: {stored}" + ("" if self.symbols is None else f" {self.symbols[dense]}")
            for dense, stored in enumerate(self.dense_to_td)
        )

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
        content_metadata: The ``content_metadata`` a torch_dist target gets
            (fsdp_dtensor sources; ``None`` when it gets none).
        layouts: How torch_dist stores a block, by prefix, as requested with
            ``--block-layout`` (fsdp_dtensor sources).
        per_layer: torch_dist stores every TransformerBlock per layer
            (``--torch-dist-layout per-layer``; fsdp_dtensor sources).
    """

    args: ArgsView
    source: str
    model: Mapping[str, TensorEntry]
    content_metadata: Mapping[str, Any] | None = None
    layouts: Mapping[str, str] = field(default_factory=dict)
    per_layer: bool = False

    @functools.cached_property
    def census(self) -> dict[str, BlockCensus]:
        """The blocks the source holds, by prefix."""
        return take_census(self.model)

    @functools.cached_property
    def lm_decoder(self) -> str:
        """The language model's decoder: ``decoder``, or ``language_model.decoder`` in a VLM."""
        return language_model_prefix(self.model) + DECODER


def language_model_prefix(model: Iterable[str]) -> str:
    """The prefix of the language model's modules: ``language_model.`` in a vision-language model.

    Args:
        model: The model-relative keys.

    Raises:
        RefusedError: Keys are under both single-root and vision-language
            top-level modules (FC-R13).
    """
    roots = {root_of(key) for key in model}
    single, vlm = sorted(roots & SINGLE_ROOTS), sorted(roots & VLM_ROOTS)
    if single and vlm:
        raise Refusal(
            errors.FC_R13,
            f"model keys are under single-model roots ({', '.join(single)}) and vision-language "
            f"roots ({', '.join(vlm)})",
            hint="check that the store holds one model",
        ).to_error()
    return f"{LANGUAGE_MODEL}." if vlm else ""


DefaultClassifier = Callable[[BlockFacts, BlockCensus], BlockLayout]
"""Classifies a block of the census that no registered classifier recognized."""


@REGISTRY.classifier
def hybrid_stack(facts: BlockFacts) -> Iterator[BlockLayout]:
    """The ``HybridStack`` decoder of a hybrid model: per layer, its modules numbered by pattern
    position in torch_dist (see the module docstring).

    Raises:
        RefusedError: The source stores tensors stacked over its layers
            (FC-R12), ``args.num_layers`` is not the pattern's number of
            layers, or shortcut-MoE grouping of the pattern is impossible (FC-R11).
    """
    pattern = hybrid_pattern_of(facts.args)
    if pattern is None:
        return
    prefix = facts.lm_decoder
    census = facts.census.get(prefix)
    if census is not None and census.stacked:
        raise Refusal(
            errors.FC_R12,
            f"{prefix} is a hybrid decoder ({pattern.text!r}), whose layers are stored one "
            "per key, but the store holds tensors stacked over its layers",
            keys=census.stacked_keys(),
            hint="check that the arguments belong to this checkpoint",
        ).to_error()
    evidence = f"hybrid layer pattern {pattern.text!r} ({_count(len(pattern.main), 'layer')})"
    _check_num_layers(facts.args, len(pattern.main), evidence)
    shortcut = bool(facts.args.get("moe_shortcut_connection", False))
    modules = group_modules(pattern.main, shortcut=shortcut, stage_starts=pattern.stage_starts)
    if shortcut:
        evidence += (
            f" grouped into {_count(len(modules), 'module')} by "
            "args.moe_shortcut_connection=True (each E joins the layer before it unless that "
            "layer is paired)"
        )
    yield BlockLayout(
        prefix=prefix,
        kind=HYBRID_STACK,
        num_modules=len(modules),
        td_layout=PER_LAYER,
        dense_to_td=tuple(position for position, _ in modules),
        symbols=tuple(symbols for _, symbols in modules),
        evidence=evidence,
    )


def _check_num_layers(args: ArgsView, count: int, evidence: str) -> None:
    """``args.num_layers`` is the hybrid decoder's number of layers (FC-R11)."""
    num_layers = args.get("num_layers", None)
    if num_layers is not None and num_layers != count:
        raise Refusal(
            errors.FC_R11,
            f"args.num_layers={num_layers} but {evidence}",
            hint="check that the source checkpoint's arguments are intact",
        ).to_error()


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
            disagree with its layout, or a hybrid pattern cannot be grouped
            into modules (FC-R11), or a block's stacking is inconsistent (FC-R12).
        AmbiguousError: A block's torch_dist layout cannot be decided (FC-A02).
        UsageError: ``facts.layouts`` names a block that is not a
            ``TransformerBlock`` of the model.
        InternalError: Two classifiers claim one block.
    """
    blocks = [block for classifier in REGISTRY.classifiers.values() for block in classifier(facts)]
    known = {block.prefix for block in blocks}
    if len(known) != len(blocks):
        raise InternalError(f"classifiers claim a block twice: {sorted(b.prefix for b in blocks)}")
    blocks.extend(
        default(facts, census) for prefix, census in facts.census.items() if prefix not in known
    )
    _check_requested_layouts(blocks, facts.layouts)
    for block in blocks:
        if facts.source == FSDP_DTENSOR or not block.stacked:
            _check_stored_indices(block, facts.model, facts.source)
    return tuple(sorted(blocks, key=lambda block: block.prefix))


def _check_requested_layouts(blocks: list[BlockLayout], layouts: Mapping[str, str]) -> None:
    """``--block-layout`` may only name the model's TransformerBlocks."""
    kinds = {block.prefix: block.kind for block in blocks}
    for prefix in sorted(layouts):
        if kinds.get(prefix) != TRANSFORMER_BLOCK:
            what = f"a {kinds[prefix]} block" if prefix in kinds else "no layer list of this model"
            raise UsageError(
                f"--block-layout {prefix}={layouts[prefix]}: {prefix} is {what}",
                hint="name TransformerBlock layer lists (the blocks plan lists)",
            )


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
        table = (
            ""
            if block.symbols is None
            else f"; its modules (dense: torch_dist index, symbols) are {block.index_table()}"
        )
        raise Refusal(
            errors.FC_R11,
            f"{block.prefix}: {block.evidence} means {source} layer indices "
            f"{index_ranges(sorted(expected))}, but the store has {index_ranges(sorted(stored))} "
            f"(missing {missing or 'none'}, unexpected {unexpected or 'none'}){table}",
            hint="check that the arguments belong to this checkpoint",
        ).to_error()


def _count(count: int, noun: str) -> str:
    """``1 layer``, ``2 layers``, ``0 layers``."""
    return f"{count} {noun}" + ("" if count == 1 else "s")


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
