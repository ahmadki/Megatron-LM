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
* ``mtp_block``: the multi-token prediction block (``mtp``), one layer per
  depth (a single layer that serves every depth with
  ``--mtp-use-repeated-layer``), numbered identically in both formats and
  never stacked. A GPT MTP layer holds a single inner layer, which both
  formats store as ``transformer_layer`` (Megatron renames the live
  ``mtp_model_layer`` on save and load); a Megatron-FSDP store saved before
  that rename holds the live name, which the block records as a
  :class:`ModuleRename` of the fsdp_dtensor side (so does
  ``--fsdp-gpt-mtp-inner-name mtp_model_layer`` for an fsdp_dtensor target).
* ``hybrid_mtp_inner``: the inner model of a hybrid multi-token prediction
  layer, ``mtp.layers.<d>.mtp_model_layer`` (the same name in both formats), a
  ``HybridStack`` of the pattern's MTP part numbered like the decoder from
  position 0 (with shortcut-MoE grouping too).

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
from .hybrid_pattern import HybridPattern, family_mismatch, group_modules, hybrid_pattern_of
from .keys import (
    INPUT_LAYERNORM,
    LAYER_SEGMENT,
    MLA_DOWN_PROJ,
    MLA_FUSED_DOWN_PROJ,
    KeyGrammarError,
    innermost_layer_key,
    join_layer_key,
    root_of,
    split_layer_key,
)
from .names import (
    FSDP_DTENSOR,
    GPT_MTP_INNER_LAYER,
    MTP_INNER_LAYER,
    MTP_INNER_LAYERS,
    PER_LAYER,
    STACKED,
    TD_LAYOUTS,
    TORCH_DIST,
)
from .registry import REGISTRY

HYBRID_STACK = "hybrid_stack"
TRANSFORMER_BLOCK = "transformer_block"
MTP_BLOCK = "mtp_block"
HYBRID_MTP_INNER = "hybrid_mtp_inner"

MTP = "mtp"
"""The language model's multi-token prediction block, under the language model's prefix."""

DECODER = "decoder"
"""The language model's decoder, under the language model's prefix."""

LANGUAGE_MODEL = "language_model"
MTP_BLOCKS = (MTP, f"{LANGUAGE_MODEL}.{MTP}")
"""Prefixes of the multi-token prediction block: a single-root model's and a vision-language
model's."""

SINGLE_ROOTS = frozenset({"embedding", DECODER, "output_layer", MTP})
"""Top-level modules of a single-root (GPT or hybrid) model."""

VLM_ROOTS = frozenset({LANGUAGE_MODEL, "vision_model", "vision_projection"})
"""Top-level modules of a vision-language (LLaVA-style) model."""

VISION_DECODER = "vision_model.decoder"
"""The vision encoder's ``TransformerBlock`` in a vision-language model."""


@dataclass(frozen=True, slots=True)
class ModuleRename:
    """A module of a block's layers that one format stores under another name than its logical one.

    Attributes:
        fmt: The format that stores the module renamed.
        logical: The module's name in logical ids (and in the other format).
        stored: The module's name in the keys of ``fmt``.
        reason: Why ``fmt`` stores it so (the plan's evidence, and a warning).
    """

    fmt: str
    logical: str
    stored: str
    reason: str


def _rename_module(name: str, old: str, new: str) -> str | None:
    """``name`` (inside a layer) with its leading module ``old`` named ``new``, if it is in it."""
    if name == old or name.startswith(old + "."):
        return new + name[len(old) :]
    return None


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
        renames: Modules of the layers that a format stores under another
            name than their logical one.

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
    renames: tuple[ModuleRename, ...] = ()

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

    def logical_name(self, name: str, fmt: str) -> str:
        """The logical name of ``name``, a name inside a layer of a store of format ``fmt``."""
        for rename in self.renames:
            renamed = _rename_module(name, rename.stored, rename.logical)
            if rename.fmt == fmt and renamed is not None:
                return renamed
        return name

    def stored_name(self, name: str, fmt: str) -> str:
        """The name inside a layer a store of format ``fmt`` gives the logical name ``name``."""
        for rename in self.renames:
            renamed = _rename_module(name, rename.logical, rename.stored)
            if rename.fmt == fmt and renamed is not None:
                return renamed
        return name

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
        fsdp_gpt_mtp_inner_name: What fsdp_dtensor names the inner layer of
            GPT MTP layers (``--fsdp-gpt-mtp-inner-name``; torch_dist sources).
    """

    args: ArgsView
    source: str
    model: Mapping[str, TensorEntry]
    content_metadata: Mapping[str, Any] | None = None
    layouts: Mapping[str, str] = field(default_factory=dict)
    per_layer: bool = False
    fsdp_gpt_mtp_inner_name: str = GPT_MTP_INNER_LAYER

    @functools.cached_property
    def census(self) -> dict[str, BlockCensus]:
        """The blocks the source holds, by prefix."""
        return take_census(self.model)

    @functools.cached_property
    def lm_decoder(self) -> str:
        """The language model's decoder: ``decoder``, or ``language_model.decoder`` in a VLM."""
        return language_model_prefix(self.model) + DECODER

    @functools.cached_property
    def lm_mtp(self) -> str:
        """The language model's multi-token prediction block: ``mtp`` (``language_model.mtp``)."""
        return language_model_prefix(self.model) + MTP


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
    evidence = f"hybrid layer pattern {pattern.text!r} ({_count(len(pattern.main), 'layer')})"
    num_layers = facts.args.get("num_layers", None)
    if num_layers is not None and num_layers != len(pattern.main):
        raise Refusal(
            errors.FC_R11,
            f"args.num_layers={num_layers} but {evidence}",
            hint="check that the source checkpoint's arguments are intact",
        ).to_error()
    yield _hybrid_layer_list(
        facts,
        facts.lm_decoder,
        HYBRID_STACK,
        pattern.main,
        evidence,
        what=f"a hybrid decoder ({pattern.text!r})",
        stage_starts=pattern.stage_starts,
    )


@REGISTRY.classifier
def multi_token_prediction(facts: BlockFacts) -> Iterator[BlockLayout]:
    """The multi-token prediction block, and the inner layer list of each hybrid MTP layer.

    The block has one layer per depth: the hybrid layer pattern's MTP parts
    (``/``), or ``args.mtp_num_layers`` without a pattern, and a single layer
    with ``args.mtp_use_repeated_layer`` (``MultiTokenPredictionBlock``). A
    hybrid MTP layer's inner model is a ``HybridStack`` of its part of the
    pattern, stored per layer under ``mtp_model_layer.layers.<j>`` in both
    formats (``MultiTokenPredictionLayer.sharded_state_dict`` renames the
    inner model of GPT MTP layers only); a GPT MTP layer's is one layer,
    stored as ``transformer_layer`` (see :func:`_gpt_mtp_renames`).

    Raises:
        RefusedError: ``args.mtp_num_layers`` disagrees with the pattern's MTP
            parts, or the store holds MTP layers the arguments do not have
            (FC-R11); an inner layer list is stored stacked (FC-R12) or cannot
            be grouped (FC-R11).
    """
    pattern = hybrid_pattern_of(facts.args)
    depths, evidence = _mtp_depths(facts.args, pattern)
    prefix = facts.lm_mtp
    if not depths:
        if prefix in facts.census:
            raise Refusal(
                errors.FC_R11,
                f"the store holds multi-token prediction layers ({prefix}{LAYER_SEGMENT}*), but "
                f"{evidence}",
                hint="check that the arguments belong to this checkpoint",
            ).to_error()
        return
    count = depths
    if facts.args.get("mtp_use_repeated_layer", False):
        count = 1
        evidence += " with args.mtp_use_repeated_layer=True (one layer serves every depth)"
    renames = _gpt_mtp_renames(facts) if is_gpt_mtp(facts.args) else ()
    yield BlockLayout(
        prefix, MTP_BLOCK, count, PER_LAYER, tuple(range(count)), None, evidence, renames
    )
    if pattern is None:
        return  # GPT MTP layers (is_gpt_mtp): the inner model is a single layer
    for depth in range(count):
        symbols = pattern.mtp[depth]
        yield _hybrid_layer_list(
            facts,
            join_layer_key(prefix, depth, MTP_INNER_LAYER),
            HYBRID_MTP_INNER,
            symbols,
            f"MTP depth pattern {symbols!r} ({_count(len(symbols), 'layer')})",
            what=f"the hybrid layer list of MTP depth {depth} ({symbols!r})",
        )


def is_gpt_mtp(args: ArgsView) -> bool:
    """Whether the multi-token prediction layers of the arguments' model, if any, are GPT MTP
    layers (no hybrid layer pattern) rather than hybrid ones.

    The one predicate of the GPT MTP naming policy, which this module keeps
    whole: a GPT MTP layer holds a single inner ``TransformerLayer``, live
    name ``mtp_model_layer`` (:func:`gpt_mtp_live_name`), which both formats
    store as ``transformer_layer`` (:func:`_gpt_mtp_renames` for the legacy
    exception, :func:`mtp_inner_keys` to find the stored names); a hybrid MTP
    layer's inner layer list is ``mtp_model_layer`` everywhere.
    """
    return hybrid_pattern_of(args) is None


def gpt_mtp_live_name(name: str) -> str:
    """The live (``named_parameters``) name of ``name``, a logical name inside a GPT MTP layer:
    its inner ``transformer_layer`` is ``mtp_model_layer`` in the live model; other names are
    unchanged (hybrid MTP layers keep their names, so this applies to GPT MTP layers only)."""
    return _rename_module(name, GPT_MTP_INNER_LAYER, MTP_INNER_LAYER) or name


def fused_mla_live_name(name: str, *, absorbs_norm: bool) -> str:
    """The live (``named_parameters``) name of ``name``, a stored model name, in a model built
    with ``--mla-down-proj-fusion``.

    ``FusedMLASelfAttention`` holds multi-latent attention's q and kv
    down-projections in one module, ``linear_qkv_down_proj``, which both
    formats store unfused (``linear_q_down_proj``, ``linear_kv_down_proj``);
    on the Transformer Engine GPT layer spec it also absorbs the layer's input
    layer norm as its ``layer_norm_<leaf>``, stored as
    ``input_layernorm.<leaf>`` (``absorbs_norm``: ``name``'s layer is such a
    layer). Other names are unchanged.
    """
    match = MLA_DOWN_PROJ.fullmatch(name)
    if match is not None:
        return f"{match.group('attention')}.{MLA_FUSED_DOWN_PROJ}.{match.group('leaf')}"
    match = INPUT_LAYERNORM.fullmatch(name)
    if match is not None and absorbs_norm:
        layer, leaf = match.group("layer"), match.group("leaf")
        return f"{layer}.self_attention.{MLA_FUSED_DOWN_PROJ}.layer_norm_{leaf}"
    return name


def _gpt_mtp_renames(facts: BlockFacts) -> tuple[ModuleRename, ...]:
    """How the fsdp_dtensor side names the inner layer of GPT MTP layers, where it is not
    ``transformer_layer`` as on current Megatron.

    Megatron stores it as ``transformer_layer`` in both formats, renaming the
    live ``mtp_model_layer`` on save and load
    (``MultiTokenPredictionLayer.sharded_state_dict``,
    ``handle_mtp_in_state_dict``), but Megatron-FSDP saved the live name
    before ``handle_mtp_in_state_dict`` existed: an fsdp_dtensor source that
    holds it (and only it, which the refusal fence checks, FC-R09) is read
    under it, and an fsdp_dtensor target is written under it with
    ``--fsdp-gpt-mtp-inner-name mtp_model_layer``.
    """
    if facts.source == FSDP_DTENSOR:
        legacy = [key for key in mtp_inner_keys(facts.model) if key.name == MTP_INNER_LAYER]
        if not legacy:
            return ()
        reason = (
            "the fsdp_dtensor source stores the inner layer of its GPT MTP layers "
            f"({_count(len(legacy), 'key')}) under the live name {MTP_INNER_LAYER}, as "
            "Megatron-FSDP saved it before handle_mtp_in_state_dict; torch_dist stores it as "
            f"{GPT_MTP_INNER_LAYER}"
        )
    elif facts.fsdp_gpt_mtp_inner_name == MTP_INNER_LAYER:
        reason = (
            f"--fsdp-gpt-mtp-inner-name {MTP_INNER_LAYER}: the fsdp_dtensor target stores the "
            "inner layer of its GPT MTP layers under the live name, which only Megatron-FSDP "
            f"versions older than handle_mtp_in_state_dict load ({GPT_MTP_INNER_LAYER} otherwise)"
        )
    else:
        return ()
    return (ModuleRename(FSDP_DTENSOR, GPT_MTP_INNER_LAYER, MTP_INNER_LAYER, reason),)


@dataclass(frozen=True, slots=True)
class MtpInnerKey:
    """A model key under the inner layer of a multi-token prediction layer.

    Attributes:
        key: The model-relative key.
        name: The inner layer's name (``transformer_layer`` or ``mtp_model_layer``).
        layer_list: The inner layer holds a layer list (``<name>.layers.``), as a
            hybrid MTP layer's does.
    """

    key: str
    name: str
    layer_list: bool


def mtp_inner_keys(model: Iterable[str]) -> list[MtpInnerKey]:
    """The keys of ``model`` under the inner layer of a multi-token prediction layer,
    ``[language_model.]mtp.layers.<d>.<name>.<...>`` with ``name`` ``transformer_layer`` or
    ``mtp_model_layer`` (sorted; malformed layer segments are left to the planner)."""
    found = []
    for key in sorted(model):
        for block in MTP_BLOCKS:
            try:
                layer_key = split_layer_key(key, block)
            except KeyGrammarError:
                continue
            if layer_key is None or layer_key.index is None:
                continue
            name = layer_key.inner.partition(".")[0]
            if name in MTP_INNER_LAYERS:
                layer_list = layer_key.inner.startswith(name + LAYER_SEGMENT)
                found.append(MtpInnerKey(key, name, layer_list))
    return found


def _mtp_depths(args: ArgsView, pattern: HybridPattern | None) -> tuple[int, str]:
    """The number of multi-token prediction depths, and the evidence (a noun phrase).

    Raises:
        RefusedError: ``args.mtp_num_layers`` disagrees with the hybrid layer
            pattern's MTP parts (FC-R11): Megatron's ``HybridModel`` builds no
            model from such arguments.
    """
    mtp_num_layers = args.get("mtp_num_layers", None) or 0
    if pattern is None:
        return mtp_num_layers, f"args.mtp_num_layers={mtp_num_layers or None}"
    evidence = f"hybrid layer pattern {pattern.text!r} ({_count(len(pattern.mtp), 'MTP depth')})"
    if mtp_num_layers and mtp_num_layers != len(pattern.mtp):
        raise Refusal(
            errors.FC_R11,
            f"args.mtp_num_layers={mtp_num_layers} but {evidence}",
            hint="check that the source checkpoint's arguments are intact",
        ).to_error()
    return len(pattern.mtp), evidence


def _hybrid_layer_list(
    facts: BlockFacts,
    prefix: str,
    kind: str,
    symbols: str,
    evidence: str,
    *,
    what: str,
    stage_starts: frozenset[int] = frozenset(),
) -> BlockLayout:
    """The layout of a ``HybridStack`` layer list of layers ``symbols`` under ``prefix``: its
    modules (grouped with ``args.moe_shortcut_connection``) stored per layer, torch_dist under
    their first layer's position.

    Raises:
        RefusedError: The source stores tensors stacked over its layers
            (FC-R12), or the layers cannot be grouped (FC-R11).
    """
    census = facts.census.get(prefix)
    if census is not None and census.stacked:
        raise Refusal(
            errors.FC_R12,
            f"{prefix} is {what}, whose layers are stored one per key, but the store holds "
            "tensors stacked over its layers",
            keys=census.stacked_keys(),
            hint="check that the arguments belong to this checkpoint",
        ).to_error()
    shortcut = bool(facts.args.get("moe_shortcut_connection", False))
    modules = group_modules(symbols, shortcut=shortcut, stage_starts=stage_starts)
    if shortcut:
        evidence += (
            f" grouped into {_count(len(modules), 'module')} by "
            "args.moe_shortcut_connection=True (each E joins the layer before it unless that "
            "layer is paired)"
        )
    return BlockLayout(
        prefix=prefix,
        kind=kind,
        num_modules=len(modules),
        td_layout=PER_LAYER,
        dense_to_td=tuple(position for position, _ in modules),
        symbols=tuple(module for _, module in modules),
        evidence=evidence,
    )


def language_model_blocks(blocks: Iterable[BlockLayout], lm_decoder: str) -> tuple[str, ...]:
    """The blocks built with the language model's configuration: its decoder ``lm_decoder`` and
    its multi-token prediction blocks (sorted)."""
    mtp = {block.prefix for block in blocks if block.kind in (MTP_BLOCK, HYBRID_MTP_INNER)}
    return tuple(sorted(mtp | {lm_decoder}))


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
            disagree with its layout, a hybrid pattern cannot be grouped into
            modules, or a hybrid module holds the tensors of other layer types
            than its symbols (FC-R11); or a block's stacking is inconsistent (FC-R12).
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
        if block.symbols is not None:
            _check_families(block, block.symbols, facts.census.get(block.prefix), facts.source)
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
        table = "" if block.symbols is None else _module_table(block)
        raise Refusal(
            errors.FC_R11,
            f"{block.prefix}: {block.evidence} means {source} layer indices "
            f"{index_ranges(sorted(expected))}, but the store has {index_ranges(sorted(stored))} "
            f"(missing {missing or 'none'}, unexpected {unexpected or 'none'}){table}",
            hint="check that the arguments belong to this checkpoint",
        ).to_error()


def _check_families(
    block: BlockLayout, modules: tuple[str, ...], census: BlockCensus | None, source: str
) -> None:
    """Every module of a hybrid block holds the tensors of its layers' symbols ``modules`` (FC-R11).

    See :func:`~.hybrid_pattern.family_mismatch`; the modules are read under
    their ``source`` index, each key in its innermost block.
    """
    layers = {} if census is None else census.layers
    wrong: list[str] = []
    keys: list[str] = []
    for dense, symbols in enumerate(modules):
        stored = block.stored_index(dense, source)
        names = sorted(layers.get(stored, {}))
        reason = family_mismatch(symbols, names)
        if reason is not None:
            wrong.append(f"module {dense} ({source} index {stored}, {symbols!r}): {reason}")
            keys.extend(join_layer_key(block.prefix, stored, name) for name in names)
    if wrong:
        raise Refusal(
            errors.FC_R11,
            f"{block.prefix}: the tensors of {len(wrong)} of its {block.num_modules} modules "
            f"contradict {block.evidence}: {'; '.join(wrong)}{_module_table(block)}",
            keys=tuple(keys),
            hint="check that the arguments belong to this checkpoint",
        ).to_error()


def _module_table(block: BlockLayout) -> str:
    """The full index table of a hybrid block, for refusals."""
    return f"; its modules (dense: torch_dist index, symbols) are {block.index_table()}"


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
