# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""State-space mixers: the family of each one, its dimensions and its fused projections.

The mixer layers of a hybrid model (symbol ``M``) hold a Mamba-2 mixer
(``MambaMixer``, which the default hybrid stack spec builds) or a Gated
DeltaProduct mixer (``GatedDeltaProductMixer``, which
``gated_delta_product_stack_spec`` builds). A mixer's input projection
``in_proj`` concatenates one projection per component along its first axis,
and its depthwise convolution ``conv1d`` runs over some of them; each
tensor-parallel rank holds its shard of every component, in order (a
:class:`~.part_layout.PartLayout`, one entry per component). torch_dist stores
every component of the weight and bias of these projections as a tensor of
its own (Megatron's split factory: ``mixer.in_proj.weight.z``, TP-canonical);
fsdp_dtensor stores each fused tensor whole, as its ranks' local tensors one
after the other (rank-major).

:func:`mixer_layout` decides the family of every mixer, never from the fused
widths alone:

1. **Names.** The torch_dist parts name one family: ``z, x, B, C, dt``
   (Mamba-2), ``z, V0..V(M-1), K0..K(M-1), Q, b0..b(M-1), a`` (Gated
   DeltaProduct with ``M`` Householder steps), ``query, key, value, z, beta,
   alpha`` (Gated DeltaNet) or ``query, key, value, z, f, b, w`` (Gated
   DeltaNet 2). fsdp_dtensor's ``mixer.conv1d_weight`` is Mamba-2's; its
   ``mixer.conv1d.weight`` is Gated DeltaProduct's or a legacy Mamba-2 store's.
2. **Spec.** ``args.spec`` builds Gated DeltaProduct mixers (its
   ``gated_delta_product_stack_spec``, or the alias ``gdp_stack_spec``), any
   other spec Mamba-2 mixers; the names must allow that family.
3. **Widths.** Of the four families, exactly one may have the fused widths
   the arguments give it (:func:`family_layouts`): the one the spec builds.
   Gated DeltaProduct needs ``args.mamba_num_heads`` (its mixer asserts it)
   and as many Householder steps as ``args.gdp_num_householder`` (3 for a
   store that predates the argument, as Megatron back-fills it).

In torch_dist, every part must then have the name and size the family's
layout gives it, and a Gated DeltaProduct mixer's ``A_log`` and ``dt_bias``
must have ``H`` rows and its ``norm.weight`` ``d`` (a Megatron-FSDP job above
TP 1 saves one rank's shard of them). A contradiction anywhere is refused
(FC-R10).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from .. import errors
from ..engine.metadata import TensorEntry
from ..errors import Refusal, RefusalError
from .args_view import ArgsView
from .blocks import BlockLayout
from .keys import CONV1D, IN_PROJ, MIXER_TENSOR, KeyGrammarError, MixerTensor, split_mixer_tensor
from .logical import LogicalId, LogicalSlice
from .names import FSDP_DTENSOR
from .part_layout import PartLayout
from .structure import decompose

MAMBA2, GDP, GDN, GDN2 = "mamba2", "gdp", "gdn", "gdn2"
FAMILY_NAMES: Mapping[str, str] = {
    MAMBA2: "Mamba-2",
    GDP: "Gated DeltaProduct",
    GDN: "Gated DeltaNet",
    GDN2: "Gated DeltaNet 2",
}
"""The families of layers with fused ``in_proj``/``conv1d`` projections, by id."""

HYBRID_SPEC_MODULES = (
    "megatron.core.models.hybrid.hybrid_layer_specs",
    "megatron.core.models.mamba.mamba_layer_specs",
)
"""Modules exporting the hybrid stack specs (the second re-exports the first)."""

GDP_SPEC_NAMES = frozenset({"gated_delta_product_stack_spec", "gdp_stack_spec"})
"""Names of the Gated DeltaProduct stack spec in those modules (the second is an alias)."""

DEFAULT_HOUSEHOLDERS = 3
"""``gdp_num_householder`` of a store that predates the argument: Megatron back-fills 3 on load
(``load_args_from_checkpoint``), its ``TransformerConfig`` default."""

MAMBA_EXPAND = 2
"""``MambaMixer``'s default ``expand``: its inner size is ``expand x hidden_size`` when
``mamba_num_heads`` is unset."""

_HOUSEHOLDER_PART = re.compile(r"V(?:0|[1-9][0-9]*)")
_MIXER_VECTOR_KEY = re.compile(r"(?:^|\.)mixer\.(?:A_log|dt_bias|norm\.weight)$")
_MIXER_VECTOR = re.compile(r"(?P<mixer>(?:.+\.)?mixer)\.(?P<name>A_log|dt_bias|norm\.weight)")
_GDP_VECTOR_PART = {"A_log": "a", "dt_bias": "a", "norm.weight": "z"}
"""The ``in_proj`` component as long as each per-head (``H``) or per-channel (``d``) parameter of a
Gated DeltaProduct mixer (tensor-parallel, stored unfused in both formats)."""
_FUSED_TENSORS = ((IN_PROJ, "weight"), (CONV1D, "weight"))
"""The fused tensors every mixer of every family has (their biases are optional)."""


def spec_name(args: ArgsView) -> str | None:
    """``--spec`` as ``<module>.<name>`` (``None`` when the default spec is used)."""
    spec = args.get("spec", None)
    if spec is None:
        return None
    return ".".join(str(part) for part in spec) if isinstance(spec, (list, tuple)) else str(spec)


def is_gdp_spec(args: ArgsView) -> bool:
    """Whether ``args.spec`` is the Gated DeltaProduct stack spec of a hybrid spec module."""
    module, _, name = (spec_name(args) or "").rpartition(".")
    return module in HYBRID_SPEC_MODULES and name in GDP_SPEC_NAMES


@dataclass(frozen=True, slots=True)
class FamilyLayout:
    """The fused projections of one family's mixers, as the arguments give them.

    Attributes:
        family: The family id.
        in_proj: The components of ``in_proj`` and their global sizes.
        conv1d: The components of ``conv1d`` and their global sizes.
        dims: The dimensions and the arguments they come from, for messages.
    """

    family: str
    in_proj: PartLayout
    conv1d: PartLayout
    dims: str

    def projection(self, name: str) -> PartLayout:
        """The layout of the projection ``name`` (``in_proj`` or ``conv1d``)."""
        return self.in_proj if name == IN_PROJ else self.conv1d

    @property
    def widths(self) -> str:
        """``in_proj <rows> rows, conv1d <rows> rows``."""
        return f"in_proj {self.in_proj.fused_rows} rows, conv1d {self.conv1d.fused_rows} rows"


def gdp_parts(
    d_inner: int, group_state: int, heads: int, householders: int
) -> tuple[PartLayout, PartLayout]:
    """The ``in_proj`` and ``conv1d`` components of a Gated DeltaProduct mixer.

    ``in_proj`` is ``z, V0..V(M-1), K0..K(M-1), Q, b0..b(M-1), a`` and
    ``conv1d`` is ``V0..V(M-1), K0..K(M-1), Q`` for ``M`` Householder steps,
    as ``_get_in_proj_checkpoint_split_layout`` and
    ``_get_conv_checkpoint_split_layout`` split them (pinned by a test).

    Args:
        d_inner: ``mamba_num_heads x mamba_head_dim`` (``z`` and each ``V``).
        group_state: ``mamba_num_groups x mamba_state_dim`` (each ``K`` and ``Q``).
        heads: ``mamba_num_heads`` (each ``b`` and ``a``).
        householders: ``M``.
    """
    values = tuple(f"V{i}" for i in range(householders))
    keys = tuple(f"K{i}" for i in range(householders))
    betas = tuple(f"b{i}" for i in range(householders))
    in_proj = PartLayout(
        ("z", *values, *keys, "Q", *betas, "a"),
        (d_inner,) * (1 + householders)
        + (group_state,) * (householders + 1)
        + (heads,) * (householders + 1),
    )
    conv1d = PartLayout(
        (*values, *keys, "Q"), (d_inner,) * householders + (group_state,) * (householders + 1)
    )
    return in_proj, conv1d


def mamba2_parts(d_inner: int, group_state: int, heads: int) -> tuple[PartLayout, PartLayout]:
    """The ``in_proj`` (``z, x, B, C, dt``) and ``conv1d`` (``x, B, C``) components of a Mamba-2
    mixer (``MambaMixer.sharded_state_dict``)."""
    return (
        PartLayout(("z", "x", "B", "C", "dt"), (d_inner, d_inner, group_state, group_state, heads)),
        PartLayout(("x", "B", "C"), (d_inner, group_state, group_state)),
    )


def gdn_parts(qk: int, v: int, value_heads: int) -> tuple[PartLayout, PartLayout]:
    """The ``in_proj`` (``query, key, value, z, beta, alpha``) and ``conv1d`` (``query, key,
    value``) components of a Gated DeltaNet (``in_proj_split_names`` of ``GatedDeltaNet``)."""
    return (
        PartLayout(
            ("query", "key", "value", "z", "beta", "alpha"),
            (qk, qk, v, v, value_heads, value_heads),
        ),
        PartLayout(("query", "key", "value"), (qk, qk, v)),
    )


def gdn2_parts(qk: int, v: int) -> tuple[PartLayout, PartLayout]:
    """The ``in_proj`` (``query, key, value, z, f, b, w``) and ``conv1d`` (``query, key, value``)
    components of a Gated DeltaNet 2 (``in_proj_split_names`` of ``GatedDeltaNet2``)."""
    return (
        PartLayout(("query", "key", "value", "z", "f", "b", "w"), (qk, qk, v, v, qk, qk, v)),
        PartLayout(("query", "key", "value"), (qk, qk, v)),
    )


def householders_of(args: ArgsView) -> tuple[int, str]:
    """The number of Householder steps of the Gated DeltaProduct mixers, and its evidence."""
    value = args.get("gdp_num_householder", None)
    if value is None:
        return DEFAULT_HOUSEHOLDERS, f"args.gdp_num_householder unset: {DEFAULT_HOUSEHOLDERS}"
    return value, f"args.gdp_num_householder={value}"


def _sizes(args: ArgsView, names: Iterable[str]) -> tuple[dict[str, int], str | None]:
    """The arguments ``names``, and why they are no sizes if they are not (``None``: they are)."""
    values = {name: args.get(name, None) for name in names}
    wrong = [f"args.{name}={value}" for name, value in values.items() if not _is_size(value)]
    return values, (", ".join(wrong) + ": sizes must be positive integers" if wrong else None)


def _is_size(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _gdp_layout(args: ArgsView) -> FamilyLayout | str:
    sizes, wrong = _sizes(
        args, ("mamba_num_heads", "mamba_head_dim", "mamba_num_groups", "mamba_state_dim")
    )
    if wrong is not None:
        return wrong
    householders, evidence = householders_of(args)
    if not _is_size(householders):
        return f"{evidence}: the number of Householder steps must be a positive integer"
    heads, head_dim = sizes["mamba_num_heads"], sizes["mamba_head_dim"]
    groups, state = sizes["mamba_num_groups"], sizes["mamba_state_dim"]
    d_inner, group_state = heads * head_dim, groups * state
    dims = (
        f"d={d_inner} (args.mamba_num_heads={heads} x args.mamba_head_dim={head_dim}), "
        f"GN={group_state} (args.mamba_num_groups={groups} x args.mamba_state_dim={state}), "
        f"H={heads}, M={householders} ({evidence})"
    )
    return FamilyLayout(GDP, *gdp_parts(d_inner, group_state, heads, householders), dims)


def _mamba2_layout(args: ArgsView) -> FamilyLayout | str:
    sizes, wrong = _sizes(args, ("mamba_head_dim", "mamba_num_groups", "mamba_state_dim"))
    if wrong is not None:
        return wrong
    head_dim, groups = sizes["mamba_head_dim"], sizes["mamba_num_groups"]
    state = sizes["mamba_state_dim"]
    heads = args.get("mamba_num_heads", None)
    if heads is None:
        hidden = args.get("hidden_size", None)
        if not _is_size(hidden) or (hidden * MAMBA_EXPAND) % head_dim:
            return f"args.mamba_num_heads is unset and args.hidden_size={hidden} gives no heads"
        heads = hidden * MAMBA_EXPAND // head_dim
        source = f"args.hidden_size={hidden} x {MAMBA_EXPAND} / args.mamba_head_dim"
    elif not _is_size(heads):
        return f"args.mamba_num_heads={heads}: sizes must be positive integers"
    else:
        source = f"args.mamba_num_heads={heads}"
    d_inner, group_state = heads * head_dim, groups * state
    dims = f"d={d_inner}, GN={group_state}, H={heads} ({source})"
    return FamilyLayout(MAMBA2, *mamba2_parts(d_inner, group_state, heads), dims)


def _gdn_layouts(args: ArgsView) -> dict[str, FamilyLayout | str]:
    names = (
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_value_head_dim",
        "linear_num_value_heads",
    )
    sizes, wrong = _sizes(args, names)
    if wrong is not None:
        return {GDN: wrong, GDN2: wrong}
    qk = sizes["linear_key_head_dim"] * sizes["linear_num_key_heads"]
    v = sizes["linear_value_head_dim"] * sizes["linear_num_value_heads"]
    value_heads = sizes["linear_num_value_heads"]
    dims = f"qk={qk}, v={v}, value heads={value_heads} (args.linear_*)"
    return {
        GDN: FamilyLayout(GDN, *gdn_parts(qk, v, value_heads), dims),
        GDN2: FamilyLayout(GDN2, *gdn2_parts(qk, v), dims),
    }


def family_layouts(args: ArgsView) -> dict[str, FamilyLayout | str]:
    """Every family's fused projections as the arguments give them, or why they cannot.

    Mamba-2 and Gated DeltaProduct: ``d = heads x mamba_head_dim``,
    ``GN = mamba_num_groups x mamba_state_dim``, ``H = heads`` (Mamba-2's heads
    default to ``expand x hidden_size / mamba_head_dim``); Gated DeltaNet:
    ``qk = linear_key_head_dim x linear_num_key_heads``,
    ``v = linear_value_head_dim x linear_num_value_heads``.
    """
    return {MAMBA2: _mamba2_layout(args), GDP: _gdp_layout(args), **_gdn_layouts(args)}


@dataclass(frozen=True, slots=True)
class FusedProjection:
    """The weight or bias of a mixer's fused ``in_proj`` or ``conv1d``.

    Attributes:
        lid: The logical id of the whole tensor.
        family: The mixer's family.
        parts: Its components, in the order each tensor-parallel rank concatenates them.
        keys: The stored source keys that hold it (its parts', in torch_dist).
    """

    lid: LogicalId
    family: str
    parts: PartLayout
    keys: tuple[str, ...]

    parts_noun: ClassVar[str] = "parts"
    """What its parts are called in messages."""

    @property
    def key(self) -> str:
        """A stored source key that holds it (for messages)."""
        return self.keys[0]

    @property
    def kind(self) -> str:
        """What it is, for messages (``Gated DeltaProduct in_proj/conv1d``)."""
        return f"{FAMILY_NAMES[self.family]} in_proj/conv1d"


Mixer = tuple[str | None, int | None, str]
"""A mixer: its block, dense layer and module path (``mixer``, ``<child>.mixer``)."""


@dataclass(frozen=True)
class MixerLayout:
    """The model's state-space mixers and their fused projections.

    Attributes:
        families: The family of every mixer.
        projections: Every fused projection tensor, by the logical id of the whole tensor.
    """

    families: Mapping[Mixer, str]
    projections: Mapping[LogicalId, FusedProjection]

    def of(self, lid: LogicalId) -> FusedProjection | None:
        """The fused projection ``lid`` is (or is a part of), if any."""
        return self.projections.get(lid.whole)

    def families_json(self) -> dict[str, dict[str, int]]:
        """The number of mixers of each family in each block."""
        counts: dict[str, dict[str, int]] = {}
        for (block, _, _), family in self.families.items():
            kinds = counts.setdefault(block or "", {})
            kinds[family] = kinds.get(family, 0) + 1
        return {block: dict(sorted(kinds.items())) for block, kinds in sorted(counts.items())}


@dataclass
class _Stored:
    """One fused projection tensor of a mixer as the source stores it: whole, or by part."""

    tensor: MixerTensor
    slices: dict[str | None, LogicalSlice]

    @property
    def rows(self) -> int:
        """Rows of the whole tensor (the sum of its parts')."""
        return sum(piece.shape[0] for piece in self.slices.values())

    @property
    def keys(self) -> tuple[str, ...]:
        """The stored keys that hold it, sorted."""
        return tuple(sorted(piece.key for piece in self.slices.values()))

    @property
    def parts(self) -> dict[str, int]:
        """The rows of each part it is stored as (none when it is stored whole)."""
        return {part: piece.shape[0] for part, piece in self.slices.items() if part is not None}


def mixer_layout(
    args: ArgsView, model: Mapping[str, TensorEntry], blocks: Sequence[BlockLayout], source: str
) -> tuple[MixerLayout, str]:
    """The family and fused projections of every state-space mixer of the model, and the evidence.

    Args:
        args: The checkpoint's training arguments.
        model: The source's model tensors by model-relative key.
        blocks: The model's blocks.
        source: The source format.

    Raises:
        RefusedError: A mixer's family cannot be told, or the names, the spec,
            the widths or the parts contradict one another (FC-R10).
    """
    stored = _stored_mixers(model, blocks, source)
    vectors = _mixer_vectors(model, blocks, source)
    if not stored:
        return MixerLayout({}, {}), "no state-space mixer (mixer.in_proj or mixer.conv1d tensors)"
    spec_family = GDP if is_gdp_spec(args) else MAMBA2
    spec = f"args.spec={spec_name(args)!r} builds {FAMILY_NAMES[spec_family]} mixers"
    layouts = family_layouts(args)
    families: dict[Mixer, str] = {}
    projections: dict[LogicalId, FusedProjection] = {}
    for mixer, tensors in stored.items():
        layout = _discriminate(mixer, tensors, spec_family, spec, layouts, args, source)
        if spec_family == GDP:
            _check_vectors(_where(mixer), vectors.get(mixer, {}), layout)
        families[mixer] = spec_family
        for found in tensors.values():
            (lid,) = {piece.lid.whole for piece in found.slices.values()}
            parts = layout.projection(found.tensor.projection)
            projections[lid] = FusedProjection(lid, spec_family, parts, found.keys)
    chosen = layouts[spec_family]
    assert isinstance(chosen, FamilyLayout)  # checked by _discriminate
    others = "; ".join(
        f"{FAMILY_NAMES[family]}: "
        + (layout.widths if isinstance(layout, FamilyLayout) else f"not derivable, {layout}")
        for family, layout in layouts.items()
        if family != spec_family
    )
    evidence = (
        f"{spec}; {len(stored)} mixers stored "
        + ("whole" if source == FSDP_DTENSOR else "as parts")
        + f"; {chosen.dims}: {chosen.widths}, the only family whose widths the stored ones match "
        f"({others})"
    )
    return MixerLayout(families, projections), evidence


def _stored_mixers(
    model: Mapping[str, TensorEntry], blocks: Sequence[BlockLayout], source: str
) -> dict[Mixer, dict[tuple[str, str], _Stored]]:
    """The fused projection tensors of every mixer, by mixer and (projection, leaf).

    Malformed keys are left out (the planner refuses them with the grammar error).
    """
    mixers: dict[Mixer, dict[tuple[str, str], _Stored]] = {}
    for relative, entry in sorted(model.items()):
        if MIXER_TENSOR.search(relative) is None:
            continue
        try:
            slices = decompose(relative, entry, blocks, source)
        except KeyGrammarError:
            continue
        for piece in slices:
            tensor = split_mixer_tensor(piece.lid.path, piece.lid.leaf)
            if tensor is None:
                continue
            tensors = mixers.setdefault((piece.lid.block, piece.lid.layer, tensor.mixer), {})
            found = tensors.setdefault((tensor.projection, tensor.leaf), _Stored(tensor, {}))
            found.slices[piece.lid.part] = piece
    return mixers


def _mixer_vectors(
    model: Mapping[str, TensorEntry], blocks: Sequence[BlockLayout], source: str
) -> dict[Mixer, dict[str, LogicalSlice]]:
    """The ``A_log``, ``dt_bias`` and ``norm.weight`` of every mixer, by mixer and name."""
    vectors: dict[Mixer, dict[str, LogicalSlice]] = {}
    for relative, entry in sorted(model.items()):
        if _MIXER_VECTOR_KEY.search(relative) is None:
            continue
        try:
            slices = decompose(relative, entry, blocks, source)
        except KeyGrammarError:
            continue
        for piece in slices:
            match = _MIXER_VECTOR.fullmatch(f"{piece.lid.path}.{piece.lid.leaf}")
            if match is not None and piece.lid.part is None:
                mixer = (piece.lid.block, piece.lid.layer, match.group("mixer"))
                vectors.setdefault(mixer, {})[match.group("name")] = piece
    return vectors


def _check_vectors(where: str, vectors: Mapping[str, LogicalSlice], layout: FamilyLayout) -> None:
    """A Gated DeltaProduct mixer's ``A_log`` and ``dt_bias`` have ``H`` rows and its
    ``norm.weight`` ``d`` (FC-R10): a Megatron-FSDP job above TP 1 holds and saves one
    tensor-parallel rank's shard of each, which no conversion can complete."""
    sizes = dict(zip(layout.in_proj.parts, layout.in_proj.sizes))
    wrong = {
        name: piece
        for name, piece in sorted(vectors.items())
        if piece.shape[:1] != (sizes[_GDP_VECTOR_PART[name]],)
    }
    if wrong:
        found = ", ".join(f"{name} {tuple(piece.shape)}" for name, piece in wrong.items())
        raise _refusal(
            f"{where}: {found}, but a Gated DeltaProduct mixer's A_log and dt_bias have "
            f"{sizes['a']} rows and its norm.weight {sizes['z']} ({layout.dims})",
            tuple(piece.key for piece in wrong.values()),
            hint="a Megatron-FSDP job above TP 1 saves one tensor-parallel rank's shard of these "
            "parameters (see the Gated DeltaProduct limitations); convert a store that holds them "
            "whole",
        )


def megatron_fsdp_warnings(mixers: MixerLayout, target: str, pack_tp: int | None) -> list[str]:
    """Why no current Megatron-FSDP job loads an fsdp_dtensor ``target`` of Gated DeltaProduct
    mixers packed above TP 1: it takes their ``A_log``, ``dt_bias`` and ``norm.weight``
    (tensor-parallel without a ``partition_dim``) for replicated parameters of one rank's shard."""
    if target != FSDP_DTENSOR or pack_tp is None or pack_tp <= 1:
        return []
    if GDP not in set(mixers.families.values()):
        return []
    return [
        f"Gated DeltaProduct mixers packed for --fsdp-tp {pack_tp}: current Megatron-FSDP jobs "
        "above TP 1 hold one tensor-parallel rank's shard of a mixer's A_log, dt_bias and "
        "norm.weight, so no current Megatron job loads this fsdp_dtensor store (see the Gated "
        "DeltaProduct limitations)"
    ]


def _discriminate(
    mixer: Mixer,
    tensors: Mapping[tuple[str, str], _Stored],
    spec_family: str,
    spec: str,
    layouts: Mapping[str, FamilyLayout | str],
    args: ArgsView,
    source: str,
) -> FamilyLayout:
    """The layout of ``mixer``'s family, which ``args.spec`` says is ``spec_family``.

    Raises:
        RefusedError: See :func:`mixer_layout` (FC-R10).
    """
    where = _where(mixer)
    keys = tuple(key for found in tensors.values() for key in found.keys)
    for found in tensors.values():
        _check_consistent(where, found)
    missing = [f"{name}.{leaf}" for name, leaf in _FUSED_TENSORS if (name, leaf) not in tensors]
    if missing:
        raise _refusal(f"{where} has no {' or '.join(missing)}", keys)
    named, names = _named_families(where, tensors, source)
    if spec_family not in named:
        raise _refusal(
            f"{where} is a {' or '.join(FAMILY_NAMES[family] for family in sorted(named))} "
            f"mixer by {names}, but {spec}",
            keys,
            hint="check args.spec against the model that saved the store",
        )
    if spec_family == GDP:
        _check_householders(where, tensors, args, keys)
    layout = layouts[spec_family]
    if not isinstance(layout, FamilyLayout):
        raise _refusal(
            f"{where} is a {FAMILY_NAMES[spec_family]} mixer, but its dimensions cannot be "
            f"derived from the arguments: {layout}",
            keys,
        )
    _check_widths(where, tensors, spec_family, spec, layouts, keys)
    for found in tensors.values():
        _check_parts(where, found, layout, keys)
    return layout


def _where(mixer: Mixer) -> str:
    block, layer, path = mixer
    return path if block is None else f"{block}[{layer}].{path}"


def _check_consistent(where: str, found: _Stored) -> None:
    """A stored tensor is whole or parts (not both), and its parts share a dtype and trailing
    dimensions (FC-R10)."""
    pieces = list(found.slices.values())
    if None in found.slices and len(pieces) > 1:
        raise _refusal(
            f"{where}: {found.tensor.projection}.{found.tensor.leaf} is stored whole and as parts",
            found.keys,
        )
    if len({piece.dtype for piece in pieces}) > 1 or len({piece.shape[1:] for piece in pieces}) > 1:
        raise _refusal(
            f"{where}: the parts of {found.tensor.projection}.{found.tensor.leaf} differ in dtype "
            "or in their dimensions after the first",
            found.keys,
        )


def _named_families(
    where: str, tensors: Mapping[tuple[str, str], _Stored], source: str
) -> tuple[frozenset[str], str]:
    """The families the stored names allow, and the names (a noun phrase, for messages).

    Raises:
        RefusedError: torch_dist stores a fused projection whole, or its
            ``in_proj`` parts are no family's (FC-R10).
    """
    if any(found.tensor.conv_parameter for found in tensors.values()):
        return frozenset({MAMBA2}), "its convolution parameter conv1d_weight"
    if source == FSDP_DTENSOR:
        return frozenset({GDP, MAMBA2}), "its fused in_proj and conv1d.weight"
    # Megatron's split factory stores every part of a Gated DeltaProduct mixer's in_proj and
    # conv1d weights and biases; a whole (TP-bound, rank-major) torch_dist tensor would need
    # identity at pack_tp == src_tp only. Mamba-2's in_proj.bias is the first such key.
    whole = [key for found in tensors.values() if None in found.slices for key in found.keys]
    if whole:
        raise _refusal(
            f"{where}: torch_dist stores a mixer's in_proj and conv1d as one tensor per part, but "
            "the store holds them whole",
            tuple(whole),
        )
    parts = tensors[IN_PROJ, "weight"].parts
    named = _family_of_parts(parts)
    if named is None:
        raise _refusal(
            f"{where}: its in_proj parts {', '.join(sorted(parts))} are no mixer family's",
            tensors[IN_PROJ, "weight"].keys,
            hint="check that the store was saved by Megatron",
        )
    family, order = named
    return frozenset({family}), f"its in_proj parts {', '.join(_ordered(parts, order))}"


def _family_of_parts(parts: Iterable[str]) -> tuple[str, tuple[str, ...]] | None:
    """The family whose ``in_proj`` components are exactly ``parts`` (any number of Householder
    steps for Gated DeltaProduct) and their order, if any."""
    names = set(parts)
    householders = sum(1 for name in names if _HOUSEHOLDER_PART.fullmatch(name))
    candidates = {
        MAMBA2: mamba2_parts(1, 1, 1)[0],
        GDP: gdp_parts(1, 1, 1, householders)[0] if householders else None,
        GDN: gdn_parts(1, 1, 1)[0],
        GDN2: gdn2_parts(1, 1)[0],
    }
    for family, layout in candidates.items():
        if layout is not None and names == set(layout.parts):
            return family, layout.parts
    return None


def _ordered(parts: Mapping[str, int], order: Sequence[str]) -> dict[str, int]:
    """``parts`` in the order ``order`` gives them, any others after them by name."""
    rank = {name: index for index, name in enumerate(order)}
    return dict(sorted(parts.items(), key=lambda item: (rank.get(item[0], len(rank)), item[0])))


def _check_householders(
    where: str, tensors: Mapping[tuple[str, str], _Stored], args: ArgsView, keys: tuple[str, ...]
) -> None:
    """A Gated DeltaProduct mixer stored as parts has ``args.gdp_num_householder`` Householder
    steps (FC-R10)."""
    parts = tensors[IN_PROJ, "weight"].parts
    stored = sum(1 for name in parts if _HOUSEHOLDER_PART.fullmatch(name))
    householders, evidence = householders_of(args)
    if parts and stored != householders:
        raise _refusal(
            f"{where}: its in_proj holds {stored} Householder steps (V0..V{stored - 1}), but "
            f"{evidence}",
            keys,
            hint="check args.gdp_num_householder against the model that saved the store",
        )


def _check_widths(
    where: str,
    tensors: Mapping[tuple[str, str], _Stored],
    spec_family: str,
    spec: str,
    layouts: Mapping[str, FamilyLayout | str],
    keys: tuple[str, ...],
) -> None:
    """Exactly one family has the stored widths as the arguments give it: ``spec_family``
    (FC-R10)."""
    matches = [
        family
        for family, layout in layouts.items()
        if isinstance(layout, FamilyLayout)
        and all(
            found.rows == layout.projection(found.tensor.projection).fused_rows
            for found in tensors.values()
        )
    ]
    if matches == [spec_family]:
        return
    stored = ", ".join(
        f"{name}.{leaf} {found.rows} rows" for (name, leaf), found in sorted(tensors.items())
    )
    layout = layouts[spec_family]
    assert isinstance(layout, FamilyLayout)  # checked by the caller
    if spec_family in matches:
        others = " and ".join(FAMILY_NAMES[family] for family in matches if family != spec_family)
        message = (
            f"{where}: its widths ({stored}) are what the arguments give "
            f"{FAMILY_NAMES[spec_family]} mixers ({layout.dims}) and {others} mixers alike, so "
            "they cannot tell the family"
        )
    else:
        match = (
            f"; they are what the arguments give {' and '.join(FAMILY_NAMES[f] for f in matches)} "
            "mixers"
            if matches
            else ""
        )
        message = (
            f"{where}: its widths ({stored}) are not what the arguments give "
            f"{FAMILY_NAMES[spec_family]} mixers ({layout.dims}: {layout.widths}), which {spec}"
            f"{match}"
        )
    raise _refusal(message, keys)


def _check_parts(where: str, found: _Stored, layout: FamilyLayout, keys: tuple[str, ...]) -> None:
    """A tensor stored as parts has exactly its family's parts, of their sizes (FC-R10)."""
    parts = found.parts
    if not parts:
        return
    expected = layout.projection(found.tensor.projection)
    if parts != dict(zip(expected.parts, expected.sizes)):
        raise _refusal(
            f"{where}: {found.tensor.projection}.{found.tensor.leaf} is stored as parts "
            f"{_rows(_ordered(parts, expected.parts))}, but a {FAMILY_NAMES[layout.family]} "
            f"{found.tensor.projection} has {_rows(dict(zip(expected.parts, expected.sizes)))} "
            f"({layout.dims})",
            keys,
        )


def _rows(parts: Mapping[str, int]) -> str:
    return ", ".join(f"{part}:{rows}" for part, rows in parts.items())


def _refusal(
    message: str, keys: tuple[str, ...], hint: str = "check that the arguments belong to this store"
) -> RefusalError:
    return Refusal(errors.FC_R10, message, keys=tuple(sorted(keys)), hint=hint).to_error()
