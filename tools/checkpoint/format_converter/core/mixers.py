# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mixers: the family of each one, its dimensions and its fused projections.

The mixer layers of a hybrid model (symbol ``M``) hold a Mamba-2 mixer
(``MambaMixer``, which the default hybrid stack spec builds) or a Gated
DeltaProduct mixer (``GatedDeltaProductMixer``, which
``gated_delta_product_stack_spec`` builds), under ``mixer``. A Gated DeltaNet
layer (hybrid symbol ``G``, or a GPT layer of linear attention through
``experimental_attention_variant``) holds a ``GatedDeltaNet`` or a
``GatedDeltaNet2`` under ``self_attention``; this module calls it a mixer too.
A mixer's input projection ``in_proj`` concatenates one projection per
component along its first axis, and its depthwise convolution ``conv1d`` runs
over some of them; each tensor-parallel rank holds its shard of every
component, in order (a :class:`~.part_layout.PartLayout`, one entry per
component). torch_dist stores every component of the weight and bias of these
projections as a tensor of its own (Megatron's split factory:
``mixer.in_proj.weight.z``, TP-canonical); fsdp_dtensor stores each fused
tensor whole, as its ranks' local tensors one after the other (rank-major).
Mamba-2's convolution is the mixer's own parameters ``conv1d_weight`` and
``conv1d_bias`` (the names fsdp_dtensor stores; ``MambaMixer.sharded_state_dict``
keeps torch_dist's ``conv1d.weight``/``conv1d.bias``). The split factories of
Mamba-2 and Gated DeltaNet leave an ``in_proj.bias`` whole (``MambaMixer``
asserts ``bias=False``, and the layer specs build no Gated DeltaNet with one,
so Megatron's stores hold none): torch_dist stores the ranks' local tensors one
after the other, as fsdp_dtensor does, which converts only at the size it was
saved at (:func:`check_tp_bound`).

:func:`mixer_layout` decides the family of every mixer, never from the fused
widths alone:

1. **Names.** The torch_dist parts name one family: ``z, x, B, C, dt``
   (Mamba-2), ``z, V0..V(M-1), K0..K(M-1), Q, b0..b(M-1), a`` (Gated
   DeltaProduct with ``M`` Householder steps), ``query, key, value, z, beta,
   alpha`` (Gated DeltaNet) or ``query, key, value, z, f, b, w`` (Gated
   DeltaNet 2). fsdp_dtensor's ``mixer.conv1d_weight`` is Mamba-2's; its
   ``mixer.conv1d.weight`` is Gated DeltaProduct's or a legacy Mamba-2 store's;
   its ``self_attention.in_proj`` is a Gated DeltaNet's of either kind.
2. **What Megatron builds there.** Under ``mixer``, ``args.spec`` builds Gated
   DeltaProduct mixers (its ``gated_delta_product_stack_spec``, or the alias
   ``gdp_stack_spec``), any other spec Mamba-2 mixers. Under
   ``self_attention``, ``args.experimental_attention_variant`` ``gdn2`` builds
   Gated DeltaNet 2 layers, any other value Gated DeltaNet layers (as the hybrid
   stack and the GPT linear-attention spec decide); a GPT decoder (a
   ``TransformerBlock``) builds them only for a Gated DeltaNet variant, at the
   layers ``args.linear_attention_freq`` makes linear attention. The names must
   allow that family: a model may mix families, one per module path.
3. **Widths.** Of the families the names allow, exactly one may have the fused
   widths the arguments give it (:func:`family_layouts`): the one Megatron
   builds. Gated DeltaProduct needs ``args.mamba_num_heads`` (its mixer
   asserts it) and as many Householder steps as ``args.gdp_num_householder``
   (3 for a store that predates the argument, as Megatron back-fills it).

In torch_dist, every part must then have the name and size the family's
layout gives it, and the mixer's tensor-parallel per-head and per-channel
parameters must be whole (:attr:`FamilyLayout.vectors`: a Megatron-FSDP job
above TP 1 saves one rank's shard of a Gated DeltaProduct mixer's). An
fsdp_dtensor store saved before ``MambaMixer`` owned its convolution names it
``mixer.conv1d.weight`` like a Gated DeltaProduct mixer's: the spec tells. A
contradiction anywhere is refused (FC-R10).
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from .. import errors
from ..engine.metadata import TensorEntry
from ..errors import InternalError, Refusal, RefusalError
from .args_view import ArgsView
from .blocks import TRANSFORMER_BLOCK, BlockLayout
from .keys import (
    CONV1D,
    IN_PROJ,
    MIXER_MODULES,
    MIXER_TENSOR,
    KeyGrammarError,
    MixerTensor,
    is_linear_attention,
    join_mixer_tensor,
    split_mixer_tensor,
)
from .logical import LogicalId, LogicalSlice
from .names import FSDP_DTENSOR, TORCH_DIST
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

FAMILY_NOUNS: Mapping[str, str] = {MAMBA2: "mixer", GDP: "mixer", GDN: "layer", GDN2: "layer"}
"""What messages call one of a family (a Gated DeltaNet is a layer, as its users know it)."""


def family_noun(family: str, plural: bool = False) -> str:
    """``Gated DeltaNet layer`` or ``Mamba-2 mixers``: a family's name and noun, for messages."""
    return f"{FAMILY_NAMES[family]} {FAMILY_NOUNS[family]}" + ("s" if plural else "")


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
"""``MambaMixer``'s default ``expand`` (no stack spec passes another): its inner size is
``expand x hidden_size`` when ``mamba_num_heads`` is unset."""

GDN2_VARIANT = "gdn2"
"""The ``experimental_attention_variant`` that builds Gated DeltaNet 2 layers; every other value
builds Gated DeltaNet ones (the hybrid stack's ``G`` layers, and the GPT linear-attention spec)."""

GDN_VARIANTS = frozenset({"gdn", GDN2_VARIANT, "gated_delta_net"})
"""``experimental_attention_variant`` values of the Gated DeltaNet family (``gated_delta_net`` is
the deprecated spelling of ``gdn``, which a store saved before the rename records): a
``TransformerBlock`` builds Gated DeltaNet layers only for these
(``is_gated_delta_net_variant``)."""

LINEAR_ATTENTION_FREQ = "linear_attention_freq"
"""The argument that places a ``TransformerBlock``'s Gated DeltaNet layers."""

TORCH_DIST_WHOLE: Mapping[str, frozenset[tuple[str, str]]] = {
    MAMBA2: frozenset({(IN_PROJ, "bias")}),
    GDN: frozenset({(IN_PROJ, "bias")}),
    GDN2: frozenset({(IN_PROJ, "bias")}),
}
"""The fused tensors torch_dist stores whole (``(projection, leaf)``, by family):
``MambaMixer.sharded_state_dict`` splits ``in_proj.weight``, ``conv1d.weight``
and ``conv1d.bias``, and so does the Gated DeltaNet's (``_GDNBase``), never
``in_proj.bias``. ``MambaMixer.__init__`` asserts it has none, and no layer
spec builds a Gated DeltaNet with ``bias=True``; a store from a module built
otherwise may hold one. Such a tensor is the tensor-parallel ranks' local
tensors one after the other in both formats."""

_HOUSEHOLDER_PART = re.compile(r"V(?:0|[1-9][0-9]*)")
_MIXER_VECTOR_KEY = re.compile(r"(?:^|\.)" + MIXER_MODULES + r"\.(?:A_log|dt_bias|D|norm\.weight)$")
_MIXER_VECTOR = re.compile(
    r"(?P<mixer>(?:.+\.)?" + MIXER_MODULES + r")\.(?P<name>A_log|dt_bias|D|norm\.weight)"
)
_SHARD_HINT = (
    "a store that holds one tensor-parallel rank's shard of these parameters cannot be "
    "completed; convert a store that holds them whole"
)
_VECTOR_HINTS: Mapping[str, str] = {
    GDP: "a Megatron-FSDP job above TP 1 saves one tensor-parallel rank's shard of these "
    "parameters (see the Gated DeltaProduct limitations); convert a store that holds them whole"
}
"""How to get a store whose vectors are whole, by family (:data:`_SHARD_HINT` otherwise)."""
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
        vectors: The rows each tensor-parallel per-head or per-channel
            parameter of a mixer may have when whole (``A_log``, ``dt_bias``,
            ...), by name; both formats store them unfused.
    """

    family: str
    in_proj: PartLayout
    conv1d: PartLayout
    dims: str
    vectors: Mapping[str, tuple[int, ...]]

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
    vectors = {"A_log": (heads,), "dt_bias": (heads,), "norm.weight": (d_inner,)}
    return FamilyLayout(GDP, *gdp_parts(d_inner, group_state, heads, householders), dims, vectors)


def _mamba2_layout(args: ArgsView) -> FamilyLayout | str:
    sizes, wrong = _sizes(args, ("mamba_head_dim", "mamba_num_groups", "mamba_state_dim"))
    if wrong is not None:
        return wrong
    head_dim, groups = sizes["mamba_head_dim"], sizes["mamba_num_groups"]
    state = sizes["mamba_state_dim"]
    # MambaMixer.__init__: d_inner = expand x d_model (d_model is hidden_size, MambaLayer); with
    # config.mamba_num_heads, nheads = mamba_num_heads and d_inner = nheads x headdim, otherwise
    # nheads = d_inner // headdim, which must divide; in_proj is z x B C dt (sizes d_inner,
    # d_inner, ngroups x d_state twice, nheads) and conv1d x B C.
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
    # MambaMixer's D has nheads rows, or d_inner with D_has_hdim.
    vectors = {
        "A_log": (heads,),
        "dt_bias": (heads,),
        "D": (heads, d_inner),
        "norm.weight": (d_inner,),
    }
    return FamilyLayout(MAMBA2, *mamba2_parts(d_inner, group_state, heads), dims, vectors)


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
    # _GDNBase.__init__: qk_dim = key_head_dim x num_key_heads, v_dim = value_head_dim x
    # num_value_heads. GatedDeltaNet's A_log and dt_bias have a row per value head;
    # GatedDeltaNet2's dt_bias one per key channel (qk_dim) and its A_log one per key head.
    key_heads, value_heads = sizes["linear_num_key_heads"], sizes["linear_num_value_heads"]
    qk = sizes["linear_key_head_dim"] * key_heads
    v = sizes["linear_value_head_dim"] * value_heads
    dims = f"qk={qk}, v={v}, key heads={key_heads}, value heads={value_heads} (args.linear_*)"
    return {
        GDN: FamilyLayout(
            GDN,
            *gdn_parts(qk, v, value_heads),
            dims,
            {"A_log": (value_heads,), "dt_bias": (value_heads,)},
        ),
        GDN2: FamilyLayout(
            GDN2, *gdn2_parts(qk, v), dims, {"A_log": (key_heads,), "dt_bias": (qk,)}
        ),
    }


def family_layouts(args: ArgsView) -> dict[str, FamilyLayout | str]:
    """Every family's fused projections as the arguments give them, or why they cannot.

    Mamba-2 and Gated DeltaProduct: ``d = heads x mamba_head_dim``,
    ``GN = mamba_num_groups x mamba_state_dim``, ``H = heads`` (Mamba-2's heads
    default to ``expand x hidden_size / mamba_head_dim``); Gated DeltaNet (and
    Gated DeltaNet 2): ``qk = linear_key_head_dim x linear_num_key_heads``,
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

    @property
    def tensor(self) -> MixerTensor:
        """Its mixer, projection and leaf."""
        tensor = split_mixer_tensor(self.lid.path, self.lid.leaf)
        if tensor is None:
            raise InternalError(f"{self.lid} is no fused projection of a mixer")
        return tensor

    @property
    def torch_dist_whole(self) -> bool:
        """Whether torch_dist stores it whole, not as parts (:data:`TORCH_DIST_WHOLE`)."""
        tensor = self.tensor
        return (tensor.projection, tensor.leaf) in TORCH_DIST_WHOLE.get(self.family, frozenset())

    def target_lid(self, fmt: str) -> LogicalId:
        """The logical id of the whole tensor as a store of format ``fmt`` names it.

        A Mamba-2 mixer's convolution is its own parameter in fsdp_dtensor
        (``conv1d_weight``), the ``conv1d`` module's in torch_dist (and in an
        fsdp_dtensor store saved before ``MambaMixer`` owned it).
        """
        tensor = self.tensor
        own = fmt == FSDP_DTENSOR and self.family == MAMBA2 and tensor.projection == CONV1D
        path, leaf = join_mixer_tensor(dataclasses.replace(tensor, conv_parameter=own))
        return dataclasses.replace(self.lid, path=path, leaf=leaf)


Mixer = tuple[str | None, int | None, str]
"""A mixer: its block, dense layer and module path (``mixer``, ``<child>.mixer``, a Gated DeltaNet's
``self_attention``)."""


@dataclass(frozen=True)
class MixerLayout:
    """The model's mixers (state-space mixers and Gated DeltaNets) and their fused projections.

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


def built_family(mixer: str, args: ArgsView) -> tuple[str, str]:
    """The family of mixer Megatron builds at the module path ``mixer``, and why (for messages).

    Under ``self_attention`` a Gated DeltaNet: ``GatedDeltaNet2`` when
    ``args.experimental_attention_variant`` is ``gdn2``, ``GatedDeltaNet``
    otherwise (``HybridBlock`` for ``G`` layers,
    ``get_gated_delta_net_module_spec`` for GPT linear attention). Under
    ``mixer`` the stack spec's mixer: Gated DeltaProduct for its spec, Mamba-2
    for any other.
    """
    if is_linear_attention(mixer):
        variant = args.get("experimental_attention_variant", None)
        family = GDN2 if variant == GDN2_VARIANT else GDN
        return family, (
            f"args.experimental_attention_variant={variant!r} builds {FAMILY_NAMES[family]} "
            "layers"
        )
    family = GDP if is_gdp_spec(args) else MAMBA2
    return family, f"args.spec={spec_name(args)!r} builds {FAMILY_NAMES[family]} mixers"


def linear_attention_layers(freq: object, layers: int) -> tuple[bool, ...] | None:
    """Which of a ``TransformerBlock``'s ``layers`` layers are Gated DeltaNets by
    ``args.linear_attention_freq`` (``get_linear_attention_pattern``), or ``None`` for a value
    that places none of them.

    An int ``N`` makes every ``N``-th layer standard attention (layer ``i`` is
    linear attention iff ``(i + 1) % N != 0``); a list of 0s and 1s, one per
    layer, marks the linear-attention layers with 1.
    """
    if isinstance(freq, int) and not isinstance(freq, bool) and freq > 0:
        return tuple((index + 1) % freq != 0 for index in range(layers))
    if isinstance(freq, list) and len(freq) == layers:
        return tuple(value == 1 for value in freq)
    return None


def _check_transformer_block_layer(
    mixer: Mixer,
    block: BlockLayout | None,
    tensors: Mapping[tuple[str, str], _Stored],
    args: ArgsView,
) -> None:
    """A Gated DeltaNet of a ``TransformerBlock`` (a GPT decoder) is one the arguments build there
    (FC-R10).

    ``HybridBlock`` builds one for every ``G`` layer whatever the variant, but
    a ``TransformerBlock`` only through the GPT linear-attention spec: for an
    ``args.experimental_attention_variant`` of :data:`GDN_VARIANTS`
    (``is_gated_delta_net_variant``), at the layers
    ``args.linear_attention_freq`` makes linear attention
    (:func:`linear_attention_layers`); every other layer is standard
    attention.
    """
    if block is None or block.kind != TRANSFORMER_BLOCK:
        return
    where = _where(mixer)
    keys = tuple(key for found in tensors.values() for key in found.keys)
    variant = args.get("experimental_attention_variant", None)
    freq = args.get(LINEAR_ATTENTION_FREQ, None)
    hint = (
        f"check args.experimental_attention_variant and args.{LINEAR_ATTENTION_FREQ} against the "
        "model that saved the store"
    )
    pattern = linear_attention_layers(freq, block.num_modules)
    if variant not in GDN_VARIANTS or pattern is None:
        raise _refusal(
            f"{where} holds a Gated DeltaNet's in_proj and conv1d, but a TransformerBlock builds "
            f"Gated DeltaNet layers only for args.experimental_attention_variant "
            f"{' or '.join(repr(name) for name in sorted(GDN_VARIANTS))} with "
            f"args.{LINEAR_ATTENTION_FREQ} placing them among its {block.num_modules} layers, "
            f"and the arguments give experimental_attention_variant={variant!r}, "
            f"{LINEAR_ATTENTION_FREQ}={freq!r}",
            keys,
            hint=hint,
        )
    layer = mixer[1]
    if layer is not None and not pattern[layer]:
        raise _refusal(
            f"{where} holds a Gated DeltaNet's in_proj and conv1d, but "
            f"args.{LINEAR_ATTENTION_FREQ}={freq!r} makes layer {layer} of {block.prefix} "
            "standard attention",
            keys,
            hint=hint,
        )


def mixer_layout(
    args: ArgsView, model: Mapping[str, TensorEntry], blocks: Sequence[BlockLayout], source: str
) -> tuple[MixerLayout, str]:
    """The family and fused projections of every mixer of the model, and the evidence.

    Args:
        args: The checkpoint's training arguments.
        model: The source's model tensors by model-relative key.
        blocks: The model's blocks.
        source: The source format.

    Raises:
        RefusedError: A mixer's family cannot be told, or the names, what
            Megatron builds there, the widths or the parts contradict one
            another, or a TransformerBlock holds a Gated DeltaNet where the
            arguments build none (FC-R10).
    """
    stored = _stored_mixers(model, blocks, source)
    vectors = _mixer_vectors(model, blocks, source)
    if not stored:
        return MixerLayout({}, {}), "no mixer (in_proj or conv1d of a mixer or a Gated DeltaNet)"
    layouts = family_layouts(args)
    families: dict[Mixer, str] = {}
    reasons: dict[str, str] = {}
    projections: dict[LogicalId, FusedProjection] = {}
    by_prefix = {block.prefix: block for block in blocks}
    for mixer, tensors in stored.items():
        if is_linear_attention(mixer[2]):
            _check_transformer_block_layer(mixer, by_prefix.get(mixer[0]), tensors, args)
        family, reason = built_family(mixer[2], args)
        layout = _discriminate(mixer, tensors, family, reason, layouts, args, source)
        _check_vectors(_where(mixer), vectors.get(mixer, {}), layout)
        families[mixer] = family
        reasons[family] = reason
        for found in tensors.values():
            (lid,) = {piece.lid.whole for piece in found.slices.values()}
            parts = layout.projection(found.tensor.projection)
            projections[lid] = FusedProjection(lid, family, parts, found.keys)
    evidence = " | ".join(
        _family_evidence(family, reason, list(families.values()).count(family), layouts, source)
        for family, reason in sorted(reasons.items())
    )
    return MixerLayout(families, projections), evidence


def _family_evidence(
    family: str, reason: str, count: int, layouts: Mapping[str, FamilyLayout | str], source: str
) -> str:
    """Why ``count`` mixers are of ``family``: what builds them, their dimensions and widths, and
    the other families' widths."""
    chosen = layouts[family]
    assert isinstance(chosen, FamilyLayout)  # checked by _discriminate
    others = "; ".join(
        f"{FAMILY_NAMES[other]}: "
        + (layout.widths if isinstance(layout, FamilyLayout) else f"not derivable, {layout}")
        for other, layout in layouts.items()
        if other != family
    )
    shared = any(
        isinstance(layout, FamilyLayout) and layout.widths == chosen.widths
        for other, layout in layouts.items()
        if other != family
    )
    match = (
        "the family the stored names give (another family has these widths too)"
        if shared
        else "the only family whose widths the stored ones match"
    )
    return (
        f"{reason}; {count} {FAMILY_NOUNS[family]}s stored "
        + ("whole" if source == FSDP_DTENSOR else "as parts")
        + f"; {chosen.dims}: {chosen.widths}, {match} ({others})"
    )


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
            mixer = (piece.lid.block, piece.lid.layer, tensor.mixer)
            tensors = mixers.setdefault(mixer, {})
            found = tensors.setdefault((tensor.projection, tensor.leaf), _Stored(tensor, {}))
            twice = found.slices.get(piece.lid.part)
            if twice is not None:
                raise _refusal(
                    f"{_where(mixer)}: {tensor.projection}.{tensor.leaf} is stored twice, as "
                    f"{twice.key} and {piece.key}",
                    (twice.key, piece.key),
                    hint="keep one of them (a Mamba-2 store names its convolution conv1d_weight "
                    "and conv1d_bias, or conv1d.weight and conv1d.bias before MambaMixer owned "
                    "them)",
                )
            found.slices[piece.lid.part] = piece
    return mixers


def _mixer_vectors(
    model: Mapping[str, TensorEntry], blocks: Sequence[BlockLayout], source: str
) -> dict[Mixer, dict[str, LogicalSlice]]:
    """The ``A_log``, ``dt_bias``, ``D`` and ``norm.weight`` of every mixer, by mixer and name."""
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
    """A mixer's tensor-parallel per-head and per-channel parameters are whole (FC-R10): each has
    rows :attr:`FamilyLayout.vectors` allows (``A_log`` and ``dt_bias`` ``H`` rows, ``norm.weight``
    ``d``, ...).

    A Megatron-FSDP job above TP 1 holds and saves one tensor-parallel rank's
    shard of a Gated DeltaProduct mixer's, which no conversion can complete.
    """
    wrong = {
        name: piece
        for name, piece in sorted(vectors.items())
        if name in layout.vectors and piece.shape[:1] not in {(n,) for n in layout.vectors[name]}
    }
    if wrong:
        found = ", ".join(f"{name} {tuple(piece.shape)}" for name, piece in wrong.items())
        raise _refusal(
            f"{where}: {found}, but a {family_noun(layout.family)}'s "
            f"{_vector_rows(layout.vectors)} ({layout.dims})",
            tuple(piece.key for piece in wrong.values()),
            hint=_VECTOR_HINTS.get(layout.family, _SHARD_HINT),
        )


def _vector_rows(vectors: Mapping[str, tuple[int, ...]]) -> str:
    """``A_log and dt_bias have 4 rows, its D 4 or 8 and its norm.weight 8``: the rows of each
    parameter, the parameters of the same rows grouped."""
    groups: dict[tuple[int, ...], list[str]] = {}
    for name, rows in vectors.items():
        groups.setdefault(rows, []).append(name)
    phrases = []
    for index, (rows, names) in enumerate(groups.items()):
        verb = ("have " if len(names) > 1 else "has ") if index == 0 else ""
        sizes = " or ".join(str(n) for n in dict.fromkeys(rows))
        phrases.append(f"{' and '.join(names)} {verb}{sizes}" + (" rows" if index == 0 else ""))
    *first, last = [phrases[0], *(f"its {phrase}" for phrase in phrases[1:])]
    return f"{', '.join(first)} and {last}" if first else last


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


def check_tp_bound(mixers: MixerLayout, source: str, src_tp: int, pack_tp: int | None) -> None:
    """A tensor torch_dist stores whole converts only at the size it was saved at (FC-R10).

    Such a tensor (:data:`TORCH_DIST_WHOLE`) is the ranks' local tensors one
    after the other in both formats, so it converts as is, which is right
    only when fsdp_dtensor packs it at the source's tensor-parallel size. An
    fsdp_dtensor source is unpacked at its own size, which its torch_dist
    target keeps.

    Args:
        mixers: The model's mixers.
        source: The source format.
        src_tp: The tensor-parallel size the source was saved with.
        pack_tp: The size fsdp_dtensor packs at (``None``: nothing is packed).
    """
    if source != TORCH_DIST or pack_tp is None or pack_tp == src_tp:
        return
    bound = sorted(
        (fused for fused in mixers.projections.values() if fused.torch_dist_whole),
        key=lambda fused: fused.key,
    )
    if bound:
        names = ", ".join(
            sorted({f"{fused.tensor.projection}.{fused.tensor.leaf}" for fused in bound})
        )
        kinds = " and ".join(sorted({family_noun(fused.family) for fused in bound}))
        raise _refusal(
            f"{len(bound)} {kinds} {names} tensors (such as "
            f"{bound[0].lid}) are stored whole, as the {src_tp} tensor-parallel ranks' local "
            f"tensors one after the other, which no fsdp_dtensor store packed at {pack_tp} holds",
            tuple(fused.key for fused in bound),
            hint=f"pass --fsdp-tp {src_tp}, the size the source was saved with",
        )


def _discriminate(
    mixer: Mixer,
    tensors: Mapping[tuple[str, str], _Stored],
    built: str,
    reason: str,
    layouts: Mapping[str, FamilyLayout | str],
    args: ArgsView,
    source: str,
) -> FamilyLayout:
    """The layout of ``mixer``'s family, which Megatron builds as ``built`` (:func:`built_family`,
    which gives the ``reason``).

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
    named, names = _named_families(where, tensors, source, is_linear_attention(mixer[2]))
    if built not in named:
        argument = "experimental_attention_variant" if is_linear_attention(mixer[2]) else "spec"
        raise _refusal(
            f"{where} is a {' or '.join(family_noun(family) for family in sorted(named))} "
            f"by {names}, but {reason}",
            keys,
            hint=f"check args.{argument} against the model that saved the store",
        )
    if built == GDP:
        _check_householders(where, tensors, args, keys)
    layout = layouts[built]
    if not isinstance(layout, FamilyLayout):
        raise _refusal(
            f"{where} is a {family_noun(built)}, but its dimensions cannot be "
            f"derived from the arguments: {layout}",
            keys,
        )
    candidates = {family: layout for family, layout in layouts.items() if family in named}
    _check_widths(where, tensors, built, reason, candidates, keys)
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
    where: str, tensors: Mapping[tuple[str, str], _Stored], source: str, attention: bool
) -> tuple[frozenset[str], str]:
    """The families the stored names allow, and the names (a noun phrase, for messages).

    Args:
        where: The mixer, for messages.
        tensors: Its fused projection tensors as stored.
        source: The source format.
        attention: Whether the mixer is a ``self_attention`` (a Gated DeltaNet's module).

    Raises:
        RefusedError: torch_dist stores a fused projection whole that its
            family splits (any but an ``in_proj.bias`` of :data:`TORCH_DIST_WHOLE`),
            or as parts one it keeps whole, or its ``in_proj`` parts are no
            family's, or it names a convolution ``conv1d_weight``/``conv1d_bias``
            (FC-R10).
    """
    own = sorted(
        key for found in tensors.values() if found.tensor.conv_parameter for key in found.keys
    )
    if own and source != FSDP_DTENSOR:
        raise _refusal(
            f"{where}: torch_dist names a Mamba-2 mixer's convolution conv1d.weight and "
            "conv1d.bias (stored as parts), but the store holds conv1d_weight or conv1d_bias",
            tuple(own),
            hint="check that the store was saved by Megatron",
        )
    if own:
        return frozenset({MAMBA2}), "its convolution parameter conv1d_weight"
    if source == FSDP_DTENSOR and attention:
        return frozenset({GDN, GDN2}), "its fused self_attention.in_proj and conv1d.weight"
    if source == FSDP_DTENSOR:
        return frozenset({GDP, MAMBA2}), "its fused in_proj and conv1d.weight"
    parts = tensors[IN_PROJ, "weight"].parts
    named = _family_of_parts(parts) if parts else None
    whole = TORCH_DIST_WHOLE.get(named[0], frozenset()) if named is not None else frozenset()
    unsplit = [
        key
        for name, found in tensors.items()
        if None in found.slices and name not in whole
        for key in found.keys
    ]
    if unsplit:
        raise _refusal(
            f"{where}: torch_dist stores a mixer's in_proj and conv1d as one tensor per part, but "
            "the store holds them whole",
            tuple(unsplit),
        )
    if named is None:
        raise _refusal(
            f"{where}: its in_proj parts {', '.join(sorted(parts))} are no mixer family's",
            tensors[IN_PROJ, "weight"].keys,
            hint="check that the store was saved by Megatron",
        )
    family, order = named
    split = sorted(name for name in whole if name in tensors and tensors[name].parts)
    if split:
        raise _refusal(
            f"{where}: torch_dist stores a {family_noun(family)}'s "
            f"{' and '.join(f'{projection}.{leaf}' for projection, leaf in split)} whole, but the "
            "store holds it as parts",
            tuple(key for name in split for key in tensors[name].keys),
            hint="check that the store was saved by Megatron",
        )
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
    built: str,
    reason: str,
    layouts: Mapping[str, FamilyLayout | str],
    keys: tuple[str, ...],
) -> None:
    """Exactly one family of ``layouts`` has the stored widths as the arguments give it:
    ``built``, the one Megatron builds there for ``reason`` (FC-R10).

    ``layouts`` holds only the families the stored names leave: the one
    torch_dist parts or an fsdp_dtensor ``conv1d_weight`` name, Gated
    DeltaProduct and Mamba-2 for an fsdp_dtensor ``mixer.conv1d.weight``, both
    Gated DeltaNets for an fsdp_dtensor ``self_attention.in_proj``. A family
    the names exclude does not count, even with the same widths: every
    Megatron argument namespace carries Gated DeltaNet's ``linear_*``
    defaults, which a Mamba-2 model's widths may match.
    """
    matches = [
        family
        for family, layout in layouts.items()
        if isinstance(layout, FamilyLayout)
        and all(
            found.rows == layout.projection(found.tensor.projection).fused_rows
            for found in tensors.values()
        )
    ]
    if matches == [built]:
        return
    stored = ", ".join(
        f"{name}.{leaf} {found.rows} rows" for (name, leaf), found in sorted(tensors.items())
    )
    layout = layouts[built]
    assert isinstance(layout, FamilyLayout)  # checked by the caller
    if built in matches:
        others = " and ".join(family_noun(f, plural=True) for f in matches if f != built)
        message = (
            f"{where}: its widths ({stored}) are what the arguments give "
            f"{family_noun(built, plural=True)} ({layout.dims}) and {others} alike, so "
            "they cannot tell the family"
        )
    else:
        match = (
            f"; they are what the arguments give "
            f"{' and '.join(family_noun(f, plural=True) for f in matches)}"
            if matches
            else ""
        )
        message = (
            f"{where}: its widths ({stored}) are not what the arguments give "
            f"{family_noun(built, plural=True)} ({layout.dims}: {layout.widths}), which {reason}"
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
