# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The key grammar of model tensors: blocks, layer indices, paths and leaves.

The grammar and every pattern over model keys that the conversion layers use
(block classifiers, refusal detectors, sentinels and rules) live in this
module, so that they agree on one grammar; each format adapter owns its
envelope and its section rules.

A *model-relative key* is a model tensor's key without its format envelope
(torch_dist keys are bare; fsdp_dtensor keys lose their ``model.module.``
prefix). Inside a layered block it reads ``<block>.layers.<index>.<path>.<leaf>``
(one layer), or ``<block>.layers.<path>.<leaf>`` for a tensor stacked over all
the layers of the block (torch_dist only); elsewhere it reads ``<path>.<leaf>``:

* ``block``: the dotted prefix of the block's layer list, e.g. ``decoder``;
* ``index``: the layer index as the format stores it (the two formats may
  number the layers of a block differently, see ``core/blocks.py``), written
  in canonical decimal (no sign, no leading zero);
* ``path``: the module path inside the layer, or from the model root outside
  blocks, e.g. ``self_attention.linear_qkv``; it may be empty;
* ``leaf``: the parameter or buffer name, e.g. ``weight`` or ``layer_norm_weight``.

Blocks nest: a layer may hold a layer list of its own (a multi-token
prediction depth holds the layers of its inner model, e.g.
``mtp.layers.0.mtp_model_layer.layers.2.<name>``). A key belongs to the
*innermost* block whose layer list holds it: the longest block prefix, and
only that block's layer segment is interpreted; the segments of the blocks
around it are part of the prefix and kept as they are. Blocks of different
model roots never share keys (``vision_model.decoder`` and
``language_model.decoder`` are different prefixes).

The routed experts of a mixture-of-experts layer live under its experts
module ``<...>.mlp.experts``, whose linear layers ``linear_fc1`` and
``linear_fc2`` hold one weight (and bias) per expert. torch_dist stores each
as one tensor over all experts, ``<...>.mlp.experts.experts.linear_fc<n>.<leaf>``,
the global expert index along its leading axis (after the layer axis of a
tensor stacked over layers). fsdp_dtensor stores one tensor per expert, named
as the experts module names its parameters with the local index replaced by
the global one (``handle_experts_in_state_dict``): *grouped*
(``TEGroupedMLP``) ``<...>.mlp.experts.linear_fc<n>.<leaf><e>``, or
*sequential* (``SequentialMLP``) ``<...>.mlp.experts.local_experts.<e>.linear_fc<n>.<leaf>``.
:class:`ExpertName` names such a tensor by the module path
``<...>.mlp.experts.linear_fc<n>`` (and one expert of it by its global index),
whatever the format.

Megatron-FSDP may store the weight and bias of a gated MLP's ``linear_fc1``
as two SwiGLU halves, ``<name>_w`` and ``<name>_v`` (``linear_fc1.weight<e>_w``
for a grouped expert): :func:`split_glu_half` names the whole tensor and the
half, and :data:`SWIGLU_KEYS` are the names it splits. The other way round,
torch_dist stores the weight and bias of the fused projections of a
state-space mixer or a Gated DeltaNet (:data:`FUSED_PROJECTION`) as one tensor
per component, e.g. ``mixer.in_proj.weight.V0`` or
``self_attention.conv1d.weight.query``, while fsdp_dtensor stores them whole:
:func:`split_factory_part` names the whole tensor and the part.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..errors import InternalError
from .names import GROUPED, SEQUENTIAL, TORCH_DIST

LAYERS = "layers"
"""Name of the layer list of a block (``nn.ModuleList`` attribute)."""

LAYER_SEGMENT = f".{LAYERS}."
"""The segment that separates a block prefix from what its layer list holds."""


_INDEX_AND_INNER = re.compile(r"(?P<index>0|[1-9][0-9]*)\.(?P<inner>.+)")
_NAME_START = re.compile(r"[A-Za-z_]")

_EXPERTS_MODULE = r"(?P<experts>(?:.+\.)?mlp\.experts)"
_EXPERT_LINEAR = r"(?P<linear>linear_fc[0-9])"
_EXPERT_LEAF = r"(?P<leaf>weight|bias)"
_EXPERT_INDEX = r"(?P<expert>0|[1-9][0-9]*)"
_EXPERT_PATH = re.compile(rf"{_EXPERTS_MODULE}\.{_EXPERT_LINEAR}")
_TORCH_DIST_EXPERT = re.compile(rf"{_EXPERTS_MODULE}\.experts\.{_EXPERT_LINEAR}\.{_EXPERT_LEAF}")
_FSDP_EXPERT = {
    GROUPED: re.compile(rf"{_EXPERTS_MODULE}\.{_EXPERT_LINEAR}\.{_EXPERT_LEAF}{_EXPERT_INDEX}"),
    SEQUENTIAL: re.compile(
        rf"{_EXPERTS_MODULE}\.local_experts\.{_EXPERT_INDEX}\.{_EXPERT_LINEAR}\.{_EXPERT_LEAF}"
    ),
}


class KeyGrammarError(ValueError):
    """A key does not follow the grammar it is parsed with."""


@dataclass(frozen=True, slots=True)
class LayerKey:
    """A model-relative key inside a block's layer list.

    Attributes:
        block: The block prefix, e.g. ``decoder``.
        index: The stored layer index, or ``None`` for a tensor stacked over
            every layer of the block (``<block>.layers.<inner>``).
        inner: The rest of the key inside the layer, ``<path>.<leaf>``.
    """

    block: str
    index: int | None
    inner: str

    @property
    def stacked(self) -> bool:
        """Whether the key holds every layer of the block (no layer index)."""
        return self.index is None

    def render(self) -> str:
        """The key: ``<block>.layers.<index>.<inner>`` (``<block>.layers.<inner>`` if stacked)."""
        return join_layer_key(self.block, self.index, self.inner)


def split_layer_key(key: str, block: str) -> LayerKey | None:
    """Parse ``key`` as a key inside the layer list of ``block``.

    Returns:
        The parsed key, or ``None`` when ``key`` is not under ``<block>.layers.``.

    Raises:
        KeyGrammarError: ``key`` is under ``<block>.layers.`` but the segment
            after it is neither a canonical layer index followed by a name nor
            a name (a stacked tensor).
    """
    head = block + LAYER_SEGMENT
    if not key.startswith(head):
        return None
    return _parse_layers(key, block, key[len(head) :])


def innermost_layer_key(key: str) -> LayerKey | None:
    """Parse ``key`` at its last layer segment: the innermost layer list that holds it.

    This finds the blocks a store holds without knowing them in advance.

    Returns:
        The parsed key, or ``None`` when ``key`` is in no layer list.

    Raises:
        KeyGrammarError: The segment after the last ``.layers.`` is malformed
            (see :func:`split_layer_key`).
    """
    at = key.rfind(LAYER_SEGMENT)
    if at <= 0:
        return None
    return _parse_layers(key, key[:at], key[at + len(LAYER_SEGMENT) :])


def attribute_layer_key(key: str, blocks: Iterable[str]) -> LayerKey | None:
    """Parse ``key`` inside the innermost of ``blocks`` whose layer list holds it.

    The innermost block is the one with the longest prefix; the layer segments
    of the blocks around it stay part of that prefix.

    Returns:
        The parsed key, or ``None`` when no block's layer list holds ``key``.

    Raises:
        KeyGrammarError: The innermost block's layer segment is malformed.
    """
    holders = [block for block in blocks if key.startswith(block + LAYER_SEGMENT)]
    if not holders:
        return None
    return split_layer_key(key, max(holders, key=len))


def _parse_layers(key: str, block: str, rest: str) -> LayerKey:
    match = _INDEX_AND_INNER.fullmatch(rest)
    if match is not None:
        return LayerKey(block, int(match.group("index")), match.group("inner"))
    if _NAME_START.match(rest) is None:
        raise KeyGrammarError(
            f"{key}: expected {block}{LAYER_SEGMENT}<index>.<name> with a canonical index, or "
            f"{block}{LAYER_SEGMENT}<name> for a tensor stacked over the layers"
        )
    return LayerKey(block, None, rest)


def join_layer_key(block: str, index: int | None, inner: str) -> str:
    """The model-relative key ``<block>.layers.<index>.<inner>`` (``index`` ``None``: stacked)."""
    if index is None:
        return f"{block}{LAYER_SEGMENT}{inner}"
    if index < 0:
        raise KeyGrammarError(f"layer index {index} of {block} is negative")
    return f"{block}{LAYER_SEGMENT}{index}.{inner}"


@dataclass(frozen=True, slots=True)
class ExpertName:
    """A tensor of the routed experts of a mixture-of-experts layer, by its logical name.

    Attributes:
        path: The module path of the experts' linear layer,
            ``<...>.mlp.experts.linear_fc<n>``.
        leaf: ``weight`` or ``bias``.
        expert: The global expert index, or ``None`` for a tensor that holds
            every expert along its expert axis (torch_dist).
        form: How fsdp_dtensor named the tensor (``grouped`` or ``sequential``;
            ``None`` for torch_dist).
    """

    path: str
    leaf: str
    expert: int | None = None
    form: str | None = None


def split_expert_name(name: str, fmt: str) -> ExpertName | None:
    """Parse ``name`` as a store of format ``fmt`` names a routed-expert tensor.

    Returns:
        The tensor, or ``None`` when ``name`` is not under an experts module
        ``<...>.mlp.experts.``.

    Raises:
        KeyGrammarError: ``name`` is under an experts module but not in a form
            of ``fmt``.
    """
    return split_torch_dist_expert(name) if fmt == TORCH_DIST else split_fsdp_expert(name)


def split_torch_dist_expert(name: str) -> ExpertName | None:
    """Parse ``name`` as torch_dist stores a routed-expert tensor (every expert in one tensor).

    Args:
        name: A name inside a layer, or a whole model-relative key.

    Returns:
        The tensor, from ``<...>.mlp.experts.experts.linear_fc<n>.<weight|bias>``,
        or ``None`` when ``name`` is not under an experts module ``<...>.mlp.experts.``.

    Raises:
        KeyGrammarError: ``name`` is under an experts module but not in that form.
    """
    if ROUTED_EXPERTS.search(name) is None:
        return None
    match = _TORCH_DIST_EXPERT.fullmatch(name)
    if match is None:
        raise KeyGrammarError(
            f"{name}: torch_dist stores routed-expert tensors as "
            "<...>.mlp.experts.experts.linear_fc<n>.weight or .bias"
        )
    return ExpertName(f"{match.group('experts')}.{match.group('linear')}", match.group("leaf"))


def join_torch_dist_expert(path: str, leaf: str) -> str:
    """The torch_dist name of the routed-expert tensor ``path``/``leaf`` (every expert).

    Raises:
        KeyGrammarError: ``path`` is not ``<...>.mlp.experts.linear_fc<n>``.
    """
    experts, linear = _split_expert_path(path)
    return f"{experts}.experts.{linear}.{leaf}"


def split_fsdp_expert(name: str) -> ExpertName | None:
    """Parse ``name`` as fsdp_dtensor names one routed expert's tensor (grouped or sequential).

    Args:
        name: A name inside a layer, or a whole model-relative key.

    Returns:
        The tensor, from ``<...>.mlp.experts.linear_fc<n>.<weight|bias><e>``
        (grouped) or ``<...>.mlp.experts.local_experts.<e>.linear_fc<n>.<weight|bias>``
        (sequential), ``e`` a canonical decimal index; ``None`` when ``name``
        is not under an experts module ``<...>.mlp.experts.``.

    Raises:
        KeyGrammarError: ``name`` is under an experts module but in neither form.
    """
    if ROUTED_EXPERTS.search(name) is None:
        return None
    for form, pattern in _FSDP_EXPERT.items():
        match = pattern.fullmatch(name)
        if match is not None:
            path = f"{match.group('experts')}.{match.group('linear')}"
            return ExpertName(path, match.group("leaf"), int(match.group("expert")), form)
    raise KeyGrammarError(
        f"{name}: fsdp_dtensor names routed-expert tensors <...>.mlp.experts.linear_fc<n>."
        "<weight|bias><e> (grouped) or <...>.mlp.experts.local_experts.<e>.linear_fc<n>."
        "<weight|bias> (sequential)"
    )


def join_fsdp_expert(path: str, leaf: str, expert: int, form: str) -> str:
    """The fsdp_dtensor name of expert ``expert`` of the tensor ``path``/``leaf`` in ``form``.

    Raises:
        KeyGrammarError: ``path`` is not ``<...>.mlp.experts.linear_fc<n>``.
        InternalError: ``form`` is neither ``grouped`` nor ``sequential``.
    """
    experts, linear = _split_expert_path(path)
    if form == GROUPED:
        return f"{experts}.{linear}.{leaf}{expert}"
    if form == SEQUENTIAL:
        return f"{experts}.local_experts.{expert}.{linear}.{leaf}"
    raise InternalError(f"unknown expert naming {form!r}")


def _split_expert_path(path: str) -> tuple[str, str]:
    """``(<...>.mlp.experts, linear_fc<n>)`` of an expert linear layer's module path."""
    match = _EXPERT_PATH.fullmatch(path)
    if match is None:
        raise KeyGrammarError(f"{path} is not the module path of an experts' linear layer")
    return match.group("experts"), match.group("linear")


def torch_dist_chunk_axes(key: str) -> int:
    """The leading axes along which Megatron saves a torch_dist tensor one index per chunk.

    A tensor stacked over the layers of a block leads with its layer axis, and
    a routed-expert tensor then has its expert axis: Megatron saves each layer,
    and each expert of a layer, as a chunk of its own.

    Raises:
        KeyGrammarError: The key's layer segment or expert name is malformed.
    """
    layer_key = innermost_layer_key(key)
    stacked = layer_key is not None and layer_key.stacked
    return int(stacked) + int(split_torch_dist_expert(key) is not None)


def split_leaf(name: str) -> tuple[str, str]:
    """Split ``<path>.<leaf>`` into ``(path, leaf)``; the path is empty for a bare leaf.

    Raises:
        KeyGrammarError: The name is empty or ends with a dot.
    """
    path, _, leaf = name.rpartition(".")
    if not leaf:
        raise KeyGrammarError(f"{name!r} has no leaf name")
    return path, leaf


def join_leaf(path: str, leaf: str) -> str:
    """``<path>.<leaf>``, or ``leaf`` alone when the path is empty."""
    return f"{path}.{leaf}" if path else leaf


def root_of(key: str) -> str:
    """The first dotted segment of a model-relative key (``embedding``, ``decoder``, ...)."""
    return key.partition(".")[0]


FACTORY_PART = re.compile(r"\.(?:weight|bias)\.[A-Za-z_][A-Za-z0-9_]*$")
"""A torch_dist factory sub-key: one part of a parameter stored as parts (``in_proj.weight.z``)."""

GLU_HALF = re.compile(r"\.(?:weight|bias)[0-9]*_[wv]$")
"""A SwiGLU half of ``linear_fc1`` (``weight_w``, ``weight3_v``, ``bias_w``): as Megatron-FSDP
splits it, or as a torch_dist save with ``singleton_local_shards`` names it."""

MIXER_MODULE = "mixer"
"""The module a state-space mixer layer holds its fused projections in."""

LINEAR_ATTENTION = "self_attention"
"""The module a Gated DeltaNet layer holds its fused projections in (a state-space mixer layer's
is :data:`MIXER_MODULE`)."""

MIXER_MODULES = rf"(?:{re.escape(MIXER_MODULE)}|{re.escape(LINEAR_ATTENTION)})"
"""A regular expression of either module (every pattern of fused projections is built from it)."""

FUSED_MIXER = re.compile(
    r"\." + MIXER_MODULES + r"\."
    r"(?:in_proj\.(?:weight|bias)|conv1d(?:_weight|_bias|\.weight|\.bias))$"
)
"""A fused state-space or linear-attention projection as fsdp_dtensor stores it."""

IN_PROJ, CONV1D = "in_proj", "conv1d"
"""The fused projections of a state-space mixer (and of a Gated DeltaNet): the input projection,
which concatenates one projection per component, and the depthwise convolution over some of them."""

FUSED_PROJECTION = re.compile(r"(?:^|\.)" + MIXER_MODULES + r"\.(?:in_proj|conv1d)$")
"""The module path of a fused projection (``<...>.mixer.in_proj``, ``<...>.self_attention.conv1d``).
Megatron's split factory saves its weight and bias in torch_dist as one tensor per component
(``mixer.in_proj.weight.z``); fsdp_dtensor stores them whole."""

_FACTORY_PART_NAME = re.compile(
    r"(?P<tensor>(?P<path>.+)\.(?:weight|bias))\.(?P<part>[A-Za-z_][A-Za-z0-9_]*)"
)


def split_factory_part(name: str) -> tuple[str, str | None]:
    """The tensor and the part of a torch_dist factory sub-key of a fused projection.

    Returns:
        ``(<...>.in_proj.weight, "V0")`` for ``<...>.in_proj.weight.V0`` (see
        :data:`FUSED_PROJECTION`), ``(name, None)`` for any other name.
    """
    match = _FACTORY_PART_NAME.fullmatch(name)
    if match is None or FUSED_PROJECTION.search(match.group("path")) is None:
        return name, None
    return match.group("tensor"), match.group("part")


def join_factory_part(name: str, part: str) -> str:
    """The torch_dist key of part ``part`` of the fused-projection tensor ``name``.

    Raises:
        InternalError: ``name`` is not the weight or bias of a fused projection.
    """
    path, _ = split_leaf(name)
    if FUSED_PROJECTION.search(path) is None:
        raise InternalError(f"{name} is not a fused projection that torch_dist stores in parts")
    return f"{name}.{part}"


def is_fused_projection(path: str) -> bool:
    """Whether ``path`` is the module path of a fused projection (:data:`FUSED_PROJECTION`)."""
    return FUSED_PROJECTION.search(path) is not None


MIXER_TENSOR = re.compile(
    r"(?:^|\.)" + MIXER_MODULES + r"\.(?:in_proj|conv1d)\.(?:weight|bias)"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*)?$"
    r"|(?:^|\.)" + re.escape(MIXER_MODULE) + r"\.conv1d_(?:weight|bias)$"
)
"""A fused projection tensor of a state-space mixer or a Gated DeltaNet, whole or one torch_dist
part of it; Mamba-2's fsdp_dtensor stores name its convolution ``mixer.conv1d_weight`` and
``mixer.conv1d_bias``. Standard and multi-latent attention have no ``in_proj`` or ``conv1d``."""

_MIXER_PROJECTION = re.compile(
    r"(?P<mixer>(?:.+\.)?" + MIXER_MODULES + r")\.(?P<projection>in_proj|conv1d)"
)
_MIXER_MODULE = re.compile(r"(?:.+\.)?" + re.escape(MIXER_MODULE))
_LINEAR_ATTENTION_MODULE = re.compile(r"(?:.+\.)?" + re.escape(LINEAR_ATTENTION))
_MAMBA_CONV_LEAVES = {"conv1d_weight": "weight", "conv1d_bias": "bias"}


def is_linear_attention(mixer: str) -> bool:
    """Whether the mixer module path ``mixer`` is a Gated DeltaNet's (``<...>.self_attention``)."""
    return _LINEAR_ATTENTION_MODULE.fullmatch(mixer) is not None


@dataclass(frozen=True, slots=True)
class MixerTensor:
    """A fused projection tensor of a state-space mixer or a Gated DeltaNet, by its logical name.

    Attributes:
        mixer: The module path of the mixer inside its layer (``mixer``, or
            ``<child>.mixer`` in a shortcut pair; a Gated DeltaNet's
            ``self_attention`` or ``<child>.self_attention``).
        projection: ``in_proj`` or ``conv1d``.
        leaf: ``weight`` or ``bias``.
        conv_parameter: Stored as the mixer's own parameter ``conv1d_weight``
            or ``conv1d_bias`` (as Mamba-2's fsdp_dtensor stores name it).
    """

    mixer: str
    projection: str
    leaf: str
    conv_parameter: bool = False


def split_mixer_tensor(path: str, leaf: str) -> MixerTensor | None:
    """The mixer tensor of the logical name ``path``/``leaf``, or ``None`` if it is none."""
    match = _MIXER_PROJECTION.fullmatch(path)
    if match is not None and leaf in ("weight", "bias"):
        return MixerTensor(match.group("mixer"), match.group("projection"), leaf)
    if leaf in _MAMBA_CONV_LEAVES and _MIXER_MODULE.fullmatch(path) is not None:
        return MixerTensor(path, CONV1D, _MAMBA_CONV_LEAVES[leaf], conv_parameter=True)
    return None


def join_mixer_tensor(tensor: MixerTensor) -> tuple[str, str]:
    """The logical ``(path, leaf)`` of ``tensor``, the inverse of :func:`split_mixer_tensor`.

    ``conv_parameter`` names the convolution as the mixer's own parameter
    (``mixer``, ``conv1d_weight``), otherwise as the ``conv1d`` module's
    (``mixer.conv1d``, ``weight``).
    """
    if tensor.conv_parameter:
        if tensor.projection != CONV1D:
            raise InternalError(f"{tensor} is no convolution, which alone is a mixer parameter")
        return tensor.mixer, f"{CONV1D}_{tensor.leaf}"
    return f"{tensor.mixer}.{tensor.projection}", tensor.leaf


ROUTED_EXPERTS = re.compile(r"(?:^|\.)mlp\.experts\.")
"""A tensor under the routed experts of a mixture-of-experts layer (``<...>.mlp.experts.``)."""

SINGLE_GROUPED_EXPERTS = re.compile(r"(?:^|\.)mlp\.experts\.linear_fc[0-9]\.(?:weight|bias)$")
"""One grouped parameter for every expert (``moe_single_grouped_weight``/``_bias``)."""

GROUPED_EXPERT_BIAS = re.compile(r"(?:^|\.)mlp\.experts\.linear_fc[0-9]\.bias(?:0|[1-9][0-9]*)$")
"""An expert bias of a grouped experts module (fsdp_dtensor ``linear_fc<n>.bias<e>``)."""

TORCH_DIST_EXPERT_BIAS = re.compile(r"(?:^|\.)mlp\.experts\.experts\.linear_fc[0-9]\.bias$")
"""The expert biases of a linear layer as torch_dist stores them (one tensor over all experts)."""

LEGACY_GROUPED_EXPERTS = re.compile(r"(?:^|\.)mlp\.experts\.weight[12]$")
"""A parameter of the legacy ``GroupedMLP``: every local expert of a linear layer in one tensor."""

SINGLETON_EXPERT_SHARD = re.compile(r"(?:^|\.)mlp\.experts\.experts\.(?:0|[1-9][0-9]*)\.")
"""An expert saved with ``singleton_local_shards`` (torch_dist ``experts.experts.<e>.<name>``)."""

_MLA_PROJECTION = r"self_attention\.(?:linear_q_proj|linear_(?:q|kv|qkv)_(?:down|up)_proj)\."

MLA_PARAM = re.compile(r"\." + _MLA_PROJECTION)
"""A projection of multi-latent attention."""


@dataclass(frozen=True, slots=True)
class LayerFamily:
    """The tensors that identify one type of hybrid layer (its *family signature*).

    Attributes:
        tensors: What they are, for messages (e.g. ``self_attention.linear_qkv``).
        pattern: Searched in the names inside a module of a hybrid layer list.
    """

    tensors: str
    pattern: re.Pattern[str]


LAYER_FAMILIES: Mapping[str, LayerFamily] = {
    "M": LayerFamily(
        "mixer.in_proj, mixer.conv1d", re.compile(r"(?:^|\.)mixer\.(?:in_proj|conv1d)[._]")
    ),
    "G": LayerFamily("self_attention.in_proj", re.compile(r"(?:^|\.)self_attention\.in_proj\.")),
    "*": LayerFamily(
        "self_attention.linear_qkv", re.compile(r"(?:^|\.)self_attention\.linear_qkv\.")
    ),
    "+": LayerFamily("self_attention.linear_q*_proj", re.compile(r"(?:^|\.)" + _MLA_PROJECTION)),
    "-": LayerFamily("mlp.linear_fc1", re.compile(r"(?:^|\.)mlp\.linear_fc1\.")),
    "E": LayerFamily("mlp.router.weight", re.compile(r"(?:^|\.)mlp\.router\.weight$")),
}
"""The family signature of the layers of each hybrid symbol: a state-space mixer's input
projection or convolution (``M``), a Gated DeltaNet's input projection (``G``), a self-attention's
fused QKV projection (``*``) or multi-latent projections (``+``), a dense MLP's first projection
(``-``) and a mixture-of-experts router (``E``)."""

TE_OP_FUSER_MLP = re.compile(r"\.mlp\.[0-9]+\.")
"""An MLP built with Transformer Engine's operation fuser (``mlp.<i>.basic_ops...``)."""

EXTRA_STATE = re.compile(r"\._extra_state\d*(/|$)")
"""A Transformer Engine ``_extra_state`` key (``_extra_state<i>`` when grouped; torch_dist
stores it as a tensor or a ``/shard_*`` byte item)."""

FC1 = "linear_fc1"
"""An MLP's first projection (a gated linear unit's gate and value projections when gated)."""

FC2 = "linear_fc2"
"""An MLP's second projection, whose inputs are the first projection's (value) outputs."""

LINEAR_FC = re.compile(
    r"(?:^|\.)(?:linear_fc1\.(?:weight|bias)(?:0|[1-9][0-9]*)?(?:_[wv])?"
    r"|linear_fc2\.weight(?:0|[1-9][0-9]*)?)$"
)
"""An MLP's first projection's weight or bias (``linear_fc1.<leaf>``, one expert's
``linear_fc1.<leaf><e>`` of a grouped experts module in fsdp_dtensor, and their SwiGLU halves
``_w``/``_v``) or its second projection's weight."""

GLU_PARTS = ("w", "v")
"""The gate and value halves of a gated ``linear_fc1``, in their order along its first axis."""

_GLU_HALF_NAME = re.compile(
    r"(?P<tensor>(?:.+\.)?linear_fc1\.(?:weight|bias)(?:0|[1-9][0-9]*)?)_(?P<part>[wv])"
)

SWIGLU_KEYS = tuple(
    re.compile(pattern)
    for pattern in (
        r"(.*)\.mlp\.linear_fc1\.weight$",
        r"(.*)\.mlp\.linear_fc1\.bias$",
        r"(.*)\.mlp\.experts\.linear_fc1\.weight(\d+)$",
        r"(.*)\.mlp\.experts\.linear_fc1\.bias(\d+)$",
        r"(.*)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.weight$",
        r"(.*)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.bias$",
        r"(.*)\.mlp\.shared_experts\.linear_fc1\.weight$",
        r"(.*)\.mlp\.shared_experts\.linear_fc1\.bias$",
    )
)
"""The ``linear_fc1`` names Megatron-FSDP splits into SwiGLU halves (``is_swiglu_key`` of
``handle_swiglu_in_state_dict``), searched in the model's state-dict names (fsdp_dtensor model
keys without ``model.module.``): of dense MLPs, grouped or sequential experts, shared experts."""


def is_fc1_path(path: str) -> bool:
    """Whether ``path`` is the module path of an MLP's first projection (``<...>.linear_fc1``)."""
    return path == FC1 or path.endswith(f".{FC1}")


def is_swiglu_key(name: str) -> bool:
    """Whether Megatron-FSDP splits the tensor it stores as ``name`` into SwiGLU halves (if its
    layer is gated; see :data:`SWIGLU_KEYS`)."""
    return any(pattern.search(name) is not None for pattern in SWIGLU_KEYS)


def is_glu_half(path: str, part: str) -> bool:
    """Whether ``part`` of the tensor at module path ``path`` is a SwiGLU half of ``linear_fc1``."""
    return part in GLU_PARTS and is_fc1_path(path)


def split_glu_half(name: str) -> tuple[str, str | None]:
    """The fused tensor and the half of an fsdp_dtensor SwiGLU half, e.g. ``linear_fc1.weight3_v``.

    Returns:
        ``(<...>.linear_fc1.<leaf>[<e>], "w" or "v")`` for a half, ``(name, None)`` for any other
        name.
    """
    match = _GLU_HALF_NAME.fullmatch(name)
    return (name, None) if match is None else (match.group("tensor"), match.group("part"))


def join_glu_half(name: str, part: str) -> str:
    """The fsdp_dtensor name of half ``part`` of the fused ``linear_fc1`` tensor ``name``.

    Raises:
        InternalError: ``part`` is not a SwiGLU half.
    """
    if part not in GLU_PARTS:
        raise InternalError(f"{part!r} is not a SwiGLU half of {name}")
    return f"{name}_{part}"


def fc2_path_of(fc1_path: str) -> str:
    """The module path of the second projection beside the first projection ``fc1_path``.

    Raises:
        KeyGrammarError: ``fc1_path`` is not the module path of a first projection.
    """
    if not is_fc1_path(fc1_path):
        raise KeyGrammarError(f"{fc1_path} is not the module path of a {FC1}")
    return fc1_path[: -len(FC1)] + FC2
