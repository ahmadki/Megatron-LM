# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-relative keys to logical ids and back, through the blocks' layer numbering.

:func:`decompose` parses a source model key with the key grammar, attributes
it to the innermost block whose layer list holds it (the longest matching
block prefix; the layer segments of enclosing blocks stay in that prefix) and
maps the stored layer index to the dense index. A stored tensor may lead with
index axes, each giving one logical tensor per index: a torch_dist key
stacked over the layers of a block has a layer axis, and a torch_dist
routed-expert tensor an expert axis after it, so ``[L, E, ...]`` becomes one
slice per (layer, expert), ``[l:l+1, e:e+1, ...]``, whose logical tensor has
one expert's shape. Keys outside every block keep their path. An fsdp_dtensor
SwiGLU half (``linear_fc1.weight_w``) is the id of that part of the fused
tensor, and so is a torch_dist part of a fused projection
(``mixer.in_proj.weight.V0``). A module of a block's layers that the format
stores under another name than its logical one (a legacy Megatron-FSDP
store's GPT MTP ``mtp_model_layer``) gets its logical name
(``transformer_layer``), and back. :func:`compose` is the inverse for the target
format: it places each
layer of a block that torch_dist stores stacked at ``[l:l+1, ...]`` of the
stacked tensor ``<block>.layers.<name>``, and each expert at its index of the
expert axis (fsdp_dtensor names each expert's tensor in the model's expert
naming instead). A SwiGLU half is a tensor of its own in fsdp_dtensor, and in
torch_dist its rows of the fused ``linear_fc1`` tensor, ``[W; V]`` (the gate
half first, as Megatron's SwiGLU factory saves it). A part of a fused
projection is a tensor of its own in torch_dist (``<name>.<part>``) and no
tensor of fsdp_dtensor, which stores the projection whole.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..engine.metadata import TensorEntry
from ..engine.regions import Region
from ..errors import InternalError
from .blocks import BlockLayout
from .experts import ExpertLayout
from .keys import (
    GLU_PARTS,
    ExpertName,
    KeyGrammarError,
    LayerKey,
    attribute_layer_key,
    is_fused_projection,
    is_glu_half,
    join_factory_part,
    join_fsdp_expert,
    join_glu_half,
    join_layer_key,
    join_torch_dist_expert,
    split_expert_name,
    split_factory_part,
    split_glu_half,
    split_leaf,
)
from .logical import LogicalId, LogicalSlice, OutputSlice
from .names import TORCH_DIST

STACKED_AXES = 1
"""Leading axes a tensor stacked over layers has beyond one layer's shape."""

_Axis = tuple[tuple[int | None, int | None], ...]
"""The indices along one leading axis: ``(logical index, position on the axis)`` pairs; a
single ``(index, None)`` when the stored tensor has no such axis."""


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
        One slice holding the whole tensor, or one per index of its leading
        layer and expert axes.

    Raises:
        KeyGrammarError: The key's layer segment or expert name is malformed,
            its form (stacked or per layer) is not the one the format stores
            the block in, its layer index is not one the format stores for the
            block, a stacked tensor does not hold every layer, a routed-expert
            tensor has no expert axis, or the key has no leaf.
    """
    by_prefix = {block.prefix: block for block in blocks}
    layer_key = attribute_layer_key(relative, by_prefix)
    block = None if layer_key is None else by_prefix[layer_key.block]
    name = (
        relative if block is None or layer_key is None else block.logical_name(layer_key.inner, fmt)
    )
    name, part = split_factory_part(name) if fmt == TORCH_DIST else split_glu_half(name)
    expert = split_expert_name(name, fmt)
    path, leaf = (expert.path, expert.leaf) if expert is not None else split_leaf(name)
    layers = _layer_axis(relative, entry, block, layer_key, fmt)
    layer_axes = int(layers[0][1] is not None)
    experts = _expert_axis(relative, entry, expert, axis=layer_axes)
    lead = layer_axes + int(experts[0][1] is not None)
    rest = entry.shape[lead:]
    prefix = None if block is None else block.prefix
    return [
        LogicalSlice(
            LogicalId(prefix, layer, path, leaf, index, part),
            entry.key,
            Region(
                tuple(p for p in (layer_at, index_at) if p is not None) + (0,) * len(rest),
                (1,) * lead + rest,
            ),
            entry.dtype,
            squeezed=lead,
        )
        for layer, layer_at in layers
        for index, index_at in experts
    ]


def _layer_axis(
    relative: str,
    entry: TensorEntry,
    block: BlockLayout | None,
    layer_key: LayerKey | None,
    fmt: str,
) -> _Axis:
    """The dense layers ``entry`` holds, and where along its layer axis if it has one."""
    if block is None or layer_key is None:
        return ((None, None),)
    stacked = fmt == TORCH_DIST and block.stacked
    if layer_key.stacked != stacked:
        form = "stacked over the layers" if layer_key.stacked else "one layer"
        raise KeyGrammarError(
            f"{relative} holds {form} of {block.prefix}, which {fmt} stores "
            + ("stacked" if stacked else "one layer per key")
        )
    if layer_key.index is not None:
        return ((block.dense_index(layer_key.index, fmt), None),)
    if entry.shape[:STACKED_AXES] != (block.num_modules,):
        raise KeyGrammarError(
            f"{relative} of shape {entry.shape} does not hold the {block.num_modules} layers "
            f"of {block.prefix}"
        )
    return tuple((block.dense_index(stored, fmt), stored) for stored in range(block.num_modules))


def _expert_axis(relative: str, entry: TensorEntry, expert: ExpertName | None, axis: int) -> _Axis:
    """The experts ``entry`` holds, and where along its expert axis (``axis``) if it has one."""
    if expert is None or expert.expert is not None:
        return ((None if expert is None else expert.expert, None),)
    if len(entry.shape) <= axis or entry.shape[axis] < 1:
        raise KeyGrammarError(
            f"{relative} of shape {entry.shape} has no expert axis (dimension {axis})"
        )
    return tuple((index, index) for index in range(entry.shape[axis]))


def compose(
    lid: LogicalId,
    shape: tuple[int, ...],
    blocks: Sequence[BlockLayout],
    fmt: str,
    experts: ExpertLayout | None = None,
) -> OutputSlice:
    """Where the logical tensor ``lid`` of shape ``shape`` goes in a store of format ``fmt``.

    Args:
        lid: The logical tensor.
        shape: Its shape.
        blocks: The model's blocks.
        fmt: The target format.
        experts: How the model's routed experts are stored (needed for their tensors).

    Returns:
        The slice with the model-relative key: the whole tensor, or its box
        of a tensor with leading index axes: ``[l:l+1, ...]`` of a tensor
        stacked over the layers of a block that torch_dist stores stacked
        (``l`` its torch_dist index), and ``[..., e:e+1, ...]`` of a torch_dist
        routed-expert tensor; a torch_dist SwiGLU half is further the box of
        its rows of the fused tensor, and a torch_dist part of a fused
        projection a tensor of its own (``<name>.<part>``).

    Raises:
        InternalError: ``lid`` names a block that is not in ``blocks``, a
            routed expert that ``experts`` does not place, or a part ``fmt``
            does not store (neither a SwiGLU half of a ``linear_fc1`` nor, in
            torch_dist, a part of a fused projection).
    """
    name, expert_axis = _output_name(lid, fmt, experts)
    tensor_shape, offsets = _part_box(lid, shape, fmt)
    leading: list[tuple[int, int]] = []  # (index, axis length) of each leading axis
    if lid.block is None or lid.layer is None:
        key = name
    else:
        block = next((block for block in blocks if block.prefix == lid.block), None)
        if block is None:
            raise InternalError(f"{lid} names an unknown block")
        stored = block.stored_index(lid.layer, fmt)
        stacked = fmt == TORCH_DIST and block.stacked
        key = join_layer_key(
            block.prefix, None if stacked else stored, block.stored_name(name, fmt)
        )
        if stacked:
            leading.append((stored, block.num_modules))
    if expert_axis is not None:
        leading.append(expert_axis)
    return OutputSlice(
        key,
        tuple(length for _, length in leading) + tensor_shape,
        Region(tuple(index for index, _ in leading) + offsets, (1,) * len(leading) + shape),
        squeezed=len(leading),
    )


def _part_box(
    lid: LogicalId, shape: tuple[int, ...], fmt: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The shape of the tensor (after its leading index axes) that holds ``lid``, and the offsets
    of ``lid``'s box in it: the tensor itself, or for a SwiGLU half in torch_dist, its rows of the
    fused tensor."""
    if lid.part is None or fmt != TORCH_DIST or not is_glu_half(lid.path, lid.part):
        return shape, (0,) * len(shape)
    rows = shape[0]
    fused = (len(GLU_PARTS) * rows,) + shape[1:]
    return fused, (GLU_PARTS.index(lid.part) * rows,) + (0,) * (len(shape) - 1)


def _output_name(
    lid: LogicalId, fmt: str, experts: ExpertLayout | None
) -> tuple[str, tuple[int, int] | None]:
    """The name of ``lid`` inside its layer (or from the root) in ``fmt``, and its expert axis.

    The expert axis is ``(expert index, number of experts)`` for a routed
    expert's tensor that ``fmt`` stores in one tensor with the others.
    """
    if lid.expert is None:
        name, axis = lid.name, None
    elif experts is None or lid.expert >= experts.num_experts:
        raise InternalError(f"{lid} is not an expert of the model's expert layout {experts}")
    elif fmt == TORCH_DIST:
        name = join_torch_dist_expert(lid.path, lid.leaf)
        axis = (lid.expert, experts.num_experts)
    else:
        name, axis = join_fsdp_expert(lid.path, lid.leaf, lid.expert, experts.fsdp_form), None
    if lid.part is not None:
        name = _part_name(lid, name, fmt)
    return name, axis


def _part_name(lid: LogicalId, name: str, fmt: str) -> str:
    """The name of the part ``lid`` names of the tensor ``name`` in ``fmt``: an fsdp_dtensor SwiGLU
    half, a torch_dist part of a fused projection, or ``name`` itself for a torch_dist SwiGLU half
    (its rows of the whole tensor).

    Raises:
        InternalError: ``fmt`` stores no such part.
    """
    assert lid.part is not None
    if is_glu_half(lid.path, lid.part):
        return name if fmt == TORCH_DIST else join_glu_half(name, lid.part)
    if fmt == TORCH_DIST and is_fused_projection(lid.path):
        return join_factory_part(name, lid.part)
    raise InternalError(f"{lid}: {fmt} stores no part {lid.part!r} of this tensor")
