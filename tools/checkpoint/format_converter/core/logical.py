# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The logical key space: format-independent identities of model tensors.

Conversion never maps one format's keys to the other's directly. Every source
tensor is *decomposed* into :class:`LogicalSlice` objects, each naming a region
of a stored key by a :class:`LogicalId`: the source format's adapter strips its
key envelope and ``core/structure.py`` maps the model-relative key to an id
(key grammar and the blocks' layer numbering). Rules turn slices into
:class:`LogicalOutput` objects (what each output tensor holds, as
:class:`LogicalCopy` boxes of source slices). Every output id is then
*composed* back: ``core/structure.py`` gives its model-relative key in the
target format, and the target format's adapter adds the envelope and supplies
the planner path and the chunk grid. Rules therefore see only logical ids and
serve both directions.

A :class:`LogicalId` names one model tensor as the live model does: the block
it belongs to, its *dense* layer index in that block (the ``nn.ModuleList``
position, whatever the stored index), the module path inside the layer and
the leaf (parameter or buffer) name.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..engine.regions import Region, corresponding, find_tiling_defect
from ..errors import InternalError
from .keys import join_leaf


@dataclass(frozen=True, slots=True)
class LogicalId:
    """Format-independent identity of one model tensor.

    Attributes:
        block: Prefix of the layered block holding the tensor (``decoder``),
            or ``None`` outside blocks.
        layer: Dense layer index inside the block (``None`` outside blocks).
        path: Module path inside the layer, or from the model root outside
            blocks (may be empty).
        leaf: Parameter or buffer name.

    Raises:
        InternalError: Exactly one of ``block`` and ``layer`` is set, the
            layer index is negative, or the leaf is empty.
    """

    block: str | None
    layer: int | None
    path: str
    leaf: str

    def __post_init__(self) -> None:
        if (self.block is None) != (self.layer is None):
            raise InternalError(f"{self}: block and layer must be set together")
        if self.layer is not None and self.layer < 0:
            raise InternalError(f"{self}: negative layer index")
        if not self.leaf:
            raise InternalError(f"logical id at {self.path!r} has no leaf")

    @property
    def name(self) -> str:
        """``<path>.<leaf>`` (the name inside the layer, or from the model root)."""
        return join_leaf(self.path, self.leaf)

    def __str__(self) -> str:
        if self.block is None:
            return self.name
        return f"{self.block}[{self.layer}].{self.name}"


@dataclass(frozen=True, slots=True)
class LogicalSlice:
    """A region of one stored source tensor that holds one logical tensor.

    The logical tensor has the region's shape without its first ``squeezed``
    axes, which have size 1: one layer ``[l:l+1, ...]`` of a tensor stacked
    over the layers of a block is a logical tensor of the layer's shape.

    Attributes:
        lid: The logical tensor.
        key: The stored source key (with the format envelope).
        region: The box of ``key`` that holds the logical tensor.
        dtype: Element type of the stored tensor.
        squeezed: Leading size-1 axes of ``region`` the logical tensor does not have.

    Raises:
        InternalError: ``region`` does not have ``squeezed`` leading size-1 axes.
    """

    lid: LogicalId
    key: str
    region: Region
    dtype: torch.dtype
    squeezed: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.squeezed <= self.region.ndim or any(
            size != 1 for size in self.region.sizes[: self.squeezed]
        ):
            raise InternalError(
                f"{self.lid} in {self.key}: {self.region} has no {self.squeezed} leading "
                "size-1 axes to squeeze"
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the logical tensor."""
        return self.region.sizes[self.squeezed :]

    def stored_box(self, box: Region) -> Region:
        """The box of ``key`` that holds ``box`` of the logical tensor."""
        return corresponding(box, Region.full(self.shape), self.region)


@dataclass(frozen=True, slots=True)
class LogicalCopy:
    """One box of a source logical tensor copied into a box of an output logical tensor.

    Attributes:
        src: The source logical tensor.
        src_region: The box to copy, in the source logical tensor's coordinates.
        dst_region: Where it goes, in the output logical tensor's coordinates.

    Raises:
        InternalError: The two boxes differ in shape.
    """

    src: LogicalId
    src_region: Region
    dst_region: Region

    def __post_init__(self) -> None:
        if self.src_region.sizes != self.dst_region.sizes:
            raise InternalError(
                f"copy from {self.src}: boxes {self.src_region} and {self.dst_region} "
                "differ in shape"
            )


@dataclass(frozen=True, slots=True)
class LogicalOutput:
    """One output logical tensor and the copies that fill it.

    Attributes:
        lid: The output logical tensor.
        shape: Its shape.
        dtype: Its element type.
        copies: Copies whose destination boxes tile ``shape`` exactly.

    Raises:
        InternalError: The copies do not tile the tensor.
    """

    lid: LogicalId
    shape: tuple[int, ...]
    dtype: torch.dtype
    copies: tuple[LogicalCopy, ...]

    def __post_init__(self) -> None:
        defect = find_tiling_defect([copy.dst_region for copy in self.copies], self.shape)
        if defect is not None:
            raise InternalError(f"copies into {self.lid} do not tile it: {defect}")
