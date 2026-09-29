# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Rank-major maps of fused tensors against packs built with explicit loops over ranks."""

from __future__ import annotations

import pytest
import torch

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.part_layout import PartLayout, PartRows, RowMove
from tools.checkpoint.format_converter.errors import InternalError, RefusedError

GLU = PartLayout(("w", "v"), (8, 8))
MIXER = PartLayout(("z", "B", "dt"), (8, 4, 12))
"""Parts of different sizes, as a state-space mixer's ``in_proj`` has."""


def parts_of(layout: PartLayout, width: int = 3) -> dict[str, torch.Tensor]:
    """Distinct values for every element of every part, ``[size, width]`` each."""
    start = 0
    parts = {}
    for part, size in zip(layout.parts, layout.sizes):
        parts[part] = torch.arange(start, start + size * width).reshape(size, width)
        start += size * width
    return parts


def packed_by_loops(parts: dict[str, torch.Tensor], order: tuple[str, ...], tp: int):
    """What ``tp`` column-parallel ranks hold, concatenated rank after rank: each rank's local
    tensor is its chunk of every part, in order."""
    ranks = [torch.cat([parts[name].chunk(tp)[rank] for name in order]) for rank in range(tp)]
    return torch.cat(ranks)


def pack(layout: PartLayout, parts: dict[str, torch.Tensor], tp: int) -> torch.Tensor:
    width = next(iter(parts.values())).shape[1]
    fused = torch.full((layout.fused_rows, width), -1, dtype=torch.long)
    for run in layout.rank_major_map(tp):
        fused[run.fused_start : run.fused_start + run.rows] = parts[run.part][
            run.part_start : run.part_start + run.rows
        ]
    return fused


def unpack(layout: PartLayout, fused: torch.Tensor, tp: int) -> dict[str, torch.Tensor]:
    parts = {
        part: torch.full((size, fused.shape[1]), -1, dtype=torch.long)
        for part, size in zip(layout.parts, layout.sizes)
    }
    for run in layout.rank_major_map(tp):
        parts[run.part][run.part_start : run.part_start + run.rows] = fused[
            run.fused_start : run.fused_start + run.rows
        ]
    return parts


@pytest.mark.parametrize("layout", [GLU, MIXER], ids=["glu", "mixer"])
@pytest.mark.parametrize("tp", [1, 2, 4])
def test_packing_and_unpacking_match_the_ranks_local_tensors(layout, tp):
    parts = parts_of(layout)
    expected = packed_by_loops(parts, layout.parts, tp)
    assert torch.equal(pack(layout, parts, tp), expected)
    unpacked = unpack(layout, expected, tp)
    assert all(torch.equal(unpacked[name], parts[name]) for name in layout.parts)


def test_the_map_of_a_gated_unit_at_two_ranks():
    assert GLU.rank_major_map(2) == (
        PartRows("w", 0, 0, 4),
        PartRows("v", 0, 4, 4),
        PartRows("w", 4, 8, 4),
        PartRows("v", 4, 12, 4),
    )
    assert GLU.rank_major_map(1) == (PartRows("w", 0, 0, 8), PartRows("v", 0, 8, 8))
    assert GLU.fused_rows == 16 and MIXER.size("dt") == 12


@pytest.mark.parametrize("layout", [GLU, MIXER], ids=["glu", "mixer"])
@pytest.mark.parametrize(("src", "dst"), [(1, 2), (2, 1), (2, 4), (4, 2), (1, 4), (2, 2)])
def test_repacking_moves_rows_between_pack_sizes(layout, src, dst):
    parts = parts_of(layout)
    source = packed_by_loops(parts, layout.parts, src)
    moved = torch.full_like(source, -1)
    for move in layout.repack_map(src, dst):
        moved[move.dst_start : move.dst_start + move.rows] = source[
            move.src_start : move.src_start + move.rows
        ]
    assert torch.equal(moved, packed_by_loops(parts, layout.parts, dst))


def test_repacking_the_canonical_order_of_a_gated_unit():
    assert GLU.repack_map(1, 2) == (
        RowMove(0, 0, 4),
        RowMove(8, 4, 4),
        RowMove(4, 8, 4),
        RowMove(12, 12, 4),
    )
    assert GLU.repack_map(2, 2) == tuple(RowMove(s, s, 4) for s in (0, 4, 8, 12))


@pytest.mark.parametrize("tp", [3, 16])
def test_parts_that_do_not_divide_by_the_pack_size_are_refused(tp):
    with pytest.raises(RefusedError) as caught:
        MIXER.rank_major_map(tp)
    assert caught.value.codes == (errors.FC_R10,)
    assert "do not divide by the tensor-parallel size" in caught.value.refusal.message
    with pytest.raises(RefusedError):
        MIXER.repack_map(1, tp)


@pytest.mark.parametrize(
    ("parts", "sizes"), [((), ()), (("w", "w"), (2, 2)), (("w", "v"), (2,)), (("w", "v"), (2, 0))]
)
def test_malformed_layouts_are_internal_errors(parts, sizes):
    with pytest.raises(InternalError):
        PartLayout(parts, sizes)


def test_pack_sizes_must_be_positive_and_parts_known():
    with pytest.raises(InternalError):
        GLU.rank_major_map(0)
    with pytest.raises(InternalError):
        GLU.size("z")
