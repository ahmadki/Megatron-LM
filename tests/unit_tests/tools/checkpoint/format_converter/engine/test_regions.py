# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Region algebra, checked exhaustively on small shapes against explicit cell sets."""

from __future__ import annotations

import itertools
import random

import pytest
import torch

from tools.checkpoint.format_converter.engine.regions import (
    Region,
    bounding_box,
    corresponding,
    dim0_only,
    find_tiling_defect,
    same_up_to_leading_ones,
)

SHAPE = (4, 3)


def _cells(region: Region) -> set[tuple[int, ...]]:
    ranges = [range(o, o + s) for o, s in zip(region.offsets, region.sizes)]
    return set(itertools.product(*ranges))


def _all_regions(shape: tuple[int, ...]) -> list[Region]:
    per_dim = [[(o, s) for o in range(n + 1) for s in range(n - o + 1)] for n in shape]
    return [
        Region(tuple(o for o, _ in combo), tuple(s for _, s in combo))
        for combo in itertools.product(*per_dim)
    ]


REGIONS = _all_regions(SHAPE)
NONEMPTY = [r for r in REGIONS if r.numel() > 0]


def test_region_count_and_numel():
    # 15 (offset, size) pairs of a length-4 axis times 10 of a length-3 axis.
    assert len(REGIONS) == 150
    for region in REGIONS:
        assert region.numel() == len(_cells(region))


def test_intersect_matches_cell_sets_exhaustively():
    for a, b in itertools.product(NONEMPTY, repeat=2):
        expected = _cells(a) & _cells(b)
        got = a.intersect(b)
        if expected:
            assert got is not None and _cells(got) == expected, (a, b)
        else:
            assert got is None, (a, b)


def test_contains_matches_cell_sets_exhaustively():
    for a, b in itertools.product(NONEMPTY, repeat=2):
        assert a.contains(b) == (_cells(b) <= _cells(a)), (a, b)


def test_bounding_box_is_the_smallest_container():
    rng = random.Random(0)
    for _ in range(500):
        picked = rng.sample(NONEMPTY, rng.randint(1, 4))
        box = bounding_box(picked)
        assert all(box.contains(r) for r in picked)
        cells = set().union(*(_cells(r) for r in picked))
        for d in range(len(SHAPE)):
            assert box.offsets[d] == min(c[d] for c in cells)
            assert box.ends[d] == max(c[d] for c in cells) + 1


def test_shift_and_relative_to_round_trip():
    container = Region((1, 1), (3, 2))
    for region in NONEMPTY:
        if not container.contains(region):
            with pytest.raises(ValueError):
                region.relative_to(container)
            continue
        local = region.relative_to(container)
        assert local.shift(container.offsets) == region
        assert _cells(local) == {(r - 1, c - 1) for r, c in _cells(region)}


def test_slices_select_the_region_cells():
    grid = torch.arange(12).reshape(SHAPE)
    for region in REGIONS:
        picked = grid[region.slices()].reshape(-1).tolist()
        expected = sorted(r * SHAPE[1] + c for r, c in _cells(region))
        assert picked == expected


@pytest.mark.parametrize("max_rows", [1, 2, 3, 4, 7])
def test_split_dim0_tiles_the_region(max_rows):
    for region in NONEMPTY:
        slabs = region.split_dim0(max_rows)
        assert all(s.sizes[0] <= max(max_rows, 0) for s in slabs)
        assert [s.offsets[0] for s in slabs] == sorted(s.offsets[0] for s in slabs)
        union = set().union(*(_cells(s) for s in slabs))
        assert union == _cells(region)
        assert sum(s.numel() for s in slabs) == region.numel()


def test_text_form():
    assert str(Region((1, 0), (2, 3))) == "[1:3, 0:3]"
    assert str(Region((), ())) == "[]"


def test_is_full_and_zero_dim():
    assert Region.full(SHAPE).is_full(SHAPE)
    assert not Region((0, 0), (4, 2)).is_full(SHAPE)
    scalar = Region.full(())
    assert scalar == Region((), ()) and scalar.numel() == 1 and scalar.is_full(())
    assert scalar.intersect(scalar) == scalar
    assert scalar.split_dim0(1) == [scalar]
    assert scalar.slices() == ()


def test_validation_errors():
    with pytest.raises(ValueError, match="differ in rank"):
        Region((0,), (1, 2))
    with pytest.raises(ValueError, match="non-negative"):
        Region((-1,), (2,))
    with pytest.raises(ValueError, match="not comparable"):
        Region((0,), (1,)).intersect(Region((0, 0), (1, 1)))
    with pytest.raises(ValueError, match="max_rows"):
        Region((0,), (4,)).split_dim0(0)
    with pytest.raises(ValueError, match="at least one"):
        bounding_box([])


def test_regions_accept_torch_sizes_as_plain_tuples():
    region = Region(torch.Size([1, 0]), torch.Size([2, 3]))
    assert region == Region((1, 0), (2, 3))
    assert type(region.offsets) is tuple and hash(region) == hash(Region((1, 0), (2, 3)))


def test_dim0_only():
    assert dim0_only([Region((0, 0), (2, 3)), Region((2, 0), (2, 3))], SHAPE)
    assert not dim0_only([Region((0, 0), (4, 2)), Region((0, 2), (4, 1))], SHAPE)
    assert dim0_only([Region((0,), (2,)), Region((2,), (2,))], (4,))
    assert dim0_only([Region((), ())], ())


def _brute_tiling_ok(regions: list[Region], shape: tuple[int, ...]) -> bool:
    full = _cells(Region.full(shape))
    seen: set[tuple[int, ...]] = set()
    for region in regions:
        cells = _cells(region)
        if not cells <= full or cells & seen:
            return False
        seen |= cells
    return seen == full


def test_find_tiling_defect_matches_brute_force():
    rng = random.Random(1)
    grids = [
        [Region((0, 0), (2, 3)), Region((2, 0), (2, 3))],
        [Region((0, 0), (4, 1)), Region((0, 1), (4, 2))],
        [Region((0, 0), (2, 2)), Region((0, 2), (2, 1)), Region((2, 0), (2, 3))],
    ]
    for _ in range(3000):
        grids.append(rng.sample(NONEMPTY, rng.randint(1, 4)))
    for regions in grids:
        defect = find_tiling_defect(regions, SHAPE)
        assert (defect is None) == _brute_tiling_ok(regions, SHAPE), (regions, defect)


def test_find_tiling_defect_messages():
    assert "outside" in find_tiling_defect([Region((0, 0), (5, 3))], SHAPE)
    overlap = [Region((0, 0), (3, 3)), Region((2, 0), (2, 3))]
    assert "overlap" in find_tiling_defect(overlap, SHAPE)
    assert "cover 9 of 12" in find_tiling_defect([Region((0, 0), (3, 3))], SHAPE)
    assert find_tiling_defect([Region((), ())], ()) is None
    assert "overlap" in find_tiling_defect([Region((), ()), Region((), ())], ())
    assert find_tiling_defect([Region((0, 0), (0, 3))], (0, 3)) is None


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ((1, 4, 3), (4, 3), True),
        ((4, 3), (1, 1, 4, 3), True),
        ((1,), (), True),
        ((), (), True),
        ((1, 1, 3), (1, 3), True),
        ((2, 4, 3), (4, 3), False),
        ((4, 1, 3), (4, 3), False),
        ((4, 3), (3, 4), False),
        ((1, 4, 3), (4, 1, 3), False),
    ],
)
def test_same_up_to_leading_ones(a, b, same):
    assert same_up_to_leading_ones(a, b) is same
    assert same_up_to_leading_ones(b, a) is same


def test_corresponding_maps_cells_between_ranks_exhaustively():
    """Every sub-box of a (4, 3) box maps to the same cells of layer 2 of a (5, 4, 3) tensor."""
    of = Region.full(SHAPE)
    onto = Region((2, 0, 0), (1, *SHAPE))
    for part in NONEMPTY:
        mapped = corresponding(part, of, onto)
        assert _cells(mapped) == {(2, *cell) for cell in _cells(part)}
        assert corresponding(mapped, onto, of) == part


def test_corresponding_follows_the_offsets_of_both_boxes():
    of = Region((1, 3), (2, 2))
    onto = Region((7, 0, 5), (1, 2, 2))
    assert corresponding(Region((2, 4), (1, 1)), of, onto) == Region((7, 1, 6), (1, 1, 1))
    assert corresponding(Region((), ()), Region((), ()), Region((4,), (1,))) == Region((4,), (1,))
    with pytest.raises(ValueError, match="do not hold the same elements"):
        corresponding(of, of, Region((0, 0), (2, 3)))
    with pytest.raises(ValueError, match="is not inside"):
        corresponding(Region((0, 0), (1, 1)), of, onto)
