# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Chunk-aligned, bitwise comparison of two stores key by key (identity key map).

For every key present on both sides the shapes and dtypes are compared, then
(depending on the mode) the values. Values are compared per *compare unit*: a
connected component of the overlap graph between A's and B's stored chunks.
Both sides of a unit cover the same elements, each stored chunk belongs to
exactly one unit, and a unit is read into one buffer per side, so every stored
chunk is decoded once. Buffers are compared as raw bytes, so ``-0.0`` versus
``0.0`` and different NaN payloads are differences.

A unit larger than ``--max-item-bytes`` (per side) is refused with exit 8
before anything is read.
"""

from __future__ import annotations

import bisect
import random
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

import torch

from ..engine.metadata import TensorEntry
from ..engine.reader import ReadReq, ReadStats, SourceReader
from ..engine.regions import Region, bounding_box
from ..engine.sizes import format_bytes
from ..errors import BudgetExceeded
from ..formats.detect import Store
from ..options import VerifyOptions
from .report import Finding, FindingKind, VerifyReport, sort_findings

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class CompareUnit:
    """A connected set of A and B stored chunks of one tensor, compared together.

    Attributes:
        key: Tensor key.
        region: Bounding box of the unit's chunks.
        a_chunks: A's stored chunks in the unit.
        b_chunks: B's stored chunks in the unit (same elements as ``a_chunks``).
        dtype: Element type (equal on both sides).
    """

    key: str
    region: Region
    a_chunks: tuple[Region, ...]
    b_chunks: tuple[Region, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        """Buffer size of one side."""
        return self.region.numel() * self.dtype.itemsize

    @property
    def fills_region(self) -> bool:
        """Whether the chunks cover the whole bounding box."""
        return sum(r.numel() for r in self.a_chunks) == self.region.numel()


@dataclass(frozen=True, slots=True)
class _UnitDifference:
    key: str
    differing: int
    first: tuple[int, ...]
    a_value: str
    b_value: str


def compare_stores(a: Store, b: Store, opts: VerifyOptions) -> VerifyReport:
    """Compare store ``a`` with store ``b`` key by key.

    Raises:
        BudgetExceeded: Before any read, if a compare unit or a byte item does
            not fit ``opts.max_item_bytes``.
    """
    a_tensors = {k: e for k, e in a.view.tensors.items() if not opts.ignores(k)}
    b_tensors = {k: e for k, e in b.view.tensors.items() if not opts.ignores(k)}
    findings = _key_set_findings(a_tensors.keys(), b_tensors.keys())
    comparable = []
    for key in sorted(a_tensors.keys() & b_tensors.keys()):
        problems = _metadata_findings(a_tensors[key], b_tensors[key], layout=opts.layout)
        findings.extend(problems)
        if not any(p.kind in (FindingKind.SHAPE, FindingKind.DTYPE) for p in problems):
            comparable.append(key)
    value_keys = _select_value_keys(comparable, opts)
    units = [u for key in value_keys for u in overlap_units(a_tensors[key], b_tensors[key])]
    for unit in units:
        _check_unit_budget(unit, opts.max_item_bytes)
    byte_sizes: dict[str, int] = {}
    if opts.common == "bytes":
        key_findings, byte_sizes = _plan_byte_items(a, b, opts)
        findings.extend(key_findings)
    reader_a = SourceReader(a.view, max_item_bytes=opts.max_item_bytes)
    reader_b = SourceReader(b.view, max_item_bytes=opts.max_item_bytes)
    differences, stats = _compare_units(reader_a, reader_b, units, opts.workers)
    findings.extend(_value_findings(differences, a_tensors))
    findings.extend(_compare_byte_items(reader_a, reader_b, byte_sizes))
    return VerifyReport(
        a=a.iter_dir,
        reference=str(b.iter_dir),
        mode=str(opts.mode),
        findings=sort_findings(findings),
        keys_compared=len(a_tensors.keys() & b_tensors.keys()),
        values_compared=len(value_keys),
        byte_items_compared=len(byte_sizes),
        read_stats=stats,
    )


def overlap_units(a: TensorEntry, b: TensorEntry) -> list[CompareUnit]:
    """Split a tensor into compare units: components of the A/B chunk-overlap graph."""
    if a.numel == 0:
        return []
    a_regions = [c.region for c in a.chunks]
    b_regions = [c.region for c in b.chunks]
    if set(a_regions) == set(b_regions):
        return [CompareUnit(a.key, r, (r,), (r,), a.dtype) for r in a_regions]
    parent = list(range(len(a_regions) + len(b_regions)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i, j in _overlapping_pairs(a_regions, b_regions):
        parent[find(i)] = find(len(a_regions) + j)
    groups: dict[int, tuple[list[Region], list[Region]]] = {}
    for i, region in enumerate(a_regions):
        groups.setdefault(find(i), ([], []))[0].append(region)
    for j, region in enumerate(b_regions):
        groups.setdefault(find(len(a_regions) + j), ([], []))[1].append(region)
    units = [
        CompareUnit(a.key, bounding_box(a_part + b_part), tuple(a_part), tuple(b_part), a.dtype)
        for a_part, b_part in groups.values()
    ]
    return sorted(units, key=lambda u: u.region.offsets)


def group_by_budget(items: Iterable[T], size_of: Callable[[T], int], budget: int) -> list[list[T]]:
    """Consecutive groups whose total size stays within ``budget`` (an oversized item is alone)."""
    groups: list[list[T]] = []
    total = 0
    for item in items:
        size = size_of(item)
        if not groups or total + size > budget:
            groups.append([])
            total = 0
        groups[-1].append(item)
        total += size
    return groups


def _overlapping_pairs(
    a_regions: Sequence[Region], b_regions: Sequence[Region]
) -> Iterable[tuple[int, int]]:
    if a_regions and a_regions[0].ndim == 0:
        yield from ((i, j) for i in range(len(a_regions)) for j in range(len(b_regions)))
        return
    order = sorted(range(len(b_regions)), key=lambda j: b_regions[j].offsets[0])
    starts = [b_regions[j].offsets[0] for j in order]
    longest = max(r.sizes[0] for r in b_regions)
    for i, region in enumerate(a_regions):
        low = bisect.bisect_left(starts, region.offsets[0] - longest + 1)
        high = bisect.bisect_left(starts, region.ends[0])
        for position in range(low, high):
            if region.intersect(b_regions[order[position]]) is not None:
                yield i, order[position]


def _key_set_findings(a_keys: Iterable[str], b_keys: Iterable[str]) -> list[Finding]:
    a_set, b_set = set(a_keys), set(b_keys)
    findings = [Finding(FindingKind.ONLY_IN_A, k, "missing in B") for k in a_set - b_set]
    findings += [Finding(FindingKind.ONLY_IN_B, k, "missing in A") for k in b_set - a_set]
    return findings


def _metadata_findings(a: TensorEntry, b: TensorEntry, *, layout: bool) -> list[Finding]:
    if a.shape != b.shape:
        return [Finding(FindingKind.SHAPE, a.key, f"A {a.shape} vs B {b.shape}")]
    if a.dtype != b.dtype:
        return [Finding(FindingKind.DTYPE, a.key, f"A {a.dtype} vs B {b.dtype}")]
    a_grid = {c.region for c in a.chunks}
    b_grid = {c.region for c in b.chunks}
    if layout and a_grid != b_grid:
        only_a = min(a_grid - b_grid, key=lambda r: r.offsets, default=None)
        only_b = min(b_grid - a_grid, key=lambda r: r.offsets, default=None)
        return [
            Finding(
                FindingKind.LAYOUT,
                a.key,
                f"chunk grids differ: A has {len(a_grid)} chunks, B has {len(b_grid)}; "
                f"first A-only chunk {only_a}, first B-only chunk {only_b}",
            )
        ]
    return []


def _select_value_keys(comparable: list[str], opts: VerifyOptions) -> list[str]:
    if opts.mode.kind == "meta":
        return []
    if opts.mode.kind == "full":
        return comparable
    count = min(opts.mode.sample_size, len(comparable))
    return sorted(random.Random(opts.seed).sample(comparable, count))


def _check_unit_budget(unit: CompareUnit, max_item_bytes: int) -> None:
    if unit.nbytes > max_item_bytes:
        raise BudgetExceeded(
            f"comparing {unit.key} needs {unit.region} ({format_bytes(unit.nbytes)} per side) "
            f"in memory, above the per-item budget of {format_bytes(max_item_bytes)}",
            hint=f"pass --max-item-bytes {unit.nbytes} or more (host memory permitting)",
        )


def _compare_units(
    reader_a: SourceReader, reader_b: SourceReader, units: list[CompareUnit], workers: int
) -> tuple[list[_UnitDifference], ReadStats]:
    jobs = group_by_budget(units, lambda u: u.nbytes, reader_a.batch_bytes)
    differences: list[_UnitDifference] = []
    stats = ReadStats()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for job_differences, job_stats in pool.map(
            lambda job: _compare_job(reader_a, reader_b, job), jobs
        ):
            differences.extend(job_differences)
            stats += job_stats
    return differences, stats


def _compare_job(
    reader_a: SourceReader, reader_b: SourceReader, units: list[CompareUnit]
) -> tuple[list[_UnitDifference], ReadStats]:
    a_buffers, a_reqs = _unit_buffers(units, lambda u: u.a_chunks)
    b_buffers, b_reqs = _unit_buffers(units, lambda u: u.b_chunks)
    stats = reader_a.read_into(a_reqs) + reader_b.read_into(b_reqs)
    differences = []
    for unit, a_buffer, b_buffer in zip(units, a_buffers, b_buffers):
        difference = _bitwise_difference(unit, a_buffer, b_buffer)
        if difference is not None:
            differences.append(difference)
    return differences, stats


def _unit_buffers(
    units: list[CompareUnit], chunks_of: Callable[[CompareUnit], tuple[Region, ...]]
) -> tuple[list[torch.Tensor], list[ReadReq]]:
    buffers, reqs = [], []
    for unit in units:
        allocate = torch.empty if unit.fills_region else torch.zeros
        buffer = allocate(unit.region.sizes, dtype=unit.dtype)
        buffers.append(buffer)
        for chunk in chunks_of(unit):
            view = buffer[chunk.relative_to(unit.region).slices()]
            reqs.append(ReadReq(unit.key, chunk, view))
    return buffers, reqs


def _bitwise_difference(
    unit: CompareUnit, a: torch.Tensor, b: torch.Tensor
) -> _UnitDifference | None:
    a_bytes = a.reshape(-1).view(torch.uint8)
    b_bytes = b.reshape(-1).view(torch.uint8)
    if torch.equal(a_bytes, b_bytes):
        return None
    itemsize = unit.dtype.itemsize
    differs = (a_bytes != b_bytes).reshape(-1, itemsize).any(dim=1)
    first = int(differs.nonzero()[0])
    local = torch.unravel_index(torch.tensor(first), unit.region.sizes) if unit.region.ndim else ()
    coords = tuple(o + int(i) for o, i in zip(unit.region.offsets, local))
    return _UnitDifference(
        key=unit.key,
        differing=int(differs.sum()),
        first=coords,
        a_value=_describe_element(a, first),
        b_value=_describe_element(b, first),
    )


def _describe_element(buffer: torch.Tensor, index: int) -> str:
    element = buffer.reshape(-1)[index : index + 1]
    raw = bytes(element.view(torch.uint8).tolist()).hex()
    return f"{element.item()!r} (bytes {raw})"


def _value_findings(
    differences: list[_UnitDifference], tensors: dict[str, TensorEntry]
) -> list[Finding]:
    by_key: dict[str, list[_UnitDifference]] = {}
    for difference in differences:
        by_key.setdefault(difference.key, []).append(difference)
    findings = []
    for key, parts in by_key.items():
        first = min(parts, key=lambda p: p.first)
        total = sum(p.differing for p in parts)
        findings.append(
            Finding(
                FindingKind.VALUE,
                key,
                f"{total} of {tensors[key].numel} elements differ bitwise; first at "
                f"{first.first}: A {first.a_value} vs B {first.b_value}",
            )
        )
    return findings


def _plan_byte_items(
    a: Store, b: Store, opts: VerifyOptions
) -> tuple[list[Finding], dict[str, int]]:
    """Key-set findings of the byte items and the payload size of each shared one.

    Raises:
        BudgetExceeded: A shared byte item is larger than ``opts.max_item_bytes``.
    """
    a_items = {k: e for k, e in a.view.byte_items.items() if not opts.ignores(k)}
    b_items = {k: e for k, e in b.view.byte_items.items() if not opts.ignores(k)}
    sizes = {
        k: max(a_items[k].length, b_items[k].length)
        for k in sorted(a_items.keys() & b_items.keys())
    }
    for key, size in sizes.items():
        if size > opts.max_item_bytes:
            raise BudgetExceeded(
                f"byte item {key} is {format_bytes(size)}, above the per-item budget of "
                f"{format_bytes(opts.max_item_bytes)}",
                hint="pass a larger --max-item-bytes or --common skip",
            )
    return _key_set_findings(a_items.keys(), b_items.keys()), sizes


def _compare_byte_items(
    reader_a: SourceReader, reader_b: SourceReader, sizes: dict[str, int]
) -> list[Finding]:
    findings = []
    for batch in group_by_budget(sizes, sizes.__getitem__, reader_a.batch_bytes):
        a_payloads = reader_a.read_bytes(batch)
        b_payloads = reader_b.read_bytes(batch)
        findings.extend(
            Finding(
                FindingKind.BYTES,
                k,
                f"payloads differ (A {len(a_payloads[k])} bytes, B {len(b_payloads[k])} bytes)",
            )
            for k in batch
            if a_payloads[k] != b_payloads[k]
        )
    return findings
