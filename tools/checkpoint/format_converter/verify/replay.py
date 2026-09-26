# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Plan replay (``verify A B --map convert``): B against A's conversion, recomputed.

The conversion plan of ``A`` is rebuilt from metadata (the same code that
converted it), every planned piece is filled from ``A`` through its copy
operations and compared bit for bit with the same box of ``B``, and the key
sets, shapes and dtypes are compared with the plan's. Because the reference
is the converter's own plan, this check is **not independent**: it detects
faults in the engine, the workers, the merge or the storage (a corrupt, lost
or misplaced chunk), never a wrong rule. Byte items are compared by key only
(the common state is re-encoded by the conversion).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..engine.pieces import ExpectedTensor, Piece, expected_tensors
from ..engine.reader import ReadReq, ReadStats, SourceReader
from ..engine.sizes import format_bytes
from ..engine.worker import read_requests
from ..errors import BudgetExceeded
from ..formats.detect import Store
from ..options import VerifyOptions
from .compare import (
    CompareUnit,
    UnitDifference,
    bitwise_difference,
    group_by_budget,
    value_findings,
)
from .report import Finding, FindingKind, VerifyReport, sort_findings

NOT_INDEPENDENT = "not independent: detects engine and I/O faults only"


def replay_conversion(
    a: Store, b: Store, pieces: Sequence[Piece], byte_keys: frozenset[str], opts: VerifyOptions
) -> VerifyReport:
    """Compare ``b`` with the conversion of ``a`` that ``pieces`` and ``byte_keys`` describe.

    Raises:
        BudgetExceeded: Before any read, if a piece does not fit ``opts.max_item_bytes``.
    """
    planned = {k: e for k, e in expected_tensors(pieces).items() if not opts.ignores(k)}
    stored = {k: e for k, e in b.view.tensors.items() if not opts.ignores(k)}
    findings = [
        Finding(FindingKind.ONLY_IN_A, k, "planned but missing in B")
        for k in planned.keys() - stored.keys()
    ]
    findings += [
        Finding(FindingKind.ONLY_IN_B, k, "in B but not planned")
        for k in stored.keys() - planned.keys()
    ]
    comparable = []
    for key in sorted(planned.keys() & stored.keys()):
        problem = _metadata_finding(planned[key], stored[key].shape, stored[key].dtype)
        if problem is None:
            comparable.append(key)
        else:
            findings.append(problem)
    items = {k for k in b.view.byte_items if not opts.ignores(k)}
    wanted = {k for k in byte_keys if not opts.ignores(k)}
    findings += [
        Finding(FindingKind.ONLY_IN_A, k, "planned but missing in B") for k in wanted - items
    ]
    findings += [Finding(FindingKind.ONLY_IN_B, k, "in B but not planned") for k in items - wanted]
    value_keys = set(opts.value_keys(comparable))
    selected = [piece for piece in pieces if piece.key in value_keys]
    for piece in selected:
        if piece.nbytes > opts.max_item_bytes:
            raise BudgetExceeded(
                f"replaying {piece.key} {piece.chunk} needs {format_bytes(piece.nbytes)} per side, "
                f"above the per-item budget of {format_bytes(opts.max_item_bytes)}",
                hint=f"pass --max-item-bytes {piece.nbytes} or more (host memory permitting)",
            )
    differences, stats = _replay(a, b, selected, opts.max_item_bytes)
    findings.extend(value_findings(differences, stored))
    return VerifyReport(
        a=b.iter_dir,
        reference=f"plan replay of {a.iter_dir} ({NOT_INDEPENDENT})",
        mode=str(opts.mode),
        findings=sort_findings(findings),
        keys_compared=len(planned.keys() & stored.keys()),
        values_compared=len(value_keys),
        byte_items_compared=0,
        read_stats=stats,
        independent=False,
    )


def _metadata_finding(
    planned: ExpectedTensor, shape: tuple[int, ...], dtype: torch.dtype
) -> Finding | None:
    if planned.global_shape != shape:
        return Finding(
            FindingKind.SHAPE, planned.key, f"planned {planned.global_shape} vs B {shape}"
        )
    if planned.dtype != dtype:
        return Finding(FindingKind.DTYPE, planned.key, f"planned {planned.dtype} vs B {dtype}")
    return None


def _replay(
    a: Store, b: Store, pieces: Sequence[Piece], budget: int
) -> tuple[list[UnitDifference], ReadStats]:
    reader_a = SourceReader(a.view, max_item_bytes=budget)
    reader_b = SourceReader(b.view, max_item_bytes=budget)
    differences: list[UnitDifference] = []
    stats = ReadStats()
    for batch in group_by_budget(pieces, lambda p: p.nbytes, reader_a.batch_bytes):
        replayed = [torch.empty(p.chunk.sizes, dtype=p.dtype) for p in batch]
        written = [torch.empty(p.chunk.sizes, dtype=p.dtype) for p in batch]
        stats += reader_a.read_into(read_requests(batch, replayed))
        stats += reader_b.read_into(
            [ReadReq(p.key, p.chunk, buffer) for p, buffer in zip(batch, written)]
        )
        for piece, mine, theirs in zip(batch, replayed, written):
            unit = CompareUnit(piece.key, piece.chunk, (), (), piece.dtype, piece.dtype)
            difference = bitwise_difference(unit, theirs, mine)
            if difference is not None:
                differences.append(difference)
    return differences, stats
