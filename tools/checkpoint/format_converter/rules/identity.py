# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The identity rule: tensors stored the same way in both formats.

It runs last and takes every slice no other rule claimed, except slices whose
stored key a sentinel of the source format reserves for another rule (those
are declined with the sentinel's reason and refused by the planner). Each
taken slice becomes one output tensor with the same logical id and shape; the
element type is decided by the conversion's dtype policy.

The sentinels are registered by ``core/refusals.py``, which this module
imports, so the rule never runs without them whoever imports it.
"""

from __future__ import annotations

from collections.abc import Iterator

from ..core import refusals  # pylint: disable=unused-import  # registers the sentinels
from ..core.context import ArchContext
from ..core.logical import LogicalCopy, LogicalOutput, LogicalSlice
from ..core.registry import REGISTRY
from ..core.rule import FamilyInstance, LeafRule, LogicalPool, Stage
from ..engine.regions import Region


@REGISTRY.rule
class IdentityRule(LeafRule):
    """Copy a tensor unchanged (same logical id, shape and elements)."""

    name = "identity"
    stage = Stage.IDENTITY

    def claim(self, pool: LogicalPool, ctx: ArchContext) -> Iterator[FamilyInstance]:
        """Take every unreserved slice left in ``pool`` and copy it whole."""
        sentinels = REGISTRY.sentinels_for(ctx.source[0])

        def unreserved(piece: LogicalSlice) -> bool:
            reserved = [sentinel for sentinel in sentinels if sentinel.matches(piece.key)]
            for sentinel in reserved:
                pool.decline(piece, f"{sentinel.reason} (owned by {', '.join(sentinel.owners)})")
            return not reserved

        for piece in pool.take(self.name, unreserved):
            full = Region.full(piece.shape)
            output = LogicalOutput(
                piece.lid, piece.shape, piece.dtype, (LogicalCopy(piece.lid, full, full),)
            )
            yield FamilyInstance(self.name, (piece,), (output,))
