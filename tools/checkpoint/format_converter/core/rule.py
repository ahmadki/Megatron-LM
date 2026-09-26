# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Leaf rules and the pool of logical slices they claim exactly once.

Planning puts every source :class:`~.logical.LogicalSlice` into one
:class:`LogicalPool`. Rules run in :class:`Stage` order (then by name); each
takes the slices it handles from the pool and returns :class:`FamilyInstance`
objects: the slices it consumed and the output tensors it builds from them.
A slice can be taken by one rule only, and every taken slice must be consumed
by exactly one instance of that rule; anything else is a bug (exit 1). The
identity rule runs last and takes what is left, except slices a sentinel
reserves for a rule that must handle them (see ``core/registry.py``); those
stay in the pool with the sentinel's reason and are refused.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

from ..errors import InternalError
from .logical import LogicalId, LogicalOutput, LogicalSlice


class Stage(enum.IntEnum):
    """When a rule claims slices: lower stages first, so specific rules precede generic ones."""

    FUSED = 20
    """Fused projections stored as parts in one format and as one tensor in the other."""

    GLU = 30
    """Gated linear-unit ``linear_fc1`` halves."""

    OPTIM = 40
    """Optimizer-state specifics."""

    IDENTITY = 90
    """Tensors stored identically in both formats (runs last)."""


@dataclass(frozen=True, slots=True)
class FamilyInstance:
    """What one rule application consumed and produced.

    Attributes:
        rule: Name of the rule.
        inputs: The source slices consumed (each exactly once).
        outputs: The output tensors built from them.

    Raises:
        InternalError: An input is listed twice, or a copy reads a logical
            tensor that is not an input.
    """

    rule: str
    inputs: tuple[LogicalSlice, ...]
    outputs: tuple[LogicalOutput, ...]

    def __post_init__(self) -> None:
        ids = [piece.lid for piece in self.inputs]
        if len(set(ids)) != len(ids):
            raise InternalError(f"rule {self.rule} lists an input twice: {sorted(map(str, ids))}")
        known = set(ids)
        for output in self.outputs:
            for copy in output.copies:
                if copy.src not in known:
                    raise InternalError(
                        f"rule {self.rule}: {output.lid} copies from {copy.src}, "
                        "which is not an input of the instance"
                    )


class LogicalPool:
    """The source slices of a plan and who claimed and consumed each.

    Args:
        slices: Every source slice; logical ids must be unique.

    Raises:
        InternalError: Two slices share a logical id.
    """

    def __init__(self, slices: Iterable[LogicalSlice]) -> None:
        self._slices: dict[LogicalId, LogicalSlice] = {}
        for piece in slices:
            known = self._slices.setdefault(piece.lid, piece)
            if known is not piece:
                raise InternalError(f"{piece.lid} is held by both {known.key} and {piece.key}")
        self._claimed: dict[LogicalId, str] = {}
        self._consumed: set[LogicalId] = set()
        self._declined: dict[LogicalId, list[str]] = {}

    def take(self, rule: str, predicate: Callable[[LogicalSlice], bool]) -> list[LogicalSlice]:
        """Claim every unclaimed slice matching ``predicate`` for ``rule`` (in pool order)."""
        taken = [
            piece
            for lid, piece in self._slices.items()
            if lid not in self._claimed and predicate(piece)
        ]
        for piece in taken:
            self._claimed[piece.lid] = rule
        return taken

    def decline(self, piece: LogicalSlice, reason: str) -> None:
        """Record why a rule left ``piece`` unclaimed (reported if nothing claims it)."""
        self._declined.setdefault(piece.lid, []).append(reason)

    def consume(self, instance: FamilyInstance) -> None:
        """Record the inputs of ``instance`` as consumed.

        Raises:
            InternalError: An input was not claimed by the instance's rule, or
                was already consumed.
        """
        for piece in instance.inputs:
            owner = self._claimed.get(piece.lid)
            if owner != instance.rule:
                raise InternalError(
                    f"rule {instance.rule} consumed {piece.lid}, which "
                    + ("nobody claimed" if owner is None else f"rule {owner} claimed")
                )
            if piece.lid in self._consumed:
                raise InternalError(f"rule {instance.rule} consumed {piece.lid} twice")
            self._consumed.add(piece.lid)

    def remaining(self) -> list[LogicalSlice]:
        """Slices no rule claimed, in pool order."""
        return [piece for lid, piece in self._slices.items() if lid not in self._claimed]

    def unconsumed_claims(self) -> list[LogicalSlice]:
        """Slices a rule claimed but no instance consumed (a rule bug)."""
        return [
            piece
            for lid, piece in self._slices.items()
            if lid in self._claimed and lid not in self._consumed
        ]

    def declined(self, lid: LogicalId) -> tuple[str, ...]:
        """The reasons rules gave for leaving ``lid`` unclaimed."""
        return tuple(self._declined.get(lid, ()))

    def claims(self) -> dict[str, int]:
        """Number of slices claimed per rule (sorted by rule name)."""
        counts: dict[str, int] = {}
        for rule in self._claimed.values():
            counts[rule] = counts.get(rule, 0) + 1
        return dict(sorted(counts.items()))


class LeafRule(ABC):
    """A conversion rule over logical slices; subclasses are registered with ``REGISTRY.rule``.

    Attributes:
        name: Unique rule name (used in reports and sentinel owners).
        stage: When the rule runs.
    """

    name: ClassVar[str]
    stage: ClassVar[Stage]

    def enabled(self, ctx: Any) -> bool:
        """Whether the rule applies to the conversion described by the context ``ctx``."""
        del ctx
        return True

    @abstractmethod
    def claim(self, pool: LogicalPool, ctx: Any) -> Iterable[FamilyInstance]:
        """Take the slices this rule handles from ``pool`` and build their outputs."""
