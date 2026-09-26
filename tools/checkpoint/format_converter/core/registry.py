# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The registry every conversion rule and sentinel is declared in.

Rule modules register their :class:`~.rule.LeafRule` subclasses with the
``@REGISTRY.rule`` decorator when they are imported; planning asks the
registry for the enabled rules in stage order.

A :class:`Sentinel` names source keys that the generic identity rule must
never convert, because they need a specific rule (for example a fused
projection stored as parts, which only a part-aware rule may reassemble). Its
owners are the rules, or while a family is not supported yet the refusal
codes, that handle such keys. A slice that matches a sentinel and that no
other rule claimed is refused with the sentinel's reason, so a key that slips
past every detector still fails closed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypeVar

from ..errors import InternalError
from .rule import LeafRule

RuleT = TypeVar("RuleT", bound=type[LeafRule])


@dataclass(frozen=True, slots=True)
class Sentinel:
    """Source keys the identity rule must leave to their owners.

    Attributes:
        name: Unique name.
        fmt: Source format whose keys the pattern applies to.
        pattern: Searched in the stored source key.
        owners: Rules, or refusal codes of pending families, that handle the keys.
        reason: Why identity may not convert them.
    """

    name: str
    fmt: str
    pattern: re.Pattern[str]
    owners: tuple[str, ...]
    reason: str

    def matches(self, key: str) -> bool:
        """Whether the stored source key ``key`` is reserved by this sentinel."""
        return self.pattern.search(key) is not None


class RuleRegistry:
    """Rules and sentinels, by name."""

    def __init__(self) -> None:
        self._rules: dict[str, LeafRule] = {}
        self._sentinels: dict[str, Sentinel] = {}

    def rule(self, cls: RuleT) -> RuleT:
        """Class decorator: register one instance of a rule class under its ``name``.

        Raises:
            InternalError: Another rule has the same name.
        """
        if cls.name in self._rules:
            raise InternalError(f"rule {cls.name} is registered twice")
        self._rules[cls.name] = cls()
        return cls

    def sentinel(self, sentinel: Sentinel) -> Sentinel:
        """Register a sentinel.

        Raises:
            InternalError: Another sentinel has the same name, or it has no owner.
        """
        if sentinel.name in self._sentinels:
            raise InternalError(f"sentinel {sentinel.name} is registered twice")
        if not sentinel.owners:
            raise InternalError(f"sentinel {sentinel.name} has no owner")
        self._sentinels[sentinel.name] = sentinel
        return sentinel

    @property
    def rules(self) -> dict[str, LeafRule]:
        """Every registered rule by name."""
        return dict(self._rules)

    @property
    def sentinels(self) -> tuple[Sentinel, ...]:
        """Every registered sentinel, by name."""
        return tuple(self._sentinels[name] for name in sorted(self._sentinels))

    def rules_for(self, ctx: Any) -> list[LeafRule]:
        """The rules enabled for the context ``ctx``, in (stage, name) order."""
        ordered = sorted(self._rules.values(), key=lambda rule: (rule.stage, rule.name))
        return [rule for rule in ordered if rule.enabled(ctx)]

    def sentinels_for(self, fmt: str) -> tuple[Sentinel, ...]:
        """The sentinels over source keys of format ``fmt``, by name."""
        return tuple(sentinel for sentinel in self.sentinels if sentinel.fmt == fmt)


REGISTRY = RuleRegistry()
"""The registry of the package's rules and sentinels."""
