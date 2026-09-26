# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The registry every conversion rule, sentinel, refusal detector and allowlist is declared in.

Rule modules register their :class:`~.rule.LeafRule` subclasses with the
``@REGISTRY.rule`` decorator when they are imported; planning asks the
registry for the enabled rules in stage order. Refusal detectors (functions of
the conversion's facts that yield refusals, see ``core/refusals.py``) register
with ``@REGISTRY.detector``, and :class:`Allowlist` entries name the layer
symbols, specs and attention variants that rules have been verified for.

A :class:`Sentinel` names source keys that the generic identity rule must
never convert, because they need a specific rule (for example a fused
projection stored as parts, which only a part-aware rule may reassemble). Its
owners are the rules, or while a family is not supported yet the refusal
codes, that handle such keys. A slice that matches a sentinel and that no
other rule claimed is refused with the sentinel's reason, so a key that slips
past every detector still fails closed.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from ..errors import InternalError, Refusal
from .rule import LeafRule

if TYPE_CHECKING:
    from .context import ArchContext
    from .refusals import Facts

RuleT = TypeVar("RuleT", bound=type[LeafRule])
Detector = Callable[["Facts"], Iterable[Refusal]]
"""A refusal detector: yields the refusals the conversion's facts call for."""


class Allowlist(enum.Enum):
    """Values a converted model may use because rules and tests cover them."""

    HYBRID_SYMBOLS = "hybrid layer symbols"
    """Symbols of ``--hybrid-layer-pattern``."""

    STACK_SPECS = "model specs"
    """``--spec`` values, as ``<module>.<name>``."""

    ATTENTION_VARIANTS = "experimental attention variants"
    """Values of ``experimental_attention_variant`` besides ``None``."""


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
    """Rules, sentinels and refusal detectors by name, and allowlists."""

    def __init__(self) -> None:
        self._rules: dict[str, LeafRule] = {}
        self._sentinels: dict[str, Sentinel] = {}
        self._detectors: dict[str, Detector] = {}
        self._allowed: dict[Allowlist, frozenset[str]] = {kind: frozenset() for kind in Allowlist}

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

    def detector(self, function: Detector) -> Detector:
        """Function decorator: register a refusal detector under its name.

        Raises:
            InternalError: Another detector has the same name.
        """
        name = function.__name__
        if name in self._detectors:
            raise InternalError(f"refusal detector {name} is registered twice")
        self._detectors[name] = function
        return function

    def allow(self, kind: Allowlist, *values: str) -> None:
        """Add verified values to an allowlist."""
        self._allowed[kind] = self._allowed[kind] | frozenset(values)

    def allowed(self, kind: Allowlist) -> frozenset[str]:
        """The verified values of an allowlist."""
        return self._allowed[kind]

    @property
    def detectors(self) -> dict[str, Detector]:
        """Every registered refusal detector by name, in registration order."""
        return dict(self._detectors)

    @property
    def rules(self) -> dict[str, LeafRule]:
        """Every registered rule by name."""
        return dict(self._rules)

    @property
    def sentinels(self) -> tuple[Sentinel, ...]:
        """Every registered sentinel, by name."""
        return tuple(self._sentinels[name] for name in sorted(self._sentinels))

    def rules_for(self, ctx: ArchContext) -> list[LeafRule]:
        """The rules enabled for ``ctx``, in (stage, name) order."""
        ordered = sorted(self._rules.values(), key=lambda rule: (rule.stage, rule.name))
        return [rule for rule in ordered if rule.enabled(ctx)]

    def sentinels_for(self, fmt: str) -> tuple[Sentinel, ...]:
        """The sentinels over source keys of format ``fmt``, by name."""
        return tuple(sentinel for sentinel in self.sentinels if sentinel.fmt == fmt)


REGISTRY = RuleRegistry()
"""The registry of the package's rules, sentinels, refusal detectors and allowlists."""
