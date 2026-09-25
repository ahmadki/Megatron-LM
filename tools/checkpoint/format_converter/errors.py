# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Error types, stable refusal codes and process exit codes.

Every failure the tool can diagnose is a :class:`ConverterError` subclass whose
``exit_code`` is the process exit status of the CLI. Refusals carry a stable
``FC-xxx`` code so that scripts and documentation can refer to them:

* ``FC-Rnn``: permanent refusal (the input is not supported), exit 3.
* ``FC-Pnn``: pending family (not supported yet), exit 3.
* ``FC-Ann``: ambiguous input that needs an explicit flag, exit 4.
* ``FC-Enn``: wrong environment, exit 3.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import ClassVar


class ExitCode(enum.IntEnum):
    """Process exit codes of the ``format_converter`` CLI."""

    OK = 0
    INTERNAL = 1
    USAGE = 2
    REFUSED = 3
    AMBIGUOUS = 4
    VERIFY_MISMATCH = 5
    INTEGRITY = 6
    FINGERPRINT_MISMATCH = 7
    BUDGET_EXCEEDED = 8
    INTERRUPTED = 130


FC_R01 = "FC-R01"
"""The store is a Megatron-FSDP v2 ``fsdp_dtensor`` store (``model.<fqn>`` keys)."""

FC_R02 = "FC-R02"
"""The store has virtual-pipeline sections (``model<i>.`` or ``optimizer.<i>.`` keys)."""

FC_R13 = "FC-R13"
"""The store layout is unknown or not a DCP store this tool can read."""

FC_R19 = "FC-R19"
"""The ``fsdp_dtensor`` store holds split fused-projection sub-keys (unknown layout)."""

FC_A05 = "FC-A05"
"""``--from`` names a different format than the one detected from the store."""

FC_E02 = "FC-E02"
"""The staging directory is not on the destination's filesystem (hard links fail)."""

MAX_LISTED_KEYS = 8
"""Number of offending keys printed in a refusal line before it is abbreviated."""

_CODE_PATTERN = re.compile(r"^FC-(?P<category>[RPAE])\d{2}$")


@dataclass(frozen=True, slots=True)
class Refusal:
    """A diagnosed, documented reason for refusing an input.

    Attributes:
        code: Stable code such as ``FC-R01``.
        message: One-sentence description of what was found.
        keys: Offending checkpoint keys, if any.
        hint: What the user can do about it (a flag or a documentation anchor).
    """

    code: str
    message: str
    keys: tuple[str, ...] = ()
    hint: str | None = None

    def __post_init__(self) -> None:
        if _CODE_PATTERN.match(self.code) is None:
            raise ValueError(f"invalid refusal code {self.code!r}; expected FC-[RPAE]nn")

    @property
    def category(self) -> str:
        """The code category letter: ``R``, ``P``, ``A`` or ``E``."""
        match = _CODE_PATTERN.match(self.code)
        assert match is not None  # validated in __post_init__
        return match.group("category")

    def format(self) -> str:
        """Render the refusal as ``FC-xxx: <message> [keys...] hint: <hint>``."""
        parts = [f"{self.code}: {self.message}"]
        if self.keys:
            listed = ", ".join(self.keys[:MAX_LISTED_KEYS])
            extra = len(self.keys) - MAX_LISTED_KEYS
            if extra > 0:
                listed += f", ... (+{extra} more)"
            parts.append(f"[{listed}]")
        if self.hint:
            parts.append(f"hint: {self.hint}")
        return " ".join(parts)

    def to_error(self) -> RefusalError:
        """Build the exception that raises this refusal with the right exit code."""
        if self.category == "A":
            return AmbiguousError(self)
        return RefusedError(self)


class ConverterError(Exception):
    """Base class of every diagnosed failure; ``exit_code`` is the CLI exit status."""

    exit_code: ClassVar[ExitCode] = ExitCode.INTERNAL

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def format(self) -> str:
        """Render the error as one line, with its hint if there is one."""
        if self.hint:
            return f"{self.message} hint: {self.hint}"
        return self.message

    def __str__(self) -> str:
        return self.format()


class InternalError(ConverterError):
    """An invariant of the tool itself was violated (a bug), exit 1."""

    exit_code = ExitCode.INTERNAL


class UsageError(ConverterError):
    """The command line or API arguments are invalid, exit 2."""

    exit_code = ExitCode.USAGE


class IntegrityError(ConverterError):
    """The store is corrupt, incomplete or unreadable, exit 6."""

    exit_code = ExitCode.INTEGRITY


class FingerprintMismatch(ConverterError):
    """Existing staging belongs to a different conversion, exit 7."""

    exit_code = ExitCode.FINGERPRINT_MISMATCH


class BudgetExceeded(ConverterError):
    """A single read unit does not fit the memory budget, exit 8."""

    exit_code = ExitCode.BUDGET_EXCEEDED


class RefusalError(ConverterError):
    """Base class of errors that carry a :class:`Refusal`."""

    def __init__(self, refusal: Refusal) -> None:
        super().__init__(refusal.message, hint=refusal.hint)
        self.refusal = refusal

    def format(self) -> str:
        """Render the carried refusal (``FC-xxx: ...``)."""
        return self.refusal.format()


class RefusedError(RefusalError):
    """The input is refused (``FC-R``, ``FC-P`` or ``FC-E`` codes), exit 3."""

    exit_code = ExitCode.REFUSED


class AmbiguousError(RefusalError):
    """The input is ambiguous and needs an explicit flag (``FC-A`` codes), exit 4."""

    exit_code = ExitCode.AMBIGUOUS
