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
from collections.abc import Sequence
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

FC_R03 = "FC-R03"
"""The model was built with a spec or implementation whose keys the rules do not know."""

FC_R06 = "FC-R06"
"""The model uses grouped tensor parallelism (weight shards or rematerialization > 1)."""

FC_R07 = "FC-R07"
"""A torch_dist target would lose FP8 delayed-scaling state that its loader requires."""

FC_R11 = "FC-R11"
"""The layer indices a store holds contradict the hybrid layer pattern."""

FC_R13 = "FC-R13"
"""The store layout is unknown or not a DCP store this tool can read."""

FC_R14 = "FC-R14"
"""Model tensors are stored in the legacy N-D reformulated layout (``mcore_data``)."""

FC_R17 = "FC-R17"
"""The checkpoint was saved with ``singleton_local_shards``."""

FC_R18 = "FC-R18"
"""The checkpoint holds a ModelOpt state (``modelopt_state``)."""

FC_R19 = "FC-R19"
"""The ``fsdp_dtensor`` store holds split fused-projection sub-keys (unknown layout)."""

FC_R20 = "FC-R20"
"""The model uses a layer type no rule is verified for (hybrid symbol or attention variant)."""

FC_P01 = "FC-P01"
"""Pending: non-hybrid ``TransformerBlock`` layers (stacked keys or a stacking decision)."""

FC_P02 = "FC-P02"
"""Pending: several model roots (``language_model.``, ``vision_model.``, ``vision_projection.``)."""

FC_P03 = "FC-P03"
"""Pending: mixture-of-experts layers."""

FC_P04 = "FC-P04"
"""Pending: shortcut-MoE layer numbering or hybrid multi-token prediction."""

FC_P05 = "FC-P05"
"""Pending: Gated DeltaProduct mixers."""

FC_P06 = "FC-P06"
"""Pending: gated ``linear_fc1`` that fsdp_dtensor keeps fused (SwiGLU split off)."""

FC_P07 = "FC-P07"
"""Pending: ``linear_fc1`` split into SwiGLU halves (``args.swiglu`` set)."""

FC_P08 = "FC-P08"
"""Pending: Mamba-2 mixers."""

FC_P09 = "FC-P09"
"""Pending: Gated DeltaNet and linear-attention layers."""

FC_P10 = "FC-P10"
"""Pending: GPT multi-token prediction or multi-latent attention."""

FC_P11 = "FC-P11"
"""Pending: converting optimizer state from fsdp_dtensor to torch_dist."""

FC_P12 = "FC-P12"
"""Pending: converting optimizer state from torch_dist to fsdp_dtensor."""

FC_A05 = "FC-A05"
"""``--from`` names a different format than the one detected from the store."""

FC_A06 = "FC-A06"
"""The source holds a conversion manifest that describes another store."""

FC_E01 = "FC-E01"
"""The common state needs Megatron's classes, which this process cannot import."""

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
        return refusal_error((self,))


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
    """Base class of errors that carry one or more :class:`Refusal` objects.

    Attributes:
        refusal: The first refusal.
        refusals: Every refusal, in report order.
    """

    def __init__(self, refusal: Refusal, *others: Refusal) -> None:
        super().__init__(refusal.message, hint=refusal.hint)
        self.refusal = refusal
        self.refusals = (refusal, *others)

    def format(self) -> str:
        """Render the carried refusals, one ``FC-xxx: ...`` line each."""
        return "\n".join(refusal.format() for refusal in self.refusals)

    @property
    def codes(self) -> tuple[str, ...]:
        """The codes of the carried refusals, in report order."""
        return tuple(refusal.code for refusal in self.refusals)


class RefusedError(RefusalError):
    """The input is refused (``FC-R``, ``FC-P`` or ``FC-E`` codes), exit 3."""

    exit_code = ExitCode.REFUSED


class AmbiguousError(RefusalError):
    """The input is ambiguous and needs an explicit flag (``FC-A`` codes), exit 4."""

    exit_code = ExitCode.AMBIGUOUS


_CATEGORY_ORDER = {"R": 0, "E": 1, "A": 2, "P": 3}


def refusal_error(refusals: Sequence[Refusal]) -> RefusalError:
    """One error carrying every refusal, permanent ones first (then by code).

    The error exits 4 (:class:`AmbiguousError`) when every refusal only needs a
    flag (``FC-A``), else 3 (:class:`RefusedError`).

    Raises:
        InternalError: ``refusals`` is empty.
    """
    if not refusals:
        raise InternalError("refusal_error() needs at least one refusal")
    ordered = sorted(refusals, key=lambda r: (_CATEGORY_ORDER[r.category], r.code))
    if all(refusal.category == "A" for refusal in ordered):
        return AmbiguousError(*ordered)
    return RefusedError(*ordered)
