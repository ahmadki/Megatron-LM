# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Semantic comparison of two stores' common state (``verify --common semantic``).

Both common states are decoded for display, with every class outside a small
allowlist replaced by an inert stub, so this runs on any host and never executes
checkpoint code. The training ``args`` are compared field by field; every other
top-level entry is compared as a whole. Values are compared structurally: stubs
by class name and pickled state, tensors by dtype, shape and bytes, NaN equal to
NaN. The common state may live in different places on the two sides (the in-DCP
item or a legacy ``common.pt`` of torch_dist, or the top-level byte items of
fsdp_dtensor), so a migrated or cross-format store compares equal.

Findings are named ``args.<field>`` or ``<key>``; ``--ignore-key`` patterns
apply to these names, for example to accept the arguments a conversion
sanitizes.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Mapping
from typing import Any

import torch

from ..engine.metadata import Stub
from ..engine.reader import SourceReader
from ..formats.common_summary import read_common_state
from ..formats.detect import Detection
from .report import Finding, FindingKind

_MAX_VALUE_TEXT = 80

ONLY_IN_A, ONLY_IN_B = "only in A", "only in B"
"""Details of a finding for a field only one side holds."""


def compare_common(
    a: Detection,
    b: Detection,
    reader_a: SourceReader,
    reader_b: SourceReader,
    ignores: Callable[[str], bool],
) -> tuple[list[Finding], int]:
    """Compare the common states of two detected stores.

    Returns:
        The findings and the number of fields compared.
    """
    a_common = read_common_state(a, reader_a)
    b_common = read_common_state(b, reader_b)
    return semantic_findings(
        None if a_common is None else a_common.state,
        None if b_common is None else b_common.state,
        ignores,
    )


def semantic_findings(
    a: Mapping[str, Any] | None, b: Mapping[str, Any] | None, ignores: Callable[[str], bool]
) -> tuple[list[Finding], int]:
    """Differences between two common dicts (``None``: the store has no common state).

    Returns:
        The findings and the number of fields compared.
    """
    a_fields, b_fields = _fields(a or {}), _fields(b or {})
    findings = []
    compared = 0
    for name in sorted(a_fields.keys() | b_fields.keys()):
        if ignores(name):
            continue
        compared += 1
        if name not in b_fields:
            findings.append(Finding(FindingKind.COMMON, name, ONLY_IN_A))
        elif name not in a_fields:
            findings.append(Finding(FindingKind.COMMON, name, ONLY_IN_B))
        elif canonical(a_fields[name]) != canonical(b_fields[name]):
            findings.append(
                Finding(
                    FindingKind.COMMON,
                    name,
                    f"A {_text(a_fields[name])} vs B {_text(b_fields[name])}",
                )
            )
    return findings, compared


def canonical(value: Any) -> Any:
    """A structural, comparable form of a decoded value."""
    if isinstance(value, Stub):
        return (
            "stub",
            value.qualified_name,
            canonical(value.args),
            canonical(dict(value.kwargs)),
            canonical(value.state),
            canonical(value.items),
            canonical(value.entries),
        )
    if isinstance(value, argparse.Namespace):
        return ("namespace", canonical(vars(value)))
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().contiguous().reshape(-1)
        return (
            "tensor",
            str(value.dtype),
            tuple(value.shape),
            flat.view(torch.uint8).numpy().tobytes(),
        )
    if isinstance(value, Mapping):
        return ("dict", {key: canonical(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(canonical(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return ("set", frozenset(value))
    if isinstance(value, float) and math.isnan(value):
        return ("nan",)
    return value


def _fields(common: Mapping[str, Any]) -> dict[str, Any]:
    """Top-level entries by name, with the args expanded into ``args.<field>``."""
    fields = {str(key): value for key, value in common.items() if key != "args"}
    args = common.get("args")
    if isinstance(args, argparse.Namespace):
        fields.update({f"args.{name}": value for name, value in vars(args).items()})
    elif "args" in common:
        fields["args"] = args
    return fields


def _text(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= _MAX_VALUE_TEXT else text[: _MAX_VALUE_TEXT - 3] + "..."
