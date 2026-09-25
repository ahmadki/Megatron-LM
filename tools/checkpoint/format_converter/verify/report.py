# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Findings and reports of ``verify`` (text summary and JSONL)."""

from __future__ import annotations

import enum
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..engine.reader import ReadStats


class FindingKind(enum.Enum):
    """What differs (declaration order is the report order)."""

    ONLY_IN_A = "only_in_a"
    ONLY_IN_B = "only_in_b"
    SHAPE = "shape"
    DTYPE = "dtype"
    LAYOUT = "layout"
    VALUE = "value"
    BYTES = "bytes"


_KIND_ORDER = {kind: index for index, kind in enumerate(FindingKind)}


@dataclass(frozen=True, slots=True)
class Finding:
    """One difference between the two sides.

    Attributes:
        kind: What differs.
        key: The key concerned.
        detail: Human-readable specifics.
    """

    kind: FindingKind
    key: str
    detail: str

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {"kind": self.kind.value, "key": self.key, "detail": self.detail}


def sort_findings(findings: list[Finding]) -> tuple[Finding, ...]:
    """Findings in report order: by kind, then key."""
    return tuple(sorted(findings, key=lambda f: (_KIND_ORDER[f.kind], f.key, f.detail)))


@dataclass(frozen=True)
class VerifyReport:
    """Outcome of a ``verify`` run.

    Attributes:
        a: The store that was verified.
        reference: What it was compared with.
        mode: ``meta``, ``full`` or ``sample:N``.
        findings: Every difference found, in report order.
        keys_compared: Tensors whose shape and dtype were compared.
        values_compared: Tensors whose values were compared.
        byte_items_compared: Byte items whose payloads were compared.
        read_stats: What reading both sides cost.
    """

    a: Path
    reference: str
    mode: str
    findings: tuple[Finding, ...]
    keys_compared: int
    values_compared: int
    byte_items_compared: int
    read_stats: ReadStats

    @property
    def ok(self) -> bool:
        """True when nothing differs."""
        return not self.findings

    def summary(self) -> dict[str, Any]:
        """The JSON summary record (last line of the JSONL report)."""
        return {
            "kind": "summary",
            "ok": self.ok,
            "a": str(self.a),
            "reference": self.reference,
            "mode": self.mode,
            "findings": len(self.findings),
            "keys_compared": self.keys_compared,
            "values_compared": self.values_compared,
            "byte_items_compared": self.byte_items_compared,
            "bytes_read": self.read_stats.bytes_read,
            "items_read": self.read_stats.items_touched,
            "read_amplification": self.read_stats.amplification,
        }

    def write_jsonl(self, path: Path) -> None:
        """Write one JSON line per finding, then the summary, atomically."""
        lines = [json.dumps(f.to_json()) for f in self.findings]
        lines.append(json.dumps(self.summary()))
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text("\n".join(lines) + "\n")
        os.replace(tmp, path)
