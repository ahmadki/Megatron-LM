# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reports of ``plan`` and ``convert``: JSON records and their text rendering."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engine.fs import write_atomic
from .engine.memory import children_peak_rss_bytes, process_peak_rss_bytes
from .engine.runner import RunResult
from .engine.sizes import format_bytes


def journal_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Totals over the units' journal records, plus a compact record per unit."""
    return {
        "units": len(records),
        "bytes_read": sum(r["bytes_read"] for r in records),
        "bytes_written": sum(r["bytes_written"] for r in records),
        "items_touched": sum(r["items_touched"] for r in records),
        "torch_load_calls": sum(r["torch_load_calls"] for r in records),
        "max_unit_vmhwm_bytes": max((r["vmhwm_bytes"] or 0 for r in records), default=0),
        "cuda_initialized": any(r["cuda_initialized"] for r in records),
        "per_unit": [
            {
                "uid": r["uid"],
                "worker": r["worker"],
                "seconds": round(r["t1"] - r["t0"], 3),
                "bytes_read": r["bytes_read"],
                "bytes_written": r["bytes_written"],
                "vmhwm_bytes": r["vmhwm_bytes"],
            }
            for r in sorted(records, key=lambda r: r["uid"])
        ],
    }


def peak_rss() -> dict[str, int]:
    """Peak resident set sizes: this process (VmHWM) and its largest child (RUSAGE_CHILDREN)."""
    return {
        "driver_bytes": process_peak_rss_bytes() or 0,
        "children_bytes": children_peak_rss_bytes(),
    }


@dataclass(frozen=True)
class ConvertReport:
    """Outcome of ``convert``.

    Attributes:
        plan: The plan summary (``ConversionPlan.to_json()``).
        run: What the run did.
        wall_seconds: Wall time of the whole conversion.
        peak_rss: Peak resident set sizes (see :func:`peak_rss`).
    """

    plan: dict[str, Any]
    run: RunResult
    wall_seconds: float
    peak_rss: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable form."""
        target = self.run.target
        tracker = self.run.tracker
        return {
            **self.plan,
            "published": str(target.iter_dir),
            "tracker": None if tracker is None else tracker.to_json(),
            "resumed": self.run.resumed,
            "units_run": self.run.units_run,
            "run": journal_summary(self.run.journal),
            "wall_seconds": round(self.wall_seconds, 3),
            "peak_rss": self.peak_rss,
        }

    def render(self) -> str:
        """Human-readable text."""
        data = self.to_json()
        run = data["run"]
        amplification = (
            run["torch_load_calls"] / run["items_touched"] if run["items_touched"] else 1.0
        )
        lines = render_plan(self.plan).splitlines()
        lines += [
            f"published:  {data['published']}{_render_tracker(data['tracker'])}",
            f"run:        {data['units_run']} of {run['units']} units run by this invocation"
            f"{' (resumed)' if data['resumed'] else ''}; read "
            f"{format_bytes(run['bytes_read'])} in {run['items_touched']} stored items "
            f"(amplification {amplification:.2f}), wrote {format_bytes(run['bytes_written'])}, "
            f"{data['wall_seconds']:.1f} s",
            f"peak rss:   driver {format_bytes(data['peak_rss']['driver_bytes'])}, largest worker "
            f"{format_bytes(data['peak_rss']['children_bytes'])}",
        ]
        return "\n".join(lines) + "\n"


def _render_tracker(tracker: Mapping[str, Any] | None) -> str:
    if tracker is None:
        return ""
    if tracker["after"] == tracker["before"]:
        return (
            f" (tracker {tracker['path']} left at {tracker['before']}: it only moves forward, "
            "edit it to resume from this iteration)"
        )
    return f" (tracker {tracker['path']}: {tracker['before'] or '-'} -> {tracker['after']})"


def render_plan(plan: Mapping[str, Any]) -> str:
    """Human-readable text of a plan summary."""
    source, target, common = plan["source"], plan["target"], plan["common"]
    units, memory = plan["units"], plan["memory"]
    lines = [
        f"source:     {source['iter_dir']} ({source['format']}, {source['dialect']}), "
        f"iteration {source['iteration'] if source['iteration'] is not None else 'unknown'}",
        f"target:     {target['format']} ({target['dialect']}), model weights only",
        f"sections:   {'section':<16} {'action':<8} {'keys':>8}  {'tensor bytes':>12}",
    ]
    for section in plan["sections"]:
        size = format_bytes(section["tensor_bytes"]) if section["tensor_bytes"] else "-"
        lines.append(
            f"            {section['section']:<16} {section['action']:<8} "
            f"{section['keys']:>8}  {size:>12}"
        )
    codec = f"{common['codec']} codec"
    if common["codec"] == "stub":
        codec += "; convert needs the Megatron container"
    lines.append(
        f"common:     from {common['source']} ({codec}); keep "
        f"{', '.join(common['kept']) or '-'}; drop {', '.join(common['dropped']) or '-'}"
    )
    lines.extend(
        f"  args.{name}: {old!r} -> {new!r}" for name, (old, new) in common["args_changes"].items()
    )
    sidecars = plan["sidecars"]
    lines.append(
        f"sidecars:   copy {', '.join(sidecars['copy']) or '-'}; drop "
        f"{', '.join(sidecars['dropped']) or '-'}; unknown {', '.join(sidecars['unknown']) or '-'}"
        + (f"; rewrite {', '.join(sidecars['regenerated'])}" if sidecars["regenerated"] else "")
    )
    lines += [
        f"units:      {units['count']} units, {units['pieces']} pieces, {units['byte_items']} byte "
        f"items; largest unit {format_bytes(units['largest_bytes'])} (unit budget "
        f"{format_bytes(units['unit_bytes'])}, hard cap {format_bytes(units['hard_cap_bytes'])})",
        f"memory:     about {format_bytes(memory['total'])} = {max(memory['workers'], 1)} x "
        f"{format_bytes(memory['worker_peak'])} + driver {format_bytes(memory['driver'])} "
        f"(budget {format_bytes(memory['budget'])})",
        f"disk:       about {format_bytes(plan['output_bytes'])} written (convert needs 1.02 x "
        "that free on the destination)",
    ]
    return "\n".join(lines) + "\n"


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Write a JSON report atomically."""
    write_atomic(path, (json.dumps(report, indent=2) + "\n").encode())
