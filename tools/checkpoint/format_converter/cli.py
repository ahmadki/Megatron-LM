# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Command line interface: ``python -m tools.checkpoint.format_converter <command>``.

Heavy modules (torch) are imported by the command handlers, so ``--help`` and
usage errors stay fast.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import __version__
from .errors import ConverterError, ExitCode

if TYPE_CHECKING:
    from .api import InspectReport
    from .verify.report import VerifyReport

PROG = "python -m tools.checkpoint.format_converter"
FORMAT_CHOICES = ("torch_dist", "fsdp_dtensor")
COMMON_CHOICES = ("skip", "bytes")
MAX_PRINTED_FINDINGS = 20

_PACKAGE_LOGGER = __name__.rpartition(".")[0]
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
_EPILOG = (
    "exit codes: 0 ok, 1 internal error, 2 usage error, 3 refused (FC-R/FC-P/FC-E), "
    "4 ambiguous (FC-A), 5 verification mismatch, 6 integrity or I/O error, "
    "7 staging fingerprint mismatch, 8 memory budget exceeded, 130 interrupted"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with every subcommand registered."""
    parser = argparse.ArgumentParser(
        prog=PROG, description="Inspect and verify Megatron DCP checkpoint stores.", epilog=_EPILOG
    )
    parser.add_argument(
        "--log-level", choices=_LOG_LEVELS, default="WARNING", help="logging verbosity"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    _add_inspect(subparsers)
    _add_verify(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    args = build_parser().parse_args(argv)
    _configure_logging(args.log_level)
    try:
        return int(args.handler(args))
    except ConverterError as err:
        sys.stderr.write(f"error: {err.format()}\n")
        return int(err.exit_code)
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return int(ExitCode.INTERRUPTED)


def _configure_logging(level: str) -> None:
    logging.getLogger(_PACKAGE_LOGGER).setLevel(level)
    if not logging.getLogger().handlers:
        logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")


def _regex(text: str) -> re.Pattern[str]:
    try:
        return re.compile(text)
    except re.error as err:
        raise argparse.ArgumentTypeError(f"invalid regular expression {text!r}: {err}") from err


def _size(text: str) -> int:
    from .engine.sizes import parse_size  # pylint: disable=import-outside-toplevel

    try:
        return parse_size(text)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from err


def _mode(text: str) -> Any:
    from .options import CompareMode  # pylint: disable=import-outside-toplevel

    try:
        return CompareMode.parse(text)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from err


def _add_store_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--iteration", type=int, metavar="N", help="iteration to open under a checkpoint root"
    )
    parser.add_argument(
        "--max-item-bytes",
        type=_size,
        metavar="SIZE",
        help="largest single read unit held in memory, e.g. 8GiB "
        "(default: a quarter of the available memory); larger units exit 8",
    )
    parser.add_argument(
        "--workers", type=int, default=4, metavar="N", help="reader threads (default: 4)"
    )


def _read_options(args: argparse.Namespace) -> dict[str, Any]:
    options = {"iteration": args.iteration, "workers": args.workers}
    if args.max_item_bytes is not None:
        options["max_item_bytes"] = args.max_item_bytes
    return options


def _add_inspect(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "inspect",
        help="summarize a checkpoint store",
        description="Summarize a torch_dist or fsdp_dtensor store from its metadata: format, "
        "key sections, dtypes, chunk grids and the common state (host-safe; Megatron classes "
        "are shown as stubs).",
        epilog=_EPILOG,
    )
    parser.add_argument("src", type=Path, metavar="SRC", help="store directory or checkpoint root")
    parser.add_argument(
        "--from",
        dest="from_format",
        choices=FORMAT_CHOICES,
        help="format to assume when the store has no format markers",
    )
    parser.add_argument("--keys", type=_regex, metavar="RE", help="list keys matching RE")
    parser.add_argument(
        "--chunks", action="store_true", help="list keys with their stored chunk grids"
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument(
        "--digests",
        type=Path,
        metavar="OUT.json",
        help="write SHA-256 digests of the selected (default: all) tensors to OUT.json",
    )
    _add_store_arguments(parser)
    parser.set_defaults(handler=_run_inspect)


def _run_inspect(args: argparse.Namespace) -> int:
    # pylint: disable=import-outside-toplevel
    from .api import inspect_checkpoint
    from .engine.hashing import write_digest_file
    from .options import InspectOptions

    opts = InspectOptions(
        from_format=args.from_format,
        keys=args.keys,
        chunks=args.chunks,
        digests=args.digests is not None,
        **_read_options(args),
    )
    report = inspect_checkpoint(args.src, opts)
    if args.digests is not None and report.digests is not None:
        write_digest_file(args.digests, report.digests)
    if args.json:
        sys.stdout.write(json.dumps(report.to_json(), indent=2) + "\n")
    else:
        sys.stdout.write(render_inspect(report, digests_path=args.digests))
    return int(ExitCode.OK)


def _add_verify(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "verify",
        help="compare two stores bitwise",
        description="Compare store A with store B key by key: key sets, shapes, dtypes and "
        "values bit for bit, reading each stored chunk once. Exits 5 on any difference.",
        epilog=_EPILOG,
    )
    parser.add_argument("a", type=Path, metavar="A", help="store (or checkpoint root) to verify")
    parser.add_argument("b", type=Path, metavar="B", nargs="?", help="store to compare with")
    parser.add_argument(
        "--mode",
        type=_mode,
        default="full",
        metavar="MODE",
        help="meta (keys, shapes, dtypes), full (all values; default) or sample:N "
        "(values of N keys chosen with --seed)",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of sample:N (default: 0)")
    parser.add_argument("--layout", action="store_true", help="also require identical chunk grids")
    parser.add_argument(
        "--common",
        choices=COMMON_CHOICES,
        default="bytes",
        help="bytes: byte items (common state) must be byte-identical (default); skip: ignore them",
    )
    parser.add_argument(
        "--ignore-key",
        type=_regex,
        action="append",
        default=[],
        metavar="RE",
        help="ignore keys matching RE (repeatable)",
    )
    parser.add_argument(
        "--report", type=Path, metavar="PATH.jsonl", help="write every finding as JSON lines"
    )
    _add_store_arguments(parser)
    parser.set_defaults(handler=_run_verify)


def _run_verify(args: argparse.Namespace) -> int:
    # pylint: disable=import-outside-toplevel
    from .api import verify_checkpoint
    from .options import VerifyOptions

    opts = VerifyOptions(
        mode=args.mode,
        seed=args.seed,
        layout=args.layout,
        common=args.common,
        ignore_keys=tuple(args.ignore_key),
        **_read_options(args),
    )
    report = verify_checkpoint(args.a, args.b, opts)
    if args.report is not None:
        report.write_jsonl(args.report)
    sys.stdout.write(render_verify(report))
    return int(ExitCode.OK if report.ok else ExitCode.VERIFY_MISMATCH)


def render_verify(report: VerifyReport) -> str:
    """Human-readable text of a ``verify`` report."""
    # pylint: disable=import-outside-toplevel
    from .engine.sizes import format_bytes

    stats = report.read_stats
    lines = [
        f"verify:     {'OK' if report.ok else 'MISMATCH'}",
        f"a:          {report.a}",
        f"reference:  {report.reference}",
        f"compared:   {report.keys_compared} tensors, values of {report.values_compared} "
        f"(mode {report.mode}), {report.byte_items_compared} byte items",
        f"read:       {format_bytes(stats.bytes_read)} in {stats.items_touched} stored items "
        f"(amplification {stats.amplification:.2f})",
        f"findings:   {len(report.findings)}",
    ]
    for finding in report.findings[:MAX_PRINTED_FINDINGS]:
        lines.append(f"  {finding.kind.value:<10} {finding.key}: {finding.detail}")
    hidden = len(report.findings) - MAX_PRINTED_FINDINGS
    if hidden > 0:
        lines.append(f"  ... {hidden} more (use --report PATH.jsonl for all)")
    return "\n".join(lines) + "\n"


def render_inspect(report: InspectReport, *, digests_path: Path | None = None) -> str:
    """Human-readable text of an ``inspect`` report."""
    # pylint: disable=import-outside-toplevel
    from .engine.sizes import format_bytes

    detection, census = report.detection, report.census
    lines = [
        f"store:      {detection.iter_dir}",
        f"format:     {detection.fmt} ({detection.dialect})",
        f"iteration:  {detection.iteration if detection.iteration is not None else 'unknown'}",
        f"evidence:   {'; '.join(detection.evidence)}",
        f"metadata:   DCP version {report.dcp_version or 'unrecorded'}; "
        f"planner data {'yes' if report.has_planner_data else 'no'}; "
        f"legacy mcore_data {'yes' if report.has_legacy_mcore_data else 'no'}",
        f"items:      {census.tensors} tensors ({format_bytes(census.tensor_bytes)}), "
        f"{census.byte_items} byte items",
        f"sections:   {'section':<16} {'keys':>8}  {'tensor bytes':>12}",
    ]
    for section, totals in census.sections.items():
        size = format_bytes(totals.tensor_bytes) if totals.tensor_bytes else "-"
        lines.append(f"            {section.value:<16} {totals.keys:>8}  {size:>12}")
    lines.append("dtypes:     " + ", ".join(f"{d} {n}" for d, n in census.dtypes.items()))
    lines.append(
        "chunks:     "
        + ", ".join(
            f"{n} chunk{'s' if n != 1 else ''}: {c}" for n, c in census.chunk_counts.items()
        )
    )
    if census.largest_chunk is not None:
        chunk = census.largest_chunk
        lines.append(
            f"largest:    {chunk.key} chunk at {chunk.offsets}, "
            f"{format_bytes(chunk.nbytes)} in {chunk.relative_path}"
        )
    if report.metadata_stubs:
        lines.append(f"stubbed:    {', '.join(report.metadata_stubs)} (in .metadata)")
    lines.extend(_render_common(report))
    for listing in report.keys:
        if listing.shape is None:
            lines.append(f"  {listing.key}  bytes  {listing.nbytes} B  [{listing.section.value}]")
            continue
        lines.append(
            f"  {listing.key}  {listing.shape} {listing.dtype}  {format_bytes(listing.nbytes)}  "
            f"{listing.num_chunks} chunk(s)  [{listing.section.value}]"
        )
        for region in listing.chunks or ():
            lines.append(f"      offsets {region.offsets} sizes {region.sizes}")
    if report.digests is not None:
        target = f" to {digests_path}" if digests_path is not None else ""
        lines.append(f"digests:    {len(report.digests)} tensors{target}")
    return "\n".join(lines) + "\n"


def _render_common(report: InspectReport) -> list[str]:
    common = report.common
    if common is None:
        return ["common:     none"]
    lines = [
        f"common:     from {common.source}: iteration {common.iteration}, "
        f"checkpoint_version {common.checkpoint_version}; keys {', '.join(common.keys)}"
    ]
    lines.extend(f"  args.{name} = {value}" for name, value in common.args.items())
    if common.stubbed:
        lines.append(f"  stubbed classes: {', '.join(common.stubbed)}")
    return lines
