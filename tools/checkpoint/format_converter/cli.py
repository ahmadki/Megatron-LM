# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Command line interface: ``python -m tools.checkpoint.format_converter <command>``.

Heavy modules (torch) are imported by the command handlers, so ``--help`` and
usage errors stay fast. ``plan`` and ``convert`` hide the GPUs from their own
process before torch is imported, so nothing they load can initialize CUDA.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
OPTIMIZER_CHOICES = ("drop", "convert")
COMMON_CHOICES = ("skip", "bytes", "semantic")
MAP_CHOICES = ("identity", "convert")
EXPECTED_DIFF_CHOICES = ("A", "B", "C", "D", "E")
RESUME_CHOICES = ("auto", "never")
TORCH_DIST_LAYOUTS = ("native", "per-layer")
EXPERT_NAMINGS = ("auto", "grouped", "sequential")
TARGET_SWIGLU = ("auto", "on", "off")
FSDP_GPT_MTP_INNER_NAMES = ("transformer_layer", "mtp_model_layer")
MAX_PRINTED_FINDINGS = 20

_PACKAGE_LOGGER = __name__.rpartition(".")[0]
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
_EPILOG = (
    "exit codes: 0 ok, 1 internal error, 2 usage error, 3 refused (FC-R/FC-P/FC-E), "
    "4 ambiguous (FC-A), 5 verification mismatch, 6 integrity or I/O error, "
    "7 staging fingerprint mismatch, 8 memory or disk budget exceeded, 130 interrupted"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with every subcommand registered."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Inspect, verify and convert Megatron DCP checkpoint stores.",
        epilog=_EPILOG,
    )
    parser.add_argument(
        "--log-level", choices=_LOG_LEVELS, default="WARNING", help="logging verbosity"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    _add_inspect(subparsers)
    _add_verify(subparsers)
    _add_plan(subparsers)
    _add_convert(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    args = build_parser().parse_args(argv)
    _configure_logging(args.log_level)
    try:
        return int(args.handler(args))
    except ConverterError as err:
        for line in err.format().splitlines():
            sys.stderr.write(f"error: {line}\n")
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


def _block_layout(text: str) -> tuple[str, str]:
    prefix, equals, layout = text.rpartition("=")
    if not equals:
        raise argparse.ArgumentTypeError(f"expected PREFIX=LAYOUT, got {text!r}")
    return prefix, layout


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
        help="compare two stores bitwise, or check one against SHA-256 digests",
        description="Compare store A with store B key by key (key sets, shapes, dtypes and "
        "values bit for bit, reading each stored chunk once), or check A against SHA-256 "
        "digest files with --oracle. Exits 5 on any difference.",
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
        help="bytes: byte items (common state) must be byte-identical (default); semantic: "
        "decode both common states and compare their fields (args field by field); "
        "skip: ignore byte items",
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
        "--expected-diff",
        choices=EXPECTED_DIFF_CHOICES,
        help="compare A (a converted store) with B under an expected-difference contract: "
        "differences it allows are reported as expected, anything else is a finding; it sets "
        "the common-state comparison, overriding --common (see the README)",
    )
    parser.add_argument(
        "--map",
        choices=MAP_CHOICES,
        default="identity",
        help="identity: compare key by key (default); convert: compare B with the replayed "
        "conversion of A (not independent: detects engine and I/O faults only)",
    )
    parser.add_argument(
        "--oracle",
        metavar="GLOB",
        help="verify A against these digest files (sorted; position = file index) instead of B",
    )
    parser.add_argument(
        "--oracle-key-map",
        type=Path,
        metavar="PY",
        help="Python file defining to_store_key(oracle_key, file_index) -> str | None",
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
        oracle=args.oracle,
        oracle_key_map=args.oracle_key_map,
        expected_diff=args.expected_diff,
        map=args.map,
        **_read_options(args),
    )
    report = verify_checkpoint(args.a, args.b, opts)
    if args.report is not None:
        report.write_jsonl(args.report)
    sys.stdout.write(render_verify(report))
    return int(ExitCode.OK if report.ok else ExitCode.VERIFY_MISMATCH)


def _add_conversion_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("src", type=Path, metavar="SRC", help="store directory or checkpoint root")
    parser.add_argument(
        "--to",
        dest="to_format",
        choices=FORMAT_CHOICES,
        required=True,
        help="target format: fsdp_dtensor (from torch_dist) or torch_dist (from either)",
    )
    parser.add_argument(
        "--from",
        dest="from_format",
        choices=FORMAT_CHOICES,
        help="format to assume when the store has no format markers",
    )
    parser.add_argument(
        "--iteration", type=int, metavar="N", help="iteration to open under a checkpoint root"
    )
    parser.add_argument(
        "--drop-common-key",
        action="append",
        default=[],
        metavar="KEY",
        help="drop this common-state key: an unknown one instead of refusing, or a kept one "
        "that is not required, such as content_metadata (repeatable)",
    )
    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZER_CHOICES,
        default="drop",
        help="drop: write model weights only (default); convert: convert optimizer and "
        "scheduler state too (fsdp_dtensor to torch_dist; torch_dist to fsdp_dtensor is refused "
        "with FC-P12)",
    )
    parser.add_argument(
        "--allow-bf16-masters",
        action="store_true",
        help="with --optimizer convert, take trained parameters stored narrower than fp32 in "
        "the fsdp_dtensor model section as their fp32 masters (upcast exactly; FC-R21 otherwise)",
    )
    parser.add_argument(
        "--allow-root",
        action="append",
        default=[],
        metavar="PREFIX",
        help="convert model keys under this top-level prefix unchanged although no rule knows "
        "them (repeatable; cross-format only)",
    )
    parser.add_argument(
        "--max-output-chunk-bytes",
        type=_size,
        metavar="SIZE",
        help="largest chunk of an fsdp_dtensor output tensor; larger tensors are split along "
        "dimension 0 (default: 1GiB)",
    )
    parser.add_argument(
        "--allow-missing-fp8-extra-state",
        action="store_true",
        help="write a torch_dist store for an FP8 delayed-scaling model without its scaling "
        "state, which fsdp_dtensor stores do not hold (the scaling history restarts)",
    )
    parser.add_argument(
        "--ignore-manifest",
        action="store_true",
        help="do not restore the arguments and content metadata the source's "
        "conversion_manifest.json records",
    )
    parser.add_argument(
        "--torch-dist-layout",
        choices=TORCH_DIST_LAYOUTS,
        default="native",
        help="fsdp_dtensor to torch_dist: native stacks each TransformerBlock as Megatron would "
        "(default); per-layer stores every TransformerBlock one layer per key and sets "
        "non_homogeneous_layers in content_metadata, so Megatron loads it per layer",
    )
    parser.add_argument(
        "--block-layout",
        type=_block_layout,
        action="append",
        default=[],
        metavar="PREFIX=LAYOUT",
        help="fsdp_dtensor to torch_dist: store the TransformerBlock layer list PREFIX (e.g. "
        "vision_model.decoder) stacked or per_layer instead of deciding it from the arguments "
        "(repeatable); content_metadata is unchanged, so the job that loads the store must "
        "build PREFIX that way",
    )
    parser.add_argument(
        "--expert-naming",
        choices=EXPERT_NAMINGS,
        default="auto",
        help="how fsdp_dtensor names routed experts: auto as the model's builder does (default; "
        "grouped for TEGroupedMLP, sequential for SequentialMLP), or as given (a hybrid model's "
        "target is always grouped); an fsdp_dtensor source must name them so",
    )
    parser.add_argument(
        "--target-swiglu",
        choices=TARGET_SWIGLU,
        default="auto",
        help="torch_dist to fsdp_dtensor: whether the job that will load the store sets --swiglu, "
        "which makes Megatron-FSDP split gated linear_fc1 into SwiGLU halves: auto as the "
        "source's arguments say (default), on or off; --swiglu also makes the MLPs SiLU-gated, "
        "so for a language model with MLPs on/off must match the source's arguments (another "
        "value is refused: that job builds another model); for one without MLPs they describe "
        "the job, and the store's arguments record it",
    )
    parser.add_argument(
        "--fsdp-gpt-mtp-inner-name",
        choices=FSDP_GPT_MTP_INNER_NAMES,
        default="transformer_layer",
        help="torch_dist to fsdp_dtensor: the name the store gives the inner layer of GPT "
        "multi-token prediction layers: transformer_layer, which current Megatron-FSDP saves "
        "and loads (default), or the live name mtp_model_layer, only for Megatron-FSDP versions "
        "older than handle_mtp_in_state_dict (an fsdp_dtensor source is read under the name it "
        "holds)",
    )
    parser.add_argument(
        "--fsdp-tp",
        type=int,
        metavar="N",
        help="torch_dist to fsdp_dtensor: the tensor-parallel size of the job that will load the "
        "store; fsdp_dtensor packs fused tensors (a gated linear_fc1 it does not split, the "
        "in_proj and conv1d of a Gated DeltaProduct or Mamba-2 mixer or of a Gated DeltaNet) per "
        "tensor-parallel rank, so the store only loads at this size, which its arguments record "
        "(default: the source's, if it is 1; required otherwise when such tensors exist)",
    )
    parser.add_argument(
        "--fsdp-etp",
        type=int,
        metavar="N",
        help="torch_dist to fsdp_dtensor: the expert tensor-parallel size of that job, which the "
        "fused tensors of routed experts are packed at (default and requirement as --fsdp-tp; "
        "also required when --fsdp-tp is not the source's tensor-parallel size, since a job's "
        "expert size follows its tensor-parallel size unless it sets "
        "--expert-tensor-parallel-size)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        metavar="N",
        help="worker processes; 0 runs the units in this process (default: min(4, cores/8))",
    )
    parser.add_argument(
        "--unit-bytes",
        type=_size,
        metavar="SIZE",
        help="target peak memory of one unit of work (default: 4GiB)",
    )
    parser.add_argument(
        "--max-unit-bytes-hard",
        type=_size,
        metavar="SIZE",
        help="largest memory any single unit may need; larger ones exit 8 "
        "(default: 0.6 x available memory / workers)",
    )
    parser.add_argument(
        "--writer-threads",
        type=int,
        metavar="N",
        help="writer threads (data files) per unit (default: 4)",
    )
    parser.add_argument(
        "--max-host-bytes",
        type=_size,
        metavar="SIZE",
        help="host-memory budget of the whole conversion; exit 8 if the estimate exceeds it "
        "(default: 0.8 x available memory)",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")


def _plan_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {
        "to_format": args.to_format,
        "from_format": args.from_format,
        "iteration": args.iteration,
        "drop_common_keys": frozenset(args.drop_common_key),
        "optimizer": args.optimizer,
        "allow_bf16_masters": args.allow_bf16_masters,
        "allow_roots": tuple(args.allow_root),
        "allow_missing_fp8_extra_state": args.allow_missing_fp8_extra_state,
        "ignore_manifest": args.ignore_manifest,
        "block_layouts": tuple(args.block_layout),
        "torch_dist_layout": args.torch_dist_layout,
        "expert_naming": args.expert_naming,
        "fsdp_tp": args.fsdp_tp,
        "fsdp_etp": args.fsdp_etp,
        "target_swiglu": args.target_swiglu,
        "fsdp_gpt_mtp_inner_name": args.fsdp_gpt_mtp_inner_name,
    }
    for name in (
        "workers",
        "unit_bytes",
        "max_unit_bytes_hard",
        "writer_threads",
        "max_host_bytes",
        "max_output_chunk_bytes",
    ):
        if getattr(args, name) is not None:
            options[name] = getattr(args, name)
    return options


def _add_plan(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "plan",
        help="dry-run a conversion: what is kept or dropped, units, memory and disk",
        description="Plan a conversion from the source's metadata and common state: "
        "refusals, the derived architecture and the rules that convert it (cross-format), "
        "sections kept, converted and dropped, the common state, units, and the memory and "
        "disk estimates. Exits 3 when convert would be refused and 8 when the conversion "
        "exceeds the memory budgets, without writing anything; the free disk space is "
        "checked by convert. On a host without Megatron, a common state that references "
        "Megatron classes is decoded with stubs (the stub codec): the plan is complete, but "
        "convert must run in the Megatron container.",
        epilog=_EPILOG,
    )
    _add_conversion_arguments(parser)
    parser.set_defaults(handler=_run_plan)


def _run_plan(args: argparse.Namespace) -> int:
    _hide_gpus()
    # pylint: disable=import-outside-toplevel
    from .api import plan_conversion
    from .options import PlanOptions
    from .report import render_plan

    plan = plan_conversion(args.src, PlanOptions(**_plan_options(args)))
    summary = plan.to_json()
    sys.stdout.write(json.dumps(summary, indent=2) + "\n" if args.json else render_plan(summary))
    return int(ExitCode.OK)


def _add_convert(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "convert",
        help="convert a store between torch_dist and fsdp_dtensor (model weights only)",
        description="Convert SRC into a new store under DST with bounded host memory and no "
        "GPU, keeping model weights only. torch_dist to fsdp_dtensor and back converts the "
        "model tensors through the rules and writes a conversion manifest; torch_dist to "
        "torch_dist rewrites the store with its chunk grid and extra state (a legacy "
        "common.pt is migrated). Optimizer, RNG and rerun state are dropped. "
        "DST is a checkpoint root (it gets iter_NNNNNNN and the tracker, written last and "
        "only moved forward) or, if named iter_NNNNNNN or release, the iteration directory "
        "itself. An interrupted run resumes when rerun with the same arguments.",
        epilog=_EPILOG,
    )
    _add_conversion_arguments(parser)
    parser.add_argument(
        "dst", type=Path, metavar="DST", help="checkpoint root or iteration directory"
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        metavar="DIR",
        help="directory in which the staging directory .fc-staging-NAME is created (only that "
        "subdirectory is ever deleted); must be on DST's file system (default: the target's "
        "parent)",
    )
    parser.add_argument(
        "--resume",
        choices=RESUME_CHOICES,
        default="auto",
        help="auto: continue staging of the same conversion (default); never: refuse if it exists",
    )
    parser.add_argument(
        "--restart", action="store_true", help="discard existing staging and start over"
    )
    parser.add_argument(
        "--keep-staging", action="store_true", help="keep the staging directory afterwards"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing target iteration directory (only that directory, and only "
        "if it is a single store)",
    )
    parser.add_argument(
        "--dst-is-iter-dir",
        action="store_true",
        help="treat DST as the iteration directory itself (no tracker)",
    )
    parser.add_argument(
        "--no-tracker",
        dest="tracker",
        action="store_false",
        help="do not write latest_checkpointed_iteration.txt in DST",
    )
    parser.add_argument(
        "--no-copy-sidecars",
        dest="copy_sidecars",
        action="store_false",
        help="do not copy run_config.yaml and tokenizer assets",
    )
    parser.add_argument(
        "--report", type=Path, metavar="PATH.json", help="also write the report as JSON"
    )
    parser.set_defaults(handler=_run_convert)


def _run_convert(args: argparse.Namespace) -> int:
    _hide_gpus()
    # pylint: disable=import-outside-toplevel
    from .api import convert_checkpoint
    from .options import ConvertOptions
    from .report import write_report

    opts = ConvertOptions(
        **_plan_options(args),
        staging_dir=args.staging_dir,
        resume=args.resume,
        restart=args.restart,
        keep_staging=args.keep_staging,
        overwrite=args.overwrite,
        dst_is_iter_dir=args.dst_is_iter_dir,
        tracker=args.tracker,
        copy_sidecars=args.copy_sidecars,
    )
    report = convert_checkpoint(args.src, args.dst, opts)
    if args.report is not None:
        write_report(args.report, report.to_json())
    sys.stdout.write(
        json.dumps(report.to_json(), indent=2) + "\n" if args.json else report.render()
    )
    return int(ExitCode.OK)


def _hide_gpus() -> None:
    """Hide the GPUs from this process, if torch is not imported yet (a fresh CLI process)."""
    if "torch" not in sys.modules:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


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
        f"(mode {report.mode}), {report.byte_items_compared} byte items, "
        f"{report.common_fields_compared} common-state fields",
        f"read:       {format_bytes(stats.bytes_read)} in {stats.items_touched} stored items "
        f"(amplification {stats.amplification:.2f})",
        f"findings:   {len(report.findings)}",
    ]
    for finding in report.findings[:MAX_PRINTED_FINDINGS]:
        lines.append(f"  {finding.kind.value:<10} {finding.key}: {finding.detail}")
    hidden = len(report.findings) - MAX_PRINTED_FINDINGS
    if hidden > 0:
        lines.append(f"  ... {hidden} more (use --report PATH.jsonl for all)")
    if report.contract is not None:
        lines.append(
            f"expected:   {len(report.expected)} differences allowed by contract {report.contract}"
        )
    if not report.independent:
        lines.append("note:       not an independent check (detects engine and I/O faults only)")
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
