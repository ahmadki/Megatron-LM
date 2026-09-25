# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Command line interface: ``python -m tools.checkpoint.format_converter <command>``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from . import __version__
from .errors import ConverterError, ExitCode

PROG = "python -m tools.checkpoint.format_converter"

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
    parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    return parser


def _configure_logging(level: str) -> None:
    logging.getLogger(_PACKAGE_LOGGER).setLevel(level)
    if not logging.getLogger().handlers:
        logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")


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
