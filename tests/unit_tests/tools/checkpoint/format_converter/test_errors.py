# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Error hierarchy, exit codes and the printable refusal format."""

from __future__ import annotations

import pytest

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.errors import (
    AmbiguousError,
    BudgetExceeded,
    ExitCode,
    IntegrityError,
    InternalError,
    Refusal,
    RefusedError,
    UsageError,
)


def test_exit_code_values_are_stable():
    assert {code.name: int(code) for code in ExitCode} == {
        "OK": 0,
        "INTERNAL": 1,
        "USAGE": 2,
        "REFUSED": 3,
        "AMBIGUOUS": 4,
        "VERIFY_MISMATCH": 5,
        "INTEGRITY": 6,
        "FINGERPRINT_MISMATCH": 7,
        "BUDGET_EXCEEDED": 8,
        "INTERRUPTED": 130,
    }


@pytest.mark.parametrize(
    "error_type, code",
    [(InternalError, 1), (UsageError, 2), (IntegrityError, 6), (BudgetExceeded, 8)],
)
def test_error_exit_codes(error_type, code):
    err = error_type("boom", hint="do this")
    assert err.exit_code == code
    assert str(err) == "boom hint: do this"


def test_refusal_format_lists_keys_and_hint():
    refusal = Refusal(errors.FC_R13, "unknown layout", keys=("a", "b"), hint="use --from")
    assert refusal.format() == "FC-R13: unknown layout [a, b] hint: use --from"


def test_refusal_format_abbreviates_long_key_lists():
    keys = tuple(f"k{i}" for i in range(errors.MAX_LISTED_KEYS + 3))
    line = Refusal(errors.FC_R01, "v2 store", keys=keys).format()
    assert line.startswith("FC-R01: v2 store [k0, k1,")
    assert line.endswith(", ... (+3 more)]")


def test_refusal_categories_map_to_exit_codes():
    refused = Refusal(errors.FC_R02, "vpp").to_error()
    ambiguous = Refusal(errors.FC_A05, "disagree", hint="drop --from").to_error()
    assert isinstance(refused, RefusedError) and refused.exit_code == ExitCode.REFUSED
    assert isinstance(ambiguous, AmbiguousError) and ambiguous.exit_code == ExitCode.AMBIGUOUS
    assert str(ambiguous) == "FC-A05: disagree hint: drop --from"
    assert ambiguous.refusal.code == "FC-A05"


@pytest.mark.parametrize("bad_code", ["R01", "FC-X01", "FC-R1", "fc-r01", "FC-R001"])
def test_refusal_rejects_malformed_codes(bad_code):
    with pytest.raises(ValueError, match="invalid refusal code"):
        Refusal(bad_code, "message")


def test_cli_prints_every_refusal_on_its_own_line(monkeypatch, capsys):
    from tools.checkpoint.format_converter import cli  # pylint: disable=import-outside-toplevel

    def refuse(args):
        raise errors.refusal_error(
            [Refusal(errors.FC_P12, "optimizer", hint="later"), Refusal(errors.FC_R20, "symbol D")]
        )

    monkeypatch.setattr(cli, "_run_inspect", refuse)
    assert cli.main(["inspect", "somewhere"]) == ExitCode.REFUSED
    assert capsys.readouterr().err.splitlines() == [
        "error: FC-R20: symbol D",
        "error: FC-P12: optimizer hint: later",
    ]
