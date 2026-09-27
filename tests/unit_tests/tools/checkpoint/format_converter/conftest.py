# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Local pytest configuration for the format_converter tests.

It imports no megatron module and never creates a process group, so the tests
run unchanged on the host and in the multi-rank ``torch.distributed.run`` CI
bucket.
"""

from __future__ import annotations

import pytest

from . import _harness


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers of this package (no pyproject changes)."""
    config.addinivalue_line(
        "markers",
        "covers(rule, evidence): the test is evidence for a conversion rule; evidence is "
        "ground_truth (hand-built expectations), parity (pinned to Megatron) or real_save "
        "(Megatron's own saves and loads); test_registry_contract.py requires both kinds",
    )


@pytest.fixture(name="scrubbed_env")
def fixture_scrubbed_env() -> dict[str, str]:
    """Environment for child processes: no launcher variables and no visible GPU."""
    return _harness.scrubbed_env()


@pytest.fixture(name="local_rank_zero")
def fixture_local_rank_zero() -> None:
    """Skip the test on every local rank but 0 (for subprocess-heavy tests)."""
    if not _harness.is_local_rank_zero():
        pytest.skip("runs on local rank 0 only")
