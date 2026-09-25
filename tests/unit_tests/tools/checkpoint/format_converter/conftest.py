# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Local pytest configuration for the format_converter tests.

It imports no megatron module and never creates a process group, so the tests
run unchanged on the host and in the multi-rank ``torch.distributed.run`` CI
bucket.
"""

from __future__ import annotations

import pytest

from . import _harness


@pytest.fixture(name="scrubbed_env")
def fixture_scrubbed_env() -> dict[str, str]:
    """Environment for child processes: no launcher variables and no visible GPU."""
    return _harness.scrubbed_env()


@pytest.fixture(name="local_rank_zero")
def fixture_local_rank_zero() -> None:
    """Skip the test on every local rank but 0 (for subprocess-heavy tests)."""
    if not _harness.is_local_rank_zero():
        pytest.skip("runs on local rank 0 only")
