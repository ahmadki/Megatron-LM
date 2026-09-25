# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Test-only crash injection at the engine's commit points.

Set ``_FC_TEST_FAULT`` to one of the points below and the process that reaches
it exits immediately with :data:`FAULT_EXIT_CODE`, as if it had been killed:

* ``unit:K``: a worker wrote unit K's part store but has not committed it;
* ``merge:link:K``: the merge placed K data files in the final store;
* ``publish:before_rename``: publishing moved any existing target to the trash
  but has not renamed the merged store into place;
* ``publish:before_tracker``: the store is in place, the tracker is not written.

Tests use it to check that a resumed conversion completes and produces the same
bytes as an uninterrupted one. The variable is only read; nothing else changes.
"""

from __future__ import annotations

import os

FAULT_ENV = "_FC_TEST_FAULT"
FAULT_EXIT_CODE = 99


def inject_fault(point: str) -> None:
    """Exit the process at once if ``_FC_TEST_FAULT`` names ``point``."""
    if os.environ.get(FAULT_ENV) == point:
        os._exit(FAULT_EXIT_CODE)  # pylint: disable=protected-access
