# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Module-level span group state — importable anywhere without circular deps.

This module is intentionally minimal.  It holds only the :class:`frozenset`
of enabled span groups so that any Megatron submodule (e.g.
:mod:`megatron.core.pipeline_parallel.schedules`) can call
:func:`is_span_group_enabled` without importing the full telemetry package
and without creating circular import chains.

Span groups are registered once at telemetry initialisation time via
:func:`set_enabled_span_groups` (called from
:func:`~megatron.core.telemetry.handle.setup_telemetry`).  Before that call
every :func:`is_span_group_enabled` query returns ``False``, so all
instrumentation sites are safely silent during startup.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
# Note: frozenset assignment is atomic under CPython's GIL, but we use a lock
# for correctness under free-threaded Python (PEP 703, Python 3.13+).
_ENABLED_GROUPS: frozenset = frozenset()


def set_enabled_span_groups(groups: frozenset) -> None:
    """Register the active span groups.

    Called once from :func:`~megatron.core.telemetry.handle.setup_telemetry`
    after the :class:`~megatron.core.telemetry.config.TelemetryConfig` has
    been resolved.  Subsequent calls override the previous value (useful for
    testing).

    Args:
        groups: A :class:`frozenset` of group-name strings.  See
            :class:`~megatron.core.telemetry.config.SpanGroup` for the
            canonical constants.
    """
    global _ENABLED_GROUPS
    with _LOCK:
        _ENABLED_GROUPS = groups


def is_span_group_enabled(group: str) -> bool:
    """Return ``True`` if the named span group is currently enabled.

    This is the primary check used at every instrumentation site.  It is a
    module-level function (not a method on any class) so that callers in
    ``megatron/core/`` can import it without pulling in the full telemetry
    package.

    Returns ``False`` before :func:`set_enabled_span_groups` is first called
    (i.e. during startup, before telemetry is initialised) — no spans are
    emitted until the user explicitly enables them.

    Args:
        group: A span group name string (one of the constants defined on
            :class:`~megatron.core.telemetry.config.SpanGroup`).

    Returns:
        bool: ``True`` if the group is enabled, ``False`` otherwise.
    """
    return group in _ENABLED_GROUPS
