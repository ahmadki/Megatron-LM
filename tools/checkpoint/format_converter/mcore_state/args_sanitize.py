# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Adjusting a checkpoint's training arguments to the state the output store holds.

Megatron's loader reads some saved arguments to decide what to load: for
example it loads optimizer state unless the checkpoint's ``no_save_optim`` is
set, and RNG state unless ``no_save_rng`` is. A store without that state must
say so. Only attributes the arguments already have are changed, and every
change is reported as ``{field: (old, new)}``.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..core.args_view import ArgsView

WEIGHTS_ONLY_FIELDS: Mapping[str, Any] = {"no_save_optim": True, "no_save_rng": True}
"""Arguments of a store holding model weights only (no optimizer or RNG state)."""


@dataclass(frozen=True)
class SanitizedArgs:
    """Arguments after sanitation.

    Attributes:
        args: A copy of the arguments with the changes applied.
        changes: ``{field: (old, new)}`` for every field whose value changed.
        consulted: Every field that was read, with its original value.
    """

    args: argparse.Namespace
    changes: dict[str, tuple[Any, Any]]
    consulted: dict[str, Any]


def sanitize_weights_only(args: argparse.Namespace) -> SanitizedArgs:
    """Arguments of a weights-only torch_dist store: ``no_save_optim`` and ``no_save_rng`` set.

    Raises:
        IntegrityError: ``args`` is not a Namespace.
        InternalError: A field to change holds a non-primitive value.
    """
    view = ArgsView(args)
    changes = {
        name: (view.get(name), value)
        for name, value in WEIGHTS_ONLY_FIELDS.items()
        if view.has(name) and view.get(name) != value
    }
    sanitized = copy.copy(args)
    for name, (_, value) in changes.items():
        setattr(sanitized, name, value)
    return SanitizedArgs(sanitized, changes, view.consumed)
