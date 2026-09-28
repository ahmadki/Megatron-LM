# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Adjusting a checkpoint's training arguments to the store they are written into.

Megatron's loader reads some saved arguments to decide what to load: for
example it loads optimizer state unless the checkpoint's ``no_save_optim`` is
set, and RNG state unless ``no_save_rng`` is; ``--use-checkpoint-args`` also
takes the format and the parallel layout from them. A converted store's
arguments must therefore describe the store:

* weights only: ``no_save_optim`` and ``no_save_rng`` are set;
* fsdp_dtensor: ``ckpt_format='fsdp_dtensor'`` and ``use_megatron_fsdp``; one
  pipeline stage (Megatron-FSDP stores have no pipeline stages: the pipeline
  and virtual-pipeline fields are reset and ``|`` stage separators are
  removed from the hybrid layer pattern); ``moe_grouped_gemm`` set to build
  the experts module whose names the store uses (a GPT model's builder
  reads it);
* torch_dist: ``ckpt_format='torch_dist'``, no ``use_megatron_fsdp``; values
  a previous conversion recorded in its manifest are restored, except that
  ``no_save_optim`` and ``no_save_rng`` are never cleared for a store without
  that state.

Only attributes the arguments already have are changed, and every change is
reported as ``{field: (old, new)}``. The field values are in ``core/format_args.py``.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..core.args_view import ArgsView
from ..core.format_args import (
    EXPERT_FORM_FIELD,
    FSDP_DTENSOR_FIELDS,
    PATTERN_FIELDS,
    SINGLE_STAGE_FIELDS,
    TORCH_DIST_FIELDS,
    WEIGHTS_ONLY_FIELDS,
)
from ..core.hybrid_pattern import PIPE


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
    return _apply(args, WEIGHTS_ONLY_FIELDS)


def sanitize_for_fsdp_dtensor(
    args: argparse.Namespace, *, grouped_gemm: bool | None = None
) -> SanitizedArgs:
    """Arguments of a weights-only fsdp_dtensor store written from a torch_dist checkpoint.

    Args:
        args: The torch_dist checkpoint's arguments.
        grouped_gemm: The ``moe_grouped_gemm`` under which Megatron builds the
            experts module whose names the store uses (``None``: left as is).

    Raises:
        IntegrityError: ``args`` is not a Namespace.
        InternalError: A field to change holds a non-primitive value.
    """
    view = ArgsView(args)
    unpiped = {
        name: view.get(name).replace(PIPE, "")
        for name in PATTERN_FIELDS
        if isinstance(view.get(name, None), str)
    }
    experts = {} if grouped_gemm is None else {EXPERT_FORM_FIELD: grouped_gemm}
    return _apply(
        args,
        {**WEIGHTS_ONLY_FIELDS, **FSDP_DTENSOR_FIELDS, **SINGLE_STAGE_FIELDS, **unpiped, **experts},
    )


def sanitize_for_torch_dist(
    args: argparse.Namespace, restore: Mapping[str, Any] | None = None
) -> SanitizedArgs:
    """Arguments of a weights-only torch_dist store written from an fsdp_dtensor checkpoint.

    Args:
        args: The fsdp_dtensor store's arguments.
        restore: Original values recorded by the conversion that produced the
            fsdp_dtensor store (its manifest); ``no_save_optim`` and
            ``no_save_rng`` are not restored.

    Raises:
        IntegrityError: ``args`` is not a Namespace.
        InternalError: A field to change holds a non-primitive value.
    """
    restored = {k: v for k, v in (restore or {}).items() if k not in WEIGHTS_ONLY_FIELDS}
    return _apply(args, {**TORCH_DIST_FIELDS, **restored, **WEIGHTS_ONLY_FIELDS})


def _apply(args: argparse.Namespace, targets: Mapping[str, Any]) -> SanitizedArgs:
    """A copy of ``args`` with the existing fields of ``targets`` set to their values."""
    view = ArgsView(args)
    changes = {
        name: (view.get(name), value)
        for name, value in targets.items()
        if view.has(name) and view.get(name) != value
    }
    sanitized = copy.copy(args)
    for name, (_, value) in changes.items():
        setattr(sanitized, name, value)
    return SanitizedArgs(sanitized, changes, view.consumed)
