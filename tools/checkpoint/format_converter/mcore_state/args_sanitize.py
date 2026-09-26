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
  removed from the hybrid layer pattern);
* torch_dist: ``ckpt_format='torch_dist'``, no ``use_megatron_fsdp``; values
  a previous conversion recorded in its manifest are restored, except that
  ``no_save_optim`` and ``no_save_rng`` are never cleared for a store without
  that state.

Only attributes the arguments already have are changed, and every change is
reported as ``{field: (old, new)}``.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..core.args_view import ArgsView
from ..core.hybrid_pattern import PIPE

WEIGHTS_ONLY_FIELDS: Mapping[str, Any] = {"no_save_optim": True, "no_save_rng": True}
"""Arguments of a store holding model weights only (no optimizer or RNG state)."""

FSDP_DTENSOR_FIELDS: Mapping[str, Any] = {"ckpt_format": "fsdp_dtensor", "use_megatron_fsdp": True}
TORCH_DIST_FIELDS: Mapping[str, Any] = {"ckpt_format": "torch_dist", "use_megatron_fsdp": False}

SINGLE_STAGE_FIELDS: Mapping[str, Any] = {
    "pipeline_model_parallel_size": 1,
    "transformer_pipeline_model_parallel_size": 1,
    "virtual_pipeline_model_parallel_size": None,
    "num_layers_per_virtual_pipeline_stage": None,
    "num_virtual_stages_per_pipeline_rank": None,
    "pipeline_model_parallel_layout": None,
    "decoder_first_pipeline_num_layers": None,
    "decoder_last_pipeline_num_layers": None,
}
"""The pipeline layout of a single-stage job (what an fsdp_dtensor store describes)."""

PATTERN_FIELDS = ("hybrid_layer_pattern", "hybrid_override_pattern")
"""Hybrid layer patterns, whose ``|`` separators mark pipeline stages."""


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


def sanitize_for_fsdp_dtensor(args: argparse.Namespace) -> SanitizedArgs:
    """Arguments of a weights-only fsdp_dtensor store written from a torch_dist checkpoint.

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
    return _apply(
        args, {**WEIGHTS_ONLY_FIELDS, **FSDP_DTENSOR_FIELDS, **SINGLE_STAGE_FIELDS, **unpiped}
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
