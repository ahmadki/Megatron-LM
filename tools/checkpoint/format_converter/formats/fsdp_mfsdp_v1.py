# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The fsdp_dtensor format as Megatron-FSDP v1 writes it.

Megatron's ``fsdp_dtensor`` save runs ``torch.distributed.checkpoint.save`` on
the training state dict directly, so a store holds:

* model: ``model.module.<fqn>`` tensors, with exactly one ``module.`` level
  for every precision (the Megatron-FSDP wrapper's state dict passes its
  prefix through ``Float16Module``); trainable parameters hold the fp32 main
  weights, buffers and frozen parameters their own dtype; no ``_extra_state``;
* common: byte items ``args``, ``iteration`` and ``checkpoint_version`` (which
  the loader requires) and ``num_floating_point_operations_so_far``, each a
  ``torch.save`` of one value;
* optimizer: ``optimizer.state.<depth><fqn>.<moment>`` tensors and
  ``optimizer.param_to_group_meta...`` byte items;
* scheduler: ``opt_param_scheduler.<field>`` byte items;
* rng: ``rng_state...``; rerun: ``rerun_state_machine...``.

A published store has no ``metadata.json``: Megatron would classify a store
with one as torch_dist. ``Metadata.planner_data`` maps every key to its path in
the saved state dict, e.g. ``('model', 'module.<fqn>')`` and ``('args',)``; a
converted store writes the same paths.

A model key whose name still starts with ``module.`` after the prefix is
stripped (``model.module.module.<fqn>``) is refused: no Megatron-FSDP v1 save
writes it, and a stale converter did.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import torch

from .. import errors
from ..core.keys import EXTRA_STATE
from ..core.names import DIALECTS, FSDP_DTENSOR, FSDP_MODEL_PREFIX, FSDP_OPTIMIZER_STATE
from ..engine.metadata import MetadataView
from ..engine.reader import SourceReader
from ..engine.regions import Region
from ..errors import IntegrityError, InternalError, Refusal
from .base import CommonLayout, RawCommon, Section
from .detect import (
    FSDP_COMMON_ITEMS,
    FSDP_OPTIONAL_COMMON_ITEMS,
    LEGACY_COMMON_FILE,
    TORCH_DIST_CONFIG_FILE,
)

MODEL_SECTION = "model"
"""First element of the planner path of every model key."""

LEFTOVER_MODULE = "module."

COMMON_ITEMS = FSDP_COMMON_ITEMS + FSDP_OPTIONAL_COMMON_ITEMS
"""The byte items that hold the common state, in output order."""


def classify_key(key: str, *, is_bytes: bool) -> Section:
    """The section of a Megatron-FSDP v1 store key (``OTHER`` if unknown).

    Model state is stored as tensors only; a byte item under ``model.module.``
    is unknown.
    """
    if key.startswith("rng_state"):
        return Section.RNG
    if key.startswith("rerun_state_machine"):
        return Section.RERUN
    if EXTRA_STATE.search(key):
        return Section.EXTRA_STATE
    if key.startswith(FSDP_OPTIMIZER_STATE):
        return Section.OPTIMIZER_STATE
    if key.startswith("optimizer."):
        return Section.OPTIMIZER_OTHER
    if key.startswith("opt_param_scheduler."):
        return Section.SCHEDULER
    if key in COMMON_ITEMS:
        return Section.COMMON
    return Section.MODEL if key.startswith(FSDP_MODEL_PREFIX) and not is_bytes else Section.OTHER


class FsdpMfsdpV1Adapter:
    """The fsdp_dtensor format of Megatron-FSDP v1."""

    fmt: ClassVar[str] = FSDP_DTENSOR
    dialect: ClassVar[str] = DIALECTS[FSDP_DTENSOR]
    common_layout: ClassVar[CommonLayout] = CommonLayout.PER_KEY

    def classify(self, key: str, *, is_bytes: bool) -> Section:
        """The section of ``key`` (see :func:`classify_key`)."""
        return classify_key(key, is_bytes=is_bytes)

    def sections(self, view: MetadataView) -> dict[str, Section]:
        """The section of every key of the store.

        Raises:
            RefusedError: A key belongs to no known section (FC-R13).
        """
        sections = {key: classify_key(key, is_bytes=False) for key in view.tensors}
        sections.update({key: classify_key(key, is_bytes=True) for key in view.byte_items})
        unknown = sorted(key for key, section in sections.items() if section is Section.OTHER)
        if unknown:
            raise Refusal(
                errors.FC_R13,
                "the store holds items that are neither model tensors, common, optimizer, "
                "scheduler, rng nor rerun state",
                keys=tuple(unknown),
                hint="this is not a store written by Megatron-FSDP v1; check the source path",
            ).to_error()
        return sections

    def read_common_raw(self, reader: SourceReader) -> RawCommon | None:
        """The common-state byte items the store has (``None`` if it has none)."""
        keys = [key for key in COMMON_ITEMS if key in reader.view.byte_items]
        if not keys:
            return None
        return RawCommon(
            f"byte items {', '.join(keys)}", CommonLayout.PER_KEY, reader.read_bytes(keys)
        )

    def common_item(self, key: str) -> tuple[str, tuple[str, ...] | None]:
        """The byte item of common-state entry ``key``: the key itself, planner path ``(key,)``.

        Raises:
            InternalError: ``key`` is not a common-state item of the format.
        """
        if key not in COMMON_ITEMS:
            raise InternalError(f"{key} is not a common-state item of {self.fmt}")
        return key, (key,)

    def stored_key(self, relative: str) -> str:
        """The store key of the model-relative key ``relative``."""
        return FSDP_MODEL_PREFIX + relative

    def relative_key(self, key: str) -> str:
        """The model-relative key of the model key ``key``.

        Raises:
            InternalError: ``key`` does not start with ``model.module.``.
            RefusedError: The name after the prefix starts with another
                ``module.`` (FC-R13).
        """
        if not key.startswith(FSDP_MODEL_PREFIX):
            raise InternalError(f"{key} is not a model key of {self.fmt}")
        relative = key[len(FSDP_MODEL_PREFIX) :]
        if relative.startswith(LEFTOVER_MODULE):
            raise Refusal(
                errors.FC_R13,
                f"model keys start with {FSDP_MODEL_PREFIX}{LEFTOVER_MODULE}, which Megatron-FSDP "
                "v1 never writes",
                keys=(key,),
                hint="the store was written by an outdated converter; convert its source again",
            ).to_error()
        return relative

    def planner_path(self, key: str) -> tuple[str, ...]:
        """The planner path of the model key ``key``: ``('model', 'module.<fqn>')``."""
        return (MODEL_SECTION, key[len(MODEL_SECTION) + 1 :])

    def output_grid(
        self, shape: tuple[int, ...], dtype: torch.dtype, *, max_chunk_bytes: int
    ) -> tuple[Region, ...]:
        """One chunk per tensor, split along dimension 0 into chunks of at most ``max_chunk_bytes``.

        A single row larger than ``max_chunk_bytes`` is a chunk of its own.
        Megatron-FSDP reads any chunk grid.
        """
        full = Region.full(shape)
        if not shape or shape[0] == 0:
            return (full,)
        row_bytes = full.numel() // shape[0] * dtype.itemsize
        rows = max(1, max_chunk_bytes // max(row_bytes, 1))
        return tuple(full.split_dim0(rows))

    def marker_files(self) -> dict[str, bytes]:
        """No marker files: a ``metadata.json`` would make Megatron read the store as torch_dist."""
        return {}

    def validate_published(self, iter_dir: Path) -> None:
        """Check the store has neither ``metadata.json`` nor ``common.pt``.

        Raises:
            IntegrityError: One of them exists.
        """
        for name in (TORCH_DIST_CONFIG_FILE, LEGACY_COMMON_FILE):
            if (iter_dir / name).exists():
                raise IntegrityError(f"{iter_dir} holds {name}, which fsdp_dtensor stores never do")
