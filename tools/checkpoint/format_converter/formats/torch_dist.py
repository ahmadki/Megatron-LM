# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The torch_dist (Megatron Core) format adapter and its weights-only rewrite rules.

Sections of a torch_dist store, and what a weights-only rewrite does with them:

* model: tensors with a bare ``ShardedTensor`` key; copied chunk for chunk.
* extra_state: ``<module>._extra_state`` (``_extra_state<i>`` when grouped), a
  tensor or a ``/shard_*`` byte item; copied, byte items verbatim.
* common: the byte item ``common_state/shard_0_1``, or a legacy ``common.pt``
  file (which wins, as in Megatron); decoded, filtered and re-encoded in DCP.
* optimizer.state: ``[chained_<i>.]optimizer.state.*``; dropped.
* optimizer other: any other ``[chained_<i>.]optimizer.*`` key; dropped.
* rng (``rng_state*``) and rerun (``rerun_state_machine_state*``); dropped.
* other: any other byte item; refused (FC-R13).

Model keys stored in the legacy N-D reformulated layout (listed in the
metadata's ``mcore_data``) are refused (FC-R14).

The common dict keeps ``args``, ``iteration``, ``checkpoint_version``,
``num_floating_point_operations_so_far`` and ``content_metadata``, and drops
``optimizer``, ``opt_param_scheduler`` and ``rerun_state_machine``. A
``modelopt_state`` (FC-R18), a ``singleton_local_shards`` layout (FC-R17) and
any other key (FC-R13, unless dropped explicitly) are refused.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch

from .. import errors
from ..core.keys import EXTRA_STATE
from ..core.names import DIALECTS, TORCH_DIST
from ..engine.metadata import MetadataView
from ..engine.reader import SourceReader
from ..engine.regions import Region
from ..errors import IntegrityError, InternalError, Refusal, UsageError
from .base import WHOLE, CommonLayout, RawCommon, Section
from .detect import LEGACY_COMMON_FILE, TORCH_DIST_COMMON_ITEM, TORCH_DIST_CONFIG_FILE

TORCH_DIST_CONFIG = {"sharded_backend": TORCH_DIST, "sharded_backend_version": 1}
"""Contents of ``metadata.json`` (Megatron's ``CheckpointingConfig``)."""

KEPT_COMMON_KEYS = (
    "args",
    "iteration",
    "checkpoint_version",
    "num_floating_point_operations_so_far",
    "content_metadata",
)
"""Common-state keys a weights-only rewrite keeps."""

DROPPED_COMMON_KEYS = ("optimizer", "opt_param_scheduler", "rerun_state_machine")
"""Common-state keys a weights-only rewrite drops (optimizer and training-loop state)."""

REQUIRED_COMMON_KEYS = ("args", "iteration", "checkpoint_version")
"""Common-state keys that ``--drop-common-key`` may not drop (the loader reads them)."""

KEPT_SECTIONS = frozenset({Section.MODEL, Section.EXTRA_STATE})
"""Sections a weights-only rewrite copies (the others but the common state are dropped)."""

_OPTIMIZER_STATE = re.compile(r"^(chained_\d+\.)?optimizer\.state\.")
_OPTIMIZER = re.compile(r"^(chained_\d+\.)?optimizer\.")


def classify_key(key: str, *, is_bytes: bool) -> Section:
    """The section of a torch_dist key (``OTHER`` for an unknown byte item)."""
    if key.startswith("rng_state"):
        return Section.RNG
    if key.startswith("rerun_state_machine_state"):
        return Section.RERUN
    if EXTRA_STATE.search(key):
        return Section.EXTRA_STATE
    if _OPTIMIZER_STATE.match(key):
        return Section.OPTIMIZER_STATE
    if _OPTIMIZER.match(key):
        return Section.OPTIMIZER_OTHER
    if is_bytes:
        return Section.COMMON if key == TORCH_DIST_COMMON_ITEM else Section.OTHER
    return Section.MODEL


class TorchDistAdapter:
    """The torch_dist format (Megatron Core ``dist_checkpointing`` with the torch backend)."""

    fmt: ClassVar[str] = TORCH_DIST
    dialect: ClassVar[str] = DIALECTS[TORCH_DIST]
    common_layout: ClassVar[CommonLayout] = CommonLayout.WRAPPED

    def classify(self, key: str, *, is_bytes: bool) -> Section:
        """The section of ``key`` (see :func:`classify_key`)."""
        return classify_key(key, is_bytes=is_bytes)

    def sections(self, view: MetadataView) -> dict[str, Section]:
        """The section of every key (see :func:`weights_only_sections`)."""
        return weights_only_sections(view)

    def stored_key(self, relative: str) -> str:
        """torch_dist model keys are bare: the model-relative key itself."""
        return relative

    def relative_key(self, key: str) -> str:
        """torch_dist model keys are bare: the key itself."""
        return key

    def planner_path(self, key: str) -> None:
        """torch_dist stores record no planner paths."""
        del key

    def output_grid(
        self, shape: tuple[int, ...], dtype: torch.dtype, *, max_chunk_bytes: int
    ) -> tuple[Region, ...]:
        """One chunk per converted tensor; Megatron's loader reads any chunk grid.

        Pieces larger than half the unit budget are split along dimension 0 by
        the scheduler; ``max_chunk_bytes`` only applies to fsdp_dtensor.
        """
        del dtype, max_chunk_bytes
        return (Region.full(shape),)

    def read_common_raw(self, reader: SourceReader) -> RawCommon | None:
        """The common state: ``common.pt`` if the store has one (as Megatron reads it), else
        the ``common_state/shard_0_1`` byte item; ``None`` if neither exists.

        Raises:
            IntegrityError: ``common.pt`` cannot be read.
        """
        legacy = reader.view.iter_dir / LEGACY_COMMON_FILE
        if legacy.is_file():
            try:
                return RawCommon(
                    LEGACY_COMMON_FILE, CommonLayout.DICT, {WHOLE: legacy.read_bytes()}
                )
            except OSError as err:
                raise IntegrityError(f"cannot read {legacy}: {err.strerror or err}") from err
        if TORCH_DIST_COMMON_ITEM not in reader.view.byte_items:
            return None
        payload = reader.read_bytes([TORCH_DIST_COMMON_ITEM])[TORCH_DIST_COMMON_ITEM]
        return RawCommon(TORCH_DIST_COMMON_ITEM, CommonLayout.WRAPPED, {WHOLE: payload})

    def common_item(self, key: str) -> tuple[str, tuple[str, ...] | None]:
        """``common_state/shard_0_1``, without a planner path (``common.pt`` is never written).

        Raises:
            InternalError: ``key`` is not the single payload of the wrapped layout.
        """
        if key != WHOLE:
            raise InternalError(f"torch_dist stores its common state in one payload, not {key!r}")
        return TORCH_DIST_COMMON_ITEM, None

    def marker_files(self) -> dict[str, bytes]:
        """``metadata.json`` declaring the torch_dist backend."""
        return {TORCH_DIST_CONFIG_FILE: json.dumps(TORCH_DIST_CONFIG).encode()}

    def validate_published(self, iter_dir: Path) -> None:
        """Check ``metadata.json`` declares torch_dist and no ``common.pt`` exists.

        Raises:
            IntegrityError: A marker is missing or wrong.
        """
        config = iter_dir / TORCH_DIST_CONFIG_FILE
        try:
            declared = json.loads(config.read_text())
        except (OSError, ValueError) as err:
            raise IntegrityError(f"cannot read {config}: {err}") from err
        if declared != TORCH_DIST_CONFIG:
            raise IntegrityError(f"{config} holds {declared}, expected {TORCH_DIST_CONFIG}")
        if (iter_dir / LEGACY_COMMON_FILE).exists():
            raise IntegrityError(f"{iter_dir} holds a {LEGACY_COMMON_FILE}, which is never written")


@dataclass(frozen=True)
class CommonRewrite:
    """The common state of a weights-only rewrite.

    Attributes:
        kept: The keys kept, with their values.
        dropped: The keys dropped (sorted).
    """

    kept: dict[str, Any]
    dropped: tuple[str, ...]


def weights_only_common(
    common: Mapping[str, Any], *, drop_keys: frozenset[str] = frozenset()
) -> CommonRewrite:
    """Filter a torch_dist common dict for a weights-only rewrite.

    Args:
        common: The decoded common dict.
        drop_keys: Keys to drop: unknown ones instead of refusing them, and
            kept ones (``content_metadata``, ...) that are not required.

    Raises:
        UsageError: ``drop_keys`` names a key the loader requires.
        RefusedError: ``modelopt_state`` (FC-R18), a ``singleton_local_shards``
            layout (FC-R17) or an unknown key not in ``drop_keys`` (FC-R13).
    """
    required = sorted(drop_keys & set(REQUIRED_COMMON_KEYS))
    if required:
        raise UsageError(f"--drop-common-key cannot drop {', '.join(required)}")
    if "modelopt_state" in common:
        raise Refusal(
            errors.FC_R18,
            "the checkpoint holds a ModelOpt state, which a weights-only rewrite would lose",
            keys=("modelopt_state",),
            hint="convert the checkpoint before applying ModelOpt",
        ).to_error()
    content = common.get("content_metadata")
    if isinstance(content, Mapping) and content.get("singleton_local_shards"):
        raise Refusal(
            errors.FC_R17,
            "the checkpoint was saved with singleton_local_shards, which this tool does not read",
            keys=("content_metadata",),
            hint="load it with current Megatron and save it again without that option",
        ).to_error()
    known = set(KEPT_COMMON_KEYS) | set(DROPPED_COMMON_KEYS)
    unknown = sorted(set(common) - known - drop_keys)
    if unknown:
        raise Refusal(
            errors.FC_R13,
            "the common state holds keys this tool does not know",
            keys=tuple(unknown),
            hint="pass --drop-common-key KEY for each key that may be dropped",
        ).to_error()
    kept = {
        key: value
        for key, value in common.items()
        if key in KEPT_COMMON_KEYS and key not in drop_keys
    }
    return CommonRewrite(kept, tuple(sorted(set(common) - set(kept))))


def weights_only_sections(view: MetadataView) -> dict[str, Section]:
    """The section of every key of a torch_dist store, checked for a weights-only rewrite.

    Raises:
        RefusedError: A byte item is unknown (FC-R13), or a model tensor is
            stored in the legacy N-D reformulated layout (FC-R14).
    """
    sections = {key: classify_key(key, is_bytes=False) for key in view.tensors}
    sections.update({key: classify_key(key, is_bytes=True) for key in view.byte_items})
    unknown = sorted(key for key, section in sections.items() if section is Section.OTHER)
    if unknown:
        raise Refusal(
            errors.FC_R13,
            "the store holds byte items that are neither common state, extra state, "
            "optimizer, rng nor rerun state",
            keys=tuple(unknown),
            hint="this is not a store written by Megatron Core; check the source path",
        ).to_error()
    reformulated = sorted(
        key for key in (view.mcore_data or {}) if sections.get(key) in KEPT_SECTIONS
    )
    if reformulated:
        raise Refusal(
            errors.FC_R14,
            "model tensors are stored in the legacy N-D reformulated layout (mcore_data)",
            keys=tuple(reformulated),
            hint="load the checkpoint with current Megatron and save it again",
        ).to_error()
    return sections
