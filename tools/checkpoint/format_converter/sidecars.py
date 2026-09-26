# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Files next to a store's data files, and which of them a converted store carries.

An iteration directory holds the DCP store itself (``.metadata``, the data files
it references, ``metadata.json`` and a legacy ``common.pt``) and possibly:

* ``run_config.yaml`` and ``tokenizer/``: format-neutral, copied;
* ``integrity.json``: Megatron's integrity manifest (SHA-256 of every file,
  checked on load with ``--verify-integrity``); the data files change, so it is
  written anew for the converted store (:func:`write_integrity_manifest`);
* ``train_dataloader_dprank*.pt``: per-rank data-loader state, tied to the
  source job's data-parallel layout; dropped with a warning;
* ``modelopt_state``: refused (FC-R18), a weights-only rewrite would lose it;
* anything else: listed in the report, not copied.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import errors
from .engine.fs import write_atomic
from .engine.metadata import METADATA_FILE, MetadataView
from .errors import Refusal
from .formats.detect import LEGACY_COMMON_FILE, TORCH_DIST_CONFIG_FILE

COPIED_SIDECARS = ("run_config.yaml", "tokenizer")
DROPPED_SIDECARS = ("train_dataloader_dprank*.pt",)
MODELOPT_STATE = "modelopt_state"
INTEGRITY_MANIFEST = "integrity.json"
"""Megatron's integrity manifest (``dist_checkpointing.validation.INTEGRITY_FNAME``)."""

_HASH_BLOCK_BYTES = 8 * 2**20

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SidecarPlan:
    """What happens to the files beside a source store.

    Attributes:
        copy: Names copied into the converted store.
        dropped: Names dropped (with a warning).
        unknown: Names neither copied nor understood.
        regenerated: Names written anew for the converted store (the integrity
            manifest).
    """

    copy: tuple[str, ...]
    dropped: tuple[str, ...]
    unknown: tuple[str, ...]
    regenerated: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {
            "copy": list(self.copy),
            "dropped": list(self.dropped),
            "unknown": list(self.unknown),
            "regenerated": list(self.regenerated),
        }


def plan_sidecars(view: MetadataView) -> SidecarPlan:
    """Classify every entry of the source iteration directory that is not part of the store.

    Raises:
        RefusedError: The directory holds a ModelOpt state (FC-R18).
    """
    store_files = {METADATA_FILE, TORCH_DIST_CONFIG_FILE, LEGACY_COMMON_FILE}
    store_files.update(c.relative_path for e in view.tensors.values() for c in e.chunks)
    store_files.update(item.relative_path for item in view.byte_items.values())
    copy, dropped, unknown, regenerated = [], [], [], []
    for name in sorted(entry.name for entry in view.iter_dir.iterdir()):
        if name in store_files:
            continue
        if name == MODELOPT_STATE:
            raise Refusal(
                errors.FC_R18,
                f"{view.iter_dir} holds a ModelOpt state, which a weights-only rewrite would lose",
                keys=(MODELOPT_STATE,),
                hint="convert the checkpoint before applying ModelOpt",
            ).to_error()
        if name == INTEGRITY_MANIFEST:
            regenerated.append(name)
        elif name in COPIED_SIDECARS:
            copy.append(name)
        elif any(fnmatch.fnmatch(name, pattern) for pattern in DROPPED_SIDECARS):
            dropped.append(name)
        else:
            unknown.append(name)
    if dropped:
        _LOGGER.warning("dropping data-loader state tied to the source job: %s", ", ".join(dropped))
    if unknown:
        _LOGGER.warning("not copying unknown files: %s", ", ".join(unknown))
    return SidecarPlan(tuple(copy), tuple(dropped), tuple(unknown), tuple(regenerated))


def copy_sidecars(plan: SidecarPlan, source_dir: Path, target_dir: Path) -> None:
    """Copy the sidecars ``plan`` keeps from ``source_dir`` into ``target_dir``."""
    for name in plan.copy:
        source = source_dir / name
        if source.is_dir():
            shutil.copytree(source, target_dir / name, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target_dir / name)


def write_integrity_manifest(store_dir: Path) -> None:
    """Write Megatron's ``integrity.json`` for ``store_dir``; call it after every other file.

    The manifest maps the name of every top-level file (except itself and
    temporary files left by an interrupted atomic write) to the SHA-256 of its
    contents, ``{"algorithm": "sha256", "files": {...}}``, as Megatron's
    ``save_integrity_manifest`` writes it. Files are hashed in bounded blocks.
    """
    files = {}
    for path in sorted(store_dir.iterdir(), key=str):
        temporary = path.name.startswith(".") and path.name.endswith(".tmp")
        if path.is_file() and path.name != INTEGRITY_MANIFEST and not temporary:
            digest = hashlib.sha256()
            with open(path, "rb") as stream:
                for block in iter(lambda stream=stream: stream.read(_HASH_BLOCK_BYTES), b""):
                    digest.update(block)
            files[path.name] = digest.hexdigest()
    manifest = {"algorithm": "sha256", "files": files}
    write_atomic(store_dir / INTEGRITY_MANIFEST, json.dumps(manifest, indent=2).encode())
