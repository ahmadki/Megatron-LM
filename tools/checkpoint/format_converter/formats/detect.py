# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Iteration-directory resolution and format/dialect detection.

A source path is either a store (a directory holding ``.metadata`` or
``metadata.json``) or a checkpoint root whose ``latest_checkpointed_iteration.txt``
names the store (``iter_%07d`` or ``release``). :func:`open_store` resolves it
and loads the store's metadata once; :func:`detect_format` classifies it:

======================================================================  ==========================
Evidence                                                                Result
======================================================================  ==========================
``metadata.json`` with ``sharded_backend == "torch_dist"``              torch_dist / mcore
``metadata.json`` with another backend                                  FC-R13
``model<i>.`` or ``optimizer.<i>.`` keys                                FC-R02 (virtual pipeline)
``model.<fqn>`` keys without ``module.``                                FC-R01 (Megatron-FSDP v2)
``model.module.*`` keys with split ``in_proj``/``conv1d`` sub-keys      FC-R19
``model.module.*`` keys and byte items args/iteration/checkpoint_version  fsdp_dtensor / mfsdp_v1
anything else (no format markers)                                       FC-R13 unless ``--from``
======================================================================  ==========================

``--from`` that disagrees with a conclusive detection is FC-A05. Legacy
``mp_rank_*`` checkpoints are not DCP stores and are always refused (FC-R13).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .. import errors
from ..core.names import DIALECTS, FORMATS, FSDP_DTENSOR, FSDP_MODEL_PREFIX, TORCH_DIST
from ..engine.metadata import METADATA_FILE, MetadataView
from ..engine.staging import PublishTarget
from ..errors import IntegrityError, Refusal, UsageError

TRACKER_FILE = "latest_checkpointed_iteration.txt"
TORCH_DIST_CONFIG_FILE = "metadata.json"
LEGACY_COMMON_FILE = "common.pt"

FSDP_COMMON_ITEMS = ("args", "iteration", "checkpoint_version")
FSDP_OPTIONAL_COMMON_ITEMS = ("num_floating_point_operations_so_far",)
TORCH_DIST_COMMON_ITEM = "common_state/shard_0_1"

_ITERATION_DIR = re.compile(r"^iter_(?P<iteration>\d+)$")
_TRACKER_VALUE = re.compile(r"^\d+$")
_VPP_KEY = re.compile(r"^(model\d+|optimizer\.\d+)\.")
_SPLIT_FUSED_KEY = re.compile(r"(in_proj|conv1d)\.(weight|bias)\.[A-Za-z]\w*$")
_LOGGER = logging.getLogger(__name__)

Iteration = int | str | None
"""An iteration number, ``"release"``, or ``None`` when the directory name does not say."""


@dataclass(frozen=True, eq=False)
class Store:
    """A resolved store with its metadata loaded.

    Attributes:
        iter_dir: The store directory.
        root: The checkpoint root it was resolved from, if any.
        iteration: The iteration it holds, if known.
        view: Its metadata view (``.metadata`` unpickled once).
    """

    iter_dir: Path
    root: Path | None
    iteration: Iteration
    view: MetadataView


@dataclass(frozen=True, slots=True)
class Detection:
    """The detected format of a store and why.

    Attributes:
        fmt: ``torch_dist`` or ``fsdp_dtensor``.
        dialect: ``mcore`` (torch_dist) or ``mfsdp_v1`` (fsdp_dtensor).
        iter_dir: The store directory.
        root: The checkpoint root it was resolved from, if any.
        iteration: The iteration it holds, if known.
        evidence: Human-readable facts the decision was based on.
    """

    fmt: str
    dialect: str
    iter_dir: Path
    root: Path | None
    iteration: Iteration
    evidence: tuple[str, ...]


def open_store(src: Path, *, iteration: int | None = None) -> Store:
    """Resolve ``src`` to a store directory and load its metadata.

    Args:
        src: A store directory or a checkpoint root.
        iteration: Iteration to open under a root (overrides its tracker file).

    Raises:
        UsageError: ``src`` is neither a store nor a resolvable root.
        RefusedError: ``src`` is a legacy ``mp_rank_*`` checkpoint (FC-R13).
        IntegrityError: The tracker file or the store is broken.
    """
    iter_dir, root, resolved = resolve_iteration_dir(Path(src), iteration=iteration)
    if not (iter_dir / METADATA_FILE).is_file():
        _refuse_legacy_layout(iter_dir)
        raise IntegrityError(f"{iter_dir} has no {METADATA_FILE}; it is not a complete DCP store")
    view = MetadataView.load(iter_dir)
    _LOGGER.info(
        "opened %s: %d tensors, %d byte items", iter_dir, len(view.tensors), len(view.byte_items)
    )
    return Store(iter_dir, root, resolved, view)


def resolve_iteration_dir(
    src: Path, *, iteration: int | None = None
) -> tuple[Path, Path | None, Iteration]:
    """Find the store directory for ``src``.

    Returns:
        ``(iter_dir, root, iteration)``; ``root`` is ``None`` when ``src`` is
        itself the store.
    """
    if not src.is_dir():
        raise UsageError(f"{src} is not a directory")
    if _is_store_dir(src):
        named = _iteration_from_name(src.name)
        if iteration is not None and iteration != named:
            raise UsageError(
                f"--iteration {iteration} does not match {src}, which is itself a store",
                hint="drop --iteration or pass the checkpoint root",
            )
        return src, None, named
    if iteration is not None:
        iter_dir = src / f"iter_{iteration:07d}"
        if not iter_dir.is_dir():
            raise UsageError(f"--iteration {iteration}: {iter_dir} does not exist")
        return iter_dir, src, iteration
    tracker = src / TRACKER_FILE
    if not tracker.is_file():
        _refuse_legacy_layout(src)
        raise UsageError(
            f"{src} is neither a DCP store (no {METADATA_FILE} or {TORCH_DIST_CONFIG_FILE}) "
            f"nor a checkpoint root with {TRACKER_FILE}",
            hint="pass the iteration directory, or the root together with --iteration N",
        )
    named = _read_tracker(tracker)
    iter_dir = src / ("release" if named == "release" else f"iter_{named:07d}")
    if not iter_dir.is_dir():
        raise IntegrityError(f"{tracker} names {iter_dir.name}, which does not exist")
    return iter_dir, src, named


def detect_format(store: Store, *, from_format: str | None = None) -> Detection:
    """Classify ``store`` as torch_dist or Megatron-FSDP v1 fsdp_dtensor.

    Args:
        store: The opened store.
        from_format: The format the user asserts (``--from``), if any.

    Raises:
        UsageError: ``from_format`` is not a known format.
        RefusedError: The store is of a refused kind (FC-R01, FC-R02, FC-R13, FC-R19).
        AmbiguousError: ``from_format`` disagrees with the detected format (FC-A05).
        IntegrityError: ``metadata.json`` is unreadable.
    """
    if from_format is not None and from_format not in FORMATS:
        raise UsageError(f"--from must be one of {', '.join(FORMATS)}, got {from_format!r}")
    config = store.iter_dir / TORCH_DIST_CONFIG_FILE
    if config.is_file():
        detected, evidence = _detect_from_config(config)
    else:
        detected, evidence = _detect_from_keys(store.view)
    if detected is None:
        if from_format is None:
            raise Refusal(
                errors.FC_R13,
                f"{store.iter_dir} has no format markers "
                f"({TORCH_DIST_CONFIG_FILE}, or {FSDP_MODEL_PREFIX}* keys with byte items "
                f"{', '.join(FSDP_COMMON_ITEMS)})",
                hint="pass --from torch_dist or --from fsdp_dtensor if you know the format",
            ).to_error()
        detected, evidence = from_format, (*evidence, f"format asserted by --from {from_format}")
    elif from_format is not None and from_format != detected:
        raise Refusal(
            errors.FC_A05,
            f"--from {from_format} disagrees with the detected format {detected} "
            f"({'; '.join(evidence)})",
            hint=f"drop --from or pass --from {detected}",
        ).to_error()
    _LOGGER.info(
        "%s is %s (%s): %s", store.iter_dir, detected, DIALECTS[detected], "; ".join(evidence)
    )
    return Detection(
        fmt=detected,
        dialect=DIALECTS[detected],
        iter_dir=store.iter_dir,
        root=store.root,
        iteration=store.iteration,
        evidence=tuple(evidence),
    )


def publish_target(
    dst: Path, iteration: Iteration, *, iter_dir_mode: bool, tracker: bool = True
) -> PublishTarget:
    """Where a converted store goes: the inverse of :func:`resolve_iteration_dir`.

    ``dst`` is an iteration directory (a bare store, no tracker) if ``iter_dir_mode``
    is set or its name is ``iter_<N>`` or ``release``; otherwise it is a checkpoint
    root that gets ``iter_%07d`` (or ``release``) and, if ``tracker``, the tracker.

    Raises:
        UsageError: ``dst`` is a root and the iteration is unknown.
    """
    dst = Path(dst).absolute()
    if iter_dir_mode or _ITERATION_DIR.match(dst.name) or dst.name == "release":
        return PublishTarget(dst)
    if iteration is None:
        raise UsageError(
            f"the source's iteration is unknown, so no iteration directory can be named in {dst}",
            hint="pass DST as the iteration directory itself (or --dst-is-iter-dir)",
        )
    name = "release" if iteration == "release" else f"iter_{iteration:07d}"
    return PublishTarget(dst / name, dst / TRACKER_FILE if tracker else None, str(iteration))


def _detect_from_config(config: Path) -> tuple[str, tuple[str, ...]]:
    try:
        backend = json.loads(config.read_text()).get("sharded_backend")
    except (OSError, ValueError, AttributeError) as err:
        raise IntegrityError(f"cannot read {config}: {err}") from err
    if backend != TORCH_DIST:
        raise Refusal(
            errors.FC_R13,
            f"{config} declares sharded_backend {backend!r}; only torch_dist is supported",
            hint="re-save the checkpoint with --ckpt-format torch_dist",
        ).to_error()
    return TORCH_DIST, (f"{TORCH_DIST_CONFIG_FILE}: sharded_backend={TORCH_DIST}",)


def _detect_from_keys(view: MetadataView) -> tuple[str | None, tuple[str, ...]]:
    keys = sorted(view.tensors.keys() | view.byte_items.keys())
    vpp = [k for k in keys if _VPP_KEY.match(k)]
    if vpp:
        raise Refusal(
            errors.FC_R02,
            "the store has virtual-pipeline sections, which fsdp_dtensor does not support",
            keys=tuple(vpp),
            hint="re-save the checkpoint without virtual pipeline parallelism",
        ).to_error()
    model = [k for k in keys if k.startswith("model.")]
    v1 = [k for k in model if k.startswith(FSDP_MODEL_PREFIX)]
    if model and not v1:
        raise Refusal(
            errors.FC_R01,
            f"model keys lack the {FSDP_MODEL_PREFIX!r} prefix; this is a Megatron-FSDP v2 store",
            keys=tuple(model),
            hint="only Megatron-FSDP v1 stores (--megatron-fsdp-version 1) are supported",
        ).to_error()
    if len(v1) != len(model):
        raise Refusal(
            errors.FC_R13,
            f"the store mixes {FSDP_MODEL_PREFIX}* keys with other model.* keys",
            keys=tuple(k for k in model if not k.startswith(FSDP_MODEL_PREFIX)),
            hint=f"Megatron-FSDP v1 stores keep every model key under {FSDP_MODEL_PREFIX}; "
            "pass a store written by one Megatron-FSDP v1 run",
        ).to_error()
    split = [k for k in v1 if _SPLIT_FUSED_KEY.search(k)]
    if split:
        raise Refusal(
            errors.FC_R19,
            "fused in_proj/conv1d projections are stored as split sub-keys, "
            "a layout Megatron-FSDP v1 does not write",
            keys=tuple(split),
            hint="pass a store saved by Megatron-FSDP v1 itself (it keeps in_proj/conv1d fused), "
            "or start from the torch_dist checkpoint",
        ).to_error()
    common = [k for k in FSDP_COMMON_ITEMS if k in view.byte_items]
    evidence = []
    if v1:
        evidence.append(f"{len(v1)} keys under {FSDP_MODEL_PREFIX}")
    if common:
        evidence.append(f"byte items {', '.join(common)}")
    if v1 and len(common) == len(FSDP_COMMON_ITEMS):
        return FSDP_DTENSOR, tuple(evidence)
    return None, tuple(evidence)


def _is_store_dir(path: Path) -> bool:
    return (path / METADATA_FILE).is_file() or (path / TORCH_DIST_CONFIG_FILE).is_file()


def _iteration_from_name(name: str) -> Iteration:
    if name == "release":
        return "release"
    match = _ITERATION_DIR.match(name)
    return int(match.group("iteration")) if match else None


def _read_tracker(tracker: Path) -> int | str:
    try:
        value = tracker.read_text().strip()
    except OSError as err:
        raise IntegrityError(f"cannot read {tracker}: {err.strerror or err}") from err
    if value == "release":
        return value
    if _TRACKER_VALUE.match(value):
        return int(value)
    raise IntegrityError(f"{tracker}: expected an iteration number or 'release', found {value!r}")


def _refuse_legacy_layout(path: Path) -> None:
    legacy = sorted(p.name for p in path.glob("mp_rank_*") if p.is_dir())
    if legacy:
        raise Refusal(
            errors.FC_R13,
            f"{path} is a legacy torch checkpoint (mp_rank_* directories), not a DCP store",
            keys=tuple(legacy),
            hint="convert it to torch_dist first (Megatron --ckpt-convert-format torch_dist)",
        ).to_error()
