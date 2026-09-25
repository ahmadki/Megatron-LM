# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Host-side, display-only summary of a store's common (non-tensor) state.

Where the common state lives:

* torch_dist: the byte item ``common_state/shard_0_1`` holding
  ``torch.save([common_dict])``, or a legacy ``common.pt`` file holding
  ``torch.save(common_dict)`` (the file wins when both exist, as in Megatron).
* fsdp_dtensor: top-level byte items ``args``, ``iteration``,
  ``checkpoint_version`` (and ``num_floating_point_operations_so_far``), each a
  ``torch.save`` of the value.

Payloads are decoded with every unknown class stubbed, so this works without
Megatron but the decoded objects must never be written back.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..engine.metadata import Stub, decode_payload
from ..engine.reader import SourceReader
from ..errors import IntegrityError
from .detect import (
    FSDP_COMMON_ITEMS,
    FSDP_OPTIONAL_COMMON_ITEMS,
    LEGACY_COMMON_FILE,
    TORCH_DIST,
    TORCH_DIST_COMMON_ITEM,
    Detection,
)

SUMMARY_ARGS = (
    "num_layers",
    "hidden_size",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "bf16",
    "fp16",
    "swiglu",
    "num_experts",
    "moe_grouped_gemm",
    "hybrid_layer_pattern",
    "mtp_num_layers",
    "ckpt_format",
    "use_megatron_fsdp",
    "megatron_fsdp_version",
)
"""Training arguments shown by ``inspect`` when the stored args have them."""


@dataclass(frozen=True)
class CommonSummary:
    """Selected facts from a store's common state.

    Attributes:
        source: Where the common state was read from.
        keys: Top-level keys of the common state.
        iteration: Stored ``iteration`` value, if present.
        checkpoint_version: Stored ``checkpoint_version`` value, if present.
        args: The :data:`SUMMARY_ARGS` fields present in the stored args (JSON-safe).
        stubbed: Classes that were stubbed while decoding.
    """

    source: str
    keys: tuple[str, ...]
    iteration: Any
    checkpoint_version: Any
    args: dict[str, Any]
    stubbed: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {
            "source": self.source,
            "keys": list(self.keys),
            "iteration": self.iteration,
            "checkpoint_version": self.checkpoint_version,
            "args": self.args,
            "stubbed": list(self.stubbed),
        }


def summarize_common_state(detection: Detection, reader: SourceReader) -> CommonSummary | None:
    """Read and summarize the common state of the detected store, if it has one.

    Raises:
        IntegrityError: The common state cannot be decoded or has the wrong shape.
    """
    if detection.fmt == TORCH_DIST:
        common, source, stubbed = _torch_dist_common(detection, reader)
    else:
        common, source, stubbed = _fsdp_common(reader)
    if common is None:
        return None
    return CommonSummary(
        source=source,
        keys=tuple(sorted(str(k) for k in common)),
        iteration=to_plain(common.get("iteration")),
        checkpoint_version=to_plain(common.get("checkpoint_version")),
        args=_selected_args(common.get("args")),
        stubbed=tuple(sorted(stubbed)),
    )


def to_plain(value: Any) -> Any:
    """A JSON-safe rendering of a decoded value (stubs and other objects as text)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): to_plain(v) for k, v in value.items()}
    return repr(value) if isinstance(value, Stub) else str(value)


def _torch_dist_common(
    detection: Detection, reader: SourceReader
) -> tuple[dict[str, Any] | None, str, frozenset[str]]:
    legacy = detection.iter_dir / LEGACY_COMMON_FILE
    if legacy.is_file():
        try:
            payload = legacy.read_bytes()
        except OSError as err:
            raise IntegrityError(f"cannot read {legacy}: {err.strerror or err}") from err
        decoded = decode_payload(payload, label=str(legacy))
        return (
            _as_common_dict(decoded.value, LEGACY_COMMON_FILE),
            LEGACY_COMMON_FILE,
            decoded.stubbed,
        )
    if TORCH_DIST_COMMON_ITEM not in reader.view.byte_items:
        return None, "", frozenset()
    payload = reader.read_bytes([TORCH_DIST_COMMON_ITEM])[TORCH_DIST_COMMON_ITEM]
    decoded = decode_payload(payload, label=f"byte item {TORCH_DIST_COMMON_ITEM}")
    value = decoded.value
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    return _as_common_dict(value, TORCH_DIST_COMMON_ITEM), TORCH_DIST_COMMON_ITEM, decoded.stubbed


def _fsdp_common(reader: SourceReader) -> tuple[dict[str, Any] | None, str, frozenset[str]]:
    keys = [
        k for k in (*FSDP_COMMON_ITEMS, *FSDP_OPTIONAL_COMMON_ITEMS) if k in reader.view.byte_items
    ]
    if not keys:
        return None, "", frozenset()
    common: dict[str, Any] = {}
    stubbed: set[str] = set()
    for key, payload in reader.read_bytes(keys).items():
        decoded = decode_payload(payload, label=f"byte item {key}")
        common[key] = decoded.value
        stubbed |= decoded.stubbed
    return common, f"byte items {', '.join(keys)}", frozenset(stubbed)


def _as_common_dict(value: Any, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise IntegrityError(
            f"{source} does not hold a common-state dict (found {type(value).__name__})"
        )
    return value


def _selected_args(args: Any) -> dict[str, Any]:
    if isinstance(args, argparse.Namespace):
        fields = vars(args)
    elif isinstance(args, Stub) and isinstance(args.state, dict):
        fields = args.state
    else:
        return {}
    return {name: to_plain(fields[name]) for name in SUMMARY_ARGS if name in fields}
