# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The conversion manifest (``conversion_manifest.json``, schema version 1).

Every cross-format conversion writes a manifest into the converted store. It
records the provenance (tool and torch versions, the source store, the options
used), the architecture derivations, the argument fields that were sanitized
(``{field: [original, written]}``), the source's ``content_metadata`` (which
fsdp_dtensor stores do not hold), the dropped sections, and the SHA-256 of the
converted store's ``.metadata``, which ties the manifest to that store.

Converting a store back reads its manifest, if it describes the store, to
restore the original arguments and ``content_metadata``: a torch_dist ->
fsdp_dtensor -> torch_dist round trip then gives the source's arguments again
(except state the weights-only store does not hold). Correctness of the
tensors never depends on the manifest, and a missing manifest is not an error.

This module builds and validates the JSON record; reading and writing the file
is the caller's.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..errors import IntegrityError, InternalError

MANIFEST_FILE = "conversion_manifest.json"
SCHEMA_VERSION = 1

_FIELDS = {
    "schema_version": int,
    "tool_version": str,
    "torch_version": str,
    "source": dict,
    "target": dict,
    "options": dict,
    "derivations": list,
    "args_sanitized": dict,
    "content_metadata": dict,
    "dropped": list,
    "output_metadata_sha256": str,
}


@dataclass(frozen=True)
class Manifest:
    """The record of one conversion.

    Attributes:
        tool_version: format_converter version.
        torch_version: torch version of the converting process.
        source: The source store: ``iter_dir``, ``format``, ``dialect``,
            ``iteration`` and ``fingerprint``.
        target: The target ``format`` and ``dialect``.
        options: The options that shaped the output.
        derivations: The architecture derivations (field, value, evidence).
        args_sanitized: ``{field: [original, written]}`` for every argument
            the conversion changed.
        content_metadata: ``{"present": bool, "value": ...}`` of the source's
            ``content_metadata`` common-state entry.
        dropped: The source sections the conversion dropped.
        output_metadata_sha256: SHA-256 of the converted store's ``.metadata``
            (``None`` until the store is merged).
    """

    tool_version: str
    torch_version: str
    source: Mapping[str, Any]
    target: Mapping[str, Any]
    options: Mapping[str, Any]
    derivations: tuple[Mapping[str, Any], ...]
    args_sanitized: Mapping[str, tuple[Any, Any]]
    content_metadata: Mapping[str, Any]
    dropped: tuple[Mapping[str, Any], ...]
    output_metadata_sha256: str | None = None

    def for_output(self, metadata_sha256: str) -> Manifest:
        """This manifest, tied to the converted store whose ``.metadata`` has this digest."""
        return dataclasses.replace(self, output_metadata_sha256=metadata_sha256)

    def to_json(self) -> dict[str, Any]:
        """The JSON record.

        Raises:
            InternalError: The manifest is not tied to a store yet.
        """
        if self.output_metadata_sha256 is None:
            raise InternalError("the manifest is written only after the store is merged")
        return {
            "schema_version": SCHEMA_VERSION,
            "tool_version": self.tool_version,
            "torch_version": self.torch_version,
            "source": dict(self.source),
            "target": dict(self.target),
            "options": dict(self.options),
            "derivations": [dict(d) for d in self.derivations],
            "args_sanitized": {k: list(v) for k, v in self.args_sanitized.items()},
            "content_metadata": dict(self.content_metadata),
            "dropped": [dict(d) for d in self.dropped],
            "output_metadata_sha256": self.output_metadata_sha256,
        }

    @classmethod
    def from_json(cls, data: Any, *, label: str) -> Manifest:
        """Parse and validate a JSON record read from ``label``.

        Raises:
            IntegrityError: The record is not a schema-1 manifest.
        """
        if not isinstance(data, dict):
            raise IntegrityError(f"{label} does not hold a JSON object")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise IntegrityError(
                f"{label} has schema version {data.get('schema_version')!r}, "
                f"this tool reads version {SCHEMA_VERSION}"
            )
        for name, kind in _FIELDS.items():
            if not isinstance(data.get(name), kind):
                raise IntegrityError(f"{label}: field {name!r} is missing or not a {kind.__name__}")
        sanitized = data["args_sanitized"]
        if not all(isinstance(v, list) and len(v) == 2 for v in sanitized.values()):
            raise IntegrityError(f"{label}: args_sanitized must map fields to [original, written]")
        if not isinstance(data["content_metadata"].get("present"), bool):
            raise IntegrityError(f"{label}: content_metadata.present must be true or false")
        return cls(
            tool_version=data["tool_version"],
            torch_version=data["torch_version"],
            source=data["source"],
            target=data["target"],
            options=data["options"],
            derivations=tuple(data["derivations"]),
            args_sanitized={k: (v[0], v[1]) for k, v in sanitized.items()},
            content_metadata=data["content_metadata"],
            dropped=tuple(data["dropped"]),
            output_metadata_sha256=data["output_metadata_sha256"],
        )

    def original_args(self) -> dict[str, Any]:
        """The original value of every sanitized argument field."""
        return {field: original for field, (original, _) in self.args_sanitized.items()}

    @property
    def source_content_metadata(self) -> Any:
        """The source's ``content_metadata`` value (``None`` when it had none)."""
        return self.content_metadata.get("value") if self.content_metadata["present"] else None
