# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The conversion manifest: schema-1 JSON records, validated on read."""

from __future__ import annotations

import json

import pytest

from tools.checkpoint.format_converter.core.manifest import MANIFEST_FILE, Manifest
from tools.checkpoint.format_converter.errors import IntegrityError, InternalError

MANIFEST = Manifest(
    tool_version="0.1.0",
    torch_version="2.8.0",
    source={
        "iter_dir": "/ckpt/iter_0000007",
        "format": "torch_dist",
        "dialect": "mcore",
        "iteration": 7,
        "fingerprint": "f" * 64,
    },
    target={"format": "fsdp_dtensor", "dialect": "mfsdp_v1"},
    options={"to_format": "fsdp_dtensor"},
    derivations=({"field": "src_tp", "value": 1, "evidence": "args.tensor_model_parallel_size=1"},),
    args_sanitized={"ckpt_format": ("torch_dist", "fsdp_dtensor"), "no_save_optim": (False, True)},
    content_metadata={"present": True, "value": {"singleton_local_shards": False}},
    dropped=({"section": "optimizer.state", "keys": 12, "tensor_bytes": 4096},),
)


def test_json_round_trip():
    record = MANIFEST.for_output("a" * 64).to_json()
    assert record["schema_version"] == 1 and record["output_metadata_sha256"] == "a" * 64
    assert record["args_sanitized"] == {
        "ckpt_format": ["torch_dist", "fsdp_dtensor"],
        "no_save_optim": [False, True],
    }
    parsed = Manifest.from_json(json.loads(json.dumps(record)), label=MANIFEST_FILE)
    assert parsed == MANIFEST.for_output("a" * 64)
    assert parsed.original_args() == {"ckpt_format": "torch_dist", "no_save_optim": False}
    assert parsed.source_content_metadata == {"singleton_local_shards": False}


def test_a_manifest_is_written_only_for_a_merged_store():
    with pytest.raises(InternalError):
        MANIFEST.to_json()


def test_absent_content_metadata():
    absent = Manifest.from_json(
        {**MANIFEST.for_output("b").to_json(), "content_metadata": {"present": False}}, label="m"
    )
    assert absent.source_content_metadata is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 2}, "schema version 2"),
        ({"source": None}, "field 'source' is missing or not a dict"),
        ({"derivations": {}}, "field 'derivations'"),
        ({"args_sanitized": {"x": [1]}}, "args_sanitized must map fields"),
        ({"content_metadata": {"value": 1}}, "content_metadata.present"),
    ],
)
def test_malformed_records_are_integrity_errors(change, message):
    record = {**MANIFEST.for_output("c").to_json(), **change}
    with pytest.raises(IntegrityError, match=message):
        Manifest.from_json(record, label="m")
    with pytest.raises(IntegrityError, match="JSON object"):
        Manifest.from_json([record], label="m")
