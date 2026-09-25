# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Iteration-directory resolution and every row of the detection table."""

from __future__ import annotations

import json
import logging

import pytest
import torch

from tools.checkpoint.format_converter.errors import (
    AmbiguousError,
    ExitCode,
    IntegrityError,
    RefusedError,
    UsageError,
)
from tools.checkpoint.format_converter.formats.detect import (
    detect_format,
    open_store,
    resolve_iteration_dir,
)

from .._synthetic import (
    torch_save_bytes,
    write_fsdp_store,
    write_store,
    write_torch_dist_store,
    write_tracker,
)

TENSORS = {"decoder.layers.0.mlp.linear_fc1.weight": torch.ones(4, 2)}


def _detect(src, **kwargs):
    from_format = kwargs.pop("from_format", None)
    return detect_format(open_store(src, **kwargs), from_format=from_format)


def _refusal(excinfo) -> tuple[str, int]:
    """Code and exit code of a refusal, which must also tell the user what to do."""
    assert excinfo.value.refusal.hint, f"{excinfo.value.refusal.code} has no hint"
    return excinfo.value.refusal.code, excinfo.value.exit_code


def test_torch_dist_store_directory(tmp_path):
    write_torch_dist_store(tmp_path / "iter_0000007", TENSORS)
    detection = _detect(tmp_path / "iter_0000007")
    assert (detection.fmt, detection.dialect) == ("torch_dist", "mcore")
    assert detection.root is None and detection.iteration == 7
    assert detection.evidence == ("metadata.json: sharded_backend=torch_dist",)


def test_tracker_with_iteration_number(tmp_path):
    write_torch_dist_store(tmp_path / "iter_0000012", TENSORS)
    write_tracker(tmp_path, 12)
    detection = _detect(tmp_path)
    assert detection.iter_dir == tmp_path / "iter_0000012"
    assert (detection.root, detection.iteration) == (tmp_path, 12)


def test_tracker_release(tmp_path):
    write_torch_dist_store(tmp_path / "release", TENSORS)
    write_tracker(tmp_path, "release")
    detection = _detect(tmp_path)
    assert detection.iter_dir == tmp_path / "release" and detection.iteration == "release"


def test_iteration_flag_resolves_a_root_and_overrides_the_tracker(tmp_path):
    write_torch_dist_store(tmp_path / "iter_0000003", TENSORS)
    write_torch_dist_store(tmp_path / "iter_0000009", TENSORS)
    assert resolve_iteration_dir(tmp_path, iteration=3) == (tmp_path / "iter_0000003", tmp_path, 3)
    write_tracker(tmp_path, 9)
    assert _detect(tmp_path, iteration=3).iter_dir == tmp_path / "iter_0000003"
    assert _detect(tmp_path).iter_dir == tmp_path / "iter_0000009"


def test_iteration_flag_must_match_a_store_directory(tmp_path):
    write_torch_dist_store(tmp_path / "iter_0000007", TENSORS)
    assert _detect(tmp_path / "iter_0000007", iteration=7).iteration == 7
    with pytest.raises(UsageError, match="does not match"):
        _detect(tmp_path / "iter_0000007", iteration=8)


def test_unresolvable_sources_are_usage_errors(tmp_path):
    with pytest.raises(UsageError, match="not a directory"):
        open_store(tmp_path / "missing")
    with pytest.raises(UsageError, match="neither a DCP store") as excinfo:
        open_store(tmp_path)
    assert excinfo.value.exit_code == ExitCode.USAGE
    with pytest.raises(UsageError, match="iter_0000004 does not exist"):
        open_store(tmp_path, iteration=4)


def test_broken_trackers_are_integrity_errors(tmp_path):
    write_tracker(tmp_path, 5)
    with pytest.raises(IntegrityError, match="names iter_0000005, which does not exist"):
        open_store(tmp_path)
    write_tracker(tmp_path, "latest")
    with pytest.raises(IntegrityError, match="expected an iteration number or 'release'"):
        open_store(tmp_path)


def test_config_without_metadata_is_an_integrity_error(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"sharded_backend": "torch_dist"}))
    with pytest.raises(IntegrityError, match="has no .metadata"):
        open_store(tmp_path)


def test_other_sharded_backend_is_refused(tmp_path):
    write_torch_dist_store(tmp_path, TENSORS)
    (tmp_path / "metadata.json").write_text(json.dumps({"sharded_backend": "zarr"}))
    with pytest.raises(RefusedError, match="sharded_backend 'zarr'") as excinfo:
        _detect(tmp_path)
    assert _refusal(excinfo) == ("FC-R13", ExitCode.REFUSED)


def test_unreadable_config_is_an_integrity_error(tmp_path):
    write_torch_dist_store(tmp_path, TENSORS)
    (tmp_path / "metadata.json").write_text("{not json")
    with pytest.raises(IntegrityError, match="cannot read"):
        _detect(tmp_path)


def test_megatron_fsdp_v1_store(tmp_path, caplog):
    write_fsdp_store(tmp_path, TENSORS)
    with caplog.at_level(logging.INFO, logger="tools.checkpoint.format_converter"):
        detection = _detect(tmp_path)
    assert (detection.fmt, detection.dialect) == ("fsdp_dtensor", "mfsdp_v1")
    assert detection.evidence == (
        "1 keys under model.module.",
        "byte items args, iteration, checkpoint_version",
    )
    assert f"{tmp_path} is fsdp_dtensor (mfsdp_v1): 1 keys under model.module." in caplog.text


@pytest.mark.parametrize(
    "items", [("args",), ("iteration", "checkpoint_version"), ("args", "iteration")]
)
def test_partial_fsdp_common_items_need_from(tmp_path, items):
    byte_items = {item: torch_save_bytes(item) for item in items}
    write_store(tmp_path, {"model.module.w": torch.ones(2)}, byte_items=byte_items)
    with pytest.raises(RefusedError, match="no format markers") as excinfo:
        _detect(tmp_path)
    assert _refusal(excinfo) == ("FC-R13", ExitCode.REFUSED)
    assert _detect(tmp_path, from_format="fsdp_dtensor").fmt == "fsdp_dtensor"


def test_megatron_fsdp_v2_store_is_refused(tmp_path):
    write_fsdp_store(tmp_path, TENSORS, prefix="model.")
    with pytest.raises(RefusedError, match="Megatron-FSDP v2") as excinfo:
        _detect(tmp_path)
    assert _refusal(excinfo) == ("FC-R01", ExitCode.REFUSED)
    assert excinfo.value.refusal.keys == ("model.decoder.layers.0.mlp.linear_fc1.weight",)


@pytest.mark.parametrize("prefix", ["model0.", "model1.module."])
def test_virtual_pipeline_model_sections_are_refused(tmp_path, prefix):
    write_fsdp_store(tmp_path, TENSORS, prefix=prefix)
    with pytest.raises(RefusedError) as excinfo:
        _detect(tmp_path)
    assert _refusal(excinfo) == ("FC-R02", ExitCode.REFUSED)


def test_virtual_pipeline_optimizer_sections_are_refused(tmp_path):
    write_fsdp_store(
        tmp_path, TENSORS, extra_byte_items={"optimizer.0.param_groups": torch_save_bytes([])}
    )
    with pytest.raises(RefusedError) as excinfo:
        _detect(tmp_path)
    assert _refusal(excinfo) == ("FC-R02", ExitCode.REFUSED)
    assert excinfo.value.refusal.keys == ("optimizer.0.param_groups",)


def test_mixed_model_prefixes_are_refused(tmp_path):
    tensors = {"module.a.weight": torch.ones(2), "b.weight": torch.ones(2)}
    write_fsdp_store(tmp_path, tensors, prefix="model.")
    with pytest.raises(RefusedError, match="mixes") as excinfo:
        _detect(tmp_path)
    assert _refusal(excinfo) == ("FC-R13", ExitCode.REFUSED)
    assert excinfo.value.refusal.keys == ("model.b.weight",)


@pytest.mark.parametrize(
    "fqn",
    [
        "decoder.layers.0.mixer.in_proj.weight.z",
        "decoder.layers.0.mixer.conv1d.bias.query",
        "decoder.layers.3.self_attention.in_proj.weight.V0",
    ],
)
def test_split_fused_projections_in_fsdp_stores_are_refused(tmp_path, fqn):
    write_fsdp_store(tmp_path, {**TENSORS, fqn: torch.ones(2, 2)})
    with pytest.raises(RefusedError) as excinfo:
        _detect(tmp_path)
    assert _refusal(excinfo) == ("FC-R19", ExitCode.REFUSED)
    assert excinfo.value.refusal.keys == (f"model.module.{fqn}",)


def test_split_projections_are_native_to_torch_dist(tmp_path):
    write_torch_dist_store(tmp_path, {"decoder.layers.0.mixer.in_proj.weight.z": torch.ones(2)})
    assert _detect(tmp_path).fmt == "torch_dist"


def test_store_without_markers_needs_from(tmp_path):
    write_store(tmp_path, {"model.module.w": torch.ones(2)})
    with pytest.raises(RefusedError, match="no format markers") as excinfo:
        _detect(tmp_path)
    assert _refusal(excinfo) == ("FC-R13", ExitCode.REFUSED)
    forced = _detect(tmp_path, from_format="fsdp_dtensor")
    assert (forced.fmt, forced.dialect) == ("fsdp_dtensor", "mfsdp_v1")
    assert forced.evidence[-1] == "format asserted by --from fsdp_dtensor"
    assert _detect(tmp_path, from_format="torch_dist").dialect == "mcore"


@pytest.mark.parametrize(
    "writer, detected, other",
    [
        (write_torch_dist_store, "torch_dist", "fsdp_dtensor"),
        (write_fsdp_store, "fsdp_dtensor", "torch_dist"),
    ],
)
def test_from_must_agree_with_detection(tmp_path, writer, detected, other):
    writer(tmp_path, TENSORS)
    assert _detect(tmp_path, from_format=detected).fmt == detected
    with pytest.raises(AmbiguousError, match=f"--from {other} disagrees") as excinfo:
        _detect(tmp_path, from_format=other)
    assert _refusal(excinfo) == ("FC-A05", ExitCode.AMBIGUOUS)
    assert f"hint: drop --from or pass --from {detected}" in str(excinfo.value)


def test_unknown_from_is_a_usage_error(tmp_path):
    write_torch_dist_store(tmp_path, TENSORS)
    with pytest.raises(UsageError, match="--from must be one of"):
        _detect(tmp_path, from_format="zarr")


def test_legacy_rank_directories_are_refused(tmp_path):
    (tmp_path / "iter_0000002" / "mp_rank_00").mkdir(parents=True)
    write_tracker(tmp_path, 2)
    for src in (tmp_path, tmp_path / "iter_0000002"):
        with pytest.raises(RefusedError, match="legacy torch checkpoint") as excinfo:
            open_store(src)
        assert _refusal(excinfo) == ("FC-R13", ExitCode.REFUSED)
