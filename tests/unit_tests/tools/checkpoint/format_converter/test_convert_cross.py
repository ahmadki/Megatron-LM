# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""torch_dist <-> fsdp_dtensor conversion of a per-layer hybrid model on synthetic stores.

The expected tensors are the ones the synthetic source was written with; the
expected keys, byte items, planner paths and arguments are written out by
hand from the two formats' layouts.
"""

from __future__ import annotations

import hashlib
import io
import json

import pytest
import torch

from tools.checkpoint.format_converter import cli
from tools.checkpoint.format_converter.api import convert_checkpoint, plan_conversion
from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.engine.reader import ReadReq, SourceReader
from tools.checkpoint.format_converter.engine.regions import Region
from tools.checkpoint.format_converter.options import ConvertOptions, PlanOptions

from ._programs import same_tree
from ._synthetic import (
    TORCH_DIST_CONFIG,
    dim0_grid,
    hybrid_args,
    hybrid_tensors,
    product_grid,
    torch_save_bytes,
    write_fsdp_store,
    write_store,
    write_tracker,
)

ITER = "iter_0000007"

pytestmark = pytest.mark.covers("identity", "ground_truth")
QKV = "decoder.layers.0.self_attention.linear_qkv.weight"
FC2 = "decoder.layers.3.mlp.linear_fc2.weight"
MODEL = hybrid_tensors()
CONTENT_METADATA = {
    "distrib_optim_sharding_type": "fully_reshardable",
    "singleton_local_shards": False,
    "chained_optim_avoid_prefix": True,
}
COMMON = {
    "args": hybrid_args(),
    "iteration": 7,
    "checkpoint_version": 3.0,
    "num_floating_point_operations_so_far": 2.5e12,
    "content_metadata": CONTENT_METADATA,
    "optimizer": {"param_groups": [{"lr": 3e-4}]},
    "opt_param_scheduler": {"num_steps": 70},
    "rerun_state_machine": {"current_iteration": 7},
}
FSDP_ITEMS = {"args", "iteration", "checkpoint_version", "num_floating_point_operations_so_far"}


def write_torch_dist_source(root, *, common=None, model=None, extra_keys=None):
    """A torch_dist training save of the hybrid model (optimizer, rng, extra state included)."""
    model = dict(MODEL if model is None else model)
    optimizer = {f"optimizer.state.exp_avg.{k}": torch.zeros(v.shape) for k, v in model.items()}
    store = root / ITER
    write_store(
        store,
        {**model, **optimizer, **(extra_keys or {})},
        grids={QKV: product_grid((24, 8), [[12], []]), FC2: product_grid((8, 16), [[], [8]])},
        byte_items={
            "common_state/shard_0_1": torch_save_bytes([COMMON if common is None else common]),
            "decoder.layers.0.self_attention.linear_qkv._extra_state/shard_0_1": b"te",
            "rng_state/shard_0.0.0_1.1.1": b"rng",
        },
    )
    (store / "metadata.json").write_text(json.dumps(TORCH_DIST_CONFIG))
    (store / "run_config.yaml").write_text("model: hybrid\n")
    write_tracker(root, 7)
    return root


def read_tensors(store) -> dict[str, torch.Tensor]:
    view = MetadataView.load(store)
    reader = SourceReader(view, max_item_bytes=2**20)
    tensors = {}
    for key, entry in view.tensors.items():
        tensors[key] = torch.empty(entry.shape, dtype=entry.dtype)
        reader.read_into([ReadReq(key, Region.full(entry.shape), tensors[key])])
    return tensors


def read_items(store) -> dict[str, object]:
    view = MetadataView.load(store)
    payloads = SourceReader(view, max_item_bytes=2**20).read_bytes(sorted(view.byte_items))
    return {k: torch.load(io.BytesIO(v), weights_only=False) for k, v in payloads.items()}


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and torch.equal(
        a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
    )


def run(*argv) -> int:
    return cli.main([str(a) for a in argv])


def test_torch_dist_to_fsdp_dtensor(tmp_path, capsys):
    src = write_torch_dist_source(tmp_path / "src")
    argv = ["convert", src, tmp_path / "fsdp", "--to", "fsdp_dtensor", "--workers", "0"]
    assert run(*argv, "--max-output-chunk-bytes", "128", "--json") == 0
    report = json.loads(capsys.readouterr().out)
    out = tmp_path / "fsdp" / ITER
    view = MetadataView.load(out)
    assert set(view.tensors) == {f"model.module.{key}" for key in MODEL}
    for key, tensor in read_tensors(out).items():
        assert bitwise_equal(tensor, MODEL[key.removeprefix("model.module.")]), key
    grid = [(c.offsets, c.sizes) for c in view.tensors[f"model.module.{QKV}"].chunks]
    assert grid == [((row, 0), (8, 8)) for row in (0, 8, 16)]  # 16-byte bf16 rows, 128-byte chunks
    assert set(view.byte_items) == FSDP_ITEMS
    assert view.planner_data == {
        **{f"model.module.{k}": ("model", f"module.{k}") for k in MODEL},
        **{item: (item,) for item in FSDP_ITEMS},
    }
    assert not (out / "metadata.json").exists() and not (out / "common.pt").exists()
    assert (tmp_path / "fsdp" / "latest_checkpointed_iteration.txt").read_text() == "7"
    items = read_items(out)
    assert (items["iteration"], items["checkpoint_version"]) == (7, 3.0)
    assert items["num_floating_point_operations_so_far"] == 2.5e12
    assert vars(items["args"]) == {
        **vars(COMMON["args"]),
        "ckpt_format": "fsdp_dtensor",
        "use_megatron_fsdp": True,
        "no_save_optim": True,
        "no_save_rng": True,
        "pipeline_model_parallel_size": 1,
        "hybrid_layer_pattern": "*-*-",
    }
    manifest = json.loads((out / "conversion_manifest.json").read_text())
    assert (
        manifest["output_metadata_sha256"]
        == hashlib.sha256((out / ".metadata").read_bytes()).hexdigest()
    )
    assert manifest["content_metadata"] == {"present": True, "value": CONTENT_METADATA}
    assert list(manifest["content_metadata"]["value"]) == list(CONTENT_METADATA)  # its order
    assert manifest["args_sanitized"]["hybrid_layer_pattern"] == ["*-|*-", "*-*-"]
    assert (manifest["source"]["format"], manifest["target"]["format"]) == (
        "torch_dist",
        "fsdp_dtensor",
    )
    assert (out / "run_config.yaml").read_text() == "model: hybrid\n"
    actions = {s["section"]: s["action"] for s in report["sections"]}
    assert actions == {
        "model": "convert",
        "optimizer.state": "drop",
        "common": "rewrite",
        "rng": "drop",
        "extra_state": "drop",
    }
    assert report["claims"] == {"identity": len(MODEL)}
    consulted = report["common"]["args_consulted"]
    assert consulted["hybrid_layer_pattern"] == "*-|*-" and consulted["transformer_impl"]
    assert {"bf16", "num_layers", "ckpt_format", "no_save_optim"} <= consulted.keys()
    assert report["common"]["dropped"] == [
        "content_metadata",
        "opt_param_scheduler",
        "optimizer",
        "rerun_state_machine",
    ]


def test_round_trip_restores_the_source(tmp_path):
    src = write_torch_dist_source(tmp_path / "src")
    convert_checkpoint(src, tmp_path / "fsdp", ConvertOptions(to_format="fsdp_dtensor", workers=0))
    report = convert_checkpoint(tmp_path / "fsdp", tmp_path / "back", ConvertOptions(workers=0))
    out = tmp_path / "back" / ITER
    view = MetadataView.load(out)
    assert set(view.tensors) == set(MODEL)
    for key, tensor in read_tensors(out).items():
        assert bitwise_equal(tensor, MODEL[key]), key
    assert all(len(entry.chunks) == 1 for entry in view.tensors.values())
    assert json.loads((out / "metadata.json").read_text()) == TORCH_DIST_CONFIG
    (common,) = read_items(out)["common_state/shard_0_1"]
    assert common["content_metadata"] == CONTENT_METADATA
    assert list(common["content_metadata"]) == list(CONTENT_METADATA)  # in the source's order
    assert vars(common["args"]) == {
        **vars(COMMON["args"]),
        "no_save_optim": True,
        "no_save_rng": True,
    }
    assert set(common) == {
        "args",
        "iteration",
        "checkpoint_version",
        "num_floating_point_operations_so_far",
        "content_metadata",
    }
    manifest = json.loads((out / "conversion_manifest.json").read_text())
    assert manifest["source"]["format"] == "fsdp_dtensor"
    derived = {d["field"]: d["value"] for d in report.plan["derivations"]}
    assert derived["manifest"] == "torch_dist" and derived["direction"] == "fsdp_to_td"


def test_a_manifest_of_another_store_is_refused_unless_ignored(tmp_path, capsys):
    src = write_torch_dist_source(tmp_path / "src")
    convert_checkpoint(src, tmp_path / "fsdp", ConvertOptions(to_format="fsdp_dtensor", workers=0))
    manifest = tmp_path / "fsdp" / ITER / "conversion_manifest.json"
    record = json.loads(manifest.read_text())
    manifest.write_text(json.dumps({**record, "output_metadata_sha256": "0" * 64}))
    assert run("plan", tmp_path / "fsdp", "--to", "torch_dist") == 4
    assert "FC-A06: the source's conversion_manifest.json describes another store" in (
        capsys.readouterr().err
    )
    report = convert_checkpoint(
        tmp_path / "fsdp", tmp_path / "back", ConvertOptions(workers=0, ignore_manifest=True)
    )
    (common,) = read_items(tmp_path / "back" / ITER)["common_state/shard_0_1"]
    assert "content_metadata" not in common
    assert common["args"].pipeline_model_parallel_size == 1  # nothing restored
    (logged,) = [d for d in report.plan["derivations"] if d["field"] == "manifest"]
    assert logged["evidence"] == "conversion_manifest.json ignored (--ignore-manifest)"
    manifest.write_text("{not json")
    assert run("plan", tmp_path / "fsdp", "--to", "torch_dist") == 6


def test_content_metadata_the_manifest_cannot_hold_is_refused_unless_dropped(tmp_path, capsys):
    common = {**COMMON, "content_metadata": {"dtype": torch.bfloat16}}
    src = write_torch_dist_source(tmp_path / "src", common=common)
    assert run("plan", src, "--to", "fsdp_dtensor") == 3
    assert "FC-R13: content_metadata holds values" in capsys.readouterr().err
    argv = ["convert", src, tmp_path / "out", "--to", "fsdp_dtensor", "--workers", "0"]
    assert run(*argv, "--drop-common-key", "content_metadata") == 0
    manifest = json.loads((tmp_path / "out" / ITER / "conversion_manifest.json").read_text())
    assert manifest["content_metadata"] == {"present": False}


def test_a_source_without_an_item_the_fsdp_dtensor_loader_requires_is_refused(tmp_path, capsys):
    common = {key: value for key, value in COMMON.items() if key != "iteration"}
    src = write_torch_dist_source(tmp_path / "src", common=common)
    _expect(
        capsys,
        3,
        "plan",
        src,
        "--to",
        "fsdp_dtensor",
        lines=["FC-R13: the common state has no iteration, which the fsdp_dtensor loader requires"],
    )


def test_drop_common_key_applies_converting_back(tmp_path):
    src = write_torch_dist_source(tmp_path / "src")
    convert_checkpoint(src, tmp_path / "fsdp", ConvertOptions(to_format="fsdp_dtensor", workers=0))
    options = ConvertOptions(workers=0, drop_common_keys=frozenset({"content_metadata"}))
    report = convert_checkpoint(tmp_path / "fsdp", tmp_path / "back", options)
    (common,) = read_items(tmp_path / "back" / ITER)["common_state/shard_0_1"]
    assert "content_metadata" not in common
    assert report.plan["common"]["dropped"] == ["content_metadata"]
    assert common["args"].pipeline_model_parallel_size == 2  # the arguments are still restored
    assert (
        run("plan", tmp_path / "fsdp", "--to", "torch_dist", "--drop-common-key", "iteration") == 2
    )


def test_an_fsdp_dtensor_training_save_converts_to_torch_dist(tmp_path):
    """An fsdp_dtensor save without a manifest, with optimizer, scheduler and rng state."""
    args = hybrid_args(
        "*-*-",
        ckpt_format="fsdp_dtensor",
        use_megatron_fsdp=True,
        pipeline_model_parallel_size=1,
        no_save_rng=True,
    )
    fp32 = {key: tensor.float() for key, tensor in MODEL.items()}  # Megatron-FSDP main weights
    store = write_fsdp_store(
        tmp_path / "fsdp" / ITER,
        fp32,
        grids={QKV: dim0_grid((24, 8), 10)},
        args=args,
        extra_byte_items={
            "num_floating_point_operations_so_far": torch_save_bytes(1.0),
            "opt_param_scheduler.num_steps": torch_save_bytes(70),
            "optimizer.param_to_group_meta.module.module.module.output_layer.weight.lr": b"x",
        },
    )
    plan = plan_conversion(store, PlanOptions(workers=0))
    actions = {a.section.value: a.action for a in plan.sections}
    assert actions == {
        "model": "convert",
        "common": "rewrite",
        "scheduler": "drop",
        "optimizer other": "drop",
    }
    convert_checkpoint(store, tmp_path / "td" / ITER, ConvertOptions(workers=0))
    out = tmp_path / "td" / ITER
    for key, tensor in read_tensors(out).items():
        assert bitwise_equal(tensor, fp32[key]), key  # fp32 kept: loaders cast on load
    (common,) = read_items(out)["common_state/shard_0_1"]
    assert vars(common["args"]) == {
        **vars(args),
        "ckpt_format": "torch_dist",
        "use_megatron_fsdp": False,
        "no_save_optim": True,
    }
    assert common["num_floating_point_operations_so_far"] == 1.0
    assert "content_metadata" not in common


def _expect(capsys, code, *argv, lines):
    assert run(*argv) == code
    err = capsys.readouterr().err
    for line in lines:
        assert line in err, err


def test_refusals_are_all_reported_before_anything_is_read(tmp_path, capsys):
    fenced = write_torch_dist_source(
        tmp_path / "fenced", common={**COMMON, "args": hybrid_args(spec=["x.y", "custom"])}
    )
    _expect(
        capsys,
        3,
        "convert",
        fenced,
        tmp_path / "out",
        "--to",
        "fsdp_dtensor",
        "--optimizer",
        "convert",
        lines=["error: FC-R03: args.spec='x.y.custom'", "error: FC-P12:"],
    )
    assert not (tmp_path / "out").exists()
    extra = {"frontend.proj.weight": torch.ones(4)}
    rooted = write_torch_dist_source(tmp_path / "rooted", extra_keys=extra)
    _expect(
        capsys, 3, "plan", rooted, "--to", "fsdp_dtensor", lines=["FC-R13", "--allow-root frontend"]
    )
    assert run("plan", rooted, "--to", "fsdp_dtensor", "--allow-root", "frontend.") == 0


def test_fp8_delayed_scaling_into_torch_dist_needs_a_flag(tmp_path, capsys):
    args = hybrid_args("*-*-", fp8="e4m3", fp8_recipe="delayed", ckpt_format="fsdp_dtensor")
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, MODEL, args=args)
    _expect(capsys, 3, "plan", store, "--to", "torch_dist", lines=["FC-R02", "FC-R07"])
    args.pipeline_model_parallel_size = 1
    store = write_fsdp_store(tmp_path / "single" / ITER, MODEL, args=args)
    _expect(capsys, 3, "plan", store, "--to", "torch_dist", lines=["FC-R07"])
    assert run("plan", store, "--to", "torch_dist", "--allow-missing-fp8-extra-state") == 0


def test_plan_prints_the_derivations_and_the_rules(tmp_path, capsys):
    src = write_torch_dist_source(tmp_path / "src")
    assert run("plan", src, "--to", "fsdp_dtensor") == 0
    text = capsys.readouterr().out
    assert "derived:    direction = \"td_to_fsdp\" (source torch_dist, target fsdp_dtensor)" in text
    assert "blocks.decoder = {" in text and "hybrid layer pattern '*-|*-' (4 layers)" in text
    assert f"rules:      model tensors claimed: identity {len(MODEL)}" in text
    assert "model            convert" in text


@pytest.mark.usefixtures("local_rank_zero")
def test_worker_processes_write_the_same_bytes(tmp_path):
    src = write_torch_dist_source(tmp_path / "src")
    out = tmp_path / "out" / ITER
    for workers in (0, 2):
        convert_checkpoint(src, out, ConvertOptions(to_format="fsdp_dtensor", workers=workers))
        if workers == 0:
            out.rename(tmp_path / "reference")
    assert same_tree(out, tmp_path / "reference")
