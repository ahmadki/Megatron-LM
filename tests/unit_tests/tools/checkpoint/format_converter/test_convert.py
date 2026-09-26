# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``plan`` and ``convert``: weights-only torch_dist rewrites of a synthetic training save."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import textwrap

import pytest
import torch

from tools.checkpoint.format_converter import cli
from tools.checkpoint.format_converter.api import (
    convert_checkpoint,
    plan_conversion,
    verify_checkpoint,
)
from tools.checkpoint.format_converter.engine import memory
from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.engine.reader import ReadReq, SourceReader
from tools.checkpoint.format_converter.engine.regions import Region
from tools.checkpoint.format_converter.engine.staging import PublishTarget
from tools.checkpoint.format_converter.errors import InternalError
from tools.checkpoint.format_converter.mcore_state.common_state import megatron_available
from tools.checkpoint.format_converter.options import ConvertOptions, PlanOptions, VerifyOptions

from ._harness import REPO_ROOT, scrubbed_env
from ._synthetic import (
    TORCH_DIST_CONFIG,
    default_args,
    dim0_grid,
    product_grid,
    torch_save_bytes,
    write_fsdp_store,
    write_store,
    write_tracker,
)

QKV = "decoder.layers.self_attention.linear_qkv.weight"
MODEL = {
    QKV: torch.arange(192, dtype=torch.float32).reshape(2, 12, 8),
    "embedding.word_embeddings.weight": torch.linspace(-2, 2, 128)
    .to(torch.bfloat16)
    .reshape(16, 8),
    "decoder.final_layernorm.weight": torch.tensor([1.0, -0.0, float("nan"), 2.0]),
    "decoder.layers.self_attention.core_attention._extra_state": torch.arange(5, dtype=torch.uint8),
}
GRIDS = {QKV: product_grid((2, 12, 8), [[1], [6], []])}
OPTIMIZER = {
    f"optimizer.state.{kind}.{key}": torch.zeros_like(value, dtype=torch.float32)
    for kind in ("param", "exp_avg", "exp_avg_sq")
    for key, value in MODEL.items()
    if not key.endswith("_extra_state")
}
OPTIMIZER["chained_1.optimizer.state.exp_avg.decoder.final_layernorm.weight"] = torch.ones(4)
EXTRA_BYTES = {
    "decoder.layers.self_attention.linear_qkv._extra_state/shard_0_2": b"te-extra-0",
    "decoder.layers.self_attention.linear_qkv._extra_state/shard_1_2": b"te-extra-1",
}
DROPPED_BYTES = {
    "rng_state/shard_0.0.0_1.1.1": b"rng",
    "rerun_state_machine_state/shard_0_1": b"rr",
}
ARGS = default_args(no_save_optim=False, no_save_rng=False, lr=3e-4, params_dtype=torch.bfloat16)
COMMON = {
    "args": ARGS,
    "iteration": 7,
    "checkpoint_version": 3.0,
    "num_floating_point_operations_so_far": 1.25e15,
    "content_metadata": {"distrib_optim_sharding_type": "fully_reshardable"},
    "optimizer": {"param_groups": [{"lr": 3e-4}]},
    "opt_param_scheduler": {"num_steps": 70},
    "rerun_state_machine": {"current_iteration": 7},
}
ITER = "iter_0000007"
TRACKER = "latest_checkpointed_iteration.txt"
IGNORED = ["^(chained_\\d+\\.)?optimizer\\.", "^rng_state", "^rerun_state_machine_state"]


def write_training_save(root, *, legacy_common=False, common=None, byte_items=None):
    """A torch_dist save with optimizer, rng, rerun and extra state, plus sidecars."""
    common = COMMON if common is None else common
    store = root / ITER
    items = {**EXTRA_BYTES, **DROPPED_BYTES, **(byte_items or {})}
    if not legacy_common:
        items["common_state/shard_0_1"] = torch_save_bytes([common])
    write_store(store, {**MODEL, **OPTIMIZER}, grids=GRIDS, byte_items=items)
    (store / "metadata.json").write_text(json.dumps(TORCH_DIST_CONFIG))
    if legacy_common:
        (store / "common.pt").write_bytes(torch_save_bytes(common))
    (store / "run_config.yaml").write_text("model:\n  num_layers: 2\n")
    (store / "tokenizer").mkdir()
    (store / "tokenizer" / "tokenizer.json").write_text('{"vocab": {}}')
    (store / "train_dataloader_dprank000.pt").write_bytes(b"loader")
    (store / "notes.txt").write_text("hand-written")
    write_tracker(root, 7)
    return root


def _run(*argv):
    return cli.main([str(a) for a in argv])


def _read_all(store) -> dict[str, torch.Tensor]:
    view = MetadataView.load(store)
    reader = SourceReader(view, max_item_bytes=2**20)
    tensors = {}
    for key, entry in view.tensors.items():
        tensors[key] = torch.empty(entry.shape, dtype=entry.dtype)
        reader.read_into([ReadReq(key, Region.full(entry.shape), tensors[key])])
    return tensors


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and torch.equal(
        a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
    )


@pytest.mark.parametrize("legacy_common", [False, True], ids=["dcp-common", "common-pt"])
def test_convert_keeps_weights_and_drops_training_state(tmp_path, capsys, legacy_common):
    src = write_training_save(tmp_path / "src", legacy_common=legacy_common)
    assert (
        _run("convert", src, tmp_path / "dst", "--to", "torch_dist", "--workers", "0", "--json")
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    out = tmp_path / "dst" / ITER
    view = MetadataView.load(out)
    assert set(view.tensors) == set(MODEL)
    assert set(view.byte_items) == set(EXTRA_BYTES) | {"common_state/shard_0_1"}
    for key, tensor in _read_all(out).items():
        assert _bitwise_equal(tensor, MODEL[key]), key
    source_view = MetadataView.load(src / ITER)
    for key, entry in view.tensors.items():
        assert [c.region for c in entry.chunks] == [
            c.region for c in source_view.tensors[key].chunks
        ]
    payloads = SourceReader(view, max_item_bytes=2**20).read_bytes(sorted(view.byte_items))
    assert {k: payloads[k] for k in EXTRA_BYTES} == EXTRA_BYTES
    (common,) = torch.load(io.BytesIO(payloads["common_state/shard_0_1"]), weights_only=False)
    assert set(common) == {
        "args",
        "iteration",
        "checkpoint_version",
        "num_floating_point_operations_so_far",
        "content_metadata",
    }
    assert vars(common["args"]) == {**vars(ARGS), "no_save_optim": True, "no_save_rng": True}
    assert common["iteration"] == 7 and common["content_metadata"] == COMMON["content_metadata"]
    assert json.loads((out / "metadata.json").read_text()) == TORCH_DIST_CONFIG
    assert not (out / "common.pt").exists()
    assert (tmp_path / "dst" / "latest_checkpointed_iteration.txt").read_text() == "7"
    assert (out / "run_config.yaml").read_text() == "model:\n  num_layers: 2\n"
    assert (out / "tokenizer" / "tokenizer.json").read_text() == '{"vocab": {}}'
    assert not (out / "train_dataloader_dprank000.pt").exists() and not (out / "notes.txt").exists()
    actions = {s["section"]: (s["action"], s["keys"]) for s in report["sections"]}
    expected = {
        "model": ("keep", 3),
        "optimizer.state": ("drop", len(OPTIMIZER)),
        "common": ("rewrite", 1),
        "rng": ("drop", 1),
        "rerun": ("drop", 1),
        "extra_state": ("keep", 3),
    }
    if legacy_common:  # the common state is a file, not a key of the store
        del expected["common"]
    assert actions == expected
    assert report["common"]["source"] == (
        "common.pt" if legacy_common else "common_state/shard_0_1"
    )
    assert report["common"]["dropped"] == [
        "opt_param_scheduler",
        "optimizer",
        "rerun_state_machine",
    ]
    assert report["common"]["args_consulted"] == {"no_save_optim": False, "no_save_rng": False}
    assert report["common"]["args_changes"] == {
        "no_save_optim": [False, True],
        "no_save_rng": [False, True],
    }
    assert report["sidecars"] == {
        "copy": ["run_config.yaml", "tokenizer"],
        "dropped": ["train_dataloader_dprank000.pt"],
        "unknown": ["notes.txt"],
        "regenerated": [],
    }
    assert report["tracker"] == {
        "path": str(tmp_path / "dst" / TRACKER),
        "before": None,
        "after": "7",
    }
    assert (
        report["run"]["torch_load_calls"]
        == report["run"]["items_touched"]
        == len([c for e in view.tensors.values() for c in e.chunks])
    )
    ignore = [arg for pattern in IGNORED for arg in ("--ignore-key", pattern)]
    assert _run("verify", out, src / ITER, "--layout", "--common", "skip", *ignore) == 0
    semantic = verify_checkpoint(
        out,
        src / ITER,
        VerifyOptions(layout=True, common="semantic", ignore_keys=tuple(map(re.compile, IGNORED))),
    )
    assert {(f.key, f.detail) for f in semantic.findings} == {
        ("args.no_save_optim", "A True vs B False"),
        ("args.no_save_rng", "A True vs B False"),
        ("opt_param_scheduler", "only in B"),
        ("optimizer", "only in B"),
        ("rerun_state_machine", "only in B"),
    }
    accepted = [*IGNORED, r"^args\.no_save_(optim|rng)$", "^(optimizer|opt_param_scheduler)$"]
    accepted.append("^rerun_state_machine$")
    ignore = [arg for pattern in accepted for arg in ("--ignore-key", pattern)]
    assert _run("verify", out, src / ITER, "--layout", "--common", "semantic", *ignore) == 0


def test_plan_reports_without_writing(tmp_path, capsys):
    src = write_training_save(tmp_path / "src")
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert _run("plan", src, "--to", "torch_dist", "--workers", "2", "--json") == 0
    plan = json.loads(capsys.readouterr().out)
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    assert plan["source"]["iteration"] == 7 and plan["target"]["format"] == "torch_dist"
    assert plan["units"]["pieces"] == 7 and plan["units"]["byte_items"] == 3
    assert plan["memory"]["workers"] == 2
    assert plan["output_bytes"] > sum(t.nbytes for t in MODEL.values())
    assert _run("plan", src, "--to", "torch_dist") == 0
    text = capsys.readouterr().out
    assert "optimizer.state  drop" in text and "args.no_save_optim: False -> True" in text


def test_convert_into_an_iteration_directory_from_a_store(tmp_path):
    src = write_training_save(tmp_path / "src")
    report = convert_checkpoint(
        src / ITER, tmp_path / "out" / "iter_0000009", ConvertOptions(workers=0)
    )
    assert report.run.target.tracker is None
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["iter_0000009"]
    tensors = _read_all(tmp_path / "out" / "iter_0000009")
    assert _bitwise_equal(tensors[QKV], MODEL[QKV])
    named = convert_checkpoint(
        src / ITER, tmp_path / "named", ConvertOptions(workers=0, dst_is_iter_dir=True)
    )
    assert named.run.target.iter_dir == (tmp_path / "named").absolute()


def _expect(capsys, code, *argv, match):
    assert _run(*argv) == code
    err = capsys.readouterr().err
    assert match in err, err


def test_refusals_and_budgets_exit_with_their_codes(tmp_path, capsys):
    src = write_training_save(tmp_path / "src")
    (src / ITER / "modelopt_state").mkdir()
    _expect(capsys, 3, "plan", src, "--to", "torch_dist", match="FC-R18")
    unknown = write_training_save(tmp_path / "u", common={**COMMON, "custom_state": 1})
    _expect(capsys, 3, "plan", unknown, "--to", "torch_dist", match="FC-R13: the common state")
    assert _run("plan", unknown, "--to", "torch_dist", "--drop-common-key", "custom_state") == 0
    good = write_training_save(tmp_path / "good")
    _expect(
        capsys,
        8,
        "plan",
        good,
        "--to",
        "torch_dist",
        "--max-host-bytes",
        "1MiB",
        match="above the budget",
    )
    _expect(
        capsys,
        8,
        "plan",
        good,
        "--to",
        "torch_dist",
        "--max-unit-bytes-hard",
        "100",
        match="hard unit cap",
    )
    fsdp = write_fsdp_store(tmp_path / "fsdp", {"decoder.final_layernorm.weight": torch.ones(2)})
    _expect(capsys, 2, "plan", fsdp, "--to", "torch_dist", match="this version converts torch_dist")
    (tmp_path / "dst" / ITER).mkdir(parents=True)
    _expect(
        capsys, 6, "convert", good, tmp_path / "dst", "--to", "torch_dist", match="already exists"
    )
    _expect(capsys, 2, "convert", good, good, "--to", "torch_dist", match="is the source store")


def test_other_staging_exits_7_until_restarted(tmp_path, capsys):
    src = write_training_save(tmp_path / "src")
    dst = tmp_path / "dst"
    argv = ["convert", src, dst, "--to", "torch_dist", "--workers", "0"]
    assert _run(*argv, "--keep-staging") == 0
    _expect(
        capsys,
        7,
        *argv,
        "--overwrite",
        "--unit-bytes",
        "1MiB",
        match="holds a different conversion",
    )
    assert _run(*argv, "--overwrite", "--unit-bytes", "1MiB", "--restart") == 0
    assert not any(p.name.startswith(".fc-") for p in dst.iterdir())


def test_plan_reads_only_the_common_state_and_cannot_be_executed(tmp_path, monkeypatch):
    src = write_training_save(tmp_path / "src")
    read = []
    read_bytes = SourceReader.read_bytes

    def recording(self, keys):
        read.extend(keys)
        return read_bytes(self, keys)

    monkeypatch.setattr(SourceReader, "read_bytes", recording)
    plan = plan_conversion(src, PlanOptions(workers=0))
    assert read == ["common_state/shard_0_1"]
    pieces = {piece.key: piece for unit in plan.units for piece in unit.byte_pieces}
    assert {key: (pieces[key].payload, pieces[key].nbytes) for key in EXTRA_BYTES} == {
        key: (None, len(payload)) for key, payload in EXTRA_BYTES.items()
    }
    with pytest.raises(InternalError, match="a dry-run plan cannot be executed"):
        plan.program(PublishTarget(tmp_path / "out"))


def test_a_changed_source_invalidates_resumable_staging(tmp_path, capsys):
    src = write_training_save(tmp_path / "src")
    dst = tmp_path / "dst"
    argv = ["convert", src, dst, "--to", "torch_dist", "--workers", "0", "--overwrite"]
    assert _run(*argv, "--keep-staging") == 0
    data = sorted((src / ITER).glob("*.distcp"))[0]
    stat = data.stat()
    os.utime(data, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    _expect(capsys, 7, *argv, match="holds a different conversion")
    assert _run(*argv, "--restart") == 0


def test_convert_refuses_an_output_that_does_not_fit_on_disk(tmp_path, capsys, monkeypatch):
    src = write_training_save(tmp_path / "src")
    monkeypatch.setattr(memory, "free_disk_bytes", lambda path: 10)
    argv = ["convert", src, tmp_path / "dst", "--to", "torch_dist", "--workers", "0"]
    _expect(capsys, 8, *argv, match="the output needs about")
    assert not (tmp_path / "dst").exists()


def _tree(path) -> dict[str, bytes]:
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_directories_around_the_conversion_are_never_deleted(tmp_path, capsys):
    """--staging-dir names a parent the tool leaves alone; a root holding the source is kept."""
    src = write_training_save(tmp_path / "src")
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "notes.txt").write_text("precious")
    dst = tmp_path / "dst"
    (dst / "iter_0000003").mkdir(parents=True)
    (dst / "iter_0000003" / ".metadata").write_bytes(b"older store")
    argv = ["convert", src, dst, "--to", "torch_dist", "--workers", "0"]
    assert _run(*argv, "--staging-dir", mine) == 0
    assert _tree(mine) == {"notes.txt": b"precious"}
    assert _run(*argv, "--staging-dir", dst, "--overwrite") == 0
    assert sorted(p.name for p in dst.iterdir()) == ["iter_0000003", ITER, TRACKER]
    before = _tree(src)
    _expect(
        capsys,
        2,
        "convert",
        src / ITER,
        src,
        "--dst-is-iter-dir",
        "--overwrite",
        "--to",
        "torch_dist",
        match="contains the source store",
    )
    assert _tree(src) == before


def test_convert_never_moves_the_tracker_back(tmp_path, capsys):
    src = write_training_save(tmp_path / "src")
    dst = tmp_path / "dst"
    dst.mkdir()
    write_tracker(dst, 9)
    assert _run("convert", src, dst, "--to", "torch_dist", "--workers", "0") == 0
    assert f"(tracker {dst / TRACKER} left at 9: it only moves forward" in capsys.readouterr().out
    assert (dst / TRACKER).read_text() == "9\n"
    assert (dst / ITER / ".metadata").is_file()


def test_integrity_manifest_is_written_for_the_new_files(tmp_path):
    src = write_training_save(tmp_path / "src")
    (src / ITER / "integrity.json").write_text('{"algorithm": "sha256", "files": {"old": "0"}}')
    report = convert_checkpoint(src, tmp_path / "dst", ConvertOptions(workers=0))
    out = tmp_path / "dst" / ITER
    expected = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in out.iterdir()
        if path.is_file() and path.name != "integrity.json"
    }
    assert {".metadata", "metadata.json", "run_config.yaml"} <= expected.keys()
    manifest = json.loads((out / "integrity.json").read_text())
    assert manifest == {"algorithm": "sha256", "files": expected}
    assert report.plan["sidecars"]["regenerated"] == ["integrity.json"]


@pytest.mark.skipif(not megatron_available(), reason="needs Megatron (the training container)")
def test_megatron_verifies_the_rewritten_integrity_manifest(tmp_path):
    from megatron.core.dist_checkpointing import validation  # pylint: disable=C0415

    src = write_training_save(tmp_path / "src")
    validation.save_integrity_manifest(str(src / ITER))
    convert_checkpoint(src, tmp_path / "dst", ConvertOptions(workers=0))
    validation._verify_integrity_manifest_impl(  # pylint: disable=protected-access
        str(tmp_path / "dst" / ITER)
    )


@pytest.mark.usefixtures("local_rank_zero")
def test_relative_paths_from_outside_the_repository(tmp_path):
    write_training_save(tmp_path / "src")
    argv = ["convert", "src", "out", "--to", "torch_dist", "--workers", "1", "--staging-dir", "st"]
    result = subprocess.run(
        [sys.executable, "-m", "tools.checkpoint.format_converter", *argv],
        cwd=tmp_path,
        env=scrubbed_env(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / ITER / ".metadata").is_file()
    assert not any((tmp_path / "st").iterdir())


def test_plan_api(tmp_path):
    src = write_training_save(tmp_path / "src")
    plan = plan_conversion(src, PlanOptions(workers=0))
    assert plan.iteration == 7 and plan.detection.fmt == "torch_dist"
    assert [unit.uid for unit in plan.units] == list(range(len(plan.units)))
    assert plan.common.codec in ("plain", "megatron")
    assert plan.common.kept == (
        "args",
        "iteration",
        "checkpoint_version",
        "num_floating_point_operations_so_far",
        "content_metadata",
    )
    assert plan.memory.workers == 0 and plan.memory.total <= plan.options.max_host_bytes


@pytest.mark.usefixtures("local_rank_zero")
def test_convert_memory_is_bounded_by_the_unit_budget(tmp_path):
    """A 64 MiB store converted with a 4 MiB unit budget raises the peak RSS by a few units."""
    store = tmp_path / "big" / ITER
    big = {"decoder.layers.mlp.linear_fc1.weight": torch.ones(64, 512, 1024, dtype=torch.bfloat16)}
    write_store(
        store,
        big,
        grids={"decoder.layers.mlp.linear_fc1.weight": dim0_grid((64, 512, 1024), 1)},
        byte_items={
            "common_state/shard_0_1": torch_save_bytes([{"args": default_args(), "iteration": 7}])
        },
    )
    (store / "metadata.json").write_text(json.dumps(TORCH_DIST_CONFIG))
    script = textwrap.dedent(f"""
        import json
        from pathlib import Path
        from tools.checkpoint.format_converter.api import convert_checkpoint, plan_conversion
        from tools.checkpoint.format_converter.engine.memory import process_peak_rss_bytes
        from tools.checkpoint.format_converter.options import ConvertOptions, PlanOptions

        options = dict(workers=0, unit_bytes=4 * 2**20, writer_threads=1)
        plan_conversion(Path({str(store)!r}), PlanOptions(**options))
        before = process_peak_rss_bytes()
        report = convert_checkpoint(
            Path({str(store)!r}), Path({str(tmp_path / "out" / ITER)!r}), ConvertOptions(**options)
        )
        print(json.dumps({{"before": before, "after": process_peak_rss_bytes(),
                          "units": report.to_json()["units"]["count"]}}))
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=scrubbed_env(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    peaks = json.loads(result.stdout.strip().splitlines()[-1])
    assert peaks["units"] >= 16
    assert peaks["after"] - peaks["before"] <= 24 * 2**20, peaks  # 6 units; the store is 64 MiB
