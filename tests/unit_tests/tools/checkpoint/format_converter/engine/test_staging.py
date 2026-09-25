# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Staging, locking, fingerprinted resume, crash injection and iteration-dir publish."""

from __future__ import annotations

import fcntl
import json
import pickle
import subprocess
import sys
import textwrap
import types

import pytest
import torch

from tools.checkpoint.format_converter.engine import staging as staging_module
from tools.checkpoint.format_converter.engine.faults import FAULT_ENV, FAULT_EXIT_CODE
from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.engine.reader import ReadReq, SourceReader
from tools.checkpoint.format_converter.engine.regions import Region
from tools.checkpoint.format_converter.engine.runner import RunOptions, run_program, save_id
from tools.checkpoint.format_converter.engine.staging import (
    PublishTarget,
    Staging,
    ensure_same_file_system,
)
from tools.checkpoint.format_converter.errors import (
    ExitCode,
    FingerprintMismatch,
    IntegrityError,
    InternalError,
    RefusedError,
    UsageError,
)

from .._harness import REPO_ROOT, scrubbed_env
from .._programs import (
    BYTE_ITEMS,
    ITERATION,
    TENSORS,
    build_program,
    convert_store,
    root_target,
    same_tree,
    write_source,
)

TRACKER = "latest_checkpointed_iteration.txt"

ITER = f"iter_{ITERATION:07d}"


@pytest.fixture(name="source")
def fixture_source(tmp_path):
    return write_source(tmp_path / "src")


def _read_store(store) -> tuple[dict, dict]:
    view = MetadataView.load(store)
    reader = SourceReader(view, max_item_bytes=2**30)
    tensors = {}
    for key, entry in view.tensors.items():
        tensors[key] = torch.empty(entry.shape, dtype=entry.dtype)
        reader.read_into([ReadReq(key, Region.full(entry.shape), tensors[key])])
    return tensors, reader.read_bytes(sorted(view.byte_items))


def _assert_is_source_copy(store, source) -> None:
    tensors, payloads = _read_store(store)
    assert set(tensors) == set(TENSORS) and payloads == BYTE_ITEMS
    for key, value in TENSORS.items():
        assert torch.equal(tensors[key], value), key
    source_view, view = MetadataView.load(source), MetadataView.load(store)
    for key, entry in view.tensors.items():
        assert [c.region for c in entry.chunks] == [
            c.region for c in source_view.tensors[key].chunks
        ]


def test_run_publishes_the_store_and_the_tracker_last(tmp_path, source):
    result = convert_store(source, tmp_path / "dst")
    store = tmp_path / "dst" / ITER
    _assert_is_source_copy(store, source)
    assert (tmp_path / "dst" / "latest_checkpointed_iteration.txt").read_text() == str(ITERATION)
    assert sorted(p.name for p in (tmp_path / "dst").iterdir()) == [
        ITER,
        "latest_checkpointed_iteration.txt",
    ]
    metadata = pickle.loads((store / ".metadata").read_bytes())
    program = build_program(source, root_target(tmp_path / "dst"))
    assert metadata.storage_meta.checkpoint_id == str(store)
    assert metadata.storage_meta.save_id == save_id(program.fingerprint)
    assert not result.resumed and result.units_run == len(program.units)
    assert result.merge.tensors == len(TENSORS) and result.merge.byte_items == 1
    assert sorted(r["uid"] for r in result.journal) == list(range(len(program.units)))


def test_iteration_dir_target_has_no_tracker(tmp_path, source):
    target = PublishTarget(tmp_path / "out" / "store")
    run_program(build_program(source, target), target, RunOptions())
    _assert_is_source_copy(tmp_path / "out" / "store", source)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["store"]


def test_existing_target_is_refused_before_anything_is_written(tmp_path, source):
    (tmp_path / "dst" / ITER).mkdir(parents=True)
    (tmp_path / "dst" / ITER / "old").write_text("keep")
    with pytest.raises(IntegrityError, match="already exists") as excinfo:
        convert_store(source, tmp_path / "dst")
    assert excinfo.value.exit_code == ExitCode.INTEGRITY
    assert sorted(p.name for p in (tmp_path / "dst").iterdir()) == [ITER]
    assert (tmp_path / "dst" / ITER / "old").read_text() == "keep"


def _old_store(path, content="old") -> None:
    """A stand-in for an existing store directory (``--overwrite`` replaces only stores)."""
    path.mkdir(parents=True)
    (path / ".metadata").write_bytes(b"old metadata")
    (path / "payload").write_text(content)


def test_overwrite_replaces_only_that_iteration(tmp_path, source):
    dst = tmp_path / "dst"
    for name in ("iter_0000001", ITER):
        _old_store(dst / name, name)
    (dst / "latest_checkpointed_iteration.txt").write_text("1")
    convert_store(source, dst, overwrite=True)
    _assert_is_source_copy(dst / ITER, source)
    assert (dst / "iter_0000001" / "payload").read_text() == "iter_0000001"
    assert (dst / "latest_checkpointed_iteration.txt").read_text() == str(ITERATION)
    assert sorted(p.name for p in dst.iterdir()) == [
        "iter_0000001",
        ITER,
        "latest_checkpointed_iteration.txt",
    ]


def _tree(path) -> dict[str, bytes]:
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_overwrite_refuses_anything_but_one_store_directory(tmp_path, source):
    """A root, a directory holding the source, or a non-store is never moved to the trash."""
    run = tmp_path / "run"
    _old_store(run / "iter_0000005", "iter_0000005")
    (run / TRACKER).write_text("5")
    before = _tree(run)
    target = PublishTarget(run)  # an iteration-dir target that is really a checkpoint root
    with pytest.raises(UsageError, match="holds no .metadata") as excinfo:
        run_program(build_program(source, target), target, RunOptions(overwrite=True))
    assert excinfo.value.exit_code == ExitCode.USAGE
    (run / ".metadata").write_bytes(b"store at the root")
    with pytest.raises(UsageError, match=r"holds other stores \(iter_0000005\)"):
        run_program(build_program(source, target), target, RunOptions(overwrite=True))
    assert _tree(run) == {**before, ".metadata": b"store at the root"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run", "src"]


def test_targets_and_staging_overlapping_the_source_are_refused(tmp_path):
    """Nothing is created when the target or staging would hold or sit inside the source."""
    run = tmp_path / "run"
    source = write_source(run / ITER)
    _old_store(run / "iter_0000005")
    before = _tree(tmp_path)
    cases = [
        (PublishTarget(source), RunOptions(), "is the source store"),
        (PublishTarget(run), RunOptions(overwrite=True), "contains the source store"),
        (PublishTarget(source / "out"), RunOptions(), "lies inside the source store"),
        (root_target(tmp_path / "dst"), RunOptions(staging_dir=source), "overlaps the source"),
        (
            PublishTarget(tmp_path / "dst" / "store"),
            RunOptions(staging_dir=tmp_path / "dst" / "store"),
            "overlaps the target",
        ),
    ]
    for target, options, message in cases:
        with pytest.raises(UsageError, match=message):
            run_program(build_program(source, target), target, options)
    assert _tree(tmp_path) == before


def test_staging_dir_is_a_parent_the_tool_never_deletes(tmp_path, source):
    """``staging_dir`` holds the tool's own ``.fc-staging-*`` directory; nothing else is touched."""
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "notes.txt").write_text("precious")
    result = convert_store(source, tmp_path / "dst", staging_dir=mine)
    assert result.staging_dir == mine / f".fc-staging-{ITER}"
    assert sorted(p.name for p in mine.iterdir()) == ["notes.txt"]
    assert (mine / "notes.txt").read_text() == "precious"
    dst = tmp_path / "dst"
    (dst / ITER).rename(tmp_path / "reference")
    _old_store(dst / "iter_0000003")
    result = convert_store(source, dst, staging_dir=dst)  # the root itself: same as the default
    assert result.staging_dir == root_target(dst).default_staging_dir()
    assert sorted(p.name for p in dst.iterdir()) == ["iter_0000003", ITER, TRACKER]
    assert same_tree(dst / ITER, tmp_path / "reference")
    with pytest.raises(InternalError, match="refusing to use"):
        Staging(mine)


@pytest.mark.parametrize(
    "before, value, after",
    [
        (None, "7", "7"),
        ("5", "7", "7"),
        ("7", "7", "7"),
        ("9", "7", "9"),
        ("release", "7", "release"),
        ("5", "release", "5"),
        ("garbage", "7", "garbage"),
    ],
)
def test_tracker_only_moves_forward(tmp_path, source, before, value, after):
    dst = tmp_path / "dst"
    dst.mkdir()
    if before is not None:
        (dst / TRACKER).write_text(f"{before}\n")
    target = PublishTarget(dst / "store", dst / TRACKER, value)
    result = run_program(build_program(source, target), target, RunOptions())
    assert (dst / TRACKER).read_text().strip() == after
    assert (result.tracker.before, result.tracker.after) == (before, after)
    assert result.tracker.written == (before != after)
    _assert_is_source_copy(dst / "store", source)


def test_post_check_catches_a_part_corrupted_after_its_digest(tmp_path, source, monkeypatch):
    """A part file changed after its unit committed is caught before anything is published."""
    target = root_target(tmp_path / "dst")
    program = build_program(source, target)
    assert ("decoder.layers.0.mlp.linear_fc1.weight", (0, 0)) in program.checked
    commit = Staging.complete

    def corrupting_complete(self, uid, digests, record):
        commit(self, uid, digests, record)
        if uid == 0:  # unit 0 holds rows 0:3 of fc1; flip one byte of one element
            rows = TENSORS["decoder.layers.0.mlp.linear_fc1.weight"][0:3].numpy().tobytes()
            (data,) = [
                path
                for path in (self.root / "parts" / "u00000").glob("*.distcp")
                if rows in path.read_bytes()
            ]
            raw = bytearray(data.read_bytes())
            raw[bytes(raw).index(rows) + 5] ^= 0x01
            data.write_bytes(bytes(raw))

    monkeypatch.setattr(Staging, "complete", corrupting_complete)
    with pytest.raises(IntegrityError, match=r"post-check: 1 items .*linear_fc1\.weight@\(0, 0\)"):
        run_program(program, target, RunOptions())
    assert not target.iter_dir.exists() and not target.tracker.exists()


def test_units_record_digests_of_exactly_the_checked_pieces(tmp_path, source):
    target = root_target(tmp_path / "dst")
    program = build_program(source, target)
    result = run_program(program, target, RunOptions(keep_staging=True))
    assert len(program.checked) == sum(len(unit.pieces) for unit in program.units)
    for unit in program.units:
        done = json.loads((result.staging_dir / "parts" / f"u{unit.uid:05d}" / "DONE").read_text())
        recorded = {
            (d["key"], None if d["offsets"] is None else tuple(d["offsets"]))
            for d in done["digests"]
        }
        expected = {(p.key, p.chunk.offsets) for p in unit.pieces} & program.checked
        expected |= {(item.key, None) for item in unit.byte_pieces}
        assert recorded == expected, unit.uid


def test_second_run_is_locked_out(tmp_path, source):
    target = root_target(tmp_path / "dst")
    staging_dir = target.default_staging_dir()
    staging_dir.mkdir(parents=True)
    with open(staging_dir / "LOCK", "a+b") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
        with pytest.raises(IntegrityError, match="another conversion is using") as excinfo:
            convert_store(source, tmp_path / "dst")
    assert excinfo.value.exit_code == ExitCode.INTEGRITY
    convert_store(source, tmp_path / "dst")


def _stage_only(program, target) -> Staging:
    staging = Staging(target.default_staging_dir())
    with staging.locked():
        staging.prepare(program.fingerprint, resume="auto", restart=False)
        staging.stage(program.staging_plan(), program.staged_units())
    return staging


def test_fingerprint_mismatch_exits_7_unless_restarted(tmp_path, source):
    target = root_target(tmp_path / "dst")
    other = build_program(source, PublishTarget(tmp_path / "elsewhere"))
    _stage_only(other, target)
    program = build_program(source, target)
    with pytest.raises(FingerprintMismatch, match="holds a different conversion") as excinfo:
        run_program(program, target, RunOptions())
    assert excinfo.value.exit_code == ExitCode.FINGERPRINT_MISMATCH == 7
    result = run_program(program, target, RunOptions(restart=True))
    assert not result.resumed
    _assert_is_source_copy(target.iter_dir, source)


def test_resume_never_refuses_existing_staging(tmp_path, source):
    target = root_target(tmp_path / "dst")
    program = build_program(source, target)
    _stage_only(program, target)
    with pytest.raises(IntegrityError, match="staging directory .* exists"):
        run_program(program, target, RunOptions(resume="never"))
    assert run_program(program, target, RunOptions()).resumed


def test_resume_treats_every_claim_as_stale(tmp_path, source):
    reference = tmp_path / "reference"
    convert_store(source, tmp_path / "dst")
    (tmp_path / "dst" / ITER).rename(reference)
    target = root_target(tmp_path / "dst")
    program = build_program(source, target)
    staging = _stage_only(program, target)
    for unit in program.units:  # a dead run (maybe in another pid namespace) claimed everything
        assert staging.claim(unit.uid, worker_id=3)
    (staging.root / "parts" / "u00000.tmp").mkdir()
    (staging.root / "parts" / "u00000.tmp" / "__0_0.distcp").write_bytes(b"torn")
    result = run_program(program, target, RunOptions())
    assert result.resumed and result.units_run == len(program.units)
    assert same_tree(target.iter_dir, reference)


def test_keep_staging_keeps_parts_and_rerun_reproduces_the_store(tmp_path, source):
    result = convert_store(source, tmp_path / "dst", keep_staging=True)
    staging_dir = result.staging_dir
    assert sorted(p.name for p in (staging_dir / "parts").iterdir()) == [
        f"u{uid:05d}" for uid in range(len(result.journal))
    ]
    assert not (staging_dir / "final").exists() and not (staging_dir / "publish.json").exists()
    first = (tmp_path / "dst" / ITER / ".metadata").read_bytes()
    rerun = convert_store(source, tmp_path / "dst", overwrite=True)
    assert rerun.resumed and rerun.units_run == 0
    assert (tmp_path / "dst" / ITER / ".metadata").read_bytes() == first
    assert not staging_dir.exists()


def test_staging_must_share_the_target_file_system(tmp_path, monkeypatch):
    ensure_same_file_system(tmp_path, tmp_path)
    devices = {str(tmp_path / "staging"): 1, str(tmp_path): 2}
    fake_os = types.SimpleNamespace(
        stat=lambda path: types.SimpleNamespace(st_dev=devices[str(path)])
    )
    monkeypatch.setattr(staging_module, "os", fake_os)
    with pytest.raises(RefusedError) as excinfo:
        ensure_same_file_system(tmp_path / "staging", tmp_path)
    assert excinfo.value.format().startswith("FC-E02: staging directory")


def _run_child(source, dst, *, fault=None, overwrite=False):
    script = textwrap.dedent(f"""
        from pathlib import Path
        from tests.unit_tests.tools.checkpoint.format_converter._programs import convert_store
        convert_store(Path({str(source)!r}), Path({str(dst)!r}), overwrite={overwrite})
        """)
    env = scrubbed_env()
    if fault is not None:
        env[FAULT_ENV] = fault
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _check_crash_state(dst, fault) -> None:
    staging_dir = root_target(dst).default_staging_dir()
    assert (staging_dir / "PLAN.json").is_file()
    assert not (dst / "latest_checkpointed_iteration.txt").exists()
    if fault == "unit:1":
        assert (staging_dir / "parts" / "u00001.tmp" / ".metadata").is_file()
        assert not (staging_dir / "parts" / "u00001").exists()
    if fault == "merge:link:2":
        assert len(list((staging_dir / "final").iterdir())) == 2
    if fault == "publish:before_rename":
        trash = json.loads((staging_dir / "publish.json").read_text())["trash"]
        assert (dst / trash / "payload").is_file() and not (dst / ITER).exists()
    if fault == "publish:before_tracker":
        assert (dst / ITER / ".metadata").is_file() and not (staging_dir / "final").exists()


@pytest.mark.usefixtures("local_rank_zero")
@pytest.mark.parametrize(
    "faults, overwrite",
    [
        (("unit:1", "merge:link:2"), False),
        (("publish:before_rename", "publish:before_tracker"), True),
    ],
    ids=["execute-and-merge", "publish"],
)
def test_crashes_then_resume_give_a_byte_identical_store(tmp_path, source, faults, overwrite):
    """Kill the run at each commit point in turn (the second crash hits the resumed run)."""
    dst = tmp_path / "dst"
    convert_store(source, dst)
    reference = tmp_path / "reference"
    (dst / ITER).rename(reference)
    (dst / "latest_checkpointed_iteration.txt").unlink()
    if overwrite:
        _old_store(dst / ITER)
    for fault in faults:
        crashed = _run_child(source, dst, fault=fault, overwrite=overwrite)
        assert crashed.returncode == FAULT_EXIT_CODE, crashed.stderr
        _check_crash_state(dst, fault)
    resumed = _run_child(source, dst, overwrite=overwrite)
    assert resumed.returncode == 0, resumed.stderr
    assert same_tree(dst / ITER, reference)
    assert (dst / "latest_checkpointed_iteration.txt").read_text() == str(ITERATION)
    assert sorted(p.name for p in dst.iterdir()) == [ITER, "latest_checkpointed_iteration.txt"]
