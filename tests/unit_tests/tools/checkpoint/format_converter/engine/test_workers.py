# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Subprocess workers: sanitized environment, byte identity with in-process runs, no CUDA."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tools.checkpoint.format_converter.engine import worker as worker_module
from tools.checkpoint.format_converter.engine import workers as workers_module
from tools.checkpoint.format_converter.engine.faults import FAULT_ENV
from tools.checkpoint.format_converter.engine.staging import Staging
from tools.checkpoint.format_converter.engine.workers import REPO_ROOT, child_env, worker_threads
from tools.checkpoint.format_converter.errors import ExitCode, IntegrityError

from .._harness import scrubbed_env
from .._programs import (
    ITERATION,
    build_program,
    convert_store,
    root_target,
    same_tree,
    write_source,
)

ITER = f"iter_{ITERATION:07d}"


@pytest.fixture(name="source")
def fixture_source(tmp_path):
    return write_source(tmp_path / "src")


def test_child_env_strips_launcher_variables_and_hides_gpus(tmp_path):
    base = {
        "RANK": "3",
        "WORLD_SIZE": "8",
        "LOCAL_RANK": "3",
        "LOCAL_WORLD_SIZE": "8",
        "GROUP_RANK": "0",
        "ROLE_RANK": "3",
        "MASTER_ADDR": "node0",
        "MASTER_PORT": "29500",
        "TORCHELASTIC_RUN_ID": "x",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "PYTHONPATH": "/somewhere",
        "HOME": "/home/user",
        "NCCL_P2P_DISABLE": "1",
    }
    env = child_env(base, tmp_path, threads=3)
    assert env == {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "3",
        "PYTHONPATH": f"{tmp_path}:/somewhere",
        "HOME": "/home/user",
        "NCCL_P2P_DISABLE": "1",
    }
    assert child_env({}, tmp_path, threads=1)["PYTHONPATH"] == str(tmp_path)
    assert (REPO_ROOT / "tools" / "checkpoint" / "format_converter").is_dir()


def test_worker_threads_split_half_the_cores(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 64)
    assert [worker_threads(n) for n in (0, 1, 4, 64)] == [32, 32, 8, 1]


@pytest.mark.usefixtures("local_rank_zero")
def test_two_workers_write_the_same_bytes_as_one_process(tmp_path, source):
    dst = tmp_path / "dst"
    in_process = convert_store(source, dst, workers=0)
    reference = tmp_path / "reference"
    (dst / ITER).rename(reference)
    pooled = convert_store(source, dst, workers=2)
    assert pooled.units_run == in_process.units_run
    assert same_tree(dst / ITER, reference)
    assert {record["worker"] for record in pooled.journal} <= {0, 1}


@pytest.mark.usefixtures("local_rank_zero")
def test_a_failing_worker_fails_the_run_and_the_run_resumes(tmp_path, source, monkeypatch):
    dst = tmp_path / "dst"
    convert_store(source, dst)
    reference = tmp_path / "reference"
    (dst / ITER).rename(reference)
    monkeypatch.setenv(FAULT_ENV, "unit:1")
    with pytest.raises(IntegrityError, match=r"worker \d exited with status 99") as excinfo:
        convert_store(source, dst, workers=2)
    assert excinfo.value.exit_code == ExitCode.INTEGRITY
    assert "rerun the same command to resume" in excinfo.value.hint
    assert (root_target(dst).default_staging_dir() / "PLAN.json").is_file()
    monkeypatch.delenv(FAULT_ENV)
    resumed = convert_store(source, dst, workers=2)
    assert resumed.resumed
    assert same_tree(dst / ITER, reference)


@pytest.mark.usefixtures("local_rank_zero")
def test_no_cuda_in_the_driver_or_the_workers(tmp_path, source):
    """A fresh driver that could see the GPUs (if any), launched as if under torchrun."""
    script = textwrap.dedent(f"""
        import json
        from pathlib import Path

        import torch
        from tests.unit_tests.tools.checkpoint.format_converter._programs import convert_store

        result = convert_store(Path({str(source)!r}), Path({str(tmp_path / "dst")!r}), workers=2)
        print(json.dumps({{
            "driver": torch.cuda.is_initialized(),
            "workers": [record["cuda_initialized"] for record in result.journal],
        }}))
        """)
    env = scrubbed_env(hide_gpus=False)
    env.update({"RANK": "0", "WORLD_SIZE": "2", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "1"})
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["driver"] is False
    assert report["workers"] and not any(report["workers"])


def test_worker_main_reports_errors_with_their_exit_code(tmp_path, capsys):
    assert worker_module.main([str(tmp_path / ".fc-staging-x"), "0"]) == ExitCode.INTEGRITY
    assert "worker 0: error: cannot read" in capsys.readouterr().err


LAUNCHER = {
    "RANK": "1",
    "WORLD_SIZE": "2",
    "LOCAL_RANK": "0",
    "LOCAL_WORLD_SIZE": "2",
    "MASTER_ADDR": "127.0.0.1",
    "MASTER_PORT": "29500",
    "TORCHELASTIC_RUN_ID": "run",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
}


@pytest.mark.usefixtures("local_rank_zero")
def test_workers_start_clean_and_inherit_the_lock(tmp_path, source, monkeypatch):
    """What run_units hands each worker: sanitized environment, the lock, the driver's pid."""
    for name, value in {**LAUNCHER, "CUDA_VISIBLE_DEVICES": "0"}.items():
        monkeypatch.setenv(name, value)
    launched = []
    start = subprocess.Popen

    def recording_popen(command, **kwargs):
        lock = Path(command[3]) / "LOCK"
        (descriptor,) = kwargs["pass_fds"]
        launched.append((command, kwargs["env"], os.fstat(descriptor).st_ino == lock.stat().st_ino))
        return start(command, **kwargs)

    monkeypatch.setattr(workers_module.subprocess, "Popen", recording_popen)
    convert_store(source, tmp_path / "dst", workers=2)
    assert len(launched) == 2
    for command, env, holds_lock in launched:
        assert holds_lock
        assert command[-2:] == ["--driver-pid", str(os.getpid())]
        assert not LAUNCHER.keys() & env.keys()
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(REPO_ROOT)


@pytest.mark.usefixtures("local_rank_zero")
def test_the_lock_outlives_the_driver_while_a_worker_runs(tmp_path):
    staging = Staging(tmp_path / ".fc-staging-x")
    with staging.locked():
        worker = subprocess.Popen(  # pylint: disable=consider-using-with
            [sys.executable, "-c", "import sys; sys.stdin.read()"],
            stdin=subprocess.PIPE,
            pass_fds=(staging.lock_fd,),
        )
    try:
        with pytest.raises(IntegrityError, match="another conversion is using"):
            with Staging(staging.root).locked():
                pass
    finally:
        worker.communicate(b"")
    with Staging(staging.root).locked():
        pass


def test_a_worker_stops_once_its_driver_is_gone(tmp_path, source):
    target = root_target(tmp_path / "dst")
    program = build_program(source, target)
    staging = Staging(target.default_staging_dir())
    with staging.locked():
        staging.prepare(program.fingerprint, resume="auto", restart=False)
        staging.stage(program.staging_plan(), program.staged_units())
    with pytest.raises(IntegrityError, match=f"the driver \\(pid {os.getpid()}\\) is gone"):
        worker_module.run_worker(staging, 0, driver_pid=os.getpid())  # not this process's parent
    assert not any((staging.root / "claims").iterdir())


@pytest.mark.usefixtures("local_rank_zero")
def test_relative_paths_work_from_any_directory(tmp_path, source, monkeypatch):
    """Workers start in the repository; the paths they get must not depend on the cwd."""
    monkeypatch.chdir(tmp_path)
    result = convert_store(
        Path("src"), Path("dst"), workers=1, staging_dir=Path("stage"), keep_staging=True
    )
    assert result.units_run == len(build_program(source, root_target(tmp_path / "dst")).units)
    assert result.staging_dir == tmp_path / "stage" / f".fc-staging-{ITER}"
    plan = json.loads((result.staging_dir / "PLAN.json").read_text())
    assert plan["source_dir"] == str(source)
