# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engine-level write programs over synthetic stores, shared by engine tests.

Importable by subprocesses (``python -c`` with the repository root on
``PYTHONPATH``) so crash-injection tests can run the same program in a child.
"""

from __future__ import annotations

import filecmp
import hashlib
from pathlib import Path

import torch

from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.engine.pieces import BytePiece, Unit, copy_pieces
from tools.checkpoint.format_converter.engine.reader import SourceReader
from tools.checkpoint.format_converter.engine.runner import (
    RunOptions,
    RunResult,
    WriteProgram,
    choose_checked,
    run_program,
)
from tools.checkpoint.format_converter.engine.staging import PublishTarget

from ._synthetic import dim0_grid, torch_save_bytes, write_store

ITERATION = 7
TENSORS = {
    "decoder.layers.0.mlp.linear_fc1.weight": torch.arange(96, dtype=torch.float32).reshape(12, 8),
    "decoder.layers.0.mlp.linear_fc2.weight": torch.linspace(-1, 1, 24).to(torch.bfloat16),
    "decoder.final_layernorm.weight": torch.tensor(0.5),
}
GRIDS = {"decoder.layers.0.mlp.linear_fc1.weight": dim0_grid((12, 8), 3)}
BYTE_ITEMS = {"blob": torch_save_bytes([{"iteration": ITERATION}])}


def write_source(path: Path) -> Path:
    """The synthetic source store used by the engine tests."""
    return write_store(path, TENSORS, grids=GRIDS, byte_items=BYTE_ITEMS)


def root_target(dst: Path) -> PublishTarget:
    """Root-mode target: ``dst/iter_0000007`` plus the tracker."""
    return PublishTarget(
        dst / f"iter_{ITERATION:07d}", dst / "latest_checkpointed_iteration.txt", str(ITERATION)
    )


def build_program(source: Path, target: PublishTarget, *, pieces_per_unit: int = 2) -> WriteProgram:
    """Copy every stored chunk and byte item of ``source``, a few pieces per unit."""
    view = MetadataView.load(source)
    pieces = [piece for key in sorted(view.tensors) for piece in copy_pieces(view.tensors[key])]
    payloads = SourceReader(view, max_item_bytes=2**30).read_bytes(sorted(view.byte_items))
    byte_pieces = tuple(BytePiece(key, payload) for key, payload in sorted(payloads.items()))
    groups = [pieces[i : i + pieces_per_unit] for i in range(0, len(pieces), pieces_per_unit)]
    units = tuple(
        Unit(uid, tuple(group), byte_pieces if uid == 0 else ()) for uid, group in enumerate(groups)
    )
    fingerprint = hashlib.sha256(f"{view.fingerprint()}|{target.iter_dir}".encode()).hexdigest()
    return WriteProgram(
        source=view,
        units=units,
        dispatch_order=tuple(reversed(range(len(units)))),
        checked=choose_checked(units, seed=fingerprint),
        writer_threads=2,
        max_item_bytes=2**30,
        fingerprint=fingerprint,
    )


def convert_store(source: Path, dst: Path, **options) -> RunResult:
    """Run the copy program of ``source`` into root-mode ``dst``."""
    target = root_target(dst)
    return run_program(build_program(source, target), target, RunOptions(**options))


def same_tree(a: Path, b: Path) -> bool:
    """True if two directory trees hold the same file names with identical bytes."""
    comparison = filecmp.dircmp(a, b)
    if comparison.left_only or comparison.right_only or comparison.funny_files:
        return False
    _, mismatch, errors = filecmp.cmpfiles(a, b, comparison.common_files, shallow=False)
    if mismatch or errors:
        return False
    return all(same_tree(a / name, b / name) for name in comparison.common_dirs)
