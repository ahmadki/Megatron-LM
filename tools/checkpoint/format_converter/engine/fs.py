# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Small crash-safe file-system primitives (atomic writes, directory fsync, journals)."""

from __future__ import annotations

import os
from pathlib import Path


def fsync_dir(path: Path) -> None:
    """Persist the directory entries of ``path`` (renames, links, new files)."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_atomic(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data``: write a temporary file, fsync, rename, fsync the directory.

    A reader sees either the old file or the complete new one, also after a crash.
    """
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def append_line(path: Path, line: str) -> None:
    """Append one line to a journal file and fsync it before returning."""
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(line.rstrip("\n") + "\n")
        stream.flush()
        os.fsync(stream.fileno())
