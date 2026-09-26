# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Key sections and a per-store census.

Keys are classified by the exact rules of each format's adapter (the ones
conversion uses).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..engine.metadata import MetadataView, StoredChunk
from . import fsdp_mfsdp_v1, torch_dist
from .base import Section
from .detect import FSDP_DTENSOR, TORCH_DIST

__all__ = ["Census", "Section", "SectionTotals", "classify_key", "take_census"]


def classify_key(fmt: str, key: str, *, is_bytes: bool) -> Section:
    """The section of ``key`` in a store of format ``fmt``."""
    if fmt == TORCH_DIST:
        return torch_dist.classify_key(key, is_bytes=is_bytes)
    if fmt == FSDP_DTENSOR:
        return fsdp_mfsdp_v1.classify_key(key, is_bytes=is_bytes)
    raise ValueError(f"unknown format {fmt!r}")


@dataclass(frozen=True, slots=True)
class SectionTotals:
    """Key count and decoded tensor bytes of one section.

    Attributes:
        keys: Number of keys (tensors and byte items).
        tensor_bytes: Decoded size of the section's tensors.
    """

    keys: int
    tensor_bytes: int


@dataclass(frozen=True)
class Census:
    """Metadata-only summary of a store.

    Attributes:
        tensors: Number of tensors.
        byte_items: Number of byte items.
        tensor_bytes: Decoded size of all tensors.
        sections: Totals per section, in :class:`Section` order.
        dtypes: Tensor count per dtype (``str(torch.dtype)``).
        chunk_counts: Tensor count per number of stored chunks.
        largest_chunk: The stored chunk with the largest decoded size, if any.
    """

    tensors: int
    byte_items: int
    tensor_bytes: int
    sections: dict[Section, SectionTotals]
    dtypes: dict[str, int]
    chunk_counts: dict[int, int]
    largest_chunk: StoredChunk | None


def take_census(view: MetadataView, fmt: str) -> Census:
    """Count keys, bytes, dtypes and chunk grids of ``view``."""
    keys: Counter[Section] = Counter()
    nbytes: Counter[Section] = Counter()
    for key, entry in view.tensors.items():
        section = classify_key(fmt, key, is_bytes=False)
        keys[section] += 1
        nbytes[section] += entry.nbytes
    for key in view.byte_items:
        keys[classify_key(fmt, key, is_bytes=True)] += 1
    chunks = [c for entry in view.tensors.values() for c in entry.chunks]
    return Census(
        tensors=len(view.tensors),
        byte_items=len(view.byte_items),
        tensor_bytes=sum(e.nbytes for e in view.tensors.values()),
        sections={s: SectionTotals(keys[s], nbytes[s]) for s in Section if keys[s]},
        dtypes=dict(sorted(Counter(str(e.dtype) for e in view.tensors.values()).items())),
        chunk_counts=dict(sorted(Counter(len(e.chunks) for e in view.tensors.values()).items())),
        largest_chunk=max(chunks, key=lambda c: c.nbytes, default=None),
    )
