# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Source reader facade: validated region reads and raw byte-item reads.

Callers describe what they need as :class:`ReadReq` objects (a box of a tensor
and a destination buffer) and get :class:`ReadStats` back. Reads go through
the read-once DCP backend in :mod:`.reader_dcp`; every stored chunk a batch
touches is decoded once, and a chunk larger than the per-item memory budget is
refused before any data is read.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..errors import BudgetExceeded, InternalError
from . import reader_dcp
from .metadata import MetadataView
from .regions import Region
from .sizes import format_bytes


@dataclass(frozen=True, slots=True)
class ReadReq:
    """Read ``region`` (global coordinates) of tensor ``key`` into ``dst``.

    Attributes:
        key: Tensor key in the store.
        region: Box to read; must lie inside the tensor.
        dst: CPU tensor of shape ``region.sizes`` and the stored dtype (may be a view).
    """

    key: str
    region: Region
    dst: torch.Tensor


@dataclass(frozen=True, slots=True)
class ReadStats:
    """What a read cost.

    Attributes:
        bytes_read: On-disk bytes of every stored item decoded (counted per decode).
        items_touched: Distinct stored items the requests overlap.
        torch_load_calls: Whole-item ``torch.load`` calls made to serve the requests.
    """

    bytes_read: int = 0
    items_touched: int = 0
    torch_load_calls: int = 0

    @property
    def amplification(self) -> float:
        """Decodes per touched item; 1.0 means every stored item was read once."""
        if self.items_touched == 0:
            return 1.0
        return self.torch_load_calls / self.items_touched

    def __add__(self, other: ReadStats) -> ReadStats:
        return ReadStats(
            bytes_read=self.bytes_read + other.bytes_read,
            items_touched=self.items_touched + other.items_touched,
            torch_load_calls=self.torch_load_calls + other.torch_load_calls,
        )


class SourceReader:
    """Reads regions and byte items of one store under a per-item memory budget.

    Args:
        view: The store's metadata view.
        max_item_bytes: Largest stored chunk (decoded size) a read may touch.
    """

    def __init__(self, view: MetadataView, *, max_item_bytes: int) -> None:
        if max_item_bytes <= 0:
            raise InternalError(f"max_item_bytes must be positive, got {max_item_bytes}")
        self.view = view
        self.max_item_bytes = max_item_bytes

    def read_into(self, reqs: Sequence[ReadReq]) -> ReadStats:
        """Fill every request's destination with one ``dcp.load``.

        Raises:
            BudgetExceeded: A touched stored chunk is larger than ``max_item_bytes``.
            InternalError: A request is malformed (unknown key, box outside the
                tensor, destination of the wrong shape, dtype or device).
        """
        for req in reqs:
            self._validate(req)
        plan = reader_dcp.plan_region_load(self.view, reqs)
        for chunk in plan.touched:
            self._check_budget(chunk.key, chunk.offsets, chunk.nbytes)
        reads = reader_dcp.execute_region_load(self.view, plan)
        lengths = {(c.key, c.offsets): c.length for c in plan.touched}
        return ReadStats(
            bytes_read=sum(lengths[source] for source in reads),
            items_touched=len(plan.touched),
            torch_load_calls=len(reads),
        )

    def read_bytes(self, keys: Sequence[str]) -> dict[str, bytes]:
        """Raw payloads of byte items ``keys`` (never unpickled)."""
        unknown = [k for k in keys if k not in self.view.byte_items]
        if unknown:
            raise InternalError(f"{self.view.iter_dir}: no byte items {unknown[:8]}")
        return reader_dcp.load_byte_items(self.view, keys)

    def _validate(self, req: ReadReq) -> None:
        entry = self.view.tensors.get(req.key)
        if entry is None:
            raise InternalError(f"{self.view.iter_dir}: no tensor {req.key!r}")
        if req.region.ndim != len(entry.shape) or not Region.full(entry.shape).contains(req.region):
            raise InternalError(f"{req.key}: {req.region} is outside shape {entry.shape}")
        if tuple(req.dst.shape) != req.region.sizes:
            raise InternalError(f"{req.key}: destination {tuple(req.dst.shape)} != {req.region}")
        if req.dst.dtype != entry.dtype or req.dst.device.type != "cpu":
            raise InternalError(
                f"{req.key}: destination must be a CPU {entry.dtype} tensor, "
                f"got {req.dst.device} {req.dst.dtype}"
            )

    def _check_budget(self, key: str, offsets: tuple[int, ...], nbytes: int) -> None:
        if nbytes > self.max_item_bytes:
            raise BudgetExceeded(
                f"stored chunk {offsets} of {key} decodes to {format_bytes(nbytes)}, "
                f"above the per-item budget of {format_bytes(self.max_item_bytes)}",
                hint=f"pass --max-item-bytes {nbytes} or more (host memory permitting)",
            )
