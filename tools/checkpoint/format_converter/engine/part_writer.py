# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Writing pieces into a DCP part store.

:class:`PieceSavePlanner` is a ``SavePlanner`` for ``dcp.save({}, ...)`` that
ignores the (empty) state dict and writes exactly the given pieces, each as one
stored chunk, plus the given byte items. Unlike ``DefaultSavePlanner`` it
accepts partial tensors, so one output tensor can be spread over several part
stores; the metadata merge checks that the parts cover every tensor exactly.

Every piece buffer must be a standalone, contiguous CPU tensor whose storage is
exactly its size: ``FileSystemWriter`` clones any tensor whose storage is larger
than the tensor itself, and serializes the buffer it is given otherwise, so a
standalone buffer is written without a copy.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

import torch
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner, WriteItem

from ..errors import InternalError
from . import torch_compat
from .pieces import BytePiece, Piece


class PieceSavePlanner(SavePlanner):
    """Save planner that writes given piece buffers and byte payloads as they are.

    Args:
        pieces: Each piece with its buffer (standalone CPU tensor of the piece's
            shape and dtype).
        byte_pieces: Byte items to write.

    Raises:
        InternalError: A buffer does not match its piece or is not standalone,
            a chunk or byte item is given twice, or a key is both a tensor and a
            byte item.
    """

    def __init__(
        self, pieces: Sequence[tuple[Piece, torch.Tensor]], byte_pieces: Sequence[BytePiece]
    ) -> None:
        self._tensors: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}
        self._payloads: dict[str, bytes] = {}
        self._items: list[WriteItem] = []
        self._planner_paths: dict[str, tuple[str, ...]] = {}
        for piece, buffer in pieces:
            self._add_piece(piece, buffer)
        tensor_keys = {key for key, _ in self._tensors}
        for byte_piece in byte_pieces:
            if byte_piece.key in self._payloads or byte_piece.key in tensor_keys:
                raise InternalError(f"byte item {byte_piece.key} is written twice")
            self._payloads[byte_piece.key] = byte_piece.payload
            self._items.append(torch_compat.bytes_write_item(byte_piece.key))
            self._record_planner_path(byte_piece.key, byte_piece.planner_path)

    def set_up_planner(
        self, state_dict: Any, storage_meta: Any = None, is_coordinator: bool = False
    ) -> None:
        """Ignore the state dict: the pieces are the whole save."""
        del state_dict, storage_meta, is_coordinator

    def create_local_plan(self) -> SavePlan:
        """One write item per piece and per byte item, in the given order."""
        return SavePlan(list(self._items))

    def create_global_plan(self, all_plans: list[SavePlan]) -> tuple[list[SavePlan], Metadata]:
        """The plans and metadata of the save; partial tensors are allowed."""
        plans, metadata = torch_compat.global_save_plan(all_plans)
        metadata.planner_data = dict(self._planner_paths) or None
        return plans, metadata

    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        """Keep the plan as it is."""
        return new_plan

    def resolve_data(self, write_item: WriteItem) -> torch.Tensor | io.BytesIO:
        """The piece buffer itself (never a copy) or the byte payload of an item."""
        key, offsets = torch_compat.write_item_target(write_item)
        if offsets is None:
            return io.BytesIO(self._payloads[key])
        return self._tensors[(key, offsets)]

    def _add_piece(self, piece: Piece, buffer: torch.Tensor) -> None:
        _check_buffer(piece, buffer)
        slot = (piece.key, piece.chunk.offsets)
        if slot in self._tensors:
            raise InternalError(f"chunk {piece.chunk} of {piece.key} is written twice")
        self._tensors[slot] = buffer
        self._items.append(
            torch_compat.tensor_write_item(piece.key, piece.chunk, piece.global_shape, piece.dtype)
        )
        self._record_planner_path(piece.key, piece.planner_path)

    def _record_planner_path(self, key: str, path: tuple[str, ...] | None) -> None:
        if path is None:
            return
        known = self._planner_paths.setdefault(key, path)
        if known != path:
            raise InternalError(f"{key} has two planner paths, {known} and {path}")


def _check_buffer(piece: Piece, buffer: torch.Tensor) -> None:
    if buffer.device.type != "cpu":
        raise InternalError(f"buffer of {piece.key} {piece.chunk} is on {buffer.device}, not cpu")
    if tuple(buffer.shape) != piece.chunk.sizes or buffer.dtype != piece.dtype:
        raise InternalError(
            f"buffer of {piece.key} {piece.chunk} is {tuple(buffer.shape)} {buffer.dtype}, "
            f"expected {piece.chunk.sizes} {piece.dtype}"
        )
    standalone = (
        buffer.is_contiguous()
        and buffer.storage_offset() == 0
        and buffer.untyped_storage().nbytes() == buffer.nbytes
    )
    if not standalone:
        raise InternalError(
            f"buffer of {piece.key} {piece.chunk} does not own exactly its contiguous storage "
            "(the writer would copy a view)"
        )
