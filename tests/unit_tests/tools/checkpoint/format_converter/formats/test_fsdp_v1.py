# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The Megatron-FSDP v1 fsdp_dtensor adapter: sections, envelope, chunk grid, common state."""

from __future__ import annotations

import pytest
import torch

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.logical import OutputSlice
from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.engine.reader import SourceReader
from tools.checkpoint.format_converter.engine.regions import Region
from tools.checkpoint.format_converter.errors import IntegrityError, InternalError, RefusedError
from tools.checkpoint.format_converter.formats.base import (
    WHOLE,
    CommonLayout,
    Section,
    model_tensors,
)
from tools.checkpoint.format_converter.formats.fsdp_mfsdp_v1 import FsdpMfsdpV1Adapter, classify_key
from tools.checkpoint.format_converter.formats.torch_dist import TorchDistAdapter

from .._synthetic import torch_save_bytes, write_fsdp_store

ADAPTER = FsdpMfsdpV1Adapter()


@pytest.mark.parametrize(
    ("key", "is_bytes", "section"),
    [
        ("model.module.decoder.layers.0.mlp.linear_fc1.weight", False, Section.MODEL),
        ("model.module.decoder.layers.0.mlp.linear_fc1.weight", True, Section.OTHER),
        ("optimizer.state.module.module.module.decoder.x.exp_avg", False, Section.OPTIMIZER_STATE),
        ("optimizer.param_to_group_meta.module.module.decoder.x.lr", True, Section.OPTIMIZER_OTHER),
        ("opt_param_scheduler.num_steps", True, Section.SCHEDULER),
        ("rng_state.(0, 0).0.random_rng_state", False, Section.RNG),
        ("rerun_state_machine", True, Section.RERUN),
        ("args", True, Section.COMMON),
        ("iteration", True, Section.COMMON),
        ("checkpoint_version", True, Section.COMMON),
        ("num_floating_point_operations_so_far", True, Section.COMMON),
        (
            "model.module.decoder.layers.0.self_attention.core_attention._extra_state",
            False,
            Section.EXTRA_STATE,
        ),
        ("content_metadata", True, Section.OTHER),
        ("decoder.final_norm.weight", False, Section.OTHER),
    ],
)
def test_classify(key, is_bytes, section):
    assert classify_key(key, is_bytes=is_bytes) is section
    assert ADAPTER.classify(key, is_bytes=is_bytes) is section


def test_envelope_adds_and_strips_exactly_one_module_level():
    key = ADAPTER.stored_key("decoder.layers.1.mlp.linear_fc2.weight")
    assert key == "model.module.decoder.layers.1.mlp.linear_fc2.weight"
    assert ADAPTER.relative_key(key) == "decoder.layers.1.mlp.linear_fc2.weight"
    assert ADAPTER.planner_path(key) == ("model", "module.decoder.layers.1.mlp.linear_fc2.weight")
    with pytest.raises(RefusedError) as excinfo:
        ADAPTER.relative_key("model.module.module.decoder.final_norm.weight")
    assert excinfo.value.refusal.code == errors.FC_R13
    assert excinfo.value.refusal.keys == ("model.module.module.decoder.final_norm.weight",)
    with pytest.raises(InternalError):
        ADAPTER.relative_key("optimizer.state.x")


def test_torch_dist_keys_are_bare():
    adapter = TorchDistAdapter()
    assert adapter.stored_key("decoder.final_norm.weight") == "decoder.final_norm.weight"
    assert adapter.relative_key("output_layer.weight") == "output_layer.weight"
    assert adapter.planner_path("output_layer.weight") is None
    whole = OutputSlice("w", (6, 4), Region.full((6, 4)))
    assert adapter.output_grid(whole, torch.float32, max_chunk_bytes=8) == (Region.full((6, 4)),)
    layer = OutputSlice("s", (3, 6, 4), Region((1, 0, 0), (1, 6, 4)), squeezed=1)
    assert adapter.output_grid(layer, torch.float32, max_chunk_bytes=8) == (layer.region,)


@pytest.mark.parametrize(
    ("shape", "limit", "grid"),
    [
        ((10, 4), 48, [(0, 3), (3, 3), (6, 3), (9, 1)]),  # 16-byte fp32 rows, 3 per chunk
        ((10, 4), 2**30, [(0, 10)]),
        ((10, 4), 8, [(row, 1) for row in range(10)]),  # a row above the limit is one chunk
        ((0, 4), 8, [(0, 0)]),
    ],
)
def test_output_grid_splits_dimension_zero(shape, limit, grid):
    placed = OutputSlice("w", shape, Region.full(shape))
    regions = ADAPTER.output_grid(placed, torch.float32, max_chunk_bytes=limit)
    assert [(r.offsets[0], r.sizes[0]) for r in regions] == grid
    assert all(r.offsets[1:] == (0,) and r.sizes[1:] == shape[1:] for r in regions)


def test_zero_dimensional_tensors_are_one_chunk():
    placed = OutputSlice("s", (), Region((), ()))
    assert ADAPTER.output_grid(placed, torch.bfloat16, max_chunk_bytes=1) == (Region((), ()),)


@pytest.fixture(name="store")
def fixture_store(tmp_path):
    return write_fsdp_store(
        tmp_path / "iter_0000007",
        {
            "decoder.final_norm.weight": torch.ones(4),
            "embedding.word_embeddings.weight": torch.ones(8, 4),
        },
        extra_byte_items={
            "num_floating_point_operations_so_far": torch_save_bytes(2.5),
            "opt_param_scheduler.num_steps": torch_save_bytes(3),
        },
    )


def test_sections_common_state_and_model_tensors(store):
    view = MetadataView.load(store)
    sections = ADAPTER.sections(view)
    assert sections["opt_param_scheduler.num_steps"] is Section.SCHEDULER
    tensors = model_tensors(view, sections, ADAPTER)
    assert list(tensors) == ["decoder.final_norm.weight", "embedding.word_embeddings.weight"]
    assert tensors["decoder.final_norm.weight"].key == "model.module.decoder.final_norm.weight"
    raw = ADAPTER.read_common_raw(SourceReader(view, max_item_bytes=2**20))
    assert raw.layout is CommonLayout.PER_KEY
    assert sorted(raw.payloads) == [
        "args",
        "checkpoint_version",
        "iteration",
        "num_floating_point_operations_so_far",
    ]
    assert ADAPTER.common_item("iteration") == ("iteration", ("iteration",))
    with pytest.raises(InternalError):
        ADAPTER.common_item(WHOLE)


def test_unknown_items_are_refused(tmp_path):
    store = write_fsdp_store(
        tmp_path / "s",
        {"decoder.final_norm.weight": torch.ones(4)},
        extra_byte_items={"custom_blob": b"x", "model.module.opaque": b"y"},
    )
    with pytest.raises(RefusedError) as excinfo:
        ADAPTER.sections(MetadataView.load(store))
    assert excinfo.value.refusal.code == errors.FC_R13
    assert excinfo.value.refusal.keys == ("custom_blob", "model.module.opaque")


def test_published_stores_have_no_torch_dist_markers(tmp_path):
    ADAPTER.validate_published(tmp_path)
    assert ADAPTER.marker_files() == {}
    (tmp_path / "metadata.json").write_text("{}")
    with pytest.raises(IntegrityError, match="metadata.json"):
        ADAPTER.validate_published(tmp_path)
