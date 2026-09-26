# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The torch_dist adapter: key sections, common state, markers and weights-only refusals."""

from __future__ import annotations

import inspect
import json

import pytest
import torch

from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.engine.reader import SourceReader
from tools.checkpoint.format_converter.errors import (
    ExitCode,
    IntegrityError,
    InternalError,
    RefusedError,
    UsageError,
)
from tools.checkpoint.format_converter.formats.base import (
    WHOLE,
    CommonLayout,
    FormatAdapter,
    Section,
)
from tools.checkpoint.format_converter.formats.sections import take_census
from tools.checkpoint.format_converter.formats.torch_dist import (
    TorchDistAdapter,
    classify_key,
    weights_only_common,
    weights_only_sections,
)

from .._synthetic import (
    TORCH_DIST_CONFIG,
    default_args,
    rewrite_metadata,
    torch_save_bytes,
    write_store,
    write_torch_dist_store,
)

ADAPTER = TorchDistAdapter()
COMMON = {"args": default_args(), "iteration": 7, "checkpoint_version": 3.0}


@pytest.mark.parametrize(
    "key, is_bytes, section",
    [
        ("decoder.layers.self_attention.linear_qkv.weight", False, Section.MODEL),
        ("embedding.word_embeddings.weight", False, Section.MODEL),
        (
            "decoder.layers.self_attention.linear_qkv._extra_state/shard_0_2",
            True,
            Section.EXTRA_STATE,
        ),
        (
            "decoder.layers.mlp.experts.experts.linear_fc1._extra_state3/shard_0_1",
            True,
            Section.EXTRA_STATE,
        ),
        ("decoder.layers.self_attention.core_attention._extra_state", False, Section.EXTRA_STATE),
        ("common_state/shard_0_1", True, Section.COMMON),
        ("common_state/shard_0_2", True, Section.OTHER),
        ("optimizer.state.exp_avg.decoder.final_layernorm.weight", False, Section.OPTIMIZER_STATE),
        (
            "chained_1.optimizer.state.param.decoder.final_layernorm.weight",
            False,
            Section.OPTIMIZER_STATE,
        ),
        (
            "optimizer.distributed.dp_group_idx_0.param_state/shard_0_2",
            True,
            Section.OPTIMIZER_OTHER,
        ),
        (
            "chained_0.optimizer.distributed.dp_group_idx_0.gbuf_idx_0.dtype_(torch.bfloat16, torch.bfloat16).bucket_idx_0.exp_avg",
            False,
            Section.OPTIMIZER_OTHER,
        ),
        ("rng_state/shard_0.0.1_1.1.2", True, Section.RNG),
        ("rerun_state_machine_state/shard_1_2", True, Section.RERUN),
        ("some_other_object/shard_0_1", True, Section.OTHER),
    ],
)
def test_classify(key, is_bytes, section):
    assert classify_key(key, is_bytes=is_bytes) is section
    assert ADAPTER.classify(key, is_bytes=is_bytes) is section


def _store(tmp_path, *, common=COMMON, legacy=None, extra_byte_items=None, tensors=None):
    store = write_torch_dist_store(
        tmp_path / "store",
        tensors or {"decoder.final_layernorm.weight": torch.ones(4)},
        common=common,
        extra_byte_items=extra_byte_items,
    )
    if legacy is not None:
        (store / "common.pt").write_bytes(torch_save_bytes(legacy))
    return SourceReader(MetadataView.load(store), max_item_bytes=2**20)


def test_common_state_is_read_from_the_dcp_item(tmp_path):
    raw = ADAPTER.read_common_raw(_store(tmp_path))
    assert (raw.source, raw.layout) == ("common_state/shard_0_1", CommonLayout.WRAPPED)
    assert raw.payloads == {WHOLE: torch_save_bytes([COMMON])}


def test_legacy_common_pt_wins(tmp_path):
    legacy = {**COMMON, "iteration": 99}
    raw = ADAPTER.read_common_raw(_store(tmp_path, legacy=legacy))
    assert (raw.source, raw.layout) == ("common.pt", CommonLayout.DICT)
    assert raw.payloads == {WHOLE: torch_save_bytes(legacy)}


def test_store_without_common_state(tmp_path):
    write_store(tmp_path / "bare", {"w": torch.ones(2)})
    view = MetadataView.load(tmp_path / "bare")
    assert ADAPTER.read_common_raw(SourceReader(view, max_item_bytes=64)) is None


def test_common_item_and_markers(tmp_path):
    assert ADAPTER.common_item(WHOLE) == ("common_state/shard_0_1", None)
    with pytest.raises(InternalError, match="one payload, not 'args'"):
        ADAPTER.common_item("args")
    markers = ADAPTER.marker_files()
    assert json.loads(markers["metadata.json"]) == TORCH_DIST_CONFIG
    store = tmp_path / "published"
    store.mkdir()
    with pytest.raises(IntegrityError, match="cannot read .*metadata.json"):
        ADAPTER.validate_published(store)
    (store / "metadata.json").write_bytes(markers["metadata.json"])
    ADAPTER.validate_published(store)
    (store / "common.pt").write_bytes(b"")
    with pytest.raises(IntegrityError, match="holds a common.pt"):
        ADAPTER.validate_published(store)
    (store / "metadata.json").write_text(json.dumps({"sharded_backend": "zarr"}))
    with pytest.raises(IntegrityError, match="expected"):
        ADAPTER.validate_published(store)


def test_weights_only_common_keeps_model_facts_and_drops_training_state():
    common = {
        **COMMON,
        "num_floating_point_operations_so_far": 12.0,
        "content_metadata": {"singleton_local_shards": False, "chained_optim_avoid_prefix": True},
        "optimizer": {"param_groups": []},
        "opt_param_scheduler": {"num_steps": 3},
        "rerun_state_machine": {"mode": 0},
    }
    rewrite = weights_only_common(common)
    assert list(rewrite.kept) == [
        "args",
        "iteration",
        "checkpoint_version",
        "num_floating_point_operations_so_far",
        "content_metadata",
    ]
    assert rewrite.kept["content_metadata"] is common["content_metadata"]
    assert rewrite.dropped == ("opt_param_scheduler", "optimizer", "rerun_state_machine")
    without = weights_only_common(common, drop_keys=frozenset({"content_metadata"}))
    assert "content_metadata" not in without.kept and "content_metadata" in without.dropped


def _refusal(common, **kwargs):
    with pytest.raises(RefusedError) as excinfo:
        weights_only_common(common, **kwargs)
    assert excinfo.value.exit_code == ExitCode.REFUSED
    return excinfo.value.format()


def test_weights_only_common_refusals():
    assert _refusal({**COMMON, "modelopt_state": {}}).startswith("FC-R18: the checkpoint holds")
    singleton = {**COMMON, "content_metadata": {"singleton_local_shards": True}}
    assert _refusal(singleton).startswith("FC-R17: the checkpoint was saved with singleton")
    unknown = {**COMMON, "tokenizer_state": 1, "zzz": 2}
    line = _refusal(unknown)
    assert line.startswith("FC-R13: the common state holds keys this tool does not know")
    assert "[tokenizer_state, zzz]" in line and "--drop-common-key" in line
    rewrite = weights_only_common(unknown, drop_keys=frozenset({"tokenizer_state", "zzz"}))
    assert rewrite.dropped == ("tokenizer_state", "zzz") and "zzz" not in rewrite.kept
    with pytest.raises(UsageError, match="cannot drop args, iteration"):
        weights_only_common(COMMON, drop_keys=frozenset({"args", "iteration"}))


def test_weights_only_sections_census_and_refusals(tmp_path):
    reader = _store(
        tmp_path,
        tensors={
            "decoder.final_layernorm.weight": torch.ones(4),
            "optimizer.state.exp_avg.decoder.final_layernorm.weight": torch.zeros(4),
        },
        extra_byte_items={
            "rng_state/shard_0_0_1": b"r",
            "rerun_state_machine_state/shard_0_1": b"s",
            "decoder.layers.self_attention._extra_state/shard_0_1": b"e",
        },
    )
    sections = weights_only_sections(reader.view)
    assert sections["decoder.final_layernorm.weight"] is Section.MODEL
    assert sections["rerun_state_machine_state/shard_0_1"] is Section.RERUN
    census = take_census(reader.view, "torch_dist")
    assert {s: t.keys for s, t in census.sections.items()} == {
        Section.MODEL: 1,
        Section.OPTIMIZER_STATE: 1,
        Section.COMMON: 1,
        Section.RNG: 1,
        Section.RERUN: 1,
        Section.EXTRA_STATE: 1,
    }
    unknown = _store(tmp_path / "u", extra_byte_items={"mystery/shard_0_1": b"?"})
    with pytest.raises(RefusedError, match=r"FC-R13: .*\[mystery/shard_0_1\]"):
        weights_only_sections(unknown.view)


def test_reformulated_model_keys_are_refused_with_fc_r14(tmp_path):
    tensors = {
        "decoder.final_layernorm.weight": torch.ones(4),
        "optimizer.state.fp32_param.decoder.final_layernorm.weight": torch.zeros(4),
    }
    reader = _store(tmp_path, tensors=tensors)

    def mark(keys):
        def mutate(metadata):
            metadata.mcore_data = {
                key: {"nd_reformulated_orig_global_shape": (2, 2)} for key in keys
            }

        return mutate

    rewrite_metadata(
        reader.view.iter_dir, mark(["optimizer.state.fp32_param.decoder.final_layernorm.weight"])
    )
    weights_only_sections(MetadataView.load(reader.view.iter_dir))
    rewrite_metadata(reader.view.iter_dir, mark(["decoder.final_layernorm.weight"]))
    with pytest.raises(RefusedError) as excinfo:
        weights_only_sections(MetadataView.load(reader.view.iter_dir))
    assert excinfo.value.format().startswith(
        "FC-R14: model tensors are stored in the legacy N-D reformulated layout (mcore_data) "
        "[decoder.final_layernorm.weight]"
    )


def test_the_adapter_implements_the_protocol():
    def parameters(function):
        return [(p.name, p.kind) for p in inspect.signature(function).parameters.values()]

    methods = [name for name, member in vars(FormatAdapter).items() if inspect.isfunction(member)]
    assert set(methods) >= {"classify", "read_common_raw", "common_item", "marker_files"}
    for name in methods:
        if not name.startswith("_"):
            assert parameters(getattr(TorchDistAdapter, name)) == parameters(
                getattr(FormatAdapter, name)
            ), name
    assert (ADAPTER.fmt, ADAPTER.dialect, ADAPTER.common_layout) == (
        "torch_dist",
        "mcore",
        CommonLayout.WRAPPED,
    )
