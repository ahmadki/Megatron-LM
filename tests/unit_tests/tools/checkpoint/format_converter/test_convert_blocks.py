# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TransformerBlock layouts (stacked and per layer) across formats, on synthetic GPT stores.

The per-layer tensors are drawn first and stacked with ``torch.stack`` to build
the torch_dist source, so the expected fsdp_dtensor tensors are the drawn ones;
keys, chunk grids and derivations are written out by hand.
"""

from __future__ import annotations

import json

import pytest
import torch

from tools.checkpoint.format_converter import cli
from tools.checkpoint.format_converter.api import convert_checkpoint, verify_checkpoint
from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.options import ConvertOptions, VerifyOptions

from ._synthetic import (
    TORCH_DIST_CONFIG,
    gpt_args,
    gpt_tensors,
    layer_grids,
    product_grid,
    stack_layers,
    torch_save_bytes,
    write_fsdp_store,
    write_store,
    write_tracker,
)
from .test_convert_cross import bitwise_equal, read_items, read_tensors

pytestmark = pytest.mark.covers("identity", "ground_truth")
ITER = "iter_0000007"
LAYERS = 3
PER_LAYER = gpt_tensors(LAYERS)
"""The GPT model one layer per key (what fsdp_dtensor stores, without the prefix)."""
STACKED = stack_layers(PER_LAYER, "decoder")
QKV = "decoder.layers.self_attention.linear_qkv.weight"


def write_gpt_source(root, model, *, grids=None, args=None, content_metadata=None):
    """A torch_dist GPT training save (common state, tracker) of ``model``."""
    common = {
        "args": args if args is not None else gpt_args(LAYERS),
        "iteration": 7,
        "checkpoint_version": 3.0,
    }
    if content_metadata is not None:
        common["content_metadata"] = content_metadata
    store = root / ITER
    write_store(
        store, model, grids=grids, byte_items={"common_state/shard_0_1": torch_save_bytes([common])}
    )
    (store / "metadata.json").write_text(json.dumps(TORCH_DIST_CONFIG))
    write_tracker(root, 7)
    return root


def stacked_grids():
    """One chunk per layer, and the attention projection split in two along its rows."""
    grids = layer_grids(STACKED, "decoder")
    grids[QKV] = product_grid((LAYERS, 24, 8), [[1, 2], [12], []])
    return grids


def run(*argv) -> int:
    return cli.main([str(a) for a in argv])


def expect_fsdp_store(out, expected):
    """``out`` holds exactly ``expected`` under ``model.module.``, bit for bit with its dtypes."""
    view = MetadataView.load(out)
    assert set(view.tensors) == {f"model.module.{key}" for key in expected}
    for key, tensor in read_tensors(out).items():
        assert bitwise_equal(tensor, expected[key.removeprefix("model.module.")]), key
    assert view.planner_data == {
        **{f"model.module.{key}": ("model", f"module.{key}") for key in expected},
        **{item: (item,) for item in view.byte_items},
    }


def test_a_stacked_gpt_store_converts_to_one_layer_per_key(tmp_path):
    src = write_gpt_source(tmp_path / "src", STACKED, grids=stacked_grids())
    report = convert_checkpoint(
        src, tmp_path / "fsdp", ConvertOptions(to_format="fsdp_dtensor", workers=0)
    ).to_json()
    expect_fsdp_store(tmp_path / "fsdp" / ITER, PER_LAYER)
    derived = {d["field"]: d for d in report["derivations"]}
    assert derived["blocks.decoder"]["value"] == {
        "kind": "transformer_block",
        "num_modules": LAYERS,
        "td_layout": "stacked",
        "dense_to_td": [0, 1, 2],
        "symbols": None,
    }
    assert report["claims"] == {"identity": len(PER_LAYER)}
    run_stats = report["run"]
    assert run_stats["torch_load_calls"] == run_stats["items_touched"]  # each chunk read once
    assert run_stats["items_touched"] == sum(len(g) for g in stacked_grids().values()) + 3
    replay = verify_checkpoint(src, tmp_path / "fsdp", VerifyOptions(map="convert"))
    assert replay.ok and replay.values_compared == len(PER_LAYER)


def test_a_per_layer_gpt_store_converts_to_the_same_store(tmp_path):
    """Saved per layer with three pipeline stages: its layer indices are global."""
    content = {"non_homogeneous_layers": True, "singleton_local_shards": False}
    args = gpt_args(LAYERS, pipeline_model_parallel_size=3)
    src = write_gpt_source(tmp_path / "src", PER_LAYER, args=args, content_metadata=content)
    report = convert_checkpoint(
        src, tmp_path / "fsdp", ConvertOptions(to_format="fsdp_dtensor", workers=0)
    ).to_json()
    expect_fsdp_store(tmp_path / "fsdp" / ITER, PER_LAYER)
    (block,) = [d for d in report["derivations"] if d["field"] == "blocks.decoder"]
    assert block["value"]["td_layout"] == "per_layer"
    assert block["evidence"] == "torch_dist stores 3 layers one per key"
    assert report["common"]["args_changes"]["pipeline_model_parallel_size"] == [3, 1]


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (
            {**STACKED, "decoder.layers.mlp.linear_fc2.bias": torch.zeros(2, 8)},
            "disagree on the number of layers",
        ),
        (
            {**STACKED, "decoder.layers.0.extra.weight": torch.zeros(2)},
            "holds both tensors stacked over its layers and tensors of single layers",
        ),
        (
            {k: v for k, v in PER_LAYER.items() if not k.startswith("decoder.layers.1.")},
            "decoder stores layers 0, 2, not 0-1",
        ),
    ],
)
def test_inconsistent_layouts_are_refused_before_reading(tmp_path, capsys, model, message):
    src = write_gpt_source(tmp_path / "src", model)
    assert run("plan", src, "--to", "fsdp_dtensor") == 3
    err = capsys.readouterr().err
    assert "error: FC-R12:" in err and message in err, err


def test_the_decoder_length_must_match_the_arguments(tmp_path, capsys):
    src = write_gpt_source(tmp_path / "src", STACKED, args=gpt_args(4))
    assert run("plan", src, "--to", "fsdp_dtensor") == 3
    assert "FC-R12: args.num_layers=4 but the store holds 3 layers of decoder" in (
        capsys.readouterr().err
    )


def read_common(store) -> dict:
    (common,) = read_items(store)["common_state/shard_0_1"]
    return common


def grid_of(store, key) -> list:
    return [(c.offsets, c.sizes) for c in MetadataView.load(store).tensors[key].chunks]


def test_a_stacked_gpt_store_round_trips_bit_for_bit(tmp_path):
    src = write_gpt_source(tmp_path / "src", STACKED, grids=stacked_grids())
    convert_checkpoint(src, tmp_path / "fsdp", ConvertOptions(to_format="fsdp_dtensor", workers=0))
    report = convert_checkpoint(tmp_path / "fsdp", tmp_path / "back", ConvertOptions(workers=0))
    out = tmp_path / "back" / ITER
    assert set(MetadataView.load(out).tensors) == set(STACKED)
    for key, tensor in read_tensors(out).items():
        assert bitwise_equal(tensor, STACKED[key]), key
    # Megatron's own grid: one chunk per layer (the row split of the source merged)
    assert grid_of(out, QKV) == [((layer, 0, 0), (1, 24, 8)) for layer in range(LAYERS)]
    fc2 = "decoder.layers.mlp.linear_fc2.bias"
    assert grid_of(out, fc2) == [((layer, 0), (1, 8)) for layer in range(LAYERS)]
    (block,) = [d for d in report.plan["derivations"] if d["field"] == "blocks.decoder"]
    assert block["value"]["td_layout"] == "stacked"
    assert verify_checkpoint(out, src / ITER, VerifyOptions(expected_diff="D")).ok


def test_a_per_layer_gpt_store_round_trips_per_layer(tmp_path):
    content = {"non_homogeneous_layers": True, "singleton_local_shards": False}
    src = write_gpt_source(tmp_path / "src", PER_LAYER, content_metadata=content)
    convert_checkpoint(src, tmp_path / "fsdp", ConvertOptions(to_format="fsdp_dtensor", workers=0))
    report = convert_checkpoint(tmp_path / "fsdp", tmp_path / "back", ConvertOptions(workers=0))
    out = tmp_path / "back" / ITER
    assert set(MetadataView.load(out).tensors) == set(PER_LAYER)
    assert read_common(out)["content_metadata"] == content
    derived = {d["field"]: d for d in report.plan["derivations"]}
    assert derived["td_content_metadata"]["value"] == content
    assert derived["blocks.decoder"]["evidence"] == (
        "Megatron builds it per layer (content_metadata.non_homogeneous_layers=True)"
    )
    assert verify_checkpoint(out, src / ITER, VerifyOptions(expected_diff="D")).ok


def test_an_fsdp_dtensor_training_save_converts_to_a_stacked_store(tmp_path):
    fp32 = {key: tensor.float() for key, tensor in PER_LAYER.items()}  # Megatron-FSDP main weights
    args = gpt_args(LAYERS, ckpt_format="fsdp_dtensor", use_megatron_fsdp=True)
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, fp32, args=args)
    convert_checkpoint(store, tmp_path / "td" / ITER, ConvertOptions(workers=0))
    out = tmp_path / "td" / ITER
    expected = stack_layers(fp32, "decoder")
    assert set(MetadataView.load(out).tensors) == set(expected)
    for key, tensor in read_tensors(out).items():
        assert bitwise_equal(tensor, expected[key]), key
    assert grid_of(out, QKV) == [((layer, 0, 0), (1, 24, 8)) for layer in range(LAYERS)]
    assert "content_metadata" not in read_common(out)


UNLOADABLE = (
    "--block-layout decoder=per_layer: a job built from these arguments and this "
    "content_metadata builds decoder stacked (no per-layer setting), so it cannot load this "
    "store strictly; --torch-dist-layout per-layer marks the per-layer blocks in "
    "content_metadata, so that Megatron loads them per layer"
)


def test_block_layout_asks_for_another_layout(tmp_path, capsys, caplog):
    args = gpt_args(LAYERS, ckpt_format="fsdp_dtensor", use_megatron_fsdp=True)
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, PER_LAYER, args=args)
    argv = ["convert", store, tmp_path / "td", "--to", "torch_dist", "--workers", "0"]
    assert run(*argv, "--block-layout", "decoder=per_layer") == 0
    assert set(MetadataView.load(tmp_path / "td" / ITER).tensors) == set(PER_LAYER)
    assert "content_metadata" not in read_common(tmp_path / "td" / ITER)  # it stays as it was
    assert UNLOADABLE in caplog.text
    assert run("plan", store, "--to", "torch_dist", "--block-layout", "encoder=stacked") == 2
    assert "encoder is no layer list of this model" in capsys.readouterr().err
    src = write_gpt_source(tmp_path / "src", STACKED)
    assert run("plan", src, "--to", "fsdp_dtensor", "--block-layout", "decoder=stacked") == 2
    assert "only fsdp_dtensor to torch_dist conversions take --block-layout" in (
        capsys.readouterr().err
    )


def test_layers_that_differ_are_not_stacked(tmp_path, capsys, caplog):
    model = {k: v for k, v in PER_LAYER.items() if k != "decoder.layers.2.mlp.linear_fc1.bias"}
    args = gpt_args(LAYERS, ckpt_format="fsdp_dtensor", use_megatron_fsdp=True)
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, model, args=args)
    assert run("plan", store, "--to", "torch_dist") == 3
    err = capsys.readouterr().err
    assert (
        "FC-R12: the arguments make Megatron stack decoder, but its layers differ: layer 0 " in err
    )
    assert "hint: pass --torch-dist-layout per-layer" in err
    assert run("plan", store, "--to", "torch_dist", "--torch-dist-layout", "per-layer") == 0
    assert run("plan", store, "--to", "torch_dist", "--block-layout", "decoder=per_layer") == 0
    assert UNLOADABLE in caplog.text


def test_the_layout_the_manifest_records_is_restored_or_refused(tmp_path, capsys, caplog):
    content = {"non_homogeneous_layers": True}
    src = write_gpt_source(tmp_path / "src", PER_LAYER, content_metadata=content)
    convert_checkpoint(src, tmp_path / "fsdp", ConvertOptions(to_format="fsdp_dtensor", workers=0))
    fsdp = tmp_path / "fsdp"
    dropped = ["--drop-common-key", "content_metadata"]
    assert run("plan", fsdp, "--to", "torch_dist", *dropped) == 4
    err = capsys.readouterr().err
    assert (
        "FC-A06: the source's conversion_manifest.json records that torch_dist stored decoder "
        "per_layer, but this conversion would store it stacked"
    ) in err
    assert (
        "hint: keep the content_metadata the manifest records or pass --torch-dist-layout "
        "per-layer (neither with --drop-common-key content_metadata) so that Megatron loads "
        "decoder per layer, or --ignore-manifest"
    ) in err
    assert run("plan", fsdp, "--to", "torch_dist", "--torch-dist-layout", "per-layer") == 0
    assert run("plan", fsdp, "--to", "torch_dist", *dropped, "--ignore-manifest") == 0
    assert run("plan", fsdp, "--to", "torch_dist") == 0
    assert "cannot load this store strictly" not in caplog.text
    assert (
        run("plan", fsdp, "--to", "torch_dist", *dropped, "--block-layout", "decoder=per_layer")
        == 0
    )
    assert UNLOADABLE in caplog.text  # without the content_metadata, Megatron stacks it


def test_torch_dist_layout_per_layer_marks_the_store(tmp_path, capsys):
    fp32 = {key: tensor.float() for key, tensor in PER_LAYER.items()}
    args = gpt_args(LAYERS, ckpt_format="fsdp_dtensor", use_megatron_fsdp=True)
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, fp32, args=args)
    argv = ["convert", store, tmp_path / "td", "--to", "torch_dist", "--workers", "0"]
    assert run(*argv, "--torch-dist-layout", "per-layer") == 0
    out = tmp_path / "td" / ITER
    assert set(MetadataView.load(out).tensors) == set(PER_LAYER)
    assert read_common(out)["content_metadata"] == {
        "non_homogeneous_layers": True,
        "singleton_local_shards": False,
        "chained_optim_avoid_prefix": True,
    }
    src = write_gpt_source(tmp_path / "src", STACKED)
    assert run("plan", src, "--to", "fsdp_dtensor", "--torch-dist-layout", "per-layer") == 2
    assert "take --torch-dist-layout per-layer" in capsys.readouterr().err


def test_a_stacked_store_comes_back_per_layer_when_asked(tmp_path):
    content = {"distrib_optim_sharding_type": "fully_reshardable", "singleton_local_shards": False}
    src = write_gpt_source(tmp_path / "src", STACKED, content_metadata=content)
    convert_checkpoint(src, tmp_path / "fsdp", ConvertOptions(to_format="fsdp_dtensor", workers=0))
    options = ConvertOptions(workers=0, torch_dist_layout="per-layer")
    convert_checkpoint(tmp_path / "fsdp", tmp_path / "back", options)  # the manifest says stacked
    out = tmp_path / "back" / ITER
    assert set(MetadataView.load(out).tensors) == set(PER_LAYER)
    for key, tensor in read_tensors(out).items():
        assert bitwise_equal(tensor, PER_LAYER[key]), key
    assert read_common(out)["content_metadata"] == {**content, "non_homogeneous_layers": True}
