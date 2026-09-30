# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shortcut-MoE hybrid decoders across formats, on synthetic stores.

The modules of the layer list are drawn one by one under their layer-list
(dense) index, as fsdp_dtensor stores them; the torch_dist source renames each
to its first layer's pattern position, written out by hand, and stacks the
experts, so the expected tensors of either format are the drawn ones.
"""

from __future__ import annotations

import pytest
import torch

from tools.checkpoint.format_converter.api import convert_checkpoint, verify_checkpoint
from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.options import ConvertOptions, VerifyOptions

from ._synthetic import (
    expert_grids,
    hybrid_args,
    renumber_layers,
    shortcut_hybrid_tensors,
    stack_experts,
    write_fsdp_store,
)
from .test_convert_blocks import (
    ITER,
    expect_fsdp_store,
    grid_of,
    read_common,
    run,
    write_gpt_source,
)
from .test_convert_cross import bitwise_equal, read_items, read_tensors

pytestmark = pytest.mark.covers("identity", "ground_truth")
PATTERN = "E*E-*EE"
MODULES = ("E", "*E", "-", "*E", "E")
"""The pattern's modules: a leading E and an E after a pair stay modules of their own."""
DENSE_TO_TD = (0, 1, 3, 4, 6)
"""The torch_dist index of each module: its first layer's position in the pattern."""
EXPERTS = 4
FSDP = shortcut_hybrid_tensors(MODULES, experts=EXPERTS)
"""The model as fsdp_dtensor stores it (without the prefix): one key per module and expert."""
TD = stack_experts(renumber_layers(FSDP, "decoder", DENSE_TO_TD))
"""The model as torch_dist stores it: modules at their pattern positions, ``[E, ...]`` experts."""


def shortcut_args(pattern: str = PATTERN, **overrides):
    """Arguments of the shortcut-MoE hybrid model (one pipeline stage unless overridden)."""
    fields = {"moe_shortcut_connection": True, "num_experts": EXPERTS}
    return hybrid_args(pattern, **{"pipeline_model_parallel_size": 1, **fields, **overrides})


def to_fsdp(tmp_path, src, name="fsdp"):
    report = convert_checkpoint(
        src, tmp_path / name, ConvertOptions(to_format="fsdp_dtensor", workers=0)
    )
    return tmp_path / name / ITER, report.to_json()


def to_torch_dist(tmp_path, store, name="td"):
    report = convert_checkpoint(store, tmp_path / name / ITER, ConvertOptions(workers=0))
    return tmp_path / name / ITER, report.to_json()


def test_the_torch_dist_indices_skip_the_moe_layers_of_the_pairs():
    assert "decoder.layers.3.mlp.linear_fc1.weight" in TD  # the dense MLP, module 2
    assert "decoder.layers.2.mlp.linear_fc1.weight" in FSDP
    assert "decoder.layers.6.mlp.experts.experts.linear_fc2.weight" in TD  # the last E, module 4
    assert "decoder.layers.4.mlp.experts.linear_fc2.weight3" in FSDP
    assert not any(key.startswith(("decoder.layers.2.", "decoder.layers.5.")) for key in TD)


def test_a_shortcut_store_converts_to_dense_module_indices(tmp_path):
    src = write_gpt_source(tmp_path / "src", TD, grids=expert_grids(TD), args=shortcut_args())
    out, report = to_fsdp(tmp_path, src)
    expect_fsdp_store(out, FSDP)
    derived = {d["field"]: d for d in report["derivations"]}
    assert derived["blocks.decoder"]["value"] == {
        "kind": "hybrid_stack",
        "num_modules": 5,
        "td_layout": "per_layer",
        "dense_to_td": list(DENSE_TO_TD),
        "symbols": list(MODULES),
    }
    assert report["claims"] == {"identity": len(FSDP)}
    back, _ = to_torch_dist(tmp_path, out)
    assert verify_checkpoint(back, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok


def test_only_an_fsdp_dtensor_target_is_warned_about(tmp_path, caplog):
    """No current Megatron job loads it: Megatron refuses --moe-shortcut-connection with
    Megatron-FSDP."""
    warning = (
        "args.moe_shortcut_connection=True: Megatron refuses --moe-shortcut-connection with "
        "Megatron-FSDP, so no current Megatron job loads this fsdp_dtensor store"
    )
    src = write_gpt_source(tmp_path / "src", TD, grids=expert_grids(TD), args=shortcut_args())
    assert run("plan", src, "--to", "fsdp_dtensor") == 0
    assert warning in caplog.text
    args = shortcut_args(ckpt_format="fsdp_dtensor", use_megatron_fsdp=True)
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, FSDP, args=args)
    caplog.clear()
    assert run("plan", store, "--to", "torch_dist") == 0
    assert "moe_shortcut_connection" not in caplog.text


def test_an_fsdp_shortcut_store_converts_to_the_module_starts(tmp_path):
    """A Megatron-FSDP-like save: fp32 main weights under the layer-list indices."""
    fp32 = {key: tensor.float() for key, tensor in FSDP.items()}
    args = shortcut_args(ckpt_format="fsdp_dtensor", use_megatron_fsdp=True)
    store = write_fsdp_store(tmp_path / "fsdp" / ITER, fp32, args=args)
    out, _ = to_torch_dist(tmp_path, store)
    expected = stack_experts(renumber_layers(fp32, "decoder", DENSE_TO_TD))
    tensors = read_tensors(out)
    assert set(tensors) == set(expected)
    for key, tensor in tensors.items():
        assert bitwise_equal(tensor, expected[key]), key
    assert grid_of(out, "decoder.layers.4.moe_layer.mlp.experts.experts.linear_fc1.weight") == [
        ((expert, 0, 0), (1, 6, 8)) for expert in range(EXPERTS)
    ]


def test_a_pipelined_shortcut_store_loses_its_stages_in_fsdp_dtensor(tmp_path):
    """Saved with two pipeline stages (pairs never cross one): the fsdp_dtensor store has one,
    and converting back restores the pattern's stage separator from the manifest."""
    args = shortcut_args("E*E|-*EE", pipeline_model_parallel_size=2)
    src = write_gpt_source(tmp_path / "src", TD, grids=expert_grids(TD), args=args)
    out, report = to_fsdp(tmp_path, src)
    expect_fsdp_store(out, FSDP)
    assert report["common"]["args_changes"]["hybrid_layer_pattern"] == ["E*E|-*EE", PATTERN]
    assert read_items(out)["args"].pipeline_model_parallel_size == 1
    back, _ = to_torch_dist(tmp_path, out)
    assert read_common(back)["args"].hybrid_layer_pattern == "E*E|-*EE"
    assert verify_checkpoint(back, src / ITER, VerifyOptions(expected_diff="D", layout=True)).ok


def test_the_children_of_a_pair_are_not_assumed(tmp_path):
    """Other child names and tensors of the pair's own (a post-norm) convert unchanged."""
    fsdp = shortcut_hybrid_tensors(MODULES, children=("paired", "routed"), experts=EXPERTS)
    for dense in (1, 3):
        fsdp[f"decoder.layers.{dense}.shortcut_post_norm.weight"] = torch.full((8,), dense * 0.5)
    source = stack_experts(renumber_layers(fsdp, "decoder", DENSE_TO_TD))
    assert "decoder.layers.4.shortcut_post_norm.weight" in source
    src = write_gpt_source(
        tmp_path / "src", source, grids=expert_grids(source), args=shortcut_args()
    )
    out, _ = to_fsdp(tmp_path, src)
    expect_fsdp_store(out, fsdp)
    back, _ = to_torch_dist(tmp_path, out)
    assert set(MetadataView.load(back).tensors) == set(source)
    assert verify_checkpoint(back, src / ITER, VerifyOptions(expected_diff="D")).ok


def test_a_store_numbered_one_index_per_layer_is_refused(tmp_path, capsys):
    """The layers of a model built without shortcut grouping, under shortcut arguments."""
    unpaired = stack_experts(shortcut_hybrid_tensors(tuple(PATTERN), experts=EXPERTS))
    src = write_gpt_source(tmp_path / "src", unpaired, args=shortcut_args())
    assert run("plan", src, "--to", "fsdp_dtensor") == 3
    err = capsys.readouterr().err
    assert (
        "error: FC-R11: decoder: hybrid layer pattern 'E*E-*EE' (7 layers) grouped into 5 "
        "modules by args.moe_shortcut_connection=True (each E joins the layer before it unless "
        "that layer is paired) means torch_dist layer indices 0-1, 3-4, 6, but the store has "
        "0-6 (missing none, unexpected "
        "[2, 5]); its modules (dense: torch_dist index, symbols) are 0: 0 E, 1: 1 *E, 2: 3 -, "
        "3: 4 *E, 4: 6 E"
    ) in err


def test_a_module_that_holds_other_layers_is_refused(tmp_path, capsys):
    """``E*E-*E-`` groups into modules at the same indices, but its last module is a dense MLP."""
    src = write_gpt_source(tmp_path / "src", TD, args=shortcut_args("E*E-*E-"))
    assert run("plan", src, "--to", "fsdp_dtensor") == 3
    err = capsys.readouterr().err
    assert (
        "error: FC-R11: decoder: the tensors of 1 of its 5 modules contradict hybrid layer "
        "pattern 'E*E-*E-'"
    ) in err
    assert "module 4 (torch_dist index 6, '-'): its tensors are those of 'E'" in err


def test_a_pattern_megatron_cannot_group_is_refused(tmp_path, capsys):
    src = write_gpt_source(tmp_path / "src", TD, args=shortcut_args("E*-E*EE"))
    assert run("plan", src, "--to", "fsdp_dtensor") == 3
    assert "layer 3 ('E') follows layer 2 ('-'), which cannot precede" in capsys.readouterr().err
