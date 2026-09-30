# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Block census, HybridStack block layouts and their cross-check against the stored indices."""

from __future__ import annotations

import argparse

import pytest
import torch

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.blocks import (
    BlockCensus,
    BlockFacts,
    BlockLayout,
    classify_blocks,
    language_model_prefix,
    take_census,
)
from tools.checkpoint.format_converter.core.keys import KeyGrammarError
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, TORCH_DIST
from tools.checkpoint.format_converter.core.stacking import transformer_block
from tools.checkpoint.format_converter.engine.metadata import TensorEntry
from tools.checkpoint.format_converter.errors import InternalError, RefusedError


def _args(**fields) -> ArgsView:
    return ArgsView(argparse.Namespace(**fields))


def _entry(key: str, shape: tuple[int, ...] = (4,)) -> TensorEntry:
    return TensorEntry(key, shape, torch.bfloat16, ())


def _keys(*indices: int) -> dict[str, TensorEntry]:
    keys = ["embedding.word_embeddings.weight", "decoder.final_norm.weight"]
    keys += [f"decoder.layers.{i}.mlp.linear_fc1.weight" for i in indices]
    return {key: _entry(key) for key in keys}


MODULE_KEYS = {
    "E": ("mlp.router.weight", "pre_mlp_layernorm.weight"),
    "*E": (
        "attn_layer.self_attention.linear_qkv.weight",
        "moe_layer.mlp.router.weight",
        "shortcut_pre_mlp_layernorm.weight",
    ),
    "*": ("self_attention.linear_qkv.weight",),
    "-": ("mlp.linear_fc1.weight", "mlp.linear_fc2.weight"),
}
"""Tensors of a module of each kind (ShortcutMoEBlock children for a pair)."""


def _module_keys(stored: dict[int, str], prefix: str = "decoder") -> dict[str, TensorEntry]:
    """Model keys of the modules ``{stored index: symbols}`` of a hybrid decoder."""
    keys = [
        f"{prefix}.layers.{index}.{name}"
        for index, sym in stored.items()
        for name in MODULE_KEYS[sym]
    ]
    return {key: _entry(key) for key in keys}


def _mtp_keys(depth: int, stored: dict[int, str], prefix: str = "mtp") -> dict[str, TensorEntry]:
    """Model keys of MTP depth ``depth``: its own tensors and its inner layer list's modules."""
    layer = f"{prefix}.layers.{depth}"
    keys = {key: _entry(key) for key in (f"{layer}.eh_proj.weight", f"{layer}.enorm.weight")}
    return {**keys, **_module_keys(stored, f"{layer}.mtp_model_layer")}


def classify_blocks_of(args: ArgsView, model, fmt: str):
    return classify_blocks(BlockFacts(args, fmt, model), default=transformer_block)


def test_the_census_puts_each_key_in_its_innermost_block():
    model = {
        key: _entry(key)
        for key in (
            "language_model.decoder.layers.0.mlp.linear_fc1.weight",
            "language_model.decoder.layers.1.mlp.linear_fc1.weight",
            "language_model.decoder.final_layernorm.weight",
            "vision_model.decoder.layers.mlp.linear_fc1.weight",
            "mtp.layers.0.eh_proj.weight",
            "mtp.layers.0.mtp_model_layer.layers.2.mixer.norm.weight",
            "decoder.layers.01.malformed.weight",
        )
    }
    census = take_census(model)
    assert list(census) == [
        "language_model.decoder",
        "mtp",
        "mtp.layers.0.mtp_model_layer",
        "vision_model.decoder",
    ]
    assert census["language_model.decoder"].indices == (0, 1)
    assert census["vision_model.decoder"] == BlockCensus(
        "vision_model.decoder",
        {},
        {"mlp.linear_fc1.weight": model["vision_model.decoder.layers.mlp.linear_fc1.weight"]},
    )
    assert census["vision_model.decoder"].stacked_keys() == (
        "vision_model.decoder.layers.mlp.linear_fc1.weight",
    )
    assert census["mtp"].layers == {0: {"eh_proj.weight": model["mtp.layers.0.eh_proj.weight"]}}
    assert census["mtp.layers.0.mtp_model_layer"].indices == (2,)


@pytest.mark.parametrize("fmt", [TORCH_DIST, FSDP_DTENSOR])
def test_a_hybrid_decoder_is_one_per_layer_block(fmt):
    model = {**_module_keys(dict(enumerate("*-*-"))), **_mtp_keys(0, {0: "*", 1: "-"})}
    block, *_ = classify_blocks_of(_args(hybrid_layer_pattern="*-|*-/*-", num_layers=4), model, fmt)
    assert block.to_json() == {
        "prefix": "decoder",
        "kind": "hybrid_stack",
        "num_modules": 4,
        "td_layout": "per_layer",
        "dense_to_td": [0, 1, 2, 3],
        "symbols": ["*", "-", "*", "-"],
        "evidence": "hybrid layer pattern '*-|*-/*-' (4 layers)",
    }
    assert [block.dense_index(i, fmt) for i in range(4)] == [0, 1, 2, 3]
    assert block.stored_index(3, fmt) == 3


def test_models_without_a_pattern_have_no_hybrid_block():
    (block,) = classify_blocks_of(_args(hybrid_layer_pattern=None), _keys(0), TORCH_DIST)
    assert (block.prefix, block.kind) == ("decoder", "transformer_block")
    assert classify_blocks_of(_args(hybrid_layer_pattern=None), {}, TORCH_DIST) == ()


def test_a_hybrid_decoder_is_never_stacked():
    stacked = "decoder.layers.mlp.linear_fc2.weight"
    model = {**_keys(0, 1, 2, 3), stacked: _entry(stacked, (4, 4))}
    with pytest.raises(RefusedError) as excinfo:
        classify_blocks_of(_args(hybrid_layer_pattern="*-*-"), model, TORCH_DIST)
    assert excinfo.value.refusal.code == errors.FC_R12
    assert excinfo.value.refusal.keys == (stacked,)
    assert "hybrid decoder ('*-*-')" in excinfo.value.refusal.message


@pytest.mark.parametrize(
    ("stored", "message"),
    [
        ((0, 1, 2), "means torch_dist layer indices 0-3, but the store has 0-2 (missing [3]"),
        ((0, 1, 2, 3, 5), "but the store has 0-3, 5 (missing none, unexpected [5])"),
        ((0, 2, 3, 4), "the store has 0, 2-4 (missing [1], unexpected [4])"),
    ],
)
def test_stored_indices_must_match_the_pattern(stored, message):
    with pytest.raises(RefusedError) as excinfo:
        classify_blocks_of(_args(hybrid_layer_pattern="*-*-"), _keys(*stored), TORCH_DIST)
    assert excinfo.value.refusal.code == errors.FC_R11
    assert message in excinfo.value.refusal.message


def test_num_layers_must_match_the_pattern():
    with pytest.raises(RefusedError) as excinfo:
        classify_blocks_of(
            _args(hybrid_layer_pattern="*-*-", num_layers=6), _keys(0, 1, 2, 3), TORCH_DIST
        )
    assert excinfo.value.refusal.code == errors.FC_R11
    assert excinfo.value.refusal.message.startswith("args.num_layers=6 but hybrid layer pattern")


SHORTCUT = "E*E-*EE"
"""A shortcut-MoE hybrid pattern: a leading E, pairs, a dense MLP and an E after a pair."""
SHORTCUT_MODULES = {0: "E", 1: "*E", 3: "-", 4: "*E", 6: "E"}
"""Its modules by torch_dist index (the position of their first layer) and their symbols."""


def _shortcut_args(pattern: str = SHORTCUT, **fields) -> ArgsView:
    layers = len(pattern.replace("|", ""))
    return _args(
        **{
            "hybrid_layer_pattern": pattern,
            "num_layers": layers,
            "moe_shortcut_connection": True,
            **fields,
        }
    )


def test_a_shortcut_decoder_numbers_its_modules_by_their_first_layer():
    td = _module_keys(SHORTCUT_MODULES)
    fsdp = _module_keys(dict(enumerate(SHORTCUT_MODULES.values())))
    (block,) = classify_blocks_of(_shortcut_args(), td, TORCH_DIST)
    assert classify_blocks_of(_shortcut_args(), fsdp, FSDP_DTENSOR) == (block,)
    assert block.to_json() == {
        "prefix": "decoder",
        "kind": "hybrid_stack",
        "num_modules": 5,
        "td_layout": "per_layer",
        "dense_to_td": [0, 1, 3, 4, 6],
        "symbols": ["E", "*E", "-", "*E", "E"],
        "evidence": "hybrid layer pattern 'E*E-*EE' (7 layers) grouped into 5 modules by "
        "args.moe_shortcut_connection=True (each E joins the layer before it unless that layer "
        "is paired)",
    }
    assert [block.dense_index(i, TORCH_DIST) for i in (0, 1, 3, 4, 6)] == [0, 1, 2, 3, 4]
    assert [block.stored_index(i, TORCH_DIST) for i in range(5)] == [0, 1, 3, 4, 6]
    assert [block.dense_index(i, FSDP_DTENSOR) for i in range(5)] == [0, 1, 2, 3, 4]
    with pytest.raises(KeyGrammarError):
        block.dense_index(2, TORCH_DIST)  # an E of a pair has no index of its own


def test_a_vision_language_shortcut_decoder_is_renumbered_too():
    model = _module_keys(SHORTCUT_MODULES, "language_model.decoder")
    (block,) = classify_blocks_of(_shortcut_args(), model, TORCH_DIST)
    assert (block.prefix, block.dense_to_td) == ("language_model.decoder", (0, 1, 3, 4, 6))


@pytest.mark.parametrize(
    ("fmt", "stored", "message"),
    [
        (
            TORCH_DIST,
            dict(enumerate("E*E-*EE")),  # one index per layer, as without shortcut grouping
            "means torch_dist layer indices 0-1, 3-4, 6, but the store has 0-6 "
            "(missing none, unexpected [2, 5])",
        ),
        (
            TORCH_DIST,
            {0: "E", 1: "*E", 3: "-", 4: "*E"},
            "means torch_dist layer indices 0-1, 3-4, 6, but the store has 0-1, 3-4 "
            "(missing [6], unexpected none)",
        ),
        (
            FSDP_DTENSOR,
            SHORTCUT_MODULES,  # torch_dist numbering in an fsdp_dtensor store
            "means fsdp_dtensor layer indices 0-4, but the store has 0-1, 3-4, 6 "
            "(missing [2], unexpected [6])",
        ),
    ],
)
def test_stored_indices_must_be_the_module_starts(fmt, stored, message):
    with pytest.raises(RefusedError) as excinfo:
        classify_blocks_of(_shortcut_args(), _module_keys(stored), fmt)
    refusal = excinfo.value.refusal
    assert refusal.code == errors.FC_R11
    assert message in refusal.message
    assert refusal.message.endswith(
        "; its modules (dense: torch_dist index, symbols) are 0: 0 E, 1: 1 *E, 2: 3 -, 3: 4 *E, "
        "4: 6 E"
    )


def test_num_layers_counts_the_layers_of_a_shortcut_pattern():
    with pytest.raises(RefusedError) as excinfo:
        classify_blocks_of(_shortcut_args(num_layers=5), _module_keys(SHORTCUT_MODULES), TORCH_DIST)
    assert excinfo.value.refusal.code == errors.FC_R11
    assert excinfo.value.refusal.message == (
        "args.num_layers=5 but hybrid layer pattern 'E*E-*EE' (7 layers)"
    )


@pytest.mark.parametrize(
    ("pattern", "reason"), [("*-E", "follows layer 1 ('-')"), ("*|E*E", "begins a pipeline stage")]
)
def test_a_pattern_megatron_cannot_group_is_refused(pattern, reason):
    with pytest.raises(RefusedError) as excinfo:
        classify_blocks_of(_shortcut_args(pattern), {}, TORCH_DIST)
    assert excinfo.value.refusal.code == errors.FC_R11
    assert reason in excinfo.value.refusal.message


@pytest.mark.parametrize("fmt", [TORCH_DIST, FSDP_DTENSOR])
def test_each_module_must_hold_the_tensors_of_its_layers(fmt):
    """Here module 2 (a dense MLP by the pattern) holds a shortcut pair's tensors."""
    modules = dict(zip((0, 1, 3, 4, 6), ["E", "*E", "*E", "*E", "E"]))
    if fmt == FSDP_DTENSOR:
        modules = dict(enumerate(modules.values()))
    with pytest.raises(RefusedError) as excinfo:
        classify_blocks_of(_shortcut_args(), _module_keys(modules), fmt)
    refusal = excinfo.value.refusal
    stored = 3 if fmt == TORCH_DIST else 2
    assert refusal.code == errors.FC_R11
    assert refusal.message == (
        "decoder: the tensors of 1 of its 5 modules contradict hybrid layer pattern 'E*E-*EE' "
        "(7 layers) "
        "grouped into 5 modules by args.moe_shortcut_connection=True (each E joins the layer "
        f"before it unless that layer is paired): module 2 ({fmt} index {stored}, '-'): its "
        "tensors are those of '*' "
        "(self_attention.linear_qkv), 'E' (mlp.router.weight), but '-' means '-' "
        "(mlp.linear_fc1); its modules (dense: torch_dist index, symbols) are 0: 0 E, 1: 1 *E, "
        "2: 3 -, 3: 4 *E, 4: 6 E"
    )
    assert refusal.keys == tuple(
        sorted(f"decoder.layers.{stored}.{name}" for name in MODULE_KEYS["*E"])
    )


def test_the_layers_of_a_hybrid_decoder_without_shortcut_are_checked_too():
    swapped = _module_keys({0: "-", 1: "*", 2: "*", 3: "-"})
    with pytest.raises(RefusedError) as excinfo:
        classify_blocks_of(_args(hybrid_layer_pattern="*-*-", num_layers=4), swapped, TORCH_DIST)
    assert excinfo.value.refusal.code == errors.FC_R11
    assert "the tensors of 2 of its 4 modules contradict" in excinfo.value.refusal.message
    assert "module 0 (torch_dist index 0, '*'): its tensors are those of '-'" in (
        excinfo.value.refusal.message
    )


def _mtp_args(pattern: str, **fields) -> ArgsView:
    return _args(
        **{
            "hybrid_layer_pattern": pattern,
            "num_layers": len(pattern.split("/")[0]),
            "mtp_num_layers": pattern.count("/"),
            **fields,
        }
    )


def _by_prefix(blocks) -> dict[str, dict]:
    return {block.prefix: block.to_json() for block in blocks}


@pytest.mark.parametrize("fmt", [TORCH_DIST, FSDP_DTENSOR])
def test_each_hybrid_mtp_depth_holds_a_hybrid_layer_list(fmt):
    model = {**_module_keys({0: "*", 1: "-"}), **_mtp_keys(0, {0: "*"}), **_mtp_keys(1, {0: "*"})}
    blocks = _by_prefix(classify_blocks_of(_mtp_args("*-/*/*"), model, fmt))
    assert list(blocks) == [
        "decoder",
        "mtp",
        "mtp.layers.0.mtp_model_layer",
        "mtp.layers.1.mtp_model_layer",
    ]
    assert blocks["mtp"] == {
        "prefix": "mtp",
        "kind": "mtp_block",
        "num_modules": 2,
        "td_layout": "per_layer",
        "dense_to_td": [0, 1],
        "symbols": None,
        "evidence": "hybrid layer pattern '*-/*/*' (2 MTP depths)",
    }
    assert blocks["mtp.layers.1.mtp_model_layer"] == {
        "prefix": "mtp.layers.1.mtp_model_layer",
        "kind": "hybrid_mtp_inner",
        "num_modules": 1,
        "td_layout": "per_layer",
        "dense_to_td": [0],
        "symbols": ["*"],
        "evidence": "MTP depth pattern '*' (1 layer)",
    }


def test_a_repeated_shortcut_mtp_layer_is_one_layer_numbered_like_the_decoder():
    """``/*EE`` twice with ``--mtp-use-repeated-layer``: one MTP layer, whose inner modules are
    ``*E`` and ``E`` (torch_dist indices 0 and 2)."""
    args = _mtp_args("*E/*EE/*EE", moe_shortcut_connection=True, mtp_use_repeated_layer=True)
    td = {**_module_keys({0: "*E"}), **_mtp_keys(0, {0: "*E", 2: "E"})}
    fsdp = {**_module_keys({0: "*E"}), **_mtp_keys(0, {0: "*E", 1: "E"})}
    blocks = _by_prefix(classify_blocks_of(args, td, TORCH_DIST))
    assert _by_prefix(classify_blocks_of(args, fsdp, FSDP_DTENSOR)) == blocks
    assert blocks["mtp"]["num_modules"] == 1
    assert blocks["mtp"]["evidence"] == (
        "hybrid layer pattern '*E/*EE/*EE' (2 MTP depths) with "
        "args.mtp_use_repeated_layer=True (one layer serves every depth)"
    )
    inner = blocks["mtp.layers.0.mtp_model_layer"]
    assert (inner["dense_to_td"], inner["symbols"]) == ([0, 2], ["*E", "E"])
    assert inner["evidence"] == (
        "MTP depth pattern '*EE' (3 layers) grouped into 2 modules by "
        "args.moe_shortcut_connection=True (each E joins the layer before it unless that layer "
        "is paired)"
    )


def test_a_vision_language_models_mtp_is_under_the_language_model():
    model = {
        **_module_keys({0: "*", 1: "-"}, "language_model.decoder"),
        **_mtp_keys(0, {0: "-"}, "language_model.mtp"),
    }
    blocks = _by_prefix(classify_blocks_of(_mtp_args("*-/-"), model, TORCH_DIST))
    assert blocks["language_model.mtp"]["kind"] == "mtp_block"
    assert blocks["language_model.mtp.layers.0.mtp_model_layer"]["kind"] == "hybrid_mtp_inner"


def test_gpt_mtp_depths_are_one_layer_each():
    model = {
        key: _entry(key)
        for key in (
            "mtp.layers.0.transformer_layer.mlp.linear_fc1.weight",
            "mtp.layers.1.eh_proj.weight",
        )
    }
    (block,) = classify_blocks_of(
        _args(hybrid_layer_pattern=None, mtp_num_layers=2), model, TORCH_DIST
    )
    assert (block.prefix, block.kind, block.dense_to_td) == ("mtp", "mtp_block", (0, 1))
    assert block.evidence == "args.mtp_num_layers=2"


@pytest.mark.parametrize(
    ("args", "model", "code", "message"),
    [
        (
            _mtp_args("*-/*", mtp_num_layers=2),
            {**_module_keys({0: "*", 1: "-"}), **_mtp_keys(0, {0: "*"})},
            errors.FC_R11,
            "args.mtp_num_layers=2 but hybrid layer pattern '*-/*' (1 MTP depth)",
        ),
        (
            _args(hybrid_layer_pattern="*-", num_layers=2),
            {**_module_keys({0: "*", 1: "-"}), **_mtp_keys(0, {0: "*"})},
            errors.FC_R11,
            "the store holds multi-token prediction layers (mtp.layers.*), but hybrid layer "
            "pattern '*-' (0 MTP depths)",
        ),
        (
            _mtp_args("*-/*"),
            {**_module_keys({0: "*", 1: "-"}), **_mtp_keys(0, {0: "*"}), **_mtp_keys(1, {0: "*"})},
            errors.FC_R11,
            "mtp: hybrid layer pattern '*-/*' (1 MTP depth) means torch_dist layer indices 0, "
            "but the store has 0-1",
        ),
        (
            _mtp_args("*-/*-"),
            {**_module_keys({0: "*", 1: "-"}), **_mtp_keys(0, {0: "*", 2: "-"})},
            errors.FC_R11,
            "mtp.layers.0.mtp_model_layer: MTP depth pattern '*-' (2 layers) means torch_dist "
            "layer indices 0-1, but the store has 0, 2 (missing [1], unexpected [2]); its modules "
            "(dense: torch_dist index, symbols) are 0: 0 *, 1: 1 -",
        ),
        (
            _mtp_args("*-/*"),
            {
                **_module_keys({0: "*", 1: "-"}),
                "mtp.layers.0.mtp_model_layer.layers.self_attention.linear_qkv.weight": _entry(
                    "mtp.layers.0.mtp_model_layer.layers.self_attention.linear_qkv.weight", (1, 4)
                ),
            },
            errors.FC_R12,
            "mtp.layers.0.mtp_model_layer is the hybrid layer list of MTP depth 0 ('*'), whose "
            "layers are stored one per key",
        ),
    ],
)
def test_mtp_layers_that_contradict_the_arguments_are_refused(args, model, code, message):
    with pytest.raises(RefusedError) as excinfo:
        classify_blocks_of(args, model, TORCH_DIST)
    assert excinfo.value.refusal.code == code
    assert message in excinfo.value.refusal.message


def test_a_layout_needs_one_distinct_stored_index_per_module():
    with pytest.raises(InternalError):
        BlockLayout("decoder", "hybrid_stack", 2, "per_layer", (0, 0), None, "")
    with pytest.raises(InternalError):
        BlockLayout("decoder", "hybrid_stack", 2, "layered", (0, 1), None, "")
    block = BlockLayout("decoder", "hybrid_stack", 2, "per_layer", (0, 2), None, "")
    assert block.dense_index(2, TORCH_DIST) == 1 and block.dense_index(1, FSDP_DTENSOR) == 1
    with pytest.raises(KeyGrammarError):
        block.dense_index(1, TORCH_DIST)


def test_the_language_model_prefix_follows_the_roots():
    assert language_model_prefix(["embedding.word_embeddings.weight", "mtp.layers.0.x"]) == ""
    vlm = ["language_model.decoder.final_norm.weight", "vision_projection.encoder.x.weight"]
    assert language_model_prefix(vlm) == "language_model."
    assert language_model_prefix([]) == ""
    with pytest.raises(RefusedError) as excinfo:
        language_model_prefix(["decoder.final_norm.weight", "vision_model.conv1.weight"])
    assert excinfo.value.refusal.code == errors.FC_R13
    assert "single-model roots (decoder) and vision-language roots (vision_model)" in (
        excinfo.value.refusal.message
    )


def test_a_vision_language_hybrid_decoder_is_under_the_language_model():
    model = _module_keys({0: "*", 1: "-"}, "language_model.decoder")
    (block,) = classify_blocks_of(_args(hybrid_layer_pattern="*-", num_layers=2), model, TORCH_DIST)
    assert (block.prefix, block.kind) == ("language_model.decoder", "hybrid_stack")
