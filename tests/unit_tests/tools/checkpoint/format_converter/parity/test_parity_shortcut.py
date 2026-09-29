# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The shortcut-MoE grouping walk against Megatron's ``group_layers_into_shortcut_blocks``.

Megatron's function runs on stand-in layers (each numbered as ``HybridStack``
numbers it, with two-stage attention support), once per pipeline stage of the
pattern, as ``HybridStack`` calls it; ``ShortcutMoEBlock``, whose norms need
Transformer Engine on a GPU, is replaced by a stand-in that records the pair.
Its modules (torch_dist index = the first layer's ``layer_number - 1``) and its
refusals must be :func:`group_modules`'s. Runs in the training container;
skipped elsewhere.
"""

from __future__ import annotations

import types
from unittest import mock

import pytest

from tools.checkpoint.format_converter.core.hybrid_pattern import (
    MOE,
    MTP_SEPARATOR,
    PIPE,
    SHORTCUT_PREDECESSORS,
    group_modules,
    parse_hybrid_pattern,
)
from tools.checkpoint.format_converter.errors import RefusedError
from tools.checkpoint.format_converter.mcore_state.common_state import megatron_available

if not megatron_available():
    pytest.skip("needs Megatron (the training container)", allow_module_level=True)

# pylint: disable=wrong-import-position
import torch

from megatron.core.models.hybrid import shortcut_block
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.transformer.module import TwoStageAttentionLayer

PATTERNS = [
    "E*E",
    "*EE",
    "*EE*",
    "M-*E",
    "*E|*E",
    "*E-*E-",
    "ME*EGE+EDECEHEWE",
    "E",
    "",
    "-E",
    "EE",
    "*-E",
    "*|E*E",
    "*E|E",
    "*-|E*E",
    "E*E|-*EE|ME",
]
"""The edge cases: leading and trailing ``E``, every predecessor, stage boundaries, refusals."""

REFUSED = frozenset({"-E", "EE", "*-E", "*|E*E", "*E|E", "*-|E*E"})
"""The patterns Megatron cannot group."""


class _Layer(torch.nn.Module, TwoStageAttentionLayer):
    """A physical layer as ``HybridStack`` numbers it, supporting two-stage attention."""

    def __init__(self, layer_number: int) -> None:
        super().__init__()
        self.layer_number = layer_number


class _Pair(torch.nn.Module):
    """Stands in for ``ShortcutMoEBlock``: the pair takes its first layer's number."""

    def __init__(self, attn_layer, moe_layer, overlap_a2a, attn_local_idx, moe_local_idx):
        super().__init__()
        del overlap_a2a
        assert moe_local_idx == attn_local_idx + 1
        assert moe_layer.layer_number == attn_layer.layer_number + 1
        self.layer_number = attn_layer.layer_number


def _megatron_modules(text: str) -> list[tuple[int, str]]:
    """Megatron's modules of the main pattern ``text``, grouped stage by stage.

    Raises:
        ValueError: Megatron cannot group a stage.
    """
    stages = text.split(MTP_SEPARATOR)[0].split(PIPE)
    symbols = "".join(stages)
    config = types.SimpleNamespace(moe_shortcut_parallel=False)
    modules, offset = [], 0
    with mock.patch.object(shortcut_block, "ShortcutMoEBlock", _Pair):
        for stage in stages:
            layers = torch.nn.ModuleList(_Layer(offset + i + 1) for i in range(len(stage)))
            grouped = shortcut_block.group_layers_into_shortcut_blocks(
                layers, list(stage), config, pp_layer_offset=offset
            )
            for module in grouped:
                position = module.layer_number - 1
                size = 2 if isinstance(module, _Pair) else 1
                modules.append((position, symbols[position : position + size]))
            offset += len(stage)
    return modules


@pytest.mark.parametrize("text", PATTERNS)
def test_the_walk_groups_and_refuses_like_megatron(text):
    pattern = parse_hybrid_pattern(text)
    try:
        theirs = _megatron_modules(text)
    except ValueError:
        theirs = None
    try:
        ours = list(group_modules(pattern.main, shortcut=True, stage_starts=pattern.stage_starts))
    except RefusedError:
        ours = None
    assert (theirs is None) == (text in REFUSED)
    assert ours == theirs


def test_the_predecessors_are_megatrons():
    assert SHORTCUT_PREDECESSORS == shortcut_block.SUPPORTED_SHORTCUT_PREDECESSORS
    assert MOE == Symbols.MOE
