# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Gated DeltaProduct layouts pinned to Megatron's own split helpers and defaults.

``GatedDeltaProductMixer`` splits its fused ``in_proj`` and ``conv1d`` per
tensor-parallel rank with ``_get_in_proj_checkpoint_split_layout`` and
``_get_conv_checkpoint_split_layout`` (checkpoint parts, and the
``partition_sizes`` of the parameters). The converter's component layouts must
give those names, and at every pack size each rank's runs must be those local
sections, in order. Runs in the training container; skipped elsewhere.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from tools.checkpoint.format_converter.core.mixers import (
    DEFAULT_HOUSEHOLDERS,
    MAMBA_EXPAND,
    gdp_parts,
)
from tools.checkpoint.format_converter.mcore_state.common_state import megatron_available

if not megatron_available():
    pytest.skip("needs Megatron (the training container)", allow_module_level=True)

gated_delta_product = pytest.importorskip("megatron.core.ssm.gated_delta_product")
mamba_mixer = pytest.importorskip("megatron.core.ssm.mamba_mixer")

# pylint: disable=wrong-import-position
from megatron.core.transformer.transformer_config import TransformerConfig

pytestmark = pytest.mark.covers("gdp", "parity")
D_INNER, GROUP_STATE, HEADS = 64, 48, 8


def _rank_runs(parts, tp: int) -> list[list[tuple[str, int]]]:
    """Each rank's runs of the tensor packed at ``tp``: ``(part, rows)`` in fused-row order."""
    per_rank = parts.fused_rows // tp
    ranks: list[list[tuple[str, int]]] = [[] for _ in range(tp)]
    for run in parts.rank_major_map(tp):
        ranks[run.fused_start // per_rank].append((run.part, run.rows))
    return ranks


@pytest.mark.parametrize("householders", [1, 2, 3])
@pytest.mark.parametrize("tp", [1, 2, 4])
def test_the_layouts_are_megatrons_checkpoint_split_layouts(householders, tp):
    in_proj, conv = gdp_parts(D_INNER, GROUP_STATE, HEADS, householders)
    sections, names = gated_delta_product._get_in_proj_checkpoint_split_layout(
        D_INNER // tp, GROUP_STATE // tp, HEADS // tp, householders
    )
    assert list(in_proj.parts) == names
    assert all(runs == list(zip(names, sections)) for runs in _rank_runs(in_proj, tp))
    sections, names = gated_delta_product._get_conv_checkpoint_split_layout(
        D_INNER // tp, GROUP_STATE // tp, householders
    )
    assert list(conv.parts) == names
    assert all(runs == list(zip(names, sections)) for runs in _rank_runs(conv, tp))


def test_the_default_number_of_householder_steps_is_megatrons():
    """A store saved before ``--gdp-num-householder`` existed has Megatron's default."""
    (field,) = [f for f in dataclasses.fields(TransformerConfig) if f.name == "gdp_num_householder"]
    assert field.default == DEFAULT_HOUSEHOLDERS


def test_mamba_mixers_expand_by_megatrons_default():
    """Without ``mamba_num_heads``, a Mamba-2 mixer's inner size is ``expand x hidden_size``."""
    assert (
        inspect.signature(mamba_mixer.MambaMixer.__init__).parameters["expand"].default
        == MAMBA_EXPAND
    )
