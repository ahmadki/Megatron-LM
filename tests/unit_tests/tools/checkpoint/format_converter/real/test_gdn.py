# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Gated DeltaNets (GDN and GDN2) against Megatron's own modules.

* ``GatedDeltaNet.sharded_state_dict`` and ``GatedDeltaNet2``'s at
  tensor-parallel sizes 1 and 2 (split factories applied), built with
  ``bias=True`` and ``conv_bias=True``: the parts of ``in_proj.weight`` and of
  the convolution's weight and bias, their names, order, global and local
  sizes, are the converter's layouts; ``in_proj.bias`` is one tensor, the
  converter's TP-bound whole tensor; ``A_log`` and ``dt_bias`` have the rows
  the converter checks.
* The torch_dist keys and shapes of a tiny hybrid model of Gated DeltaNet
  layers (``G``, default stack spec) and of a tiny GPT model whose
  linear-attention layers are Gated DeltaNets, at tensor-parallel sizes 1 and
  2, for both variants: the converter's context finds the layers from the real
  keys, their parts are the ones written out below, and the fsdp_dtensor names
  it composes are the model's state-dict names.
* A real Gated DeltaNet of each variant at tensor-parallel size 2: each rank's
  fused ``in_proj`` weight and convolution weight and bias are its block of the
  rank-major pack, by the converter's layout, of the canonical parts gathered
  from the module's own torch_dist factory sub-tensors (not their
  concatenation), and its whole torch_dist ``in_proj.bias`` is its ranks' local
  biases one after the other.

Runs only in the Megatron container with GPUs and flash-linear-attention
(``fla``); cases that need more ranks than the world has are skipped.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from tools.checkpoint.format_converter.mcore_state.common_state import megatron_available

if not (megatron_available() and torch.cuda.is_available()):
    pytest.skip("needs Megatron and CUDA (the training container)", allow_module_level=True)

pytest.importorskip("fla")

# pylint: disable=wrong-import-position
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA_GDN2, GatedDeltaNet, GatedDeltaNet2
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.context import build_arch_context
from tools.checkpoint.format_converter.core.keys import EXTRA_STATE
from tools.checkpoint.format_converter.core.logical import LogicalId
from tools.checkpoint.format_converter.core.mixers import GDN, GDN2, mixer_layout
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, TORCH_DIST
from tools.checkpoint.format_converter.core.structure import compose, decompose
from tools.checkpoint.format_converter.engine.metadata import TensorEntry

from .._real_harness import (
    GPT_LAYERS,
    HYBRID_HIDDEN,
    Utils,
    canonical_tensors,
    gathered_ranks,
    hybrid_provider,
    initialize_tensor_parallel,
    linear_attention_gpt_provider,
    main_layers,
    randomize_parameters,
    rank_major_pack,
    sharded_tensors,
)

pytestmark = [pytest.mark.covers("gdn", "parity"), pytest.mark.covers("gdn2", "parity")]
NEEDS_GDN2 = pytest.mark.skipif(not HAVE_FLA_GDN2, reason="GatedDeltaNet2 needs fla.ops.gdn2")
FAMILIES = [GDN, pytest.param(GDN2, marks=NEEDS_GDN2)]
PATTERN = "G*G-"
LINEAR = {
    "linear_key_head_dim": 8,
    "linear_num_key_heads": 2,
    "linear_value_head_dim": 8,
    "linear_num_value_heads": 4,
    "linear_conv_kernel_dim": 4,
}
"""TransformerConfig (and argument) fields of the tiny Gated DeltaNets: qk = 16, v = 32, 2 key and
4 value heads."""
IN_PROJ = {
    GDN: {"query": 16, "key": 16, "value": 32, "z": 32, "beta": 4, "alpha": 4},
    GDN2: {"query": 16, "key": 16, "value": 32, "z": 32, "f": 16, "b": 16, "w": 32},
}
CONV = {"query": 16, "key": 16, "value": 32}
VECTORS = {GDN: {"A_log": 4, "dt_bias": 4}, GDN2: {"A_log": 2, "dt_bias": 16}}
"""``A_log`` and ``dt_bias`` rows: a value head each (GDN), a key head and a key channel (GDN2)."""
VARIANTS = {GDN: None, GDN2: "gdn2"}
"""The ``experimental_attention_variant`` of a hybrid model of each family's ``G`` layers."""
LINEAR_ATTENTION_FREQ = 3
"""A GPT model's layers 0 and 1 are Gated DeltaNets, layer 2 standard attention."""


def _args(tp: int, variant: str | None, **fields) -> ArgsView:
    """The arguments a job of the tiny hybrid model saves (those the converter reads)."""
    return ArgsView(
        argparse.Namespace(
            hybrid_layer_pattern=PATTERN,
            num_layers=main_layers(PATTERN),
            spec=None,
            hidden_size=HYBRID_HIDDEN,
            tensor_model_parallel_size=tp,
            experimental_attention_variant=variant,
            **{**LINEAR, **fields},
        )
    )


def _module(family: str, seed: int):
    """A Gated DeltaNet of ``family`` with an ``in_proj`` bias and a convolution bias, with distinct
    values in every parameter (and every tensor-parallel shard)."""
    model_parallel_cuda_manual_seed(seed)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HYBRID_HIDDEN,
        num_attention_heads=4,
        use_cpu_initialization=True,
        **LINEAR,
    )
    layer = (
        hybrid_stack_spec.submodules.gdn2_layer
        if family == GDN2
        else hybrid_stack_spec.submodules.gdn_layer
    )
    module = (GatedDeltaNet2 if family == GDN2 else GatedDeltaNet)(
        config,
        layer.submodules.self_attention.submodules,
        layer_number=1,
        bias=True,
        conv_bias=True,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"]),
    ).cuda()
    randomize_parameters(module, seed)
    return module


class TestMegatronFactories:
    """``_GDNBase.sharded_state_dict``'s parts against the converter's layouts."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("family", FAMILIES)
    @pytest.mark.parametrize("tp", [1, 2])
    def test_the_factories_cut_the_converters_parts(self, tp, family):
        initialize_tensor_parallel(tp)
        module = _module(family, 5)
        shapes = {
            t.key: (tuple(t.global_shape), tuple(t.local_shape))
            for t in sharded_tensors(module.sharded_state_dict(prefix="self_attention."))
        }
        rows = sum(IN_PROJ[family].values())
        expected = {
            f"self_attention.in_proj.weight.{part}": ((n, HYBRID_HIDDEN), (n // tp, HYBRID_HIDDEN))
            for part, n in IN_PROJ[family].items()
        }
        expected["self_attention.in_proj.bias"] = ((rows,), (rows // tp,))
        for leaf, rest in (("weight", (1, 4)), ("bias", ())):
            expected.update(
                {
                    f"self_attention.conv1d.{leaf}.{part}": ((n, *rest), (n // tp, *rest))
                    for part, n in CONV.items()
                }
            )
        expected.update(
            {f"self_attention.{name}": ((n,), (n // tp,)) for name, n in VECTORS[family].items()}
        )
        fused = (
            "self_attention.in_proj.weight",
            "self_attention.in_proj.bias",
            "self_attention.conv1d.",
        )
        checked = fused + ("self_attention.A_log", "self_attention.dt_bias")
        assert {key: shape for key, shape in shapes.items() if key.startswith(checked)} == expected
        census = {
            key: TensorEntry(key, global_shape, torch.bfloat16, ())
            for key, (global_shape, _) in shapes.items()
        }
        layout, _ = mixer_layout(_args(tp, VARIANTS[family]), census, (), TORCH_DIST)
        assert {
            lid.name: (dict(zip(f.parts.parts, f.parts.sizes)), f.torch_dist_whole)
            for lid, f in layout.projections.items()
        } == {
            "self_attention.in_proj.weight": (IN_PROJ[family], False),
            "self_attention.in_proj.bias": (IN_PROJ[family], True),
            "self_attention.conv1d.weight": (CONV, False),
            "self_attention.conv1d.bias": (CONV, False),
        }


def _census(model) -> dict[str, TensorEntry]:
    return {
        t.key: TensorEntry(t.key, tuple(t.global_shape), t.dtype, ())
        for t in sharded_tensors(model.sharded_state_dict())
    }


def _fsdp_names(census, ctx) -> set[str]:
    """The fsdp_dtensor names the converter composes for the torch_dist ``census``."""
    names = set()
    for key, entry in census.items():
        for piece in decompose(key, entry, ctx.blocks, TORCH_DIST):
            fused = ctx.mixers.of(piece.lid)
            shape = piece.shape if fused is None else (fused.parts.fused_rows,)
            lid = piece.lid if fused is None else fused.target_lid(FSDP_DTENSOR)
            names.add(compose(lid, shape, ctx.blocks, FSDP_DTENSOR).key)
    return names


def _projections(ctx) -> dict[LogicalId, dict[str, int]]:
    return {
        lid: dict(zip(fused.parts.parts, fused.parts.sizes))
        for lid, fused in ctx.mixers.projections.items()
    }


def _expected_projections(family: str, layers) -> dict[LogicalId, dict[str, int]]:
    """The fused projections of the Gated DeltaNets of ``layers`` (no biases: the specs build
    none in ``in_proj`` and the convolution)."""
    return {
        LogicalId("decoder", layer, f"self_attention.{name}", "weight"): parts
        for layer in layers
        for name, parts in (("in_proj", IN_PROJ[family]), ("conv1d", CONV))
    }


def _state_dict_names(model) -> set[str]:
    return {name for name in model.state_dict() if not EXTRA_STATE.search(name)}


class TestMegatronKeys:
    """The converter's view of the torch_dist keys of real models of Gated DeltaNets."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("family", FAMILIES)
    @pytest.mark.parametrize("tp", [1, 2])
    def test_a_hybrid_models_gated_deltanets(self, tp, family):
        initialize_tensor_parallel(tp)
        model = hybrid_provider(
            seed=1, pattern=PATTERN, experimental_attention_variant=VARIANTS[family], **LINEAR
        )
        census = _census(model)
        assert "decoder.layers.2.self_attention.in_proj.weight.query" in census
        ctx = build_arch_context(
            source=TORCH_DIST,
            target=FSDP_DTENSOR,
            args=_args(tp, VARIANTS[family]),
            model=census,
            fsdp_tp=tp,
        )
        assert ctx.mixers.families_json() == {"decoder": {family: 2}}
        assert _projections(ctx) == _expected_projections(family, (0, 2))
        assert _fsdp_names(census, ctx) == _state_dict_names(model)

    @pytest.mark.parametrize(
        ("family", "variant"), [(GDN, "gdn"), pytest.param(GDN2, "gdn2", marks=NEEDS_GDN2)]
    )
    @pytest.mark.parametrize("tp", [1, 2])
    def test_a_gpt_models_linear_attention(self, tp, family, variant):
        """GPT linear attention: ``linear_attention_freq`` makes Megatron save the decoder one
        layer per key, its Gated DeltaNets' projections as parts."""
        initialize_tensor_parallel(tp)
        config = {
            **LINEAR,
            "experimental_attention_variant": variant,
            "linear_attention_freq": LINEAR_ATTENTION_FREQ,
        }
        model = linear_attention_gpt_provider(seed=1, **config)
        census = _census(model)
        assert "decoder.layers.2.self_attention.linear_qkv.weight" in census
        assert "decoder.layers.1.self_attention.conv1d.weight.value" in census
        args = argparse.Namespace(
            hybrid_layer_pattern=None,
            num_layers=GPT_LAYERS,
            spec=None,
            hidden_size=HYBRID_HIDDEN,
            tensor_model_parallel_size=tp,
            swiglu=False,
            **config,
        )
        ctx = build_arch_context(
            source=TORCH_DIST, target=FSDP_DTENSOR, args=ArgsView(args), model=census, fsdp_tp=tp
        )
        assert ctx.mixers.families_json() == {"decoder": {family: 2}}
        assert _projections(ctx) == _expected_projections(family, (0, 1))
        assert _fsdp_names(census, ctx) == _state_dict_names(model)


class TestMegatronPack:
    """A real Gated DeltaNet's fused tensors against the pack of its own torch_dist parts."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("family", FAMILIES)
    def test_each_ranks_fused_projections_are_its_block_of_the_rank_major_pack(self, family):
        initialize_tensor_parallel(2)
        module = _module(family, 3)
        canonical = canonical_tensors(
            sharded_tensors(module.sharded_state_dict(prefix="self_attention."))
        )
        assert not any(torch.isnan(t).any() for t in canonical.values() if t.is_floating_point())
        census = {k: TensorEntry(k, tuple(t.shape), t.dtype, ()) for k, t in canonical.items()}
        layout, _ = mixer_layout(_args(2, VARIANTS[family]), census, (), TORCH_DIST)
        assert len(layout.projections) == 4  # in_proj and conv1d, weight and bias
        for lid, fused in layout.projections.items():
            param = module.get_parameter(lid.name.removeprefix("self_attention."))
            live = gathered_ranks(param.detach()).float()
            if fused.torch_dist_whole:
                # TP-bound identity: the whole torch_dist tensor is the ranks' local ones in order.
                assert torch.equal(live, canonical[lid.name].float()), lid
                continue
            parts = {part: canonical[f"{lid.name}.{part}"] for part in fused.parts.parts}
            packed = rank_major_pack(fused.parts, parts, 2)
            assert torch.equal(live, packed.float()), lid
            assert not torch.equal(packed, torch.cat(list(parts.values()))), lid
