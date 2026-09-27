# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The refusal fence: every detector code on a hand-built model, and nothing on a supported one.

Each case starts from a supported per-layer hybrid ``*-*-`` model and changes
one argument or key; the fence must then report exactly that case's code.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, TORCH_DIST
from tools.checkpoint.format_converter.core.refusals import PENDING_FAMILIES, Facts, detect_refusals
from tools.checkpoint.format_converter.engine.metadata import TensorEntry
from tools.checkpoint.format_converter.errors import (
    AmbiguousError,
    Refusal,
    RefusedError,
    refusal_error,
)

H, F = 8, 16
HYBRID_ARGS = {
    "hybrid_layer_pattern": "*-*-",
    "num_layers": 4,
    "transformer_impl": "transformer_engine",
    "swiglu": False,
    "num_experts": None,
    "spec": None,
    "megatron_fsdp_version": 1,
    "pipeline_model_parallel_size": 1,
    "virtual_pipeline_model_parallel_size": None,
    "fp8": None,
    "fp8_recipe": "delayed",
    "experimental_attention_variant": None,
    "moe_shortcut_connection": False,
    "mtp_num_layers": None,
    "multi_latent_attention": False,
}


def hybrid_keys(fc1_rows: int = F) -> dict[str, tuple[int, ...]]:
    """Model-relative keys and shapes of a per-layer ``*-*-`` hybrid model."""
    keys = {
        "embedding.word_embeddings.weight": (32, H),
        "decoder.final_norm.weight": (H,),
        "output_layer.weight": (32, H),
    }
    for layer in (0, 2):
        attention = f"decoder.layers.{layer}.self_attention."
        keys[attention + "linear_qkv.weight"] = (3 * H, H)
        keys[attention + "linear_qkv.layer_norm_weight"] = (H,)
        keys[attention + "linear_proj.weight"] = (H, H)
    for layer in (1, 3):
        mlp = f"decoder.layers.{layer}.mlp."
        keys[mlp + "linear_fc1.weight"] = (fc1_rows, H)
        keys[mlp + "linear_fc1.layer_norm_weight"] = (H,)
        keys[mlp + "linear_fc2.weight"] = (H, F)
    return keys


def facts(
    *,
    args: dict | None = None,
    keys: dict[str, tuple[int, ...]] | None = None,
    source: str = TORCH_DIST,
    target: str = FSDP_DTENSOR,
    **options,
) -> Facts:
    """Facts of the hybrid model with ``args`` overriding its arguments and ``keys`` added."""
    model = {**hybrid_keys(), **(keys or {})}
    entries = {
        key: TensorEntry(key, shape, torch.bfloat16, ())
        for key, shape in model.items()
        if shape is not None
    }
    namespace = argparse.Namespace(**{**HYBRID_ARGS, **(args or {})})
    return Facts(source, target, ArgsView(namespace), entries, **options)


def codes(of: Facts) -> list[str]:
    return sorted(refusal.code for refusal in detect_refusals(of))


def test_a_supported_hybrid_model_passes_in_both_directions():
    assert codes(facts()) == []
    assert codes(facts(source=FSDP_DTENSOR, target=TORCH_DIST)) == []


GPT_ARGS = {"hybrid_layer_pattern": None}
FSDP_SWIGLU_KEYS = {
    **{f"decoder.layers.{layer}.mlp.linear_fc1.weight": None for layer in (1, 3)},
    **{
        f"decoder.layers.{layer}.mlp.linear_fc1.weight_{half}": (F, H)
        for layer in (1, 3)
        for half in "wv"
    },
}
"""An fsdp_dtensor SwiGLU store: each ``linear_fc1.weight`` split into ``weight_w``/``_v``."""
CASES = {
    "R01 v2 args": (errors.FC_R01, facts(args={"megatron_fsdp_version": 2})),
    "R02 fsdp pp": (
        errors.FC_R02,
        facts(args={"pipeline_model_parallel_size": 2}, source=FSDP_DTENSOR, target=TORCH_DIST),
    ),
    "R02 fsdp vpp": (
        errors.FC_R02,
        facts(
            args={"virtual_pipeline_model_parallel_size": 2}, source=FSDP_DTENSOR, target=TORCH_DIST
        ),
    ),
    "R03 impl": (errors.FC_R03, facts(args={"transformer_impl": "local"})),
    "R03 spec": (
        errors.FC_R03,
        facts(args={"spec": ["megatron.core.models.gpt.gpt_layer_specs", "custom_spec"]}),
    ),
    "R03 heterogeneous": (
        errors.FC_R03,
        facts(args={"heterogeneous_layers_config_path": "layers.json"}),
    ),
    "R03 op fuser": (
        errors.FC_R03,
        facts(keys={"decoder.layers.1.mlp.1.basic_ops.0.weight": (F, H)}),
    ),
    "R06 gtp": (errors.FC_R06, facts(args={"tensor_parallel_num_weight_shards": 2})),
    "R06 remat": (errors.FC_R06, facts(args={"gtp_weight_remat_size": 2})),
    "R06 shards above tp": (
        errors.FC_R06,
        facts(args={"tensor_model_parallel_size": 2, "tensor_parallel_num_weight_shards": 4}),
    ),
    "R06 expert remat": (errors.FC_R06, facts(args={"expert_gtp_weight_remat_size": 2})),
    "R06 expert shards above expert tp": (
        errors.FC_R06,
        facts(
            args={
                "tensor_model_parallel_size": 2,
                "expert_tensor_parallel_size": 1,
                "expert_tensor_parallel_num_weight_shards": 2,
            }
        ),
    ),
    "R06 expert shards above the default expert tp": (
        errors.FC_R06,
        facts(
            args={
                "tensor_model_parallel_size": 2,
                "expert_tensor_parallel_size": None,
                "expert_tensor_parallel_num_weight_shards": 4,
            }
        ),
    ),
    "R07 fp8 delayed": (
        errors.FC_R07,
        facts(args={"fp8": "e4m3"}, source=FSDP_DTENSOR, target=TORCH_DIST),
    ),
    "R13 root": (errors.FC_R13, facts(keys={"frontend.proj.weight": (H, H)})),
    "R20 symbol": (errors.FC_R20, facts(args={"hybrid_layer_pattern": "*-*D"})),
    "R20 unknown symbol": (errors.FC_R20, facts(args={"hybrid_layer_pattern": "*-*Z"})),
    "R20 variant": (errors.FC_R20, facts(args={"experimental_attention_variant": "dsa"})),
    "P01 gpt": (errors.FC_P01, facts(args=GPT_ARGS)),
    "P01 stacked": (
        errors.FC_P01,
        facts(keys={"decoder.layers.self_attention.linear_qkv.weight": (4, 3 * H, H)}),
    ),
    "P01 other block": (
        errors.FC_P01,
        facts(keys={"encoder.layers.0.linear.weight": (H, H)}, allowed_roots=("encoder",)),
    ),
    "P01 nested block": (
        errors.FC_P01,
        facts(keys={"decoder.layers.1.mlp.layers.0.linear.weight": (H, H)}),
    ),
    "P02 roots": (
        errors.FC_P02,
        facts(keys={"language_model.decoder.layers.0.self_attention.linear_proj.weight": (H, H)}),
    ),
    "P03 args": (errors.FC_P03, facts(args={"num_experts": 8})),
    "P03 symbol": (errors.FC_P03, facts(args={"hybrid_layer_pattern": "*E*E"})),
    "P03 keys": (errors.FC_P03, facts(keys={"decoder.layers.1.mlp.router.weight": (8, H)})),
    "P04 shortcut": (errors.FC_P04, facts(args={"moe_shortcut_connection": True})),
    "P04 hybrid mtp": (errors.FC_P04, facts(args={"hybrid_layer_pattern": "*-*-/*-"})),
    "P04 legacy mtp": (
        errors.FC_P04,
        facts(args={"mtp_hybrid_override_pattern": "*-", "mtp_num_layers": 2}),
    ),
    "P04 hybrid mtp keys": (
        errors.FC_P04,
        facts(
            args={"hybrid_layer_pattern": "*-*-/*-"},
            keys={"mtp.layers.0.mtp_model_layer.layers.1.mlp.linear_fc2.weight": (H, F)},
        ),
    ),
    "P05 spec": (
        errors.FC_P05,
        facts(args={"spec": ["megatron.core.models.hybrid.hybrid_layer_specs", "gdp_stack_spec"]}),
    ),
    "P05 parts": (errors.FC_P05, facts(keys={"decoder.layers.0.mixer.in_proj.weight.V0": (H, H)})),
    "P06 gated fused": (
        errors.FC_P06,
        facts(keys={"decoder.layers.1.mlp.linear_fc1.weight": (2 * F, H)}),
    ),
    "P07 swiglu": (errors.FC_P07, facts(args={"swiglu": True})),
    "P07 fsdp halves": (
        errors.FC_P07,
        facts(args={"swiglu": True}, keys=FSDP_SWIGLU_KEYS, source=FSDP_DTENSOR, target=TORCH_DIST),
    ),
    "P07 halves without args.swiglu": (
        errors.FC_P07,
        facts(keys=FSDP_SWIGLU_KEYS, source=FSDP_DTENSOR, target=TORCH_DIST),
    ),
    "P08 symbol": (errors.FC_P08, facts(args={"hybrid_layer_pattern": "M-*-"})),
    "P08 keys": (
        errors.FC_P08,
        facts(keys={"decoder.layers.0.mixer.conv1d_weight": (H, 1, 4)}, source=FSDP_DTENSOR),
    ),
    "P09 symbol": (errors.FC_P09, facts(args={"hybrid_layer_pattern": "G-*-"})),
    "P09 variant": (errors.FC_P09, facts(args={"experimental_attention_variant": "gdn2"})),
    "P09 keys": (
        errors.FC_P09,
        facts(keys={"decoder.layers.0.self_attention.in_proj.weight": (4 * H, H)}),
    ),
    "P10 mla": (errors.FC_P10, facts(args={"multi_latent_attention": True})),
    "P10 symbol": (errors.FC_P10, facts(args={"hybrid_layer_pattern": "+-+-"})),
    "P11 optimizer": (
        errors.FC_P11,
        facts(source=FSDP_DTENSOR, target=TORCH_DIST, optimizer="convert"),
    ),
    "P12 optimizer": (errors.FC_P12, facts(optimizer="convert")),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_each_case_is_refused_with_exactly_its_code(case):
    code, case_facts = CASES[case]
    assert codes(case_facts) == [code]
    (refusal,) = detect_refusals(case_facts)
    assert refusal.hint, refusal


def test_every_detector_code_has_a_case():
    assert {code for code, _ in CASES.values()} >= set(PENDING_FAMILIES) | {
        errors.FC_R01,
        errors.FC_R02,
        errors.FC_R03,
        errors.FC_R06,
        errors.FC_R07,
        errors.FC_R13,
        errors.FC_R20,
    }


def test_swiglu_halves_are_pending_whatever_the_arguments_say():
    for case in ("P07 fsdp halves", "P07 halves without args.swiglu"):
        (refusal,) = detect_refusals(CASES[case][1])
        assert "linear_fc1 stored as SwiGLU halves" in refusal.message, case
        assert refusal.keys == tuple(sorted(k for k, v in FSDP_SWIGLU_KEYS.items() if v))


def test_escape_hatches_and_harmless_values_pass():
    assert codes(facts(keys={"frontend.proj.weight": (H, H)}, allowed_roots=("frontend",))) == []
    fp8 = {"fp8": "e4m3", "fp8_recipe": "delayed"}
    assert (
        codes(
            facts(
                args=fp8, source=FSDP_DTENSOR, target=TORCH_DIST, allow_missing_fp8_extra_state=True
            )
        )
        == []
    )
    assert (
        codes(facts(args={**fp8, "fp8_recipe": "mxfp8"}, source=FSDP_DTENSOR, target=TORCH_DIST))
        == []
    )
    assert codes(facts(args=fp8)) == []  # the fsdp_dtensor target never stores it
    assert (
        codes(facts(args={"pipeline_model_parallel_size": 4})) == []
    )  # torch_dist keys are global
    assert codes(facts(args={"hybrid_layer_pattern": "*-|*-"})) == []
    hybrid_spec = ["megatron.core.models.mamba.mamba_layer_specs", "hybrid_stack_spec"]
    assert codes(facts(args={"spec": hybrid_spec})) == []
    assert (
        codes(
            facts(
                args={"swiglu": True},
                keys={
                    "decoder.layers.1.mlp.linear_fc1.weight": None,
                    "decoder.layers.3.mlp.linear_fc1.weight": None,
                },
            )
        )
        == []
    )  # no linear_fc1 to split


def _weight_shards(tp, shards, remat, expert_tp, expert_shards, expert_remat) -> dict:
    return {
        "tensor_model_parallel_size": tp,
        "tensor_parallel_num_weight_shards": shards,
        "gtp_weight_remat_size": remat,
        "expert_tensor_parallel_size": expert_tp,
        "expert_tensor_parallel_num_weight_shards": expert_shards,
        "expert_gtp_weight_remat_size": expert_remat,
    }


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param(_weight_shards(1, 1, 1, 1, 1, 1), id="tp1"),
        pytest.param(_weight_shards(2, 2, 1, 2, 2, 1), id="tp2"),
        pytest.param(_weight_shards(4, 4, 1, 4, 4, 1), id="tp4"),
        pytest.param(_weight_shards(4, 4, 1, 1, 1, 1), id="tp4-expert-tp1"),
        pytest.param(_weight_shards(2, 2, 1, None, 2, 1), id="tp2-expert-tp-unset"),
        pytest.param(_weight_shards(2, None, None, None, None, None), id="tp2-unset"),
        pytest.param({"tensor_model_parallel_size": 2}, id="tp2-no-gtp-fields"),
    ],
)
def test_weight_shards_of_plain_tensor_parallelism_are_not_grouped(fields):
    """Megatron's validate_args stores as many weight shards as tensor-parallel ranks (and the
    expert pair likewise) in every job without grouped tensor parallelism."""
    assert codes(facts(args=fields)) == []
    assert codes(facts(args=fields, source=FSDP_DTENSOR, target=TORCH_DIST)) == []


def test_grouped_tensor_parallelism_names_its_arguments():
    (refusal,) = detect_refusals(facts(args=_weight_shards(2, 4, 2, None, 4, 1)))
    assert refusal.message == (
        "grouped tensor parallelism may pad stored tensors (args.gtp_weight_remat_size=2, "
        "args.tensor_parallel_num_weight_shards=4 > tensor_model_parallel_size=2, "
        "args.expert_tensor_parallel_num_weight_shards=4 > expert_tensor_parallel_size=2)"
    )


def test_a_vision_language_hybrid_is_refused_with_every_pending_family_it_uses():
    """A shortcut-MoE hybrid language model with a stacked GLU vision encoder."""
    args = {
        "hybrid_layer_pattern": "M-*E/*E",
        "moe_shortcut_connection": True,
        "num_experts": 8,
        "spec": [
            "megatron.core.models.hybrid.hybrid_layer_specs",
            "gated_delta_product_stack_spec",
        ],
    }
    keys = {
        "language_model.decoder.layers.0.mixer.in_proj.weight.z": (H, H),
        "language_model.decoder.layers.2.attn_layer.self_attention.linear_qkv.weight": (3 * H, H),
        "language_model.decoder.layers.2.moe_layer.mlp.router.weight": (8, H),
        "language_model.mtp.layers.0.eh_proj.weight": (H, 2 * H),
        "vision_model.decoder.layers.mlp.linear_fc1.weight": (2, 2 * F, H),
        "vision_model.decoder.layers.mlp.linear_fc2.weight": (2, H, F),
        "vision_projection.encoder.linear_fc1.weight": (H, H),
        "vision_projection.encoder.linear_fc2.weight": (H, H),
    }
    found = facts(args=args, keys=keys)
    assert codes(found) == [
        errors.FC_P01,
        errors.FC_P02,
        errors.FC_P03,
        errors.FC_P04,
        errors.FC_P05,
        errors.FC_P06,
    ]


def test_refusals_carry_the_evidence():
    (refusal,) = detect_refusals(facts(args={"num_experts": 8, "hybrid_layer_pattern": "*E*-"}))
    assert refusal.message == (
        "not supported yet: mixture-of-experts layers (args.num_experts=8; hybrid layer symbol E)"
    )
    (refusal,) = detect_refusals(facts(args={"megatron_fsdp_version": 2}))
    assert "args.megatron_fsdp_version=2" in refusal.message


def test_several_refusals_form_one_error():
    pending = Refusal(errors.FC_P03, "experts", hint="later")
    permanent = Refusal(errors.FC_R20, "symbol D", hint="no")
    ambiguous = Refusal(errors.FC_A05, "from", hint="drop --from")
    error = refusal_error([pending, ambiguous, permanent])
    assert isinstance(error, RefusedError) and error.exit_code == errors.ExitCode.REFUSED
    assert error.codes == (errors.FC_R20, errors.FC_A05, errors.FC_P03)
    assert error.format().splitlines() == [
        "FC-R20: symbol D hint: no",
        "FC-A05: from hint: drop --from",
        "FC-P03: experts hint: later",
    ]
    assert isinstance(refusal_error([ambiguous]), AmbiguousError)
    with pytest.raises(errors.InternalError):
        refusal_error([])
