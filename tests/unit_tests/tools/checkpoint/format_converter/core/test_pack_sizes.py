# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The sizes fsdp_dtensor packs fused tensors at: given, defaulted, required and cross-checked."""

from __future__ import annotations

import argparse

import pytest
import torch

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.context import build_arch_context
from tools.checkpoint.format_converter.core.manifest import Manifest
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, TORCH_DIST
from tools.checkpoint.format_converter.engine.metadata import TensorEntry
from tools.checkpoint.format_converter.errors import AmbiguousError, RefusedError

H, F, E = 8, 8, 2
DENSE_FC1 = "decoder.layers.0.mlp.linear_fc1.weight"
EXPERT_FC1 = "decoder.layers.1.mlp.experts.experts.linear_fc1.weight"
GEGLU = {
    DENSE_FC1: (2 * F, H),
    "decoder.layers.0.mlp.linear_fc2.weight": (H, F),
    EXPERT_FC1: (E, 2 * F, H),
    "decoder.layers.1.mlp.experts.experts.linear_fc2.weight": (E, H, F),
}
"""A GEGLU GPT (gated without args.swiglu): Megatron-FSDP keeps every linear_fc1 fused."""
GEGLU_FSDP = {
    DENSE_FC1: (2 * F, H),
    "decoder.layers.0.mlp.linear_fc2.weight": (H, F),
    **{f"decoder.layers.1.mlp.experts.linear_fc1.weight{e}": (2 * F, H) for e in range(E)},
    **{f"decoder.layers.1.mlp.experts.linear_fc2.weight{e}": (H, F) for e in range(E)},
}
"""The same model as Megatron-FSDP saves it (one tensor per expert)."""


def context(shapes=None, *, source=TORCH_DIST, manifest=None, **options):
    fields = {
        "num_layers": 2,
        "quick_geglu": True,
        "num_experts": E,
        "moe_layer_freq": [0, 1],
        "moe_grouped_gemm": True,
        "tensor_model_parallel_size": 1,
        "expert_tensor_parallel_size": None,
    }
    fields.update(options.pop("args", {}))
    if shapes is None:
        shapes = GEGLU if source == TORCH_DIST else GEGLU_FSDP
    prefix = "model.module." if source == FSDP_DTENSOR else ""
    model = {k: TensorEntry(prefix + k, shape, torch.bfloat16, ()) for k, shape in shapes.items()}
    target = FSDP_DTENSOR if source == TORCH_DIST else TORCH_DIST
    return build_arch_context(
        source=source,
        target=target,
        args=ArgsView(argparse.Namespace(**fields)),
        model=model,
        manifest=manifest,
        metadata_sha256="abc",
        **options,
    )


def logged(ctx, field):
    (derivation,) = [d for d in ctx.derivations if d.field == field]
    return derivation.value, derivation.evidence


def test_a_source_saved_without_tensor_parallelism_packs_at_one():
    ctx = context()
    assert (ctx.pack_tp, ctx.pack_etp) == (1, 1)
    assert logged(ctx, "pack_tp") == (
        1,
        "args.tensor_model_parallel_size=1 of the source; 1 gated linear_fc1 that fsdp_dtensor "
        "keeps fused, packed per rank",
    )
    assert logged(ctx, "pack_etp") == (
        1,
        "args.expert_tensor_parallel_size=1 of the source; 2 gated linear_fc1 that fsdp_dtensor "
        "keeps fused, packed per rank",
    )


def test_a_tensor_parallel_source_needs_the_loading_jobs_size():
    with pytest.raises(AmbiguousError) as caught:
        context(args={"tensor_model_parallel_size": 2})
    refusal = caught.value.refusal
    assert refusal.code == errors.FC_A01 and refusal.keys == (DENSE_FC1,)
    assert refusal.message == (
        "fsdp_dtensor packs 1 gated linear_fc1 per rank of the job that loads it, and the source was "
        "saved with args.tensor_model_parallel_size=2: the store loads right only at the size it "
        "is packed at"
    )
    assert refusal.hint.startswith("pass --fsdp-tp N with the tensor_model_parallel_size")
    with pytest.raises(AmbiguousError) as caught:  # the expert size defaults to the tensor one
        context(args={"tensor_model_parallel_size": 2}, fsdp_tp=2)
    assert caught.value.refusal.keys == (EXPERT_FC1,)
    assert caught.value.refusal.hint.startswith("pass --fsdp-etp N")
    ctx = context(args={"tensor_model_parallel_size": 2}, fsdp_tp=4, fsdp_etp=1)
    assert (ctx.pack_tp, ctx.pack_etp) == (4, 1)
    assert logged(ctx, "pack_tp")[1].startswith("--fsdp-tp 4; ")


def test_halves_that_do_not_divide_by_the_pack_size_are_refused():
    with pytest.raises(RefusedError) as caught:
        context(fsdp_tp=16)
    refusal = caught.value.refusal
    assert refusal.code == errors.FC_R10 and refusal.keys == (DENSE_FC1,)
    assert refusal.message == (
        "the gate and value halves of 1 stored gated linear_fc1 tensors do not divide by "
        "--fsdp-tp 16"
    )
    assert refusal.hint == "pass a --fsdp-tp that divides them"
    with pytest.raises(RefusedError) as caught:  # the store's own size, converting back
        context(source=FSDP_DTENSOR, args={"tensor_model_parallel_size": 16})
    assert caught.value.refusal.message.endswith(
        "do not divide by the store's args.tensor_model_parallel_size=16"
    )


def test_split_and_plain_first_projections_are_packed_at_no_size():
    swiglu = context(args={"quick_geglu": False, "swiglu": True, "tensor_model_parallel_size": 2})
    assert (swiglu.pack_tp, swiglu.pack_etp) == (None, None)
    assert logged(swiglu, "pack_tp") == (None, "no fused tensor to pack per rank")
    plain = {key: (F, H) if "fc1" in key else shape for key, shape in GEGLU.items()}
    plain[EXPERT_FC1] = (E, F, H)
    ctx = context(plain, args={"quick_geglu": False}, fsdp_tp=2)
    assert ctx.pack_tp is None
    assert (
        logged(ctx, "pack_tp")[1] == "no fused tensor to pack per rank; --fsdp-tp 2 has no effect"
    )
    assert ctx.warnings[-1] == (
        "--fsdp-tp 2 has no effect: the fsdp_dtensor store packs no tensor per tensor-parallel "
        "rank, so it loads at any tensor_model_parallel_size"
    )


EXPERTS_ONLY = {
    "decoder.layers.0.self_attention.linear_proj.weight": (H, H),
    **{key: shape for key, shape in GEGLU.items() if ".experts." in key},
}
"""A GEGLU model whose only MLP is a mixture-of-experts layer without shared experts."""


def test_another_tensor_parallel_size_needs_the_expert_size_too():
    """A job at --fsdp-tp N runs its experts at expert tensor-parallel size N unless it sets
    --expert-tensor-parallel-size, so the source's expert size 1 is no default then."""
    with pytest.raises(AmbiguousError) as caught:
        context(fsdp_tp=2)
    refusal = caught.value.refusal
    assert refusal.code == errors.FC_A01 and refusal.keys == (EXPERT_FC1,)
    assert refusal.message == (
        "fsdp_dtensor packs 2 gated linear_fc1 of routed experts per expert tensor-parallel rank "
        "of the job that loads it, and --fsdp-tp 2 is not the size the source was saved with "
        "(args.tensor_model_parallel_size=1): a job at tensor-parallel size 2 runs its routed "
        "experts at expert tensor-parallel size 2 unless it sets --expert-tensor-parallel-size"
    )
    assert refusal.hint.startswith("pass --fsdp-etp N with the expert_tensor_parallel_size")
    assert (context(fsdp_tp=2, fsdp_etp=1).pack_tp, context(fsdp_tp=2, fsdp_etp=1).pack_etp) == (
        2,
        1,
    )
    assert context(fsdp_tp=1).pack_etp == 1  # the source's own size: its expert size stays 1
    with pytest.raises(AmbiguousError):  # nothing packed per tensor-parallel rank: still needed
        context(EXPERTS_ONLY, fsdp_tp=2)


def test_a_tensor_parallel_flag_without_dense_packs_warns_about_the_expert_size():
    ctx = context(EXPERTS_ONLY, fsdp_tp=2, fsdp_etp=2)
    assert (ctx.pack_tp, ctx.pack_etp) == (None, 2)
    assert ctx.warnings[-1] == (
        "--fsdp-tp 2 has no effect: the fsdp_dtensor store packs no tensor per tensor-parallel "
        "rank, but it packs 2 gated linear_fc1 of routed experts per expert tensor-parallel rank: "
        "they load right only at the expert_tensor_parallel_size the store's arguments record "
        "(pack_etp), which the loading job must set with --expert-tensor-parallel-size (Megatron "
        "does not take it from a checkpoint)"
    )


def vocabulary(**args):
    return {"padded_vocab_size": 64, "make_vocab_size_divisible_by": 16, **args}


@pytest.mark.parametrize(
    ("args", "fsdp_tp", "warned", "evidence"),
    [
        (vocabulary(vocab_size=50), 4, False, "is args.vocab_size=50 padded to a multiple"),
        (
            vocabulary(padded_vocab_size=32, vocab_size=20, tensor_model_parallel_size=2),
            4,
            True,
            "pads args.vocab_size=20 to 64 (",
        ),
        (vocabulary(vocab_size=20, tensor_model_parallel_size=4), 1, True, "to 32 ("),
        (vocabulary(), 2, False, "is a multiple of make_vocab_size_divisible_by x 2 = 32"),
        (vocabulary(padded_vocab_size=48), 2, True, "pads it to a multiple of "),
        (vocabulary(tensor_model_parallel_size=4), 2, True, "which may be fewer rows"),
    ],
)
def test_the_padded_vocabulary_is_checked_at_another_tensor_parallel_size(
    args, fsdp_tp, warned, evidence
):
    """A job at the store's tensor-parallel size that pads its vocabulary itself pads it to a
    multiple of make_vocab_size_divisible_by x that size; --use-checkpoint-args keeps it."""
    ctx = context(args=args, fsdp_tp=fsdp_tp, fsdp_etp=1)
    assert ctx.pack_tp == fsdp_tp
    assert evidence in logged(ctx, "pack_tp")[1]
    vocabulary_warnings = [w for w in ctx.warnings if "padded_vocab_size" in w]
    assert len(vocabulary_warnings) == int(warned)
    if warned:
        assert vocabulary_warnings[0].startswith(f"--fsdp-tp {fsdp_tp}: a job at that tensor-")
        assert vocabulary_warnings[0].endswith(
            "load the store with --use-checkpoint-args, which takes padded_vocab_size from it"
        )
    unchecked = context(fsdp_tp=2, fsdp_etp=1)  # no padded vocabulary recorded
    assert "the padded vocabulary is not checked" in logged(unchecked, "pack_tp")[1]
    same = context(args=vocabulary(padded_vocab_size=48), fsdp_tp=1)  # the source's own size
    assert "vocabulary" not in logged(same, "pack_tp")[1]


def test_a_padded_vocabulary_no_job_at_the_size_can_shard_is_refused():
    shapes = {DENSE_FC1: (24, H), "decoder.layers.0.mlp.linear_fc2.weight": (H, 12)}
    with pytest.raises(RefusedError) as caught:
        context(shapes, args={**vocabulary(), "num_layers": 1, "num_experts": None}, fsdp_tp=3)
    refusal = caught.value.refusal
    assert refusal.code == errors.FC_R10
    assert refusal.message == (
        "args.padded_vocab_size=64 does not divide by --fsdp-tp 3: a job at that tensor-parallel "
        "size cannot shard the embedding and the output layer"
    )


def _manifest(**sizes) -> Manifest:
    return Manifest(
        tool_version="0.1.0",
        torch_version="2.8.0",
        source={"iter_dir": "/ckpt/iter_0000007", "format": TORCH_DIST},
        target={"format": FSDP_DTENSOR, "dialect": "mfsdp_v1"},
        options={},
        derivations=tuple({"field": f, "value": v, "evidence": ""} for f, v in sizes.items()),
        args_sanitized={},
        content_metadata={"present": False},
        dropped=(),
        output_metadata_sha256="abc",
    )


def test_an_fsdp_source_is_unpacked_at_its_arguments_sizes():
    ctx = context(source=FSDP_DTENSOR, args={"tensor_model_parallel_size": 2})
    assert (ctx.pack_tp, ctx.pack_etp) == (2, 2)
    assert logged(ctx, "pack_tp")[1].startswith(
        "args.tensor_model_parallel_size=2 of the job that saved the source; "
    )
    tp2 = {"tensor_model_parallel_size": 2}
    agreeing = context(source=FSDP_DTENSOR, manifest=_manifest(pack_tp=2, pack_etp=2), args=tp2)
    assert (agreeing.pack_tp, agreeing.pack_etp) == (2, 2)
    unpacked = _manifest(pack_tp=None, pack_etp=None)  # records nothing to check
    assert context(source=FSDP_DTENSOR, manifest=unpacked, args=tp2).pack_tp == 2
    with pytest.raises(RefusedError) as caught:
        context(source=FSDP_DTENSOR, manifest=_manifest(pack_tp=1, pack_etp=2), args=tp2)
    refusal = caught.value.refusal
    assert refusal.code == errors.FC_R10 and refusal.keys == ("model.module." + DENSE_FC1,)
    assert refusal.message == (
        "the source's conversion_manifest.json records that its fused tensors were packed at "
        "pack_tp=1, but its args.tensor_model_parallel_size=2"
    )
    assert "--ignore-manifest" in refusal.hint
    ignored = context(
        source=FSDP_DTENSOR, manifest=_manifest(pack_tp=1), args=tp2, manifest_ignored=True
    )
    assert ignored.pack_tp == 2


GDP_SPEC = ["megatron.core.models.hybrid.hybrid_layer_specs", "gated_delta_product_stack_spec"]
GDP_ARGS = {
    "hybrid_layer_pattern": "M-",
    "spec": GDP_SPEC,
    "num_experts": None,
    "quick_geglu": False,
    "mamba_num_heads": 4,
    "mamba_head_dim": 2,
    "mamba_num_groups": 2,
    "mamba_state_dim": 2,
    "gdp_num_householder": 1,
}
"""A plain hybrid model with one Gated DeltaProduct mixer: d = 8, GN = 4, H = 4, M = 1."""
GDP_PARTS = {
    "in_proj": {"z": 8, "V0": 8, "K0": 4, "Q": 4, "b0": 4, "a": 4},
    "conv1d": {"V0": 8, "K0": 4, "Q": 4},
}
GDP_MIXER = {
    **{
        f"decoder.layers.0.mixer.{name}.weight.{part}": (
            (rows, H) if name == "in_proj" else (rows, 1, 4)
        )
        for name, parts in GDP_PARTS.items()
        for part, rows in parts.items()
    },
    "decoder.layers.1.mlp.linear_fc1.weight": (F, H),
    "decoder.layers.1.mlp.linear_fc2.weight": (H, F),
}
GDP_FSDP = {
    "decoder.layers.0.mixer.in_proj.weight": (32, H),
    "decoder.layers.0.mixer.conv1d.weight": (16, 1, 4),
    "decoder.layers.1.mlp.linear_fc1.weight": (F, H),
    "decoder.layers.1.mlp.linear_fc2.weight": (H, F),
}


def test_fused_mixer_projections_are_packed_per_tensor_parallel_rank():
    ctx = context(GDP_MIXER, args=GDP_ARGS)
    assert (ctx.pack_tp, ctx.pack_etp) == (1, None)
    assert logged(ctx, "pack_tp") == (
        1,
        "args.tensor_model_parallel_size=1 of the source; 2 Gated DeltaProduct in_proj/conv1d "
        "that fsdp_dtensor keeps fused, packed per rank",
    )
    assert logged(ctx, "mixers")[0] == {"decoder": {"gdp": 1}}
    with pytest.raises(AmbiguousError) as caught:
        context(GDP_MIXER, args={**GDP_ARGS, "tensor_model_parallel_size": 2})
    refusal = caught.value.refusal
    assert refusal.code == errors.FC_A01
    assert refusal.message == (
        "fsdp_dtensor packs 2 Gated DeltaProduct in_proj/conv1d per rank of the job that loads "
        "it, and the source was saved with args.tensor_model_parallel_size=2: the store loads "
        "right only at the size it is packed at"
    )
    assert refusal.keys == (
        "decoder.layers.0.mixer.conv1d.weight.K0",
        "decoder.layers.0.mixer.in_proj.weight.K0",
    )
    tp4 = context(GDP_MIXER, args={**GDP_ARGS, "tensor_model_parallel_size": 2}, fsdp_tp=4)
    assert tp4.pack_tp == 4


def test_mixer_parts_that_do_not_divide_by_the_pack_size_are_refused():
    with pytest.raises(RefusedError) as caught:
        context(GDP_MIXER, args=GDP_ARGS, fsdp_tp=8)
    refusal = caught.value.refusal
    assert refusal.code == errors.FC_R10
    assert refusal.message == (
        "the parts of 2 stored Gated DeltaProduct in_proj/conv1d tensors do not divide by "
        "--fsdp-tp 8"
    )
    with pytest.raises(RefusedError) as caught:  # the store's own size, converting back
        context(GDP_FSDP, source=FSDP_DTENSOR, args={**GDP_ARGS, "tensor_model_parallel_size": 8})
    assert caught.value.refusal.message.endswith(
        "do not divide by the store's args.tensor_model_parallel_size=8"
    )


def test_gated_linear_units_and_mixers_are_packed_at_one_size():
    """A GEGLU hybrid model: its gated linear_fc1 and its mixer both packed per rank."""
    geglu = {**GDP_MIXER, "decoder.layers.1.mlp.linear_fc1.weight": (2 * F, H)}
    ctx = context(
        geglu, args={**GDP_ARGS, "quick_geglu": True, "tensor_model_parallel_size": 2}, fsdp_tp=2
    )
    assert logged(ctx, "pack_tp")[1] == (
        "--fsdp-tp 2; 1 gated linear_fc1 and 2 Gated DeltaProduct in_proj/conv1d that "
        "fsdp_dtensor keeps fused, packed per rank"
    )
