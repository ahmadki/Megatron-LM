# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The architecture context: every field derived from the arguments, with its evidence."""

from __future__ import annotations

import argparse
import dataclasses

import pytest
import torch

from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.context import (
    FSDP_TO_TD,
    TD_TO_FSDP,
    build_arch_context,
)
from tools.checkpoint.format_converter.core.manifest import Manifest
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, TORCH_DIST
from tools.checkpoint.format_converter.engine.metadata import TensorEntry
from tools.checkpoint.format_converter.errors import (
    AmbiguousError,
    IntegrityError,
    RefusedError,
    UsageError,
)


def _model(**dtypes: torch.dtype) -> dict[str, TensorEntry]:
    return {key: TensorEntry(key, (2,), dtype, ()) for key, dtype in dtypes.items()}


def _args(**fields) -> ArgsView:
    return ArgsView(argparse.Namespace(**fields))


def test_torch_dist_to_fsdp_context_and_its_log():
    args = _args(
        bf16=True,
        fp16=False,
        tensor_model_parallel_size=2,
        expert_tensor_parallel_size=None,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=None,
    )
    model = _model(a=torch.bfloat16, b=torch.bfloat16, c=torch.float32)
    ctx = build_arch_context(source=TORCH_DIST, target=FSDP_DTENSOR, args=args, model=model)
    assert ctx.direction == TD_TO_FSDP
    assert ctx.source == (TORCH_DIST, "mcore") and ctx.target == (FSDP_DTENSOR, "mfsdp_v1")
    assert ctx.params_dtype == torch.bfloat16
    assert (ctx.src_tp, ctx.src_etp, ctx.ep, ctx.pp, ctx.vpp) == (2, 2, 1, 4, None)
    assert ctx.fsdp_model_prefix == "model.module."
    assert ctx.fsdp_optim_depth == "module.module.module."
    assert ctx.derivation_log() == [
        {
            "field": "direction",
            "value": TD_TO_FSDP,
            "evidence": "source torch_dist, target fsdp_dtensor",
        },
        {
            "field": "params_dtype",
            "value": "torch.bfloat16",
            "evidence": "args.bf16=True, args.fp16=False; stored model dtypes: bfloat16 x2, float32 x1",
        },
        {"field": "src_tp", "value": 2, "evidence": "args.tensor_model_parallel_size=2"},
        {
            "field": "src_etp",
            "value": 2,
            "evidence": "args.expert_tensor_parallel_size=None (unset: 2)",
        },
        {"field": "ep", "value": 1, "evidence": "args.expert_model_parallel_size=None (unset: 1)"},
        {"field": "pp", "value": 4, "evidence": "args.pipeline_model_parallel_size=4"},
        {
            "field": "vpp",
            "value": None,
            "evidence": "args.virtual_pipeline_model_parallel_size=None",
        },
        {"field": "lm_decoder", "value": "decoder", "evidence": "model keys under a, b, c"},
        {
            "field": "experts",
            "value": None,
            "evidence": "args.num_experts=None and no routed-expert tensors",
        },
        {
            "field": "gated_fc1",
            "value": {},
            "evidence": "0 linear_fc1 weights against their linear_fc2 inputs (gated: twice as "
            "many rows); args.swiglu=False, args.quick_geglu=False make the language model's MLPs "
            "(in decoder) plain",
        },
        {
            "field": "fsdp_swiglu",
            "value": False,
            "evidence": "args.swiglu=False of the job that loads the fsdp_dtensor target",
        },
        {
            "field": "swiglu_split",
            "value": {},
            "evidence": "Megatron-FSDP splits a linear_fc1 into SwiGLU halves iff its job's "
            "args.swiglu (here False) is set, its name is a SwiGLU name of "
            "handle_swiglu_in_state_dict and its layer is gated: 0 split, 0 gated kept fused",
        },
        {
            "field": "mixers",
            "value": {},
            "evidence": "no state-space mixer (mixer.in_proj or mixer.conv1d tensors)",
        },
        {"field": "pack_tp", "value": None, "evidence": "no fused tensor to pack per rank"},
        {"field": "pack_etp", "value": None, "evidence": "no fused tensor to pack per rank"},
        {"field": "fsdp_model_prefix", "value": "model.module.", "evidence": "dialect mfsdp_v1"},
        {
            "field": "fsdp_optim_depth",
            "value": "module.module.module.",
            "evidence": "params_dtype bfloat16",
        },
        {"field": "dtype_policy", "value": "preserve", "evidence": "weights-only conversion"},
        {
            "field": "manifest",
            "value": None,
            "evidence": "the source has no conversion_manifest.json",
        },
    ]
    assert args.consumed.keys() >= {"bf16", "fp16", "tensor_model_parallel_size"}


@pytest.mark.parametrize(
    ("fields", "keys", "depth", "note"),
    [
        (
            {},
            ["optimizer.state.module.module.decoder.final_norm.weight.exp_avg"],
            "module.module.",
            "; source optimizer keys use module.module.",
        ),
        (
            {"fp16": True},
            ["optimizer.state.module.module.module.embedding.word_embeddings.weight.exp_avg"],
            "module.module.module.",
            "; source optimizer keys use module.module.module.",
        ),
        (
            {"bf16": True},
            ["optimizer.state.module.module.decoder.final_norm.weight.exp_avg"],
            "module.module.module.",
            "; source optimizer keys use module.module. (disagrees; that state is not converted)",
        ),
        ({"bf16": True}, [], "module.module.module.", ""),
    ],
)
def test_fsdp_optimizer_depth_follows_the_precision_and_notes_the_keys(fields, keys, depth, note):
    ctx = build_arch_context(
        source=FSDP_DTENSOR,
        target=TORCH_DIST,
        args=_args(**fields),
        model=_model(w=torch.float32),
        optimizer_keys=keys,
    )
    assert ctx.direction == FSDP_TO_TD
    assert ctx.fsdp_optim_depth == depth
    (logged,) = [d for d in ctx.derivations if d.field == "fsdp_optim_depth"]
    assert logged.evidence.endswith(note) and logged.evidence.startswith("params_dtype")


def test_contradictory_precision_is_refused():
    with pytest.raises(RefusedError) as excinfo:
        build_arch_context(
            source=TORCH_DIST, target=FSDP_DTENSOR, args=_args(bf16=True, fp16=True), model={}
        )
    assert excinfo.value.refusal.code == errors.FC_R13
    assert "args.bf16=True and args.fp16=True" in excinfo.value.refusal.message


def test_same_format_is_not_a_cross_format_context():
    with pytest.raises(UsageError):
        build_arch_context(source=TORCH_DIST, target=TORCH_DIST, args=_args(), model={})


def _manifest(fmt: str = FSDP_DTENSOR, digest: str = "abc") -> Manifest:
    return Manifest(
        tool_version="0.1.0",
        torch_version="2.8.0",
        source={"iter_dir": "/ckpt/iter_0000007", "format": TORCH_DIST},
        target={"format": fmt, "dialect": "mfsdp_v1"},
        options={},
        derivations=(),
        args_sanitized={"ckpt_format": ("torch_dist", "fsdp_dtensor")},
        content_metadata={"present": False},
        dropped=(),
        output_metadata_sha256=digest,
    )


def test_a_manifest_must_describe_the_source_store():
    def build(manifest, digest="abc"):
        return build_arch_context(
            source=FSDP_DTENSOR,
            target=TORCH_DIST,
            args=_args(),
            model={},
            manifest=manifest,
            metadata_sha256=digest,
        )

    manifest = _manifest()
    ctx = build(manifest)
    assert ctx.manifest is manifest and ctx.dtype_policy.value == "preserve"
    (logged,) = [d for d in ctx.derivations if d.field == "manifest"]
    assert logged.value == TORCH_DIST and "from /ckpt/iter_0000007" in logged.evidence
    for stale in (_manifest(digest="other"), _manifest(fmt=TORCH_DIST)):
        with pytest.raises(AmbiguousError) as excinfo:
            build(stale)
        assert excinfo.value.refusal.code == errors.FC_A06
        assert "--ignore-manifest" in excinfo.value.refusal.hint


def test_an_ignored_manifest_is_logged_as_ignored():
    ctx = build_arch_context(
        source=FSDP_DTENSOR,
        target=TORCH_DIST,
        args=_args(),
        model={},
        manifest=_manifest(digest="other"),  # not checked either
        metadata_sha256="abc",
        manifest_ignored=True,
    )
    assert ctx.manifest is None
    (logged,) = [d for d in ctx.derivations if d.field == "manifest"]
    assert (logged.value, logged.evidence) == (
        None,
        "conversion_manifest.json ignored (--ignore-manifest)",
    )


def test_a_torch_dist_target_gets_the_content_metadata_the_manifest_records():
    def derived(manifest, **options):
        ctx = build_arch_context(
            source=FSDP_DTENSOR,
            target=TORCH_DIST,
            args=_args(),
            model={},
            manifest=manifest,
            metadata_sha256="abc",
            **options,
        )
        (logged,) = [d for d in ctx.derivations if d.field == "td_content_metadata"]
        assert logged.value == ctx.td_content_metadata
        return ctx.td_content_metadata, logged.evidence

    recorded = dataclasses.replace(
        _manifest(), content_metadata={"present": True, "value": {"non_homogeneous_layers": True}}
    )
    assert derived(recorded) == (
        {"non_homogeneous_layers": True},
        "restored from the source's conversion_manifest.json",
    )
    assert derived(recorded, drop_content_metadata=True) == (
        None,
        "--drop-common-key content_metadata",
    )
    assert derived(_manifest())[0] is None and derived(None)[0] is None
    with pytest.raises(IntegrityError, match="not a mapping"):
        derived(dataclasses.replace(recorded, content_metadata={"present": True, "value": [1]}))
    assert derived(recorded, per_layer=True) == (
        {"non_homogeneous_layers": True},
        "restored from the source's conversion_manifest.json; non_homogeneous_layers set by "
        "--torch-dist-layout per-layer",
    )
    kept = {"distrib_optim_sharding_type": "fully_reshardable", "non_homogeneous_layers": False}
    restored = dataclasses.replace(recorded, content_metadata={"present": True, "value": kept})
    assert derived(restored, per_layer=True)[0] == {**kept, "non_homogeneous_layers": True}
    assert derived(None, per_layer=True)[0] == {
        "non_homogeneous_layers": True,
        "singleton_local_shards": False,
        "chained_optim_avoid_prefix": True,
    }
    ctx = build_arch_context(source=TORCH_DIST, target=FSDP_DTENSOR, args=_args(), model={})
    assert ctx.td_content_metadata is None
    assert all(d.field != "td_content_metadata" for d in ctx.derivations)
