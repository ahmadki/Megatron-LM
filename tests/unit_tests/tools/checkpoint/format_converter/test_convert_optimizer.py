# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optimizer state from fsdp_dtensor to torch_dist, on synthetic Megatron-FSDP-like stores.

Every moment of every parameter is drawn on its own, under the names
Megatron-FSDP saves it (``optimizer.state.<depth><model key>.<moment>``), and
the torch_dist tensors it must become are built from the drawn ones by hand:
stacked with ``torch.stack`` over layers and experts, SwiGLU halves
concatenated gate first, modules renumbered by maps written out here, mixer
parts cut by explicit loops over ranks. The expected stores never come from
the converter.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Callable, Collection, Mapping
from pathlib import Path

import pytest
import torch

from tools.checkpoint.format_converter import rules  # pylint: disable=unused-import
from tools.checkpoint.format_converter import errors, optimizer_plan
from tools.checkpoint.format_converter.api import convert_checkpoint, verify_checkpoint
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.context import build_arch_context
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, MOMENTS, TORCH_DIST
from tools.checkpoint.format_converter.core.optimizer import plan_masters, plan_moments
from tools.checkpoint.format_converter.core.registry import REGISTRY
from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.engine.runner import RunOptions, run_program
from tools.checkpoint.format_converter.errors import InternalError, RefusedError
from tools.checkpoint.format_converter.formats.base import Section, model_tensors
from tools.checkpoint.format_converter.formats.fsdp_mfsdp_v1 import FsdpMfsdpV1Adapter
from tools.checkpoint.format_converter.formats.fsdp_optimizer import (
    live_names,
    parse_optimizer_state,
)
from tools.checkpoint.format_converter.formats.torch_dist import TorchDistAdapter
from tools.checkpoint.format_converter.options import ConvertOptions, VerifyOptions

from ._programs import packed_program, root_target
from ._synthetic import (
    fsdp_optimizer_items,
    gdn_tensors,
    renumber_layers,
    rne_bfloat16,
    sequential_experts,
    stack_experts,
    stack_layers,
    torch_save_bytes,
    write_fsdp_store,
    write_tracker,
)
from .test_convert_blocks import ITER, read_common
from .test_convert_cross import bitwise_equal, read_tensors, run
from .test_convert_gdn import GDN, LINEAR
from .test_convert_gdp import ARCHETYPE, archetype, archetype_args
from .test_convert_glu import MOE_HALVES, swiglu_args
from .test_convert_mamba import FUSED as MAMBA_FUSED
from .test_convert_mamba import OTHERS as MAMBA_OTHERS
from .test_convert_mamba import fsdp_expected as mamba_fsdp
from .test_convert_mamba import mamba_args
from .test_convert_mamba import torch_dist_store as mamba_torch_dist
from .test_convert_mtp import INNER, SHORTCUT, mtp_args

pytestmark = [
    pytest.mark.covers("identity", "ground_truth"),
    pytest.mark.covers("glu", "ground_truth"),
    pytest.mark.covers("mamba2", "ground_truth"),
]
BF16 = "module.module.module."
ADAPTER = FsdpMfsdpV1Adapter()
SHORTCUT_PATTERN = "E*E-*EE/*EE/*EE"
DECODER_TO_TD, INNER_TO_TD = (0, 1, 3, 4, 6), (0, 2)
"""The torch_dist index of each module of the shortcut-MoE model's layer lists (by hand)."""


def redraw(tensors: Mapping[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    """Fresh fp32 values of every tensor's shape (an optimizer state of the same parameters)."""
    generator = torch.Generator().manual_seed(seed)
    return {key: torch.randn(t.shape, generator=generator) for key, t in sorted(tensors.items())}


def draw_moments(model: Mapping[str, torch.Tensor], seed: int, frozen=()) -> dict[str, dict]:
    """Moments of every parameter of ``model`` but the ``frozen`` ones and the router expert
    biases (buffers)."""
    trained = {
        k: t for k, t in model.items() if k not in frozen and not k.endswith("router.expert_bias")
    }
    return {moment: redraw(trained, seed + i) for i, moment in enumerate(MOMENTS)}


def fuse_halves(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """SwiGLU halves ``<key>_w``/``<key>_v`` concatenated gate first into ``<key>``."""
    fused = {}
    for key, tensor in tensors.items():
        if key.endswith("_w"):
            fused[key[:-2]] = torch.cat([tensor, tensors[key[:-2] + "_v"]])
        elif not key.endswith("_v"):
            fused[key] = tensor
    return fused


def write_optimizer_store(
    root: Path,
    model: Mapping[str, torch.Tensor],
    moments: Mapping[str, dict],
    args,
    *,
    frozen: Collection[str] = (),
    group_meta: Mapping[str, Mapping] | None = None,
    scheduler: Mapping | None = None,
    steps: bool = False,
) -> Path:
    """A Megatron-FSDP-like bf16 save of ``model`` (fp32 main weights, but the ``frozen``
    tensors keep their dtype), the ``moments`` of the parameters they name, each parameter's
    ``group_meta`` by live name and the ``scheduler`` state, under Megatron-FSDP's keys and
    planner paths (with ``steps``, ``torch.optim.Adam``'s step per parameter too); with a
    tracker."""
    state = {name: {m: moments[m][name] for m in MOMENTS} for name in moments[MOMENTS[0]]}
    for leaves in state.values():
        if steps:
            leaves["step"] = torch.tensor(30.0)
    tensors, byte_items, paths = fsdp_optimizer_items(state, group_meta or {}, BF16)
    for field, value in (scheduler or {}).items():
        byte_items[f"opt_param_scheduler.{field}"] = torch_save_bytes(value)
        paths[f"opt_param_scheduler.{field}"] = ("opt_param_scheduler", field)
    fsdp_args = copy.copy(args)
    fsdp_args.ckpt_format, fsdp_args.use_megatron_fsdp = "fsdp_dtensor", True
    store = write_fsdp_store(
        root / "iter_0000007",
        {key: tensor if key in frozen else tensor.float() for key, tensor in model.items()},
        args=fsdp_args,
        extra_tensors=tensors,
        extra_byte_items=byte_items,
        planner_paths=paths,
    )
    write_tracker(root, 7)
    return store


def plan_inputs(store: Path, args, *, convert_optimizer: bool = True):
    """What planning a store's optimizer state starts from: its metadata, the architecture
    context, its model tensors and its parsed optimizer state."""
    view = MetadataView.load(store)
    sections = ADAPTER.sections(view)
    model = model_tensors(view, sections, ADAPTER)
    optimizer = [key for key, section in sections.items() if section is Section.OPTIMIZER_STATE]
    ctx = build_arch_context(
        source=FSDP_DTENSOR,
        target=TORCH_DIST,
        args=ArgsView(args),
        model=model,
        optimizer_keys=optimizer,
        convert_optimizer=convert_optimizer,
    )
    parsed = parse_optimizer_state(
        view, optimizer, depth=ctx.fsdp_optim_depth, live=live_names(model, ArgsView(args))
    )
    return view, ctx, model, parsed


def run_pieces(tmp_path, view: MetadataView, pieces) -> tuple[dict[str, torch.Tensor], tuple]:
    """Run ``pieces`` over ``view`` into a fresh store; its tensors and the units' journal."""
    target = root_target(tmp_path / "out")
    program = packed_program(view, target, list(pieces), unit_bytes=2**20)
    run = run_program(program, target, RunOptions())
    return read_tensors(target.iter_dir), run.journal


def convert_moments(tmp_path, store: Path, args) -> dict[str, torch.Tensor]:
    """Plan the moments of ``store`` for torch_dist, run the plan and read the output back."""
    view, ctx, model, parsed = plan_inputs(store, args)
    plan = plan_moments(
        ctx,
        model,
        parsed.moments,
        TorchDistAdapter(),
        rules=REGISTRY.rules_for(ctx),
        max_chunk_bytes=2**30,
    )
    return run_pieces(tmp_path, view, plan.pieces)[0]


def convert_masters(tmp_path, store: Path, args, **options) -> tuple[dict, tuple]:
    """Plan the model section and masters of ``store`` for torch_dist and run the plan."""
    view, ctx, model, parsed = plan_inputs(store, args)
    plan = plan_masters(
        ctx,
        model,
        parsed.trainable,
        TorchDistAdapter(),
        rules=REGISTRY.rules_for(ctx),
        max_chunk_bytes=2**30,
        **options,
    )
    return run_pieces(tmp_path, view, plan.pieces)


def expected_state(
    moments: Mapping[str, dict], to_torch_dist: Callable[[dict], dict]
) -> dict[str, torch.Tensor]:
    """``optimizer.state.<moment>.<key>`` of each moment laid out as torch_dist stores it."""
    return {
        f"optimizer.state.{moment}.{key}": tensor
        for moment in MOMENTS
        for key, tensor in to_torch_dist(moments[moment]).items()
    }


def assert_tensors(actual: dict, expected: dict) -> None:
    assert set(actual) == set(expected)
    for key, tensor in actual.items():
        assert bitwise_equal(tensor, expected[key]), key


def test_moments_of_a_swiglu_moe_gpt_are_stacked_and_merged(tmp_path):
    """Stacked layers, grouped experts under their global indices, SwiGLU halves."""
    args = swiglu_args(2, num_experts=4, moe_grouped_gemm=True)
    moments = draw_moments(MOE_HALVES, 0)
    store = write_optimizer_store(tmp_path / "src", MOE_HALVES, moments, args)
    actual = convert_moments(tmp_path, store, args)
    assert_tensors(
        actual,
        expected_state(moments, lambda m: stack_layers(stack_experts(fuse_halves(m)), "decoder")),
    )
    assert "optimizer.state.exp_avg.decoder.layers.mlp.experts.experts.linear_fc1.weight" in actual


def test_moments_of_a_shortcut_moe_hybrid_are_renumbered_like_its_modules(tmp_path):
    args = mtp_args(
        SHORTCUT_PATTERN, moe_shortcut_connection=True, num_experts=4, mtp_use_repeated_layer=True
    )
    moments = draw_moments(SHORTCUT, 10)
    store = write_optimizer_store(tmp_path / "src", SHORTCUT, moments, args)

    def to_torch_dist(tensors):
        decoder = renumber_layers(tensors, "decoder", DECODER_TO_TD)
        return stack_experts(renumber_layers(decoder, INNER, INNER_TO_TD))

    actual = convert_moments(tmp_path, store, args)
    assert_tensors(actual, expected_state(moments, to_torch_dist))
    assert "optimizer.state.exp_avg_sq.decoder.layers.6.mlp.router.weight" in actual
    assert f"optimizer.state.exp_avg.{INNER}.layers.2.mlp.router.weight" in actual


def test_moments_of_packed_mamba_mixers_are_cut_into_their_parts(tmp_path):
    """Mixers saved at tensor-parallel size 2: every packed moment is unpacked per part."""
    generator = torch.Generator().manual_seed(3)
    moments_by_part = {
        moment: (
            {
                name: {part: torch.randn(t.shape, generator=generator) for part, t in parts.items()}
                for name, parts in MAMBA_FUSED.items()
            },
            redraw(MAMBA_OTHERS, 20 + seed),
        )
        for seed, moment in enumerate(MOMENTS)
    }
    args = mamba_args(tensor_model_parallel_size=2)
    model = mamba_fsdp(2)
    moments = {moment: mamba_fsdp(2, *drawn) for moment, drawn in moments_by_part.items()}
    store = write_optimizer_store(tmp_path / "src", model, moments, args)
    actual = convert_moments(tmp_path, store, args)
    expected = {
        f"optimizer.state.{moment}.{key}": tensor
        for moment, drawn in moments_by_part.items()
        for key, tensor in mamba_torch_dist(*drawn, tp=2).items()
    }
    assert_tensors(actual, expected)
    assert "optimizer.state.exp_avg.decoder.layers.0.mixer.in_proj.weight.dt" in actual


def test_a_parameter_without_every_moment_is_refused(tmp_path):
    args = swiglu_args(2, num_experts=4, moe_grouped_gemm=True)
    store = write_optimizer_store(tmp_path / "src", MOE_HALVES, draw_moments(MOE_HALVES, 0), args)
    view = MetadataView.load(store)
    model = model_tensors(view, ADAPTER.sections(view), ADAPTER)
    ctx = build_arch_context(
        source=FSDP_DTENSOR, target=TORCH_DIST, args=ArgsView(args), model=model
    )
    missing = "decoder.layers.0.mlp.router.weight"
    entries = {
        "exp_avg": {name: model[name] for name in model},
        "exp_avg_sq": {name: model[name] for name in model if name != missing},
    }
    with pytest.raises(RefusedError) as caught:
        plan_moments(
            ctx,
            model,
            entries,
            TorchDistAdapter(),
            rules=REGISTRY.rules_for(ctx),
            max_chunk_bytes=1,
        )
    assert caught.value.refusal.message == "parameters with optimizer state but no exp_avg_sq"
    assert caught.value.refusal.keys == (missing,)


FROZEN = "embedding.word_embeddings.weight"
"""A frozen parameter: no optimizer state, stored in its own dtype (bf16)."""
HEAD = "decoder.hc_head_fn"
"""A parameter Megatron keeps in fp32 in a bf16 model (``mark_keep_in_fp32``)."""


def masters_model() -> dict[str, torch.Tensor]:
    """The SwiGLU MoE GPT with a keep-fp32 head; values with many mantissa bits to round."""
    generator = torch.Generator().manual_seed(8)
    model = {key: torch.randn(t.shape, generator=generator) for key, t in MOE_HALVES.items()}
    model[FROZEN] = MOE_HALVES[FROZEN]
    model[HEAD] = torch.randn(2, 8, generator=generator)
    return model


def test_masters_and_rounded_weights_of_trainable_parameters(tmp_path):
    """Masters as optimizer.state.param; weights rounded to nearest even (by integer arithmetic
    on the bits), but the keep-fp32 head; the frozen parameter and the router expert biases as
    stored; every stored chunk read once for both."""
    args = swiglu_args(2, num_experts=4, moe_grouped_gemm=True)
    model = masters_model()
    store = write_optimizer_store(
        tmp_path / "src", model, draw_moments(model, 0, frozen=(FROZEN,)), args, frozen=(FROZEN,)
    )
    actual, journal = convert_masters(tmp_path, store, args)
    fp32 = {key: tensor if key == FROZEN else tensor.float() for key, tensor in model.items()}
    trained = {k: t for k, t in fp32.items() if k != FROZEN and not k.endswith("expert_bias")}

    def layout(tensors):
        return stack_layers(stack_experts(fuse_halves(tensors)), "decoder")

    weights = {
        key: tensor if key not in layout(trained) or key == HEAD else rne_bfloat16(tensor)
        for key, tensor in layout(fp32).items()
    }
    masters = {f"optimizer.state.param.{key}": t for key, t in layout(trained).items()}
    assert_tensors(actual, {**weights, **masters})
    assert actual[FROZEN].dtype == torch.bfloat16 and actual[HEAD].dtype == torch.float32
    assert actual["decoder.layers.mlp.router.expert_bias"].dtype == torch.float32
    assert sum(r["torch_load_calls"] for r in journal) == sum(r["items_touched"] for r in journal)
    assert sum(r["items_touched"] for r in journal) == len(model)


def test_narrow_masters_are_refused_unless_allowed(tmp_path):
    args = swiglu_args(2, num_experts=4, moe_grouped_gemm=True)
    narrow = "output_layer.weight"  # a bf16 "master"
    store = write_optimizer_store(
        tmp_path / "src", MOE_HALVES, draw_moments(MOE_HALVES, 0), args, frozen=(narrow,)
    )
    with pytest.raises(RefusedError) as caught:
        convert_masters(tmp_path, store, args)
    assert caught.value.codes == (errors.FC_R21,)
    assert caught.value.refusal.keys == (f"model.module.{narrow}",)
    actual, _ = convert_masters(tmp_path, store, args, allow_bf16_masters=True)
    assert bitwise_equal(actual[f"optimizer.state.param.{narrow}"], MOE_HALVES[narrow].float())
    assert bitwise_equal(actual[narrow], MOE_HALVES[narrow])


def test_a_tensor_built_from_trained_and_frozen_halves_is_refused(tmp_path):
    args = swiglu_args(2, num_experts=4, moe_grouped_gemm=True)
    half = "decoder.layers.0.mlp.experts.linear_fc1.weight2_v"
    moments = draw_moments(MOE_HALVES, 0, frozen=(half,))
    store = write_optimizer_store(tmp_path / "src", MOE_HALVES, moments, args)
    with pytest.raises(RefusedError) as caught:
        convert_masters(tmp_path, store, args)
    assert caught.value.codes == (errors.FC_R13,)
    assert f"model.module.{half}" in caught.value.refusal.keys


SCHEDULER = {
    "max_lr": 1e-3,
    "lr_warmup_steps": 2,
    "num_steps": 30,
    "lr_decay_style": "cosine",
    "lr_decay_steps": 100,
    "min_lr": 1e-5,
    "start_wd": 0.01,
    "end_wd": 0.02,
    "wd_incr_steps": 100,
    "wd_incr_style": "linear",
}
DECAY = {
    "lr": 9.9e-4,
    "betas": (0.9, 0.95),
    "eps": 1e-8,
    "weight_decay": 0.0103,
    "bias_correction": True,
    "wd_mult": 1.0,
    "lr_mult": 1.0,
    "max_lr": 1e-3,
    "min_lr": 1e-5,
    "is_expert_parallel": False,
    "is_decoupled_lr": False,
    "default_config": True,
    "step": 30,
}
NO_DECAY = {**DECAY, "wd_mult": 0.0, "weight_decay": 0.0}
FLAGS = {
    "distrib_optim_sharding_type": "fully_reshardable",
    "distrib_optim_fully_reshardable_mem_efficient": False,
    "singleton_local_shards": False,
    "chained_optim_avoid_prefix": True,
}
"""The content_metadata of a fully reshardable torch_dist save (written out from Megatron's)."""


def expert_parallel(hparams: dict) -> dict:
    return {**hparams, "is_expert_parallel": True}


def default_groups(live: Mapping[str, torch.Tensor], *, experts_parallel: bool = False) -> dict:
    """Megatron's default groups of the live parameters ``live``: no weight decay for 1-D ones,
    and with expert parallelism the routed experts' own groups."""
    meta = {}
    for name, tensor in live.items():
        if name.endswith("router.expert_bias"):
            continue  # a buffer
        hparams = NO_DECAY if tensor.ndim == 1 else DECAY
        meta[name] = (
            expert_parallel(hparams) if experts_parallel and ".mlp.experts." in name else hparams
        )
    return meta


def optimizer_state(groups: list[list[dict]]) -> dict:
    """The common state's optimizer entry of chained optimizers' ``groups`` (by hand)."""
    states = [
        {"optimizer": {"param_groups": g}, "param_state_sharding_type": "fully_reshardable"}
        for g in groups
    ]
    return states[0] if len(states) == 1 else dict(enumerate(states))


def convert_with_optimizer(tmp_path, root: Path, **options) -> tuple[Path, dict]:
    report = convert_checkpoint(
        root, tmp_path / "td", ConvertOptions(workers=0, optimizer="convert", **options)
    )
    return tmp_path / "td" / ITER, report.to_json()


def test_a_training_save_converts_with_its_optimizer_and_scheduler(tmp_path):
    """Masters, rounded weights, moments, parameter groups, scheduler, flags and group map."""
    args = swiglu_args(2, num_experts=4, moe_grouped_gemm=True)
    model = masters_model()
    trained = {key: tensor for key, tensor in model.items() if key != FROZEN}
    store = write_optimizer_store(
        tmp_path / "src",
        model,
        draw_moments(model, 0, frozen=(FROZEN,)),
        args,
        frozen=(FROZEN,),
        group_meta=default_groups(fuse_halves(trained)),
        scheduler=SCHEDULER,
    )
    out, report = convert_with_optimizer(tmp_path, store.parent)
    actual = read_tensors(out)
    moments = draw_moments(model, 0, frozen=(FROZEN,))
    layout = lambda t: stack_layers(stack_experts(fuse_halves(t)), "decoder")  # noqa: E731
    fp32 = {key: tensor if key == FROZEN else tensor.float() for key, tensor in model.items()}
    masters = layout({k: t for k, t in fp32.items() if k in moments["exp_avg"]})
    weights = {
        key: rne_bfloat16(tensor) if key in masters and key != HEAD else tensor
        for key, tensor in layout(fp32).items()
    }
    assert_tensors(
        actual,
        {
            **weights,
            **{f"optimizer.state.param.{key}": tensor for key, tensor in masters.items()},
            **expected_state(moments, layout),
        },
    )
    common = read_common(out)
    assert common["optimizer"] == optimizer_state([[NO_DECAY, DECAY]])
    assert common["opt_param_scheduler"] == SCHEDULER
    assert common["content_metadata"] == FLAGS
    assert common["iteration"] == 7
    args_out = common["args"]
    assert (args_out.no_save_optim, args_out.no_save_rng) == (False, True)
    assert args_out.ckpt_format == "torch_dist" and not args_out.use_megatron_fsdp
    group_map = json.loads((out / "param_to_param_group_map.json").read_text())
    live = default_groups(fuse_halves(trained))
    assert group_map == {
        f"model.module.{name}": 0 if hparams is NO_DECAY else 1 for name, hparams in live.items()
    }
    assert report["optimizer"] == {
        "parameters": len(moments["exp_avg"]),
        "param_groups": [2],
        "dropped_steps": 0,
    }
    actions = {a["section"]: a["action"] for a in report["sections"]}
    assert actions["optimizer.state"] == actions["optimizer other"] == "convert"
    assert actions["scheduler"] == "convert"
    assert (out / "metadata.json").is_file()
    assert verify_checkpoint(store, out, VerifyOptions(map="convert")).ok  # plan replay


def test_routed_experts_under_expert_parallelism_get_a_chained_optimizer(tmp_path):
    """Sequential experts at expert-parallel size 2: {0: dense groups, 1: expert groups}."""
    args = swiglu_args(2, num_experts=4, moe_grouped_gemm=False, expert_model_parallel_size=2)
    model = sequential_experts(MOE_HALVES)
    meta = default_groups(fuse_halves(model), experts_parallel=True)
    store = write_optimizer_store(
        tmp_path / "src", model, draw_moments(model, 0), args, group_meta=meta, scheduler=SCHEDULER
    )
    out, _ = convert_with_optimizer(tmp_path, store.parent)
    common = read_common(out)
    expert_groups = [expert_parallel(NO_DECAY), expert_parallel(DECAY)]
    assert common["optimizer"] == optimizer_state([[NO_DECAY, DECAY], expert_groups])
    index = {0: NO_DECAY, 1: DECAY, 2: expert_groups[0], 3: expert_groups[1]}
    group_map = json.loads((out / "param_to_param_group_map.json").read_text())
    assert {name: index[gid] for name, gid in group_map.items()} == {
        f"model.module.{name}": hparams for name, hparams in meta.items()
    }
    assert "optimizer.state.exp_avg.decoder.layers.mlp.experts.experts.linear_fc1.weight" in (
        read_tensors(out)
    )


def archetype_store(root: Path, family: str, tp: int) -> tuple[Path, object, dict, dict]:
    """The vision-language archetype with ``family`` mixers (``gdp`` or ``gdn``) packed at
    ``tp``, saved by Megatron-FSDP with moments that are elementwise affine maps of its weights,
    and every parameter's default group; its store, arguments, torch_dist tensors and groups by
    live name."""
    if family == "gdp":
        fsdp, source, _ = archetype(tp)
        args = archetype_args(tp)
    else:
        fsdp, source, _ = archetype(tp, gdn_tensors, GDN, module="self_attention")
        args = archetype_args(tp, ARCHETYPE.replace("M", "G"), spec=None, **LINEAR)
    moments = {
        moment: {
            key: affine(tensor)
            for key, tensor in fsdp.items()
            if not key.endswith("router.expert_bias")
        }
        for moment, affine in AFFINE.items()
    }
    live = live_names(fsdp, ArgsView(args))
    trained = {live.get(key, key): fsdp[key] for key in moments["exp_avg"]}
    meta = default_groups(trained)
    store = write_optimizer_store(root, fsdp, moments, args, group_meta=meta, scheduler=SCHEDULER)
    return store, args, source, meta


AFFINE = {"exp_avg": lambda t: t.float() * 3 + 1, "exp_avg_sq": lambda t: t.float() * -2 + 0.5}
"""Moments that are elementwise maps of the weights: they commute with stacking, renumbering,
concatenation and packing, so the torch_dist source's own tensors give the expected moments."""


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("family", ["gdp", "gdn"])
def test_moments_of_the_vision_language_archetype_follow_its_tensors(tmp_path, family, tp):
    """Gated DeltaProduct or Gated DeltaNet packs, shortcut renumbering, the gated vision
    linear_fc1 and the ``language_model.`` prefix, for the moments alone and in a whole
    conversion with masters and groups."""
    store, args, source, meta = archetype_store(tmp_path / "src", family, tp)
    trained = {k: t for k, t in source.items() if not k.endswith("router.expert_bias")}
    expected = {
        f"optimizer.state.{moment}.{key}": affine(tensor)
        for moment, affine in AFFINE.items()
        for key, tensor in trained.items()
    }
    assert_tensors(convert_moments(tmp_path / "moments", store, args), expected)
    out, _ = convert_with_optimizer(tmp_path, store.parent)
    actual = read_tensors(out)
    moments = {key: t for key, t in actual.items() if key.startswith("optimizer.state.exp_avg")}
    assert_tensors(moments, expected)
    for key, tensor in trained.items():
        assert bitwise_equal(actual[f"optimizer.state.param.{key}"], tensor.float()), key
    group_map = json.loads((out / "param_to_param_group_map.json").read_text())
    assert set(group_map) == {f"model.module.{name}" for name in meta}


@pytest.mark.parametrize("kept", [True, False], ids=["groups-keep-the-step", "groups-without"])
def test_per_parameter_steps_are_dropped_only_where_the_groups_keep_the_step(tmp_path, kept):
    """``torch.optim.Adam`` keeps a step per parameter; the torch_dist loader reads the groups'."""
    args = swiglu_args(2, num_experts=4, moe_grouped_gemm=True)
    meta = default_groups(fuse_halves(MOE_HALVES))
    if not kept:
        meta = {
            name: {k: v for k, v in hparams.items() if k != "step"}
            for name, hparams in meta.items()
        }
    store = write_optimizer_store(
        tmp_path / "src",
        MOE_HALVES,
        draw_moments(MOE_HALVES, 0),
        args,
        group_meta=meta,
        scheduler=SCHEDULER,
        steps=True,
    )
    if kept:
        _, report = convert_with_optimizer(tmp_path, store.parent)
        assert report["optimizer"]["dropped_steps"] == len(draw_moments(MOE_HALVES, 0)["exp_avg"])
        return
    with pytest.raises(RefusedError) as caught:
        convert_with_optimizer(tmp_path, store.parent)
    assert caught.value.codes == (errors.FC_R16,)


SHORTCUT_ARGS = mtp_args(
    SHORTCUT_PATTERN, moe_shortcut_connection=True, num_experts=4, mtp_use_repeated_layer=True
)


def shortcut_torch_dist(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The shortcut-MoE model's tensors as torch_dist stores them: the modules at the hand-written
    pattern positions, experts stacked."""
    decoder = renumber_layers(tensors, "decoder", DECODER_TO_TD)
    return stack_experts(renumber_layers(decoder, INNER, INNER_TO_TD))


def optimizer_findings(out: Path, model: Mapping, moments: Mapping) -> list[str]:
    """Where the torch_dist store's optimizer state differs from the drawn state laid out by the
    hand-written module map."""
    trained = {key: model[key].float() for key in moments["exp_avg"]}
    expected = {
        **{f"optimizer.state.param.{k}": t for k, t in shortcut_torch_dist(trained).items()},
        **expected_state(moments, shortcut_torch_dist),
    }
    actual = {k: t for k, t in read_tensors(out).items() if k.startswith("optimizer.state.")}
    findings = [f"missing {key}" for key in sorted(set(expected) - set(actual))]
    findings += [f"unexpected {key}" for key in sorted(set(actual) - set(expected))]
    findings += [
        f"other values {key}"
        for key in sorted(set(actual) & set(expected))
        if not bitwise_equal(actual[key], expected[key])
    ]
    return findings


def write_shortcut_store(root: Path) -> tuple[Path, dict]:
    moments = draw_moments(SHORTCUT, 30)
    store = write_optimizer_store(
        root,
        SHORTCUT,
        moments,
        SHORTCUT_ARGS,
        group_meta=default_groups(SHORTCUT),
        scheduler=SCHEDULER,
    )
    return store, moments


def test_shortcut_moe_optimizer_state_is_renumbered_like_the_model(tmp_path):
    """Masters and moments of torch_dist modules 0, 1, 3, 4, 6 (MTP inner 0, 2) are the drawn
    state of the layer-list modules 0-4 (MTP inner 0, 1), as the model's tensors are."""
    store, moments = write_shortcut_store(tmp_path / "src")
    out, _ = convert_with_optimizer(tmp_path, store.parent)
    assert optimizer_findings(out, SHORTCUT, moments) == []
    tensors = read_tensors(out)
    router = "mlp.router.weight"
    assert bitwise_equal(
        tensors[f"optimizer.state.param.decoder.layers.6.{router}"],
        SHORTCUT[f"decoder.layers.4.{router}"].float(),
    )
    assert bitwise_equal(
        tensors[f"optimizer.state.exp_avg_sq.{INNER}.layers.2.{router}"],
        moments["exp_avg_sq"][f"{INNER}.layers.1.{router}"],
    )
    assert bitwise_equal(
        tensors[f"decoder.layers.6.{router}"],
        rne_bfloat16(SHORTCUT[f"decoder.layers.4.{router}"].float()),
    )
    group_map = json.loads((out / "param_to_param_group_map.json").read_text())
    assert f"model.module.decoder.layers.4.{router}" in group_map  # live names are dense
    assert f"model.module.{INNER}.layers.1.{router}" in group_map
    assert f"model.module.decoder.layers.6.{router}" not in group_map


def test_the_renumbering_check_fails_when_only_the_model_section_is_renumbered(
    tmp_path, monkeypatch
):
    """A converter that renumbers the model section but keeps the moments' layer-list indices."""
    real = optimizer_plan.plan_moments

    def unrenumbered(ctx, *args, **kwargs):
        dense = tuple(
            dataclasses.replace(block, dense_to_td=tuple(range(block.num_modules)))
            for block in ctx.blocks
        )
        return real(dataclasses.replace(ctx, blocks=dense), *args, **kwargs)

    monkeypatch.setattr(optimizer_plan, "plan_moments", unrenumbered)
    store, moments = write_shortcut_store(tmp_path / "src")
    with pytest.raises(InternalError, match="exp_avg keys of the plan are not its masters' keys"):
        convert_with_optimizer(tmp_path, store.parent)  # the converter's own cross-check
    monkeypatch.setattr(optimizer_plan, "_check_states", lambda masters, moments: None)
    out, _ = convert_with_optimizer(tmp_path, store.parent)
    findings = optimizer_findings(out, SHORTCUT, moments)  # and the test's hand-written map
    assert f"missing optimizer.state.exp_avg.decoder.layers.6.mlp.router.weight" in findings
    assert any(
        f.startswith("unexpected optimizer.state.exp_avg.decoder.layers.2.") for f in findings
    )
    assert not any("optimizer.state.param." in f for f in findings)  # the masters follow the map


def test_plan_reports_the_optimizer_conversion(tmp_path, capsys):
    args = swiglu_args(2, num_experts=4, moe_grouped_gemm=True)
    store = write_optimizer_store(
        tmp_path / "src",
        MOE_HALVES,
        draw_moments(MOE_HALVES, 0),
        args,
        group_meta=default_groups(fuse_halves(MOE_HALVES)),
        scheduler=SCHEDULER,
    )
    assert run("plan", store.parent, "--to", "torch_dist", "--optimizer", "convert", "--json") == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["optimizer"]["param_groups"] == [2]
    derived = {d["field"]: d["value"] for d in plan["derivations"]}
    assert derived["dtype_policy"] == "masters"
    assert plan["common"]["args_changes"]["no_save_rng"] == [False, True]


def _refused(tmp_path, capsys, store: Path, *flags: str) -> str:
    argv = ("plan", store.parent, "--to", "torch_dist", "--optimizer", "convert", *flags)
    code = run(*argv)
    return f"{code} {capsys.readouterr().err}"


def test_incomplete_or_foreign_optimizer_state_is_refused(tmp_path, capsys):
    args = swiglu_args(2, num_experts=4, moe_grouped_gemm=True)
    moments = draw_moments(MOE_HALVES, 0)
    meta = default_groups(fuse_halves(MOE_HALVES))
    weights = write_optimizer_store(tmp_path / "w", MOE_HALVES, {m: {} for m in MOMENTS}, args)
    assert "3 error: FC-R13: " in _refused(tmp_path, capsys, weights)
    assert "holds no optimizer state to convert" in _refused(tmp_path, capsys, weights)
    unscheduled = write_optimizer_store(tmp_path / "u", MOE_HALVES, moments, args, group_meta=meta)
    assert "no learning-rate scheduler state" in _refused(tmp_path, capsys, unscheduled)
    groupless = {k: v for k, v in meta.items() if not k.startswith("output_layer")}
    partial = write_optimizer_store(
        tmp_path / "p", MOE_HALVES, moments, args, group_meta=groupless, scheduler=SCHEDULER
    )
    error = _refused(tmp_path, capsys, partial)
    assert (
        "FC-R13: parameters with optimizer state but no parameter group [output_layer.weight]"
        in error
    )
    sgd = swiglu_args(2, num_experts=4, moe_grouped_gemm=True, optimizer="sgd")
    other = write_optimizer_store(
        tmp_path / "s", MOE_HALVES, moments, sgd, group_meta=meta, scheduler=SCHEDULER
    )
    assert "3 error: FC-R16: args.optimizer='sgd'" in _refused(tmp_path, capsys, other)
    complete = write_optimizer_store(
        tmp_path / "c", MOE_HALVES, moments, args, group_meta=meta, scheduler=SCHEDULER
    )
    flagless = _refused(tmp_path, capsys, complete, "--drop-common-key", "content_metadata")
    assert flagless.startswith("2 ") and "writes content_metadata" in flagless
    code = run("plan", complete.parent, "--to", "torch_dist", "--allow-bf16-masters")
    assert code == 2 and "--allow-bf16-masters applies to converted optimizer state" in (
        capsys.readouterr().err
    )


def test_the_plan_text_names_the_optimizer_state(tmp_path, capsys):
    args = swiglu_args(2, num_experts=4, moe_grouped_gemm=False, expert_model_parallel_size=2)
    model = sequential_experts(MOE_HALVES)
    store = write_optimizer_store(
        tmp_path / "src",
        model,
        draw_moments(model, 0),
        args,
        group_meta=default_groups(fuse_halves(model), experts_parallel=True),
        scheduler=SCHEDULER,
    )
    assert run("plan", store.parent, "--to", "torch_dist", "--optimizer", "convert") == 0
    text = capsys.readouterr().out
    assert "model weights and optimizer state" in text
    assert "2 + 2 parameter groups, 0 per-parameter steps dropped" in text
