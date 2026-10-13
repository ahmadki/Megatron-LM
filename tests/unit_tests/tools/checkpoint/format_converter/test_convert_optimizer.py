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
from collections.abc import Callable, Collection, Mapping
from pathlib import Path

import pytest
import torch

from tools.checkpoint.format_converter import rules  # pylint: disable=unused-import
from tools.checkpoint.format_converter import errors
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.context import build_arch_context
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, MOMENTS, TORCH_DIST
from tools.checkpoint.format_converter.core.optimizer import plan_masters, plan_moments
from tools.checkpoint.format_converter.core.registry import REGISTRY
from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.engine.runner import RunOptions, run_program
from tools.checkpoint.format_converter.errors import RefusedError
from tools.checkpoint.format_converter.formats.base import Section, model_tensors
from tools.checkpoint.format_converter.formats.fsdp_mfsdp_v1 import FsdpMfsdpV1Adapter
from tools.checkpoint.format_converter.formats.fsdp_optimizer import (
    live_names,
    parse_optimizer_state,
)
from tools.checkpoint.format_converter.formats.torch_dist import TorchDistAdapter

from ._programs import packed_program, root_target
from ._synthetic import (
    fsdp_optimizer_items,
    renumber_layers,
    rne_bfloat16,
    stack_experts,
    stack_layers,
    write_fsdp_store,
)
from .test_convert_cross import bitwise_equal, read_tensors
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
) -> Path:
    """A Megatron-FSDP-like bf16 save of ``model`` (fp32 main weights, but the ``frozen``
    tensors keep their dtype) and the ``moments`` of the parameters they name."""
    state = {name: {m: moments[m][name] for m in MOMENTS} for name in moments[MOMENTS[0]]}
    tensors, _, paths = fsdp_optimizer_items(state, {}, BF16)
    fsdp_args = copy.copy(args)
    fsdp_args.ckpt_format, fsdp_args.use_megatron_fsdp = "fsdp_dtensor", True
    return write_fsdp_store(
        root / "iter_0000007",
        {key: tensor if key in frozen else tensor.float() for key, tensor in model.items()},
        args=fsdp_args,
        extra_tensors=tensors,
        planner_paths=paths,
    )


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
