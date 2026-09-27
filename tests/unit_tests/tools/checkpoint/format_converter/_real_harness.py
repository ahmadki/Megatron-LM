# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real Megatron saves and loads for the real-save (T3) tests.

Classic (DDP) and Megatron-FSDP v1 models are built with
``setup_model_and_optimizer``, saved with Megatron's own ``save_checkpoint``
and loaded with its own ``load_checkpoint`` under mocked training arguments.
:func:`run_chain` runs the whole cross-format chain on one kind of model.

It imports Megatron at module level, so a test module must gate the import:
skip the module unless ``megatron_available()`` (from
``mcore_state.common_state``) and ``torch.cuda.is_available()``, take its shared
directory from the root conftest's ``tmp_path_dist_ckpt`` fixture (skipping when
that conftest is not loaded), and after initializing the process groups skip
when :func:`nccl_skip_reason` gives a reason (no nvidia-resiliency-ext, or a
default process group that is not NCCL). ``real/test_same_format.py`` shows the
pattern.
"""

from __future__ import annotations

import contextlib
import copy
import re
import zlib
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from unittest import mock

import torch
from torch.distributed.tensor import DTensor

from megatron.core import parallel_state
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    uneven_dtensor_to_full_tensor,
)
from megatron.core.model_parallel_config import resolve_tensor_parallel_weight_shards
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from tests.unit_tests.dist_checkpointing.utils import (
    HAVE_NVRX,
    init_basic_mock_args,
    init_checkpointing_mock_args,
    initialize_gpt_model,
    setup_model_and_optimizer,
)
from tests.unit_tests.test_utilities import Utils
from tools.checkpoint.format_converter.api import convert_checkpoint, verify_checkpoint
from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.options import ConvertOptions, VerifyOptions

__all__ = [
    "CHAIN_CHECKS",
    "Utils",
    "build_gpt",
    "build_hybrid",
    "build_llava",
    "build_te_gpt",
    "full_params",
    "live_params",
    "load_weights_only",
    "nccl_skip_reason",
    "on_rank_zero",
    "randomize_parameters",
    "real_save",
    "resolve_weight_shards",
    "run_chain",
    "save_weights_only",
    "set_format",
    "training_args",
]

HYBRID_HIDDEN = 64
HYBRID_HEADS = 4
HYBRID_VOCAB = 128
GPT_LAYERS = 3
"""Layers of the tiny GPT (TE spec) and of the language model of the tiny LLaVA."""
VISION_LAYERS = 2
VISION_HIDDEN = 16


def nccl_skip_reason() -> str | None:
    """Why real-save tests cannot run on the initialized default process group, if they cannot."""
    if not HAVE_NVRX:
        return "needs nvidia-resiliency-ext (Megatron's checkpoint tests require it)"
    backend = torch.distributed.get_backend()
    if backend != "nccl":
        return f"the default process group is {backend}, not nccl (left by an earlier test?)"
    return None


@contextlib.contextmanager
def training_args(ckpt_dir: Path, *, tp: int, pp: int = 1) -> Iterator:
    """Mocked training arguments for saving to and loading from ``ckpt_dir``.

    Distributed optimizer, fully reshardable optimizer state, bf16, no tokenizer
    assets, and the weight-shard fields a real job saves
    (:func:`resolve_weight_shards`); ``megatron.training.checkpointing.get_args``
    returns them.
    """
    args = parse_args(ignore_unknown_args=True)
    args.save_tokenizer_assets = False
    with mock.patch("megatron.training.checkpointing.get_args", new=lambda: args):
        init_basic_mock_args(args, tp, pp)
        init_checkpointing_mock_args(args, str(ckpt_dir), fully_parallel=True)
        args.dist_ckpt_optim_fully_reshardable = True
        resolve_weight_shards(args)
        yield args


def resolve_weight_shards(args) -> None:
    """Set the (expert) tensor-parallel weight-shard fields as ``validate_args`` does.

    The mocked arguments skip ``validate_args``, which stores
    ``tensor_parallel_num_weight_shards = tensor_model_parallel_size *
    gtp_weight_remat_size`` (the expert pair likewise, the expert
    tensor-parallel size defaulting to the tensor-parallel size) in every
    real save, grouped tensor parallelism or not.
    """
    if args.expert_tensor_parallel_size is None:
        args.expert_tensor_parallel_size = args.tensor_model_parallel_size
    args.tensor_parallel_num_weight_shards, args.gtp_weight_remat_size = (
        resolve_tensor_parallel_weight_shards(
            args.tensor_model_parallel_size,
            args.tensor_parallel_num_weight_shards,
            getattr(args, "gtp_weight_remat_size", 1),
        )
    )
    args.expert_tensor_parallel_num_weight_shards, args.expert_gtp_weight_remat_size = (
        resolve_tensor_parallel_weight_shards(
            args.expert_tensor_parallel_size,
            args.expert_tensor_parallel_num_weight_shards,
            getattr(args, "expert_gtp_weight_remat_size", 1),
        )
    )


def build_gpt(seed: int, *, tp: int, pp: int = 1):
    """A tiny GPT (local spec, bf16) with a distributed Adam optimizer and random moments."""
    return setup_model_and_optimizer(seed=seed, tp=tp, pp=pp, initialize_fn=initialize_gpt_model)


def real_save(model, optimizer, *, iteration: int) -> None:
    """Megatron's own ``save_checkpoint`` (model, optimizer, RNG and common state)."""
    save_checkpoint(iteration, model, optimizer, None, 0)


def live_params(model) -> dict[str, torch.Tensor]:
    """A copy of every parameter of the first model chunk, by name."""
    return {name: param.detach().clone() for name, param in model[0].named_parameters()}


def load_weights_only(model, optimizer, args, load_dir: Path) -> int:
    """Load ``load_dir`` with ``--no-load-optim --no-load-rng`` and strict key checking.

    Returns:
        The loaded iteration.
    """
    args.load = str(load_dir)
    args.no_load_optim = True
    args.no_load_rng = True
    args.finetune = False
    args.dist_ckpt_strictness = "raise_all"
    args.strict_fsdp_dtensor_load = True
    with mock.patch("megatron.training.checkpointing.check_checkpoint_args"):
        with mock.patch("megatron.training.checkpointing.update_num_microbatches"):
            iteration, _ = load_checkpoint(model, optimizer, None)
    return iteration


def hybrid_provider(
    pre_process: bool = True, post_process: bool = True, *, seed: int, pattern: str, **kwargs
) -> HybridModel:
    """A tiny hybrid model of ``pattern`` with the default (Transformer Engine) stack spec.

    Linear layers have biases, the MLP is not gated, and the output layer is
    untied from the embedding, so every kind of tensor these layers have is
    in the checkpoint. Every parameter holds distinct values that depend on
    ``seed`` (:func:`randomize_parameters`).
    """
    for extra in ("pg_collection", "config", "vp_stage"):
        kwargs.pop(extra, None)
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    config = TransformerConfig(
        num_layers=len(pattern.replace("|", "")),
        hidden_size=HYBRID_HIDDEN,
        num_attention_heads=HYBRID_HEADS,
        use_cpu_initialization=True,
        add_bias_linear=True,
        gated_linear_unit=False,
        **kwargs,
    )
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=HYBRID_VOCAB,
        max_sequence_length=4,
        hybrid_layer_pattern=pattern,
        pre_process=pre_process,
        post_process=post_process,
        position_embedding_type="rope",
        share_embeddings_and_output_weights=False,
    )
    randomize_parameters(model, seed)
    return model


def randomize_parameters(model: torch.nn.Module, seed: int) -> None:
    """Overwrite every parameter with random values drawn from ``(seed, name)``.

    Megatron initializes biases and norms to constants, so two models of
    different seeds would agree on them and a converter that wrote them
    under the wrong layer would go unnoticed. Tensor-parallel shards also
    draw from their rank, so replicated parameters stay equal across ranks
    and the shards of one parameter differ.
    """
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    with torch.no_grad():
        for name, param in model.named_parameters():
            shard = tp_rank if getattr(param, "tensor_model_parallel", False) else "all"
            stream = zlib.crc32(f"{seed}/{name}/{shard}".encode())
            param.copy_(torch.randn(param.shape, generator=torch.Generator().manual_seed(stream)))


def build_hybrid(
    seed: int, *, tp: int, pattern: str, use_megatron_fsdp: bool = False, bf16: bool = True
):
    """A tiny hybrid model (classic or Megatron-FSDP v1) with a distributed optimizer."""
    return setup_model_and_optimizer(
        seed=seed,
        tp=tp,
        pp=1,
        initialize_fn=partial(hybrid_provider, pattern=pattern),
        use_megatron_fsdp=use_megatron_fsdp,
        bf16=bf16,
    )


def _transformer_config(num_layers: int, hidden: int, heads: int, **kwargs) -> TransformerConfig:
    """A TransformerConfig of linear layers with biases and no gated linear unit."""
    for extra in ("pg_collection", "config", "vp_stage"):
        kwargs.pop(extra, None)
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden,
        num_attention_heads=heads,
        use_cpu_initialization=True,
        add_bias_linear=True,
        gated_linear_unit=False,
        **kwargs,
    )


def te_gpt_provider(
    pre_process: bool = True, post_process: bool = True, *, seed: int, **kwargs
) -> GPTModel:
    """A tiny GPT with the Transformer Engine layer spec (stacked in torch_dist).

    Linear layers have biases, the MLP is not gated and the output layer is
    untied, and every parameter holds distinct values (:func:`randomize_parameters`).
    """
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    model = GPTModel(
        config=_transformer_config(GPT_LAYERS, HYBRID_HIDDEN, HYBRID_HEADS, **kwargs),
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=HYBRID_VOCAB,
        max_sequence_length=4,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=False,
    )
    randomize_parameters(model, seed)
    return model


def llava_provider(
    pre_process: bool = True, post_process: bool = True, *, seed: int, **kwargs
) -> LLaVAModel:
    """A tiny LLaVA: a GPT language model, a CLIP vision encoder and an MLP projection.

    All three use Transformer Engine layers with biases and no gated linear
    unit, and every parameter holds distinct values (:func:`randomize_parameters`).
    """
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    language = _transformer_config(GPT_LAYERS, HYBRID_HIDDEN, HYBRID_HEADS, **kwargs)
    language.language_model_type = "dummy"
    vision = _transformer_config(VISION_LAYERS, VISION_HIDDEN, 2, **kwargs)
    vision.vision_model_type = "clip"
    projection = _transformer_config(1, HYBRID_HIDDEN, 2, ffn_hidden_size=32, **kwargs)
    submodules = get_gpt_layer_with_transformer_engine_submodules()
    model = LLaVAModel(
        language_transformer_config=language,
        language_transformer_layer_spec=ModuleSpec(module=TransformerLayer, submodules=submodules),
        language_vocab_size=HYBRID_VOCAB,
        language_max_sequence_length=64,
        vision_transformer_config=vision,
        vision_transformer_layer_spec=ModuleSpec(
            module=TransformerLayer, submodules=copy.deepcopy(submodules)
        ),
        drop_vision_class_token=False,
        vision_projection_config=projection,
        vision_projection_layer_spec=copy.deepcopy(get_submodules(submodules.mlp)),
        pre_process=pre_process,
        post_process=post_process,
        img_h=28,
        img_w=28,
        patch_dim=14,
    )
    randomize_parameters(model, seed)
    return model


def build_te_gpt(seed: int, *, tp: int, use_megatron_fsdp: bool = False, bf16: bool = True):
    """A tiny GPT (Transformer Engine spec; classic or Megatron-FSDP v1) with its optimizer."""
    return setup_model_and_optimizer(
        seed=seed,
        tp=tp,
        pp=1,
        initialize_fn=te_gpt_provider,
        use_megatron_fsdp=use_megatron_fsdp,
        bf16=bf16,
    )


def build_llava(seed: int, *, tp: int, use_megatron_fsdp: bool = False, bf16: bool = True):
    """A tiny LLaVA (classic or Megatron-FSDP v1) with a distributed optimizer."""
    return setup_model_and_optimizer(
        seed=seed,
        tp=tp,
        pp=1,
        initialize_fn=llava_provider,
        use_megatron_fsdp=use_megatron_fsdp,
        bf16=bf16,
    )


def set_format(args, *, fsdp: bool, pattern: str | None = None, bf16: bool = True) -> None:
    """Point the mocked arguments at a classic torch_dist or a Megatron-FSDP fsdp_dtensor job.

    The model is the hybrid model of ``pattern``, or without one the tiny GPT
    (or the language model of the tiny LLaVA).
    """
    args.bf16 = bf16
    args.use_megatron_fsdp = fsdp
    args.ckpt_format = "fsdp_dtensor" if fsdp else "torch_dist"
    args.data_parallel_sharding_strategy = "optim_grads_params" if fsdp else "no_shard"
    args.use_distributed_optimizer = True
    args.hybrid_layer_pattern = pattern
    args.num_layers = GPT_LAYERS if pattern is None else len(pattern.replace("|", ""))
    args.hidden_size = HYBRID_HIDDEN
    args.num_attention_heads = HYBRID_HEADS


def save_weights_only(model, optimizer, args, save_dir: Path, *, iteration: int) -> None:
    """Megatron's ``save_checkpoint`` of the model only.

    No optimizer, RNG or redundant extra state (a model-only reference save).
    """
    args.save = str(save_dir)
    args.no_save_optim = True
    args.no_save_rng = True
    args.ckpt_drop_redundant_extra_state = True
    save_checkpoint(iteration, model, optimizer, None, 0)


def full_params(model) -> dict[str, torch.Tensor]:
    """Every parameter of the first model chunk as a full CPU tensor, by unwrapped name.

    Megatron-FSDP parameters (unevenly sharded DTensors) are gathered, and so
    are the tensor-parallel shards of classic parameters (along their partition
    dimension, in rank order); both are collectives: every rank must call this.
    """
    params = {}
    for name, param in model[0].named_parameters():
        while name.startswith("module."):
            name = name[len("module.") :]
        if isinstance(param, DTensor):
            value = uneven_dtensor_to_full_tensor(param)
        elif getattr(param, "tensor_model_parallel", False):
            value = _gather_tensor_parallel(param.detach(), param.partition_dim)
        else:
            value = param
        params[name] = value.detach().cpu().clone()
    return params


def _gather_tensor_parallel(shard: torch.Tensor, dim: int) -> torch.Tensor:
    group = parallel_state.get_tensor_model_parallel_group()
    shards = [torch.empty_like(shard) for _ in range(torch.distributed.get_world_size(group))]
    torch.distributed.all_gather(shards, shard.contiguous(), group=group)
    return torch.cat(shards, dim=dim)


ITERATION = 10
ITER = f"iter_{ITERATION:07d}"
CHAIN_CHECKS = ("C1 vs S2 (A)", "S2 vs C1 (C)", "C2 vs S3 (B)", "S3 vs S1 (C)", "back vs S1 (D)")
"""The expected-difference checks of :func:`run_chain`: which store against which, and the contract."""

JOB_FIELDS = (
    "bf16",
    "data_parallel_sharding_strategy",
    "save",
    "load",
    "no_load_optim",
    "no_load_rng",
    "dist_ckpt_strictness",
)
"""Arguments the chain's jobs set differently (job settings, not conversion differences)."""


def on_rank_zero(function: Callable, *args, **kwargs):
    """Run ``function`` on rank 0; every rank gets its (picklable) result or its error text."""
    outcome = [None]
    if torch.distributed.get_rank() == 0:
        try:
            outcome[0] = {"ok": function(*args, **kwargs)}
        except Exception as err:  # pylint: disable=broad-exception-caught  # re-raised on all ranks
            outcome[0] = {"error": f"{type(err).__name__}: {err}"}
    torch.distributed.broadcast_object_list(outcome, src=0)
    assert "error" not in outcome[0], outcome[0]["error"]
    return outcome[0]["ok"]


def _convert(src: Path, dst: Path, to_format: str, **options) -> dict:
    return convert_checkpoint(
        src, dst, ConvertOptions(to_format=to_format, workers=0, **options)
    ).to_json()


def _verify(a: Path, b: Path, contract: str, *, jobs_differ: bool = False) -> list[str]:
    ignored = [re.compile(rf"^args\.{name}$") for name in JOB_FIELDS] if jobs_differ else []
    report = verify_checkpoint(
        a, b, VerifyOptions(expected_diff=contract, ignore_keys=tuple(ignored))
    )
    return [f"{f.kind.value} {f.key}: {f.detail}" for f in report.findings]


def _bits(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def assert_params(actual: dict, expected: dict, *, dtype: torch.dtype) -> None:
    """``actual`` holds ``expected``'s parameters bit for bit after the exact cast to ``dtype``."""
    assert set(actual) == set(expected)
    for name, value in actual.items():
        assert value.dtype == dtype, name
        assert _bits(value) == _bits(expected[name].to(dtype)), name


def assert_one_module_level(store: Path) -> None:
    """A real Megatron-FSDP v1 save's keys are ``model.module.<name>`` (one ``module.``)."""
    keys = on_rank_zero(lambda: sorted(MetadataView.load(store / ITER).tensors))
    assert keys and all(
        k.startswith("model.module.") and not k.startswith("model.module.module.") for k in keys
    ), keys


def run_chain(
    root: Path,
    *,
    tp: int,
    build: Callable[..., tuple],
    configure: Callable[..., None],
    per_layer: bool = False,
) -> dict[str, list[str]]:
    """The real-save chain through torch_dist -> fsdp_dtensor -> torch_dist, in ``root``.

    1. S1: model A (classic, bf16) saved as torch_dist, model only.
    2. C1 = convert(S1) to fsdp_dtensor.
    3. Model B (Megatron-FSDP, other seed, fp32) loads C1 strictly; its
       parameters equal A's exactly upcast.
    4. S2: B saved as fsdp_dtensor (keys ``model.module.<name>``).
    5. C2 = convert(S2) to torch_dist; model C (classic, another seed) loads
       it strictly, and its parameters equal A's (the fp32 main weights round
       back exactly); S3: C saved as torch_dist.
    6. With ``per_layer``: C2 again with ``--torch-dist-layout per-layer``,
       loaded strictly by model D (Megatron then builds every block per layer).
    7. back = convert(C1) to torch_dist.

    Model B trains in fp32: on main, a bf16 Megatron-FSDP job cannot load a
    store without optimizer state.

    Args:
        root: The directory shared by the ranks.
        tp: Tensor-parallel size.
        build: ``build(seed, use_megatron_fsdp=..., bf16=...)`` gives the model
            chunks and optimizer.
        configure: ``configure(args, fsdp=..., bf16=...)`` points the mocked
            arguments at a job of that kind.
        per_layer: Also run step 6.

    Returns:
        The findings of each check of :data:`CHAIN_CHECKS` (all empty when the
        chain holds).
    """
    s1, c1, s2, c2, s3, c2_per_layer, back = (
        root / name for name in ("s1", "c1", "s2", "c2", "s3", "c2_per_layer", "back")
    )
    with training_args(s1, tp=tp) as args:
        configure(args, fsdp=False)
        model_a, optimizer_a = build(2)
        expected = full_params(model_a)
        save_weights_only(model_a, optimizer_a, args, s1, iteration=ITERATION)
        torch.distributed.barrier()
        on_rank_zero(_convert, s1, c1, "fsdp_dtensor")

        configure(args, fsdp=True, bf16=False)
        model_b, optimizer_b = build(3, use_megatron_fsdp=True, bf16=False)
        assert load_weights_only(model_b, optimizer_b, args, c1) == ITERATION
        assert_params(full_params(model_b), expected, dtype=torch.float32)
        save_weights_only(model_b, optimizer_b, args, s2, iteration=ITERATION)
        torch.distributed.barrier()
        assert_one_module_level(s2)
        on_rank_zero(_convert, s2, c2, "torch_dist")

        configure(args, fsdp=False)
        model_c, optimizer_c = build(4)
        assert load_weights_only(model_c, optimizer_c, args, c2) == ITERATION
        assert_params(full_params(model_c), expected, dtype=torch.bfloat16)
        save_weights_only(model_c, optimizer_c, args, s3, iteration=ITERATION)
        if per_layer:
            on_rank_zero(_convert, s2, c2_per_layer, "torch_dist", torch_dist_layout="per-layer")
            model_d, optimizer_d = build(5)
            assert load_weights_only(model_d, optimizer_d, args, c2_per_layer) == ITERATION
            assert_params(full_params(model_d), expected, dtype=torch.bfloat16)
        torch.distributed.barrier()
        on_rank_zero(_convert, c1, back, "torch_dist")
        return on_rank_zero(
            lambda: {
                "C1 vs S2 (A)": _verify(c1, s2, "A"),
                "S2 vs C1 (C)": _verify(s2, c1, "C", jobs_differ=True),
                "C2 vs S3 (B)": _verify(c2, s3, "B"),
                "S3 vs S1 (C)": _verify(s3, s1, "C", jobs_differ=True),
                "back vs S1 (D)": _verify(back, s1, "D"),
            }
        )
