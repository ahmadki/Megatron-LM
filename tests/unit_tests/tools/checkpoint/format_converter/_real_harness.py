# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real Megatron saves and loads for the real-save (T3) tests.

Classic (DDP) and Megatron-FSDP v1 models are built with
``setup_model_and_optimizer``, saved with Megatron's own ``save_checkpoint``
and loaded with its own ``load_checkpoint`` under mocked training arguments.

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
import zlib
from collections.abc import Iterator
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
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
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

__all__ = [
    "Utils",
    "build_gpt",
    "build_hybrid",
    "full_params",
    "live_params",
    "load_weights_only",
    "nccl_skip_reason",
    "randomize_parameters",
    "real_save",
    "resolve_weight_shards",
    "save_weights_only",
    "set_format",
    "training_args",
]

HYBRID_HIDDEN = 64
HYBRID_HEADS = 4
HYBRID_VOCAB = 128


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


def set_format(args, *, fsdp: bool, pattern: str, bf16: bool = True) -> None:
    """Point the mocked arguments at a classic torch_dist or a Megatron-FSDP fsdp_dtensor job."""
    args.bf16 = bf16
    args.use_megatron_fsdp = fsdp
    args.ckpt_format = "fsdp_dtensor" if fsdp else "torch_dist"
    args.data_parallel_sharding_strategy = "optim_grads_params" if fsdp else "no_shard"
    args.use_distributed_optimizer = True
    args.hybrid_layer_pattern = pattern
    args.num_layers = len(pattern.replace("|", ""))
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
