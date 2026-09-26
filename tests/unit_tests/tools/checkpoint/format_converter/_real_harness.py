# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real Megatron saves and loads for the real-save (T3) tests.

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
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import torch

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
    "live_params",
    "load_weights_only",
    "nccl_skip_reason",
    "real_save",
    "training_args",
]


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
    assets; ``megatron.training.checkpointing.get_args`` returns them.
    """
    args = parse_args(ignore_unknown_args=True)
    args.save_tokenizer_assets = False
    with mock.patch("megatron.training.checkpointing.get_args", new=lambda: args):
        init_basic_mock_args(args, tp, pp)
        init_checkpointing_mock_args(args, str(ckpt_dir), fully_parallel=True)
        args.dist_ckpt_optim_fully_reshardable = True
        yield args


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
    with mock.patch("megatron.training.checkpointing.check_checkpoint_args"):
        with mock.patch("megatron.training.checkpointing.update_num_microbatches"):
            iteration, _ = load_checkpoint(model, optimizer, None)
    return iteration
