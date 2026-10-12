# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optimizer state from a real Megatron-FSDP save, resumed by a classic job (T3).

A Megatron-FSDP job with a distributed Adam optimizer (random moments, a
learning-rate scheduler that has stepped, a step count in its groups) saves
an fsdp_dtensor store with its optimizer state; the store is converted with
``--optimizer convert`` and loaded strictly by a classic job with
``--dist-ckpt-optim-fully-reshardable`` (no RNG, redundant extra state
dropped). The classic job's live state must then equal the Megatron-FSDP
job's, compared by a module map written out here (layers stacked, experts
under their global index), never through the converter:

* every parameter's fp32 master, ``exp_avg`` and ``exp_avg_sq`` (the classic
  optimizer's fully reshardable state, gathered over every rank, against the
  Megatron-FSDP optimizer's state and main parameters, gathered to full);
* every parameter group's hyperparameters, step included, paired by Megatron's
  own identifier, and the scheduler's state;
* every model parameter, the master rounded to the parameter dtype.

A classic save of the resumed job must then hold the converted store's keys
and values (contract ``B``). Cases: a SwiGLU GPT in bf16 (``module.module.module.``
optimizer keys), a GPT in fp32 (``module.module.``), and a mixture-of-experts
GPT at expert-parallel size 2, whose classic job chains a dense and an expert
optimizer.

Runs only in the Megatron container with GPUs and the root
``tests/unit_tests/conftest.py``; skipped elsewhere. Cases that need more
ranks than the world has are skipped.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import partial
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from tools.checkpoint.format_converter.mcore_state.common_state import megatron_available

if not (megatron_available() and torch.cuda.is_available()):
    pytest.skip("needs Megatron and CUDA (the training container)", allow_module_level=True)

# pylint: disable=wrong-import-position
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedTensor, apply_factories
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    uneven_dtensor_to_full_tensor,
)
from megatron.core.optimizer.optimizer import param_group_identifier_keys
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.training.checkpointing import (
    _build_sharded_state_dict_metadata,
    load_checkpoint,
    save_checkpoint,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tools.checkpoint.format_converter.engine.metadata import MetadataView

from .._real_harness import (
    ITER,
    ITERATION,
    Utils,
    assert_params,
    build_moe_gpt,
    build_te_gpt,
    convert_store,
    full_params,
    nccl_skip_reason,
    on_rank_zero,
    set_format,
    set_moe_format,
    training_args,
    verify_store,
)

pytestmark = pytest.mark.covers("glu", "real_save")
SWIGLU = {"gated_linear_unit": True, "activation_func": F.silu}
STEP = 7
"""The step count the Megatron-FSDP job's groups hold (as Transformer Engine's FusedAdam keeps it)."""
LAYER = re.compile(r"^decoder\.layers\.(?P<layer>\d+)\.(?P<rest>.+)$")
EXPERT = re.compile(
    r"^(?P<head>.+\.mlp\.experts)\.(?P<linear>linear_fc\d)\.(?P<leaf>weight|bias)(?P<expert>\d+)$"
)
STATES = ("param", "exp_avg", "exp_avg_sq")


@pytest.fixture(name="shared_dir")
def fixture_shared_dir(request):
    """The root conftest's directory shared by all ranks (absent under ``--confcutdir``)."""
    try:
        return request.getfixturevalue("tmp_path_dist_ckpt")
    except pytest.FixtureLookupError:
        pytest.skip("needs the root tests/unit_tests/conftest.py (run without --confcutdir)")


def _initialize(ep: int = 1) -> None:
    if Utils.world_size % ep:
        pytest.skip(f"needs a world size divisible by ep={ep}")
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=ep)
    reason = nccl_skip_reason()
    if reason is not None:
        pytest.skip(reason)


def _scheduler(optimizer) -> OptimizerParamScheduler:
    """A cosine schedule with weight-decay increments; the groups get its learning rates."""
    for group in optimizer.param_groups:
        group["max_lr"], group["min_lr"] = 1e-3, 1e-5
    return OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=1e-3,
        min_lr=1e-5,
        lr_warmup_steps=2,
        lr_decay_steps=100,
        lr_decay_style="cosine",
        start_wd=0.01,
        end_wd=0.02,
        wd_incr_steps=100,
        wd_incr_style="linear",
    )


def _td_place(name: str) -> tuple[str, tuple[int, ...]]:
    """Where torch_dist keeps the parameter ``name`` (written out by hand): its key, and its index
    along the key's leading layer and expert axes."""
    index: tuple[int, ...] = ()
    match = LAYER.match(name)
    if match is not None:
        name, index = f"decoder.layers.{match['rest']}", (int(match["layer"]),)
    match = EXPERT.match(name)
    if match is not None:
        name = f"{match['head']}.experts.{match['linear']}.{match['leaf']}"
        index += (int(match["expert"]),)
    return name, index


def _stack(placed: dict[str, dict[tuple[int, ...], torch.Tensor]]) -> dict[str, torch.Tensor]:
    """The tensors of each key, stacked along its leading axes in index order."""
    stacked = {}
    for key, parts in placed.items():
        lead = len(next(iter(parts)))
        sizes = [max(index[axis] for index in parts) + 1 for axis in range(lead)]
        tensor = torch.empty(*sizes, *next(iter(parts.values())).shape)
        for index, part in parts.items():
            tensor[index] = part
        stacked[key] = tensor
    return stacked


def _gathered(local: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Every rank's tensors by name (a collective; replicas are equal)."""
    everyone: list = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(everyone, local)
    return {name: tensor for theirs in everyone for name, tensor in theirs.items()}


def fsdp_optimizer_state(model, optimizer) -> dict[str, torch.Tensor]:
    """The Megatron-FSDP job's live master, ``exp_avg`` and ``exp_avg_sq`` of every parameter,
    gathered to full and placed by :func:`_td_place` (a collective)."""
    names = {param: optimizer._param_name(param) for param in model[0].parameters()}
    moments = optimizer.sharded_param_state_fsdp_dtensor()["state"]
    local = {}
    for param, name in names.items():
        live = re.sub(r"^(module\.)+", "", name)
        local[f"param.{live}"] = uneven_dtensor_to_full_tensor(param).float().cpu()
        for moment in STATES[1:]:
            state = moments[name][moment]
            local[f"{moment}.{live}"] = uneven_dtensor_to_full_tensor(state).cpu()
    placed: dict[str, dict] = {}
    for name, tensor in _gathered(local).items():
        state, live = name.split(".", 1)
        key, index = _td_place(live)
        placed.setdefault(f"optimizer.state.{state}.{key}", {})[index] = tensor
    return _stack(placed)


def classic_optimizer_state(model, optimizer, args) -> dict[str, torch.Tensor]:
    """The classic job's live fully reshardable state, every rank's sharded tensors assembled into
    whole tensors (a collective)."""
    metadata = _build_sharded_state_dict_metadata(args)
    state = optimizer.sharded_state_dict(model[0].sharded_state_dict(), metadata=metadata)
    apply_factories(state)
    mine = [
        (t.key, tuple(t.global_shape), tuple(t.global_offset), t.data.detach().float().cpu())
        for t in nested_values(state)
        if isinstance(t, ShardedTensor) and t.key.startswith("optimizer.state.")
    ]
    everyone: list = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(everyone, mine)
    whole: dict[str, torch.Tensor] = {}
    for entries in everyone:
        for key, shape, offset, data in entries:
            tensor = whole.setdefault(key, torch.full(shape, float("nan")))
            lead = len(shape) - data.dim()
            box = tuple(slice(o, o + n) for o, n in zip(offset[lead:], data.shape))
            tensor[tuple(offset[:lead]) + box] = data
    return whole


def _groups(optimizer) -> dict[tuple, dict]:
    """Every parameter group's hyperparameters (``params`` left out) by Megatron's identifier."""
    groups = {}
    for group in optimizer.param_groups:
        identifier = tuple(group.get(key) for key in param_group_identifier_keys)
        groups[identifier] = {k: v for k, v in group.items() if k != "params"}
    return groups


def _depth_findings(store: Path, *, bf16: bool) -> list[str]:
    """Optimizer keys of a real Megatron-FSDP save not under its precision's ``module.`` levels."""
    depth = "module." * (3 if bf16 else 2)
    keys = [*MetadataView.load(store / ITER).tensors, *MetadataView.load(store / ITER).byte_items]
    optimizer = [key for key in keys if key.startswith("optimizer.")]
    assert optimizer
    return [
        key
        for key in optimizer
        if not key.startswith(
            (f"optimizer.state.{depth}", f"optimizer.param_to_group_meta.{depth}")
        )
        or key.startswith(
            (f"optimizer.state.{depth}module.", f"optimizer.param_to_group_meta.{depth}module.")
        )
    ]


def _differences(actual: dict, expected: dict) -> list[str]:
    found = [f"{key}: missing" for key in sorted(set(expected) - set(actual))]
    found += [f"{key}: unexpected" for key in sorted(set(actual) - set(expected))]
    found += [
        f"{key}: other values"
        for key in sorted(set(actual) & set(expected))
        if not torch.equal(actual[key], expected[key])
    ]
    return found


def run_optimizer_chain(
    root: Path,
    *,
    build: Callable[..., tuple],
    configure: Callable[..., None],
    bf16: bool,
    chained: int = 1,
) -> dict[str, list[str]]:
    """S1: a Megatron-FSDP save with optimizer state; C = convert(S1, --optimizer convert); a
    classic job resumes from C; S2: its classic save. Returns the findings of every check
    (``chained``: the number of optimizers the classic job chains)."""
    s1, converted, s2 = root / "s1", root / "c", root / "s2"
    with training_args(s1, tp=1) as args:
        configure(args, fsdp=True, bf16=bf16)
        model_b, optimizer_b = build(3, use_megatron_fsdp=True, bf16=bf16)
        scheduler_b = _scheduler(optimizer_b)
        for _ in range(STEP):
            scheduler_b.step(1)
        for group in optimizer_b.param_groups:
            group["step"] = STEP
        expected_state = fsdp_optimizer_state(model_b, optimizer_b)
        expected_params = full_params(model_b)
        expected_groups = _groups(optimizer_b)
        expected_scheduler = scheduler_b.state_dict()
        save_checkpoint(ITERATION, model_b, optimizer_b, scheduler_b, 0)
        torch.distributed.barrier()
        report = on_rank_zero(convert_store, s1, converted, "torch_dist", optimizer="convert")

        configure(args, fsdp=False, bf16=bf16)
        model_c, optimizer_c = build(4, bf16=bf16)
        scheduler_c = _scheduler(optimizer_c)
        args.load, args.save = str(converted), str(s2)
        args.no_load_optim, args.no_load_rng, args.finetune = False, True, False
        args.dist_ckpt_strictness = "raise_all"
        args.ckpt_drop_redundant_extra_state = True
        with mock.patch("megatron.training.checkpointing.check_checkpoint_args"):
            with mock.patch("megatron.training.checkpointing.update_num_microbatches"):
                iteration, _ = load_checkpoint(model_c, optimizer_c, scheduler_c)
        assert iteration == ITERATION
        actual_state = classic_optimizer_state(model_c, optimizer_c, args)
        dtype = torch.bfloat16 if bf16 else torch.float32
        assert_params(full_params(model_c), expected_params, dtype=dtype)
        groups, scheduled = _groups(optimizer_c), scheduler_c.state_dict()
        findings = {
            "optimizer keys": on_rank_zero(_depth_findings, s1, bf16=bf16),
            "optimizer state": _differences(actual_state, expected_state),
            "param groups": [] if groups == expected_groups else [f"{groups} != {expected_groups}"],
            "scheduler": [] if scheduled == expected_scheduler else [f"{scheduled}"],
            "chained": (
                []
                if len(report["optimizer"]["param_groups"]) == chained
                else [f"{report['optimizer']['param_groups']}"]
            ),
        }
        save_checkpoint(ITERATION, model_c, optimizer_c, scheduler_c, 0)
        torch.distributed.barrier()
        findings["C vs S2 (B)"] = on_rank_zero(verify_store, converted, s2, "B")
        return findings


CHECKS = (
    "optimizer keys",
    "optimizer state",
    "param groups",
    "scheduler",
    "chained",
    "C vs S2 (B)",
)


class TestOptimizerResume:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("bf16", [True, False], ids=["bf16-swiglu", "fp32"])
    def test_a_gpt_resumes_its_optimizer_state_in_a_classic_job(self, shared_dir, bf16):
        """bf16 with SwiGLU (only a bf16 Megatron-FSDP job builds its state dicts); fp32 plain."""
        _initialize()
        config = SWIGLU if bf16 else {}

        def configure(args, *, fsdp: bool, bf16: bool) -> None:
            set_format(args, fsdp=fsdp, bf16=bf16)
            args.swiglu = bool(config)

        with TempNamedDir(shared_dir / f"fc_optimizer_gpt_{'bf16' if bf16 else 'fp32'}") as root:
            findings = run_optimizer_chain(
                root, build=partial(build_te_gpt, tp=1, **config), configure=configure, bf16=bf16
            )
        assert findings == {check: [] for check in CHECKS}

    def test_experts_at_expert_parallel_size_2_resume_in_chained_optimizers(self, shared_dir):
        _initialize(ep=2)
        build = partial(build_moe_gpt, tp=1, ep=2, grouped=True, add_bias_linear=False)
        configure = partial(set_moe_format, grouped=True, ep=2)
        with TempNamedDir(shared_dir / "fc_optimizer_moe_ep2") as root:
            findings = run_optimizer_chain(
                root, build=build, configure=configure, bf16=True, chained=2
            )
        assert findings == {check: [] for check in CHECKS}
