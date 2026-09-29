# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Gated linear units against Megatron's own saves, loads, SwiGLU split and tensor-parallel layout.

* T3 real-save chains (``run_chain`` in ``_real_harness.py``) of gated MLPs
  Megatron-FSDP keeps fused, packed per tensor-parallel rank: a GEGLU GPT
  (gated without ``--swiglu``) at TP 2, converted with ``--fsdp-tp 2``, and a
  LLaVA whose vision encoder is gated under a plain language model at TP 1
  (TP 2 extended). Megatron saves and strictly loads every store in both
  formats, every parameter must equal model A's after each load, and every
  store is checked against a real save under its contract.
* Real saves of SwiGLU models, whose MLPs Megatron-FSDP splits into halves:
  a SwiGLU GPT at TP 1 (TP 2 extended) and a SwiGLU mixture-of-experts GPT at
  expert-parallel size 2 (grouped experts ``weight<e>_w`` under the global
  index, with shared experts; sequential experts with biases). On current
  Megatron only a bf16 Megatron-FSDP job builds SwiGLU state dicts (an fp32
  one fails, pinned below as a strict expected failure) and a bf16 job cannot
  load a store without optimizer state, so these run ``run_bf16_fsdp_chain``:
  the converted store must hold a real Megatron-FSDP save's keys, shapes and
  dtypes, the real save converted back must load strictly into a classic
  model with every parameter equal, and the round trip must give the source.
* T2 pins on live bf16 Megatron-FSDP models: the state dict
  ``preprocess_fsdp_dtensor_state_dict`` builds (``handle_swiglu_in_state_dict``
  and the expert renumbering) holds SwiGLU halves exactly where the
  converter's predicate puts them, for a SwiGLU mixture-of-experts GPT with
  shared experts (at expert-parallel size 1 and 2) and for a LLaVA with a
  SwiGLU language model, a plain vision encoder and a gated projection
  (``encoder.linear_fc1`` stays fused); and a fused gated ``linear_fc1`` is the
  rank-major pack of the halves Megatron's own split gives,
  ``[W_0; V_0; W_1; V_1]``, by the converter's ``PartLayout``, at TP 2 and for
  routed experts at expert TP 2.

Runs only in the Megatron container with GPUs and the root
``tests/unit_tests/conftest.py``; skipped elsewhere. Cases that need more
ranks than the world has are skipped.
"""

from __future__ import annotations

import argparse
import os
from functools import partial

import pytest
import torch
import torch.nn.functional as F

from tools.checkpoint.format_converter.mcore_state.common_state import megatron_available

if not (megatron_available() and torch.cuda.is_available()):
    pytest.skip("needs Megatron and CUDA (the training container)", allow_module_level=True)

# pylint: disable=wrong-import-position
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    uneven_dtensor_to_full_tensor,
)
from megatron.core.transformer.fsdp_dtensor_checkpoint import handle_swiglu_in_state_dict
from megatron.training.checkpointing import preprocess_fsdp_dtensor_state_dict
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.context import build_arch_context
from tools.checkpoint.format_converter.core.keys import EXTRA_STATE, GLU_PARTS
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, TORCH_DIST
from tools.checkpoint.format_converter.core.part_layout import PartLayout
from tools.checkpoint.format_converter.engine.metadata import TensorEntry

from .._real_harness import (
    BF16_CHAIN_CHECKS,
    CHAIN_CHECKS,
    GPT_LAYERS,
    MOE_EXPERTS,
    Utils,
    build_llava,
    build_moe_gpt,
    build_te_gpt,
    nccl_skip_reason,
    on_rank_zero,
    run_bf16_fsdp_chain,
    run_chain,
    set_format,
    set_moe_format,
    training_args,
)

pytestmark = [pytest.mark.covers("glu", "real_save"), pytest.mark.covers("glu", "parity")]
EXTENDED = os.environ.get("MCORE_FORMAT_CONVERTER_EXTENDED") == "1"
TP_CASES = [1, pytest.param(2, marks=pytest.mark.skipif(not EXTENDED, reason="extended matrix"))]
SWIGLU = {"gated_linear_unit": True, "activation_func": F.silu}
"""The TransformerConfig of a SwiGLU MLP (``--swiglu``)."""
GEGLU = {"gated_linear_unit": True}
"""A gated MLP without SwiGLU (gelu: gated like ``--quick-geglu``, which sets no ``swiglu``)."""
SHARED_EXPERTS = {"moe_shared_expert_intermediate_size": 32}
"""Shared experts (which Megatron builds without biases only)."""


@pytest.fixture(name="shared_dir")
def fixture_shared_dir(request):
    """The root conftest's directory shared by all ranks (absent under ``--confcutdir``)."""
    try:
        return request.getfixturevalue("tmp_path_dist_ckpt")
    except pytest.FixtureLookupError:
        pytest.skip("needs the root tests/unit_tests/conftest.py (run without --confcutdir)")


def _initialize(tp: int = 1, ep: int = 1, etp: int | None = None) -> None:
    if Utils.world_size % (tp * ep) or Utils.world_size % (etp or 1):
        pytest.skip(f"needs a world size divisible by tp={tp} x ep={ep} and etp={etp}")
    Utils.initialize_model_parallel(
        tp, 1, expert_model_parallel_size=ep, expert_tensor_parallel_size=etp
    )
    reason = nccl_skip_reason()
    if reason is not None:
        pytest.skip(reason)


def gated_job(configure, *, swiglu: bool = False, quick_geglu: bool = False):
    """``configure`` for a job whose language model is gated (``--swiglu``, ``--quick-geglu``)
    or plain."""

    def configure_gated(args, **kwargs):
        configure(args, **kwargs)
        args.swiglu = swiglu
        args.quick_geglu = quick_geglu

    return configure_gated


class TestPackedChains:
    """Gated MLPs Megatron-FSDP keeps fused, packed per tensor-parallel rank."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_geglu_gpt_at_two_ranks(self, shared_dir):
        _initialize(tp=2)
        with TempNamedDir(shared_dir / "fc_glu_geglu_gpt_tp2") as root:
            findings = run_chain(
                root,
                tp=2,
                build=partial(build_te_gpt, tp=2, **GEGLU),
                configure=gated_job(set_format, quick_geglu=True),
                options={"fsdp_tp": 2},
            )
        assert findings == {check: [] for check in CHAIN_CHECKS}

    @pytest.mark.parametrize("tp", TP_CASES)
    def test_glu_vision_encoder_under_a_plain_language_model(self, shared_dir, tp):
        _initialize(tp)
        with TempNamedDir(shared_dir / f"fc_glu_vision_tp{tp}") as root:
            findings = run_chain(
                root,
                tp=tp,
                build=partial(build_llava, tp=tp, gated={"vision": GEGLU}),
                configure=gated_job(set_format),
                options={"fsdp_tp": tp},
            )
        assert findings == {check: [] for check in CHAIN_CHECKS}


class TestSplitSaves:
    """SwiGLU MLPs Megatron-FSDP splits into halves, through the saves a bf16 job permits."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("tp", TP_CASES)
    def test_swiglu_gpt(self, shared_dir, tp):
        _initialize(tp)
        with TempNamedDir(shared_dir / f"fc_glu_swiglu_gpt_tp{tp}") as root:
            findings = run_bf16_fsdp_chain(
                root,
                tp=tp,
                build=partial(build_te_gpt, tp=tp, **SWIGLU),
                configure=gated_job(set_format, swiglu=True),
            )
        assert findings == {check: [] for check in BF16_CHAIN_CHECKS}

    @pytest.mark.parametrize(
        ("grouped", "config"),
        [
            pytest.param(True, {"add_bias_linear": False, **SHARED_EXPERTS}, id="grouped"),
            pytest.param(False, {"add_bias_linear": True}, id="sequential-bias"),
        ],
    )
    def test_swiglu_moe_at_expert_parallel_size_two(self, shared_dir, grouped, config):
        """Grouped expert biases are refused at expert-parallel size 2 (FC-R05)."""
        _initialize(ep=2)
        name = f"fc_glu_swiglu_moe_{'grouped' if grouped else 'sequential'}_ep2"
        with TempNamedDir(shared_dir / name) as root:
            findings = run_bf16_fsdp_chain(
                root,
                tp=1,
                build=partial(build_moe_gpt, tp=1, ep=2, grouped=grouped, **SWIGLU, **config),
                configure=gated_job(partial(set_moe_format, grouped=grouped, ep=2), swiglu=True),
            )
        assert findings == {check: [] for check in BF16_CHAIN_CHECKS}

    @pytest.mark.xfail(
        strict=True,
        raises=AttributeError,
        reason="handle_swiglu_in_state_dict resolves 'module.' + key, which misses the "
        "parameters of an fp32 Megatron-FSDP job (no Float16Module); when this passes, run the "
        "SwiGLU saves through run_chain",
    )
    def test_an_fp32_megatron_fsdp_job_builds_swiglu_state_dicts(self, tmp_path):
        _initialize()
        with training_args(tmp_path, tp=1) as args:
            gated_job(set_format, swiglu=True)(args, fsdp=True, bf16=False)
            model, _ = build_te_gpt(1, tp=1, use_megatron_fsdp=True, bf16=False, **SWIGLU)
            preprocess_fsdp_dtensor_state_dict(args, {"model": model[0].state_dict()}, model[0])


def _preprocessed_entries(model, args) -> dict[str, TensorEntry]:
    """The model tensors of what a Megatron-FSDP job saves (``preprocess_fsdp_dtensor_state_dict``
    of its live state dict), by model-relative key, from every rank (a collective)."""
    state = preprocess_fsdp_dtensor_state_dict(args, {"model": model[0].state_dict()}, model[0])
    mine = [
        (key.removeprefix("module."), tuple(value.shape), value.dtype)
        for key, value in state["model"].items()
        if isinstance(value, torch.Tensor) and not EXTRA_STATE.search(key)
    ]
    everyone: list = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(everyone, mine)
    return {
        key: TensorEntry(f"model.module.{key}", shape, dtype, ())
        for entries in everyone
        for key, shape, dtype in entries
    }


def _split_by_converter(args: argparse.Namespace, entries: dict[str, TensorEntry]) -> dict:
    """The converter's view of the store: which first projections it splits or keeps fused. Its
    context refuses (FC-R12) a store that holds halves anywhere else than its predicate says."""
    ctx = build_arch_context(
        source=FSDP_DTENSOR, target=TORCH_DIST, args=ArgsView(args), model=entries
    )
    return ctx.glu.split_json()


class TestMegatronSwigluSplit:
    """The converter's SwiGLU predicate against the state dicts Megatron-FSDP saves (bf16 jobs)."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("ep", [1, 2])
    def test_the_converter_splits_what_megatron_fsdp_splits_in_moe_layers(self, tmp_path, ep):
        _initialize(ep=ep)
        with training_args(tmp_path, tp=1) as args:
            gated_job(partial(set_moe_format, grouped=True, ep=ep), swiglu=True)(args, fsdp=True)
            model, _ = build_moe_gpt(
                1,
                tp=1,
                ep=ep,
                grouped=True,
                use_megatron_fsdp=True,
                add_bias_linear=False,
                **SWIGLU,
                **SHARED_EXPERTS,
            )
            entries = _preprocessed_entries(model, args)
            halves = {k for k in entries if k.endswith("_w")}
            assert f"decoder.layers.2.mlp.experts.linear_fc1.weight{MOE_EXPERTS - 1}_w" in halves
            assert "decoder.layers.0.mlp.shared_experts.linear_fc1.weight_w" in halves
            assert len(halves) == GPT_LAYERS * (MOE_EXPERTS + 1)
            split = on_rank_zero(_split_by_converter, args, entries)
        assert split == {"decoder": {"split": GPT_LAYERS * (MOE_EXPERTS + 1)}}

    def test_the_converter_keeps_fused_what_megatron_fsdp_does_not_split(self, tmp_path):
        """A SwiGLU language model, a plain vision encoder and a gated projection: the encoder's
        layers are not gated and ``encoder.linear_fc1`` is no SwiGLU name."""
        _initialize()
        with training_args(tmp_path, tp=1) as args:
            gated_job(set_format, swiglu=True)(args, fsdp=True)
            gated = {"language": SWIGLU, "projection": SWIGLU}
            model, _ = build_llava(1, tp=1, use_megatron_fsdp=True, gated=gated)
            entries = _preprocessed_entries(model, args)
            assert "vision_projection.encoder.linear_fc1.weight" in entries
            assert "language_model.decoder.layers.0.mlp.linear_fc1.weight_w" in entries
            assert "vision_model.decoder.layers.0.mlp.linear_fc1.weight" in entries
            split = on_rank_zero(_split_by_converter, args, entries)
        assert split == {
            "language_model.decoder": {"split": GPT_LAYERS},
            "vision_projection.encoder": {"fused": 1},
        }


def _assert_packed_rank_major(model, fused_keys: list[str], pack: int) -> None:
    """Each fused gated ``linear_fc1`` of a live Megatron-FSDP model, gathered, is the rank-major
    pack at ``pack`` of the halves Megatron's own SwiGLU split gives, not their concatenation (a
    collective: every rank gathers every key)."""
    state = model[0].state_dict()
    halves, _ = handle_swiglu_in_state_dict(model[0], state, None)
    for key in fused_keys:
        fused = uneven_dtensor_to_full_tensor(state[key]).cpu()
        gate = uneven_dtensor_to_full_tensor(halves[f"{key}_w"]).cpu()
        value = uneven_dtensor_to_full_tensor(halves[f"{key}_v"]).cpu()
        parts = dict(zip(GLU_PARTS, (gate, value)))
        packed = torch.full_like(fused, float("nan"))
        layout = PartLayout(GLU_PARTS, (gate.shape[0], value.shape[0]))
        for run in layout.rank_major_map(pack):
            packed[run.fused_start : run.fused_start + run.rows] = parts[run.part][
                run.part_start : run.part_start + run.rows
            ]
        assert torch.equal(fused, packed), key
        assert not torch.equal(fused, torch.cat([gate, value])), key


class TestMegatronFsdpPack:
    """Megatron-FSDP's fused gated ``linear_fc1`` against the halves its own SwiGLU split gives."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_a_fused_gated_fc1_is_packed_by_tensor_parallel_rank(self):
        _initialize(tp=2)
        model, _ = build_te_gpt(1, tp=2, use_megatron_fsdp=True, **SWIGLU)
        keys = sorted(
            key
            for key in model[0].state_dict()
            if key.endswith(("mlp.linear_fc1.weight", "mlp.linear_fc1.bias"))
        )
        assert len(keys) == 2 * GPT_LAYERS
        _assert_packed_rank_major(model, keys, pack=2)

    def test_a_fused_gated_expert_fc1_is_packed_by_expert_tensor_parallel_rank(self):
        """Expert tensor-parallel size 2 under tensor-parallel size 1: the pack size of routed
        experts is their own."""
        _initialize(etp=2)
        model, _ = build_moe_gpt(
            1,
            tp=1,
            ep=1,
            etp=2,
            grouped=True,
            use_megatron_fsdp=True,
            add_bias_linear=False,
            **SWIGLU,
        )
        keys = sorted(k for k in model[0].state_dict() if ".mlp.experts.linear_fc1.weight" in k)
        assert len(keys) == GPT_LAYERS * MOE_EXPERTS
        _assert_packed_rank_major(model, keys, pack=2)
