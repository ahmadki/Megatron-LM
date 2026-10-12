# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPT multi-token prediction (MTP) and multi-latent attention (MLA) against Megatron's own saves.

* T3 real-save chains (``run_chain`` in ``_real_harness.py``) of a GPT with two
  MTP layers, whose inner layer is ``transformer_layer`` in both formats
  (Megatron renames the live ``mtp_model_layer`` on save and load), and of a
  GPT with multi-latent attention (TP 1; TP 2 extended): Megatron saves and
  strictly loads every store in both formats, every parameter equals model
  A's after each load, and every store is checked against a real save under
  its contract.
* A GPT with ``--mla-down-proj-fusion`` and MTP layers (each holding fused
  MLA, so the MLA handler runs before the MTP rename): both formats store the
  fused down-projection unfused and its absorbed input layer norm as
  ``input_layernorm``. Only a bf16 Megatron-FSDP job builds its state dicts
  (``handle_mla_down_proj_in_state_dict`` resolves ``module.`` + key, which an
  fp32 job lacks; pinned below as a strict expected failure), and a bf16 job
  cannot load a store without optimizer state, so it runs
  ``run_bf16_fsdp_chain``: the converted store holds a real Megatron-FSDP
  save's keys, shapes and dtypes, and that save converted back loads strictly
  into a classic model with every parameter equal.
* The legacy GPT MTP name: a store converted with ``--fsdp-gpt-mtp-inner-name
  mtp_model_layer`` holds a real Megatron-FSDP save's keys with the inner layer
  renamed by hand, and converted back it loads strictly into a classic model.

Runs only in the Megatron container with GPUs and the root
``tests/unit_tests/conftest.py``; skipped elsewhere.
"""

from __future__ import annotations

import os
from functools import partial

import pytest
import torch

from tools.checkpoint.format_converter.mcore_state.common_state import megatron_available

if not (megatron_available() and torch.cuda.is_available()):
    pytest.skip("needs Megatron and CUDA (the training container)", allow_module_level=True)

# pylint: disable=wrong-import-position
from megatron.training.checkpointing import preprocess_fsdp_dtensor_state_dict
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tools.checkpoint.format_converter.engine.metadata import MetadataView

from .._real_harness import (
    BF16_CHAIN_CHECKS,
    CHAIN_CHECKS,
    ITER,
    ITERATION,
    Utils,
    assert_params,
    build_mla_mtp_gpt,
    convert_store,
    full_params,
    load_weights_only,
    nccl_skip_reason,
    on_rank_zero,
    run_bf16_fsdp_chain,
    run_chain,
    save_weights_only,
    set_format,
    training_args,
    verify_store,
)

pytestmark = pytest.mark.covers("identity", "real_save")
MLA = {
    "q_lora_rank": 16,
    "kv_lora_rank": 16,
    "qk_head_dim": 16,
    "qk_pos_emb_head_dim": 8,
    "v_head_dim": 16,
    "qk_layernorm": True,
    "rope_type": "rope",
}
"""The multi-latent attention of the tiny GPT (``MLATransformerConfig`` fields and arguments)."""
MTP = {"mtp_num_layers": 2}
FUSED_MLA_MTP = {"mla": {**MLA, "mla_down_proj_fusion": True}, **MTP}
EXTENDED = os.environ.get("MCORE_FORMAT_CONVERTER_EXTENDED") == "1"
TP2 = pytest.mark.skipif(not EXTENDED, reason="extended matrix")
CASES = [
    pytest.param(MTP, 1, id="mtp"),
    pytest.param({"mla": MLA}, 1, id="mla"),
    pytest.param(MTP, 2, id="mtp-tp2", marks=TP2),
    pytest.param({"mla": MLA}, 2, id="mla-tp2", marks=TP2),
]
"""``mla_mtp_gpt_provider`` fields of each model, and its tensor-parallel size."""


@pytest.fixture(name="shared_dir")
def fixture_shared_dir(request):
    """The root conftest's directory shared by all ranks (absent under ``--confcutdir``)."""
    try:
        return request.getfixturevalue("tmp_path_dist_ckpt")
    except pytest.FixtureLookupError:
        pytest.skip("needs the root tests/unit_tests/conftest.py (run without --confcutdir)")


def _configure(args, *, fsdp: bool, config: dict, bf16: bool = True) -> None:
    """The mocked arguments of a job of the model ``config``."""
    set_format(args, fsdp=fsdp, bf16=bf16)
    args.mtp_num_layers = config.get("mtp_num_layers")
    mla = config.get("mla")
    args.multi_latent_attention = mla is not None
    for name, value in (mla or {}).items():
        setattr(args, name, value)


def _initialize(tp: int = 1) -> None:
    if Utils.world_size % tp:
        pytest.skip(f"needs a world size divisible by tp={tp}")
    Utils.initialize_model_parallel(tp, 1)
    reason = nccl_skip_reason()
    if reason is not None:
        pytest.skip(reason)


class TestMtpMlaChain:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(("config", "tp"), CASES)
    def test_torch_dist_fsdp_dtensor_torch_dist(self, shared_dir, config, tp):
        _initialize(tp)
        name = f"fc_mtp_mla_{'_'.join(sorted(config))}_tp{tp}"
        with TempNamedDir(shared_dir / name) as root:
            findings = run_chain(
                root,
                tp=tp,
                build=partial(build_mla_mtp_gpt, tp=tp, **config),
                configure=partial(_configure, config=config),
            )
        assert findings == {check: [] for check in CHAIN_CHECKS}

    def test_fused_mla_with_mtp_layers(self, shared_dir):
        _initialize()
        with TempNamedDir(shared_dir / "fc_mtp_fused_mla") as root:
            findings = run_bf16_fsdp_chain(
                root,
                tp=1,
                build=partial(build_mla_mtp_gpt, tp=1, **FUSED_MLA_MTP),
                configure=partial(_configure, config=FUSED_MLA_MTP),
            )
        assert findings == {check: [] for check in BF16_CHAIN_CHECKS}

    # Not strict: a Megatron change that fixes the handler must not fail this package's tests.
    # An XPASS here is the cue to delete this test and run the fused MLA saves through run_chain.
    @pytest.mark.xfail(
        strict=False,
        raises=AttributeError,
        reason="handle_mla_down_proj_in_state_dict resolves 'module.' + key, which misses the "
        "parameters of an fp32 Megatron-FSDP job (no Float16Module); when this passes (XPASS), "
        "delete this test and run the fused MLA saves through run_chain",
    )
    def test_an_fp32_megatron_fsdp_job_builds_fused_mla_state_dicts(self, tmp_path):
        _initialize()
        with training_args(tmp_path, tp=1) as args:
            _configure(args, fsdp=True, config=FUSED_MLA_MTP, bf16=False)
            model, _ = build_mla_mtp_gpt(
                1, tp=1, use_megatron_fsdp=True, bf16=False, **FUSED_MLA_MTP
            )
            preprocess_fsdp_dtensor_state_dict(args, {"model": model[0].state_dict()}, model[0])


def _legacy_name(key: str) -> str:
    return key.replace(".transformer_layer.", ".mtp_model_layer.")


def _shapes(store) -> dict[str, tuple[int, ...]]:
    return {key: entry.shape for key, entry in MetadataView.load(store / ITER).tensors.items()}


class TestLegacyGptMtpName:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_the_legacy_name_holds_a_real_save_renamed_and_loads_back(self, shared_dir):
        _initialize()
        with TempNamedDir(shared_dir / "fc_mtp_legacy") as root:
            s1, legacy, s2, back = (root / name for name in ("s1", "legacy", "s2", "back"))
            with training_args(s1, tp=1) as args:
                _configure(args, fsdp=False, config=MTP)
                model_a, optimizer_a = build_mla_mtp_gpt(2, tp=1, **MTP)
                expected = full_params(model_a)
                save_weights_only(model_a, optimizer_a, args, s1, iteration=ITERATION)
                torch.distributed.barrier()
                on_rank_zero(
                    convert_store,
                    s1,
                    legacy,
                    "fsdp_dtensor",
                    fsdp_gpt_mtp_inner_name="mtp_model_layer",
                )
                _configure(args, fsdp=True, config=MTP, bf16=False)
                model_b, optimizer_b = build_mla_mtp_gpt(
                    3, tp=1, use_megatron_fsdp=True, bf16=False, **MTP
                )
                save_weights_only(model_b, optimizer_b, args, s2, iteration=ITERATION)
                torch.distributed.barrier()
                real = on_rank_zero(_shapes, s2)
                assert any(".transformer_layer." in key for key in real)
                assert on_rank_zero(_shapes, legacy) == {
                    _legacy_name(k): v for k, v in real.items()
                }
                on_rank_zero(convert_store, legacy, back, "torch_dist")
                assert on_rank_zero(verify_store, back, s1, "D") == []
                _configure(args, fsdp=False, config=MTP)
                model_c, optimizer_c = build_mla_mtp_gpt(4, tp=1, **MTP)
                assert load_weights_only(model_c, optimizer_c, args, back) == ITERATION
                assert_params(full_params(model_c), expected, dtype=torch.bfloat16)
