# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real-save chain (T3): a hybrid ``*-`` model through torch_dist -> fsdp_dtensor -> torch_dist.

Every step uses Megatron's own saves and strict loads (never ``--finetune``,
``log_all`` or a partial fsdp_dtensor load), and every conversion is checked
against something the converter did not produce (``run_chain`` in
``_real_harness.py``):

1. S1: model A (classic, bf16) saved as torch_dist, model only.
2. C1 = convert(S1) to fsdp_dtensor.
3. Model B (Megatron-FSDP, other seed, fp32) loads C1 strictly.
4. B's parameters equal A's exactly upcast.
5. S2: B saved as fsdp_dtensor; its model keys are ``model.module.<name>``
   (exactly one ``module.``), and it matches C1 under contract C (values after
   the exact upcast) and C1 matches it under contract A.
6. C2 = convert(S2) to torch_dist.
7. Model C (classic, another seed) loads C2 strictly; its parameters equal A's
   (the fp32 main weights round back exactly).
8. S3: C saved as torch_dist; it matches S1 under contract C, and C2 matches
   it under contract B.
9. convert(C1) to torch_dist matches S1 under contract D: tensors, dtypes,
   the chunk grid up to merged tensor-parallel splits, and the whole common
   state including the arguments.

Model B trains in fp32: on main, a bf16 Megatron-FSDP job cannot load a store
without optimizer state (``load_checkpoint`` asks the optimizer to copy model
parameters into main parameters, which Megatron-FSDP does not implement); a
strict xfail pins that.

Runs only in the Megatron container with GPUs and the root
``tests/unit_tests/conftest.py`` (its shared directory and process groups);
skipped elsewhere. The default case is TP 1; ``MCORE_FORMAT_CONVERTER_EXTENDED=1``
adds TP 2.
"""

from __future__ import annotations

import os
from functools import partial

import pytest
import torch

from tools.checkpoint.format_converter.mcore_state.common_state import megatron_available

if not (megatron_available() and torch.cuda.is_available()):
    pytest.skip("needs Megatron and CUDA (the training container)", allow_module_level=True)

from tests.unit_tests.dist_checkpointing import TempNamedDir

from .._real_harness import (
    CHAIN_CHECKS,
    ITERATION,
    Utils,
    assert_one_module_level,
    build_hybrid,
    load_weights_only,
    nccl_skip_reason,
    run_chain,
    save_weights_only,
    set_format,
    training_args,
)

PATTERN = "*-*-"

pytestmark = pytest.mark.covers("identity", "real_save")
EXTENDED = os.environ.get("MCORE_FORMAT_CONVERTER_EXTENDED") == "1"
TP_CASES = [1, pytest.param(2, marks=pytest.mark.skipif(not EXTENDED, reason="extended matrix"))]


@pytest.fixture(name="shared_dir")
def fixture_shared_dir(request):
    """The root conftest's directory shared by all ranks (absent under ``--confcutdir``)."""
    try:
        return request.getfixturevalue("tmp_path_dist_ckpt")
    except pytest.FixtureLookupError:
        pytest.skip("needs the root tests/unit_tests/conftest.py (run without --confcutdir)")


class TestHybridChain:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("tp", TP_CASES)
    def test_torch_dist_fsdp_dtensor_torch_dist(self, shared_dir, tp):
        if Utils.world_size % tp:
            pytest.skip(f"needs a world size divisible by tp={tp}")
        Utils.initialize_model_parallel(tp, 1)
        reason = nccl_skip_reason()
        if reason is not None:
            pytest.skip(reason)
        with TempNamedDir(shared_dir / f"fc_cross_tp{tp}") as root:
            findings = run_chain(
                root,
                tp=tp,
                build=partial(build_hybrid, tp=tp, pattern=PATTERN),
                configure=partial(set_format, pattern=PATTERN),
            )
        assert findings == {check: [] for check in CHAIN_CHECKS}


class TestBf16MegatronFsdpWeightsOnlyLoad:
    """Pins the core limitation that makes the chain's Megatron-FSDP model train in fp32.

    The bf16 save before the failing load also pins the key prefix of a bf16
    Megatron-FSDP save (``Float16Module`` sits in its wrapper chain): only a
    ``NotImplementedError`` is the expected failure, so a prefix mismatch fails.
    """

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="bf16 Megatron-FSDP loads without optimizer state call reload_model_params, "
        "which Megatron-FSDP does not implement",
    )
    def test_weights_only_load_into_bf16_megatron_fsdp(self, shared_dir):
        Utils.initialize_model_parallel(1, 1)
        reason = nccl_skip_reason()
        if reason is not None:
            pytest.skip(reason)
        with TempNamedDir(shared_dir / "fc_bf16_mfsdp") as root:
            with training_args(root / "s1", tp=1) as args:
                set_format(args, fsdp=True, pattern=PATTERN)
                model, optimizer = build_hybrid(2, tp=1, pattern=PATTERN, use_megatron_fsdp=True)
                save_weights_only(model, optimizer, args, root / "s1", iteration=ITERATION)
                torch.distributed.barrier()
                assert_one_module_level(root / "s1")
                load_weights_only(model, optimizer, args, root / "s1")
