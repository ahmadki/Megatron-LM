# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real-save chain (T3): a hybrid ``*-`` model through torch_dist -> fsdp_dtensor -> torch_dist.

Every step uses Megatron's own saves and strict loads (never ``--finetune``,
``log_all`` or a partial fsdp_dtensor load), and every conversion is checked
against something the converter did not produce:

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
import re

import pytest
import torch

from tools.checkpoint.format_converter.mcore_state.common_state import megatron_available

if not (megatron_available() and torch.cuda.is_available()):
    pytest.skip("needs Megatron and CUDA (the training container)", allow_module_level=True)

from tests.unit_tests.dist_checkpointing import TempNamedDir
from tools.checkpoint.format_converter.api import convert_checkpoint, verify_checkpoint
from tools.checkpoint.format_converter.engine.metadata import MetadataView
from tools.checkpoint.format_converter.options import ConvertOptions, VerifyOptions

from .._real_harness import (
    Utils,
    build_hybrid,
    full_params,
    load_weights_only,
    nccl_skip_reason,
    save_weights_only,
    set_format,
    training_args,
)

PATTERN = "*-*-"

pytestmark = pytest.mark.covers("identity", "real_save")
ITERATION = 10
ITER = f"iter_{ITERATION:07d}"
EXTENDED = os.environ.get("MCORE_FORMAT_CONVERTER_EXTENDED") == "1"
TP_CASES = [1, pytest.param(2, marks=pytest.mark.skipif(not EXTENDED, reason="extended matrix"))]


@pytest.fixture(name="shared_dir")
def fixture_shared_dir(request):
    """The root conftest's directory shared by all ranks (absent under ``--confcutdir``)."""
    try:
        return request.getfixturevalue("tmp_path_dist_ckpt")
    except pytest.FixtureLookupError:
        pytest.skip("needs the root tests/unit_tests/conftest.py (run without --confcutdir)")


def _on_rank_zero(function, *args, **kwargs):
    """Run ``function`` on rank 0; every rank gets its JSON result or its error text."""
    outcome = [None]
    if torch.distributed.get_rank() == 0:
        try:
            outcome[0] = {"ok": function(*args, **kwargs)}
        except Exception as err:  # pylint: disable=broad-exception-caught  # re-raised on all ranks
            outcome[0] = {"error": f"{type(err).__name__}: {err}"}
    torch.distributed.broadcast_object_list(outcome, src=0)
    assert "error" not in outcome[0], outcome[0]["error"]
    return outcome[0]["ok"]


def _convert(src, dst, to_format):
    return convert_checkpoint(src, dst, ConvertOptions(to_format=to_format, workers=0)).to_json()


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


def _verify(a, b, contract, *, jobs_differ=False):
    ignored = [re.compile(rf"^args\.{name}$") for name in JOB_FIELDS] if jobs_differ else []
    options = VerifyOptions(expected_diff=contract, ignore_keys=tuple(ignored))
    report = verify_checkpoint(a, b, options)
    return [f"{f.kind.value} {f.key}: {f.detail}" for f in report.findings]


def _bits(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def _assert_one_module_level(store):
    """A real Megatron-FSDP v1 save's keys are ``model.module.<name>`` (one ``module.``)."""
    keys = _on_rank_zero(lambda: sorted(MetadataView.load(store / ITER).tensors))
    assert keys and all(
        k.startswith("model.module.") and not k.startswith("model.module.module.") for k in keys
    ), keys


def _assert_params(actual, expected, *, dtype):
    assert set(actual) == set(expected)
    for name, value in actual.items():
        assert value.dtype == dtype, name
        assert _bits(value) == _bits(expected[name].to(dtype)), name


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
            s1, c1, s2, c2, s3, back = (root / n for n in ("s1", "c1", "s2", "c2", "s3", "back"))
            with training_args(s1, tp=tp) as args:
                set_format(args, fsdp=False, pattern=PATTERN)
                model_a, optimizer_a = build_hybrid(2, tp=tp, pattern=PATTERN)
                expected = full_params(model_a)
                save_weights_only(model_a, optimizer_a, args, s1, iteration=ITERATION)
                torch.distributed.barrier()
                _on_rank_zero(_convert, s1, c1, "fsdp_dtensor")

                set_format(args, fsdp=True, pattern=PATTERN, bf16=False)
                model_b, optimizer_b = build_hybrid(
                    3, tp=tp, pattern=PATTERN, use_megatron_fsdp=True, bf16=False
                )
                assert load_weights_only(model_b, optimizer_b, args, c1) == ITERATION
                _assert_params(full_params(model_b), expected, dtype=torch.float32)
                save_weights_only(model_b, optimizer_b, args, s2, iteration=ITERATION)
                torch.distributed.barrier()
                _assert_one_module_level(s2)
                _on_rank_zero(_convert, s2, c2, "torch_dist")

                set_format(args, fsdp=False, pattern=PATTERN)
                model_c, optimizer_c = build_hybrid(4, tp=tp, pattern=PATTERN)
                assert load_weights_only(model_c, optimizer_c, args, c2) == ITERATION
                _assert_params(full_params(model_c), expected, dtype=torch.bfloat16)
                save_weights_only(model_c, optimizer_c, args, s3, iteration=ITERATION)
                torch.distributed.barrier()
                _on_rank_zero(_convert, c1, back, "torch_dist")
                findings = _on_rank_zero(
                    lambda: {
                        "C1 vs S2 (A)": _verify(c1, s2, "A"),
                        "S2 vs C1 (C)": _verify(s2, c1, "C", jobs_differ=True),
                        "C2 vs S3 (B)": _verify(c2, s3, "B"),
                        "S3 vs S1 (C)": _verify(s3, s1, "C", jobs_differ=True),
                        "back vs S1 (D)": _verify(back, s1, "D"),
                    }
                )
        assert findings == {
            "C1 vs S2 (A)": [],
            "S2 vs C1 (C)": [],
            "C2 vs S3 (B)": [],
            "S3 vs S1 (C)": [],
            "back vs S1 (D)": [],
        }


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
                _assert_one_module_level(root / "s1")
                load_weights_only(model, optimizer, args, root / "s1")
