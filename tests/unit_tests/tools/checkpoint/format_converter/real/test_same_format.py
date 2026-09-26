# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real-save regression (T3): a weights-only rewrite of Megatron's own save loads bit-exactly.

A tiny GPT is saved by ``megatron.training.checkpointing.save_checkpoint`` with the
distributed optimizer in the fully reshardable format, rewritten by the
converter on rank 0 (in process, no workers), and loaded strictly into a fresh,
differently seeded model with ``--no-load-optim --no-load-rng``. Every live
parameter must equal the saving model's bit for bit. Runs only in the Megatron
container with GPUs, with the root ``tests/unit_tests/conftest.py`` loaded (it
provides the directory shared by the ranks and the process groups); skipped
elsewhere, including under the host command's ``--confcutdir``.
"""

from __future__ import annotations

import pytest
import torch

from tools.checkpoint.format_converter.mcore_state.common_state import megatron_available

if not (megatron_available() and torch.cuda.is_available()):
    pytest.skip("needs Megatron and CUDA (the training container)", allow_module_level=True)

from tests.unit_tests.dist_checkpointing import TempNamedDir
from tools.checkpoint.format_converter.api import convert_checkpoint
from tools.checkpoint.format_converter.options import ConvertOptions

from .._real_harness import (
    Utils,
    build_gpt,
    live_params,
    load_weights_only,
    nccl_skip_reason,
    real_save,
    training_args,
)

ITERATION = 10


@pytest.fixture(name="shared_dir")
def fixture_shared_dir(request):
    """The root conftest's directory shared by all ranks (absent under ``--confcutdir``)."""
    try:
        return request.getfixturevalue("tmp_path_dist_ckpt")
    except pytest.FixtureLookupError:
        pytest.skip("needs the root tests/unit_tests/conftest.py (run without --confcutdir)")


def _convert_on_rank_zero(src, dst):
    """Convert on rank 0 only; every rank gets the report summary or the error text."""
    outcome = [None]
    if torch.distributed.get_rank() == 0:
        try:
            report = convert_checkpoint(src, dst, ConvertOptions(workers=0))
            outcome[0] = {"ok": report.to_json()}
        except Exception as err:  # pylint: disable=broad-exception-caught  # re-raised on all ranks
            outcome[0] = {"error": f"{type(err).__name__}: {err}"}
    torch.distributed.broadcast_object_list(outcome, src=0)
    assert "error" not in outcome[0], outcome[0]["error"]
    return outcome[0]["ok"]


class TestWeightsOnlyRewriteOfARealSave:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("tp", [1, 2])
    def test_rewrite_loads_strictly_and_bit_exactly(self, shared_dir, tp):
        if Utils.world_size % tp:
            pytest.skip(f"needs a world size divisible by tp={tp}")
        Utils.initialize_model_parallel(tp, 1)
        reason = nccl_skip_reason()
        if reason is not None:
            pytest.skip(reason)
        with TempNamedDir(shared_dir / f"fc_weights_only_tp{tp}") as root:
            src, dst = root / "src", root / "dst"
            with training_args(src, tp=tp) as args:
                model, optimizer = build_gpt(seed=2, tp=tp)
                real_save(model, optimizer, iteration=ITERATION)
                expected = live_params(model)
                torch.distributed.barrier()
                report = _convert_on_rank_zero(src, dst)
                fresh, fresh_optimizer = build_gpt(seed=3, tp=tp)
                assert any(
                    not torch.equal(param, expected[name])
                    for name, param in live_params(fresh).items()
                ), "the fresh model already holds the saved weights"
                assert load_weights_only(fresh, fresh_optimizer, args, dst) == ITERATION
                loaded = live_params(fresh)
        assert set(loaded) == set(expected)
        for name, param in loaded.items():
            assert param.dtype == expected[name].dtype, name
            assert torch.equal(
                param.reshape(-1).view(torch.uint8), expected[name].reshape(-1).view(torch.uint8)
            ), name
        actions = {s["section"]: s["action"] for s in report["sections"]}
        assert actions["model"] == "keep" and actions["optimizer.state"] == "drop"
        assert "optimizer" in report["common"]["dropped"]
        assert report["common"]["codec"] == "megatron"
