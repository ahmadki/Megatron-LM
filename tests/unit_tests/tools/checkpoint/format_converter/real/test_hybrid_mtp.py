# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real-save chains (T3) of hybrid models with multi-token prediction (MTP) layers.

Each MTP depth's inner model is a hybrid layer list (``mtp.layers.<d>.
mtp_model_layer.layers.<j>``, the same name in both formats): a plain one, one
with mixture-of-experts layers, and one layer serving every depth
(``--mtp-use-repeated-layer``). ``run_chain`` (``_real_harness.py``) takes each
through Megatron's own saves and strict loads in both formats, every
parameter (and router expert bias) equal across the models and every store
checked against a real save under its contract.

Runs only in the Megatron container with GPUs and the root
``tests/unit_tests/conftest.py``; skipped elsewhere.
"""

from __future__ import annotations

from functools import partial

import pytest
import torch

from tools.checkpoint.format_converter.mcore_state.common_state import megatron_available

if not (megatron_available() and torch.cuda.is_available()):
    pytest.skip("needs Megatron and CUDA (the training container)", allow_module_level=True)

# pylint: disable=wrong-import-position
from tests.unit_tests.dist_checkpointing import TempNamedDir

from .._real_harness import (
    CHAIN_CHECKS,
    MOE_CONFIG,
    Utils,
    build_hybrid,
    nccl_skip_reason,
    run_chain,
    set_format,
    set_moe_format,
)

pytestmark = pytest.mark.covers("identity", "real_save")
CASES = [
    pytest.param("*-*-/*-/*-", {}, id="two-depths"),
    pytest.param("*E-/*E", MOE_CONFIG, id="moe"),
    pytest.param("*-/*-/*-", {"mtp_use_repeated_layer": True}, id="repeated"),
]
"""(pattern, TransformerConfig fields) of each model."""


@pytest.fixture(name="shared_dir")
def fixture_shared_dir(request):
    """The root conftest's directory shared by all ranks (absent under ``--confcutdir``)."""
    try:
        return request.getfixturevalue("tmp_path_dist_ckpt")
    except pytest.FixtureLookupError:
        pytest.skip("needs the root tests/unit_tests/conftest.py (run without --confcutdir)")


def _configure(args, *, fsdp: bool, pattern: str, config: dict, bf16: bool = True) -> None:
    if "num_moe_experts" in config:
        set_moe_format(args, fsdp=fsdp, grouped=True, ep=1, pattern=pattern, bf16=bf16)
    else:
        set_format(args, fsdp=fsdp, pattern=pattern, bf16=bf16)
    args.mtp_use_repeated_layer = bool(config.get("mtp_use_repeated_layer", False))


class TestHybridMtpChain:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(("pattern", "config"), CASES)
    def test_torch_dist_fsdp_dtensor_torch_dist(self, shared_dir, pattern, config):
        Utils.initialize_model_parallel(1, 1)
        reason = nccl_skip_reason()
        if reason is not None:
            pytest.skip(reason)
        name = "fc_mtp_" + "".join(c for c in pattern if c.isalnum()) + str(len(config))
        with TempNamedDir(shared_dir / name) as root:
            findings = run_chain(
                root,
                tp=1,
                build=partial(build_hybrid, tp=1, pattern=pattern, **config),
                configure=partial(_configure, pattern=pattern, config=config),
                buffers="num_moe_experts" in config,
            )
        assert findings == {check: [] for check in CHAIN_CHECKS}
