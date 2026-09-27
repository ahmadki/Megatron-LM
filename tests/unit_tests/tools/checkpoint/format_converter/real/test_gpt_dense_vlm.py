# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real-save chains (T3) of stacked TransformerBlocks: a dense GPT and a tiny LLaVA.

Megatron saves both models' ``TransformerBlock`` layers stacked in torch_dist
(one key per tensor, one chunk per layer) and one layer per key in
fsdp_dtensor. Each goes through ``run_chain`` (``_real_harness.py``): Megatron's
own saves and strict loads in both formats, the converted stores checked
against real saves under their expected-difference contracts and the round
trip against the source under contract D (layout included). The chain also
converts the Megatron-FSDP save with ``--torch-dist-layout per-layer`` and
loads it strictly into a classic model: the ``content_metadata`` it writes
makes Megatron build every block (the vision encoder too) per layer.

The default cases run at TP 1; ``MCORE_FORMAT_CONVERTER_EXTENDED=1`` adds TP 2.
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

from tests.unit_tests.dist_checkpointing import TempNamedDir
from tools.checkpoint.format_converter.engine.metadata import MetadataView

from .._real_harness import (
    CHAIN_CHECKS,
    ITER,
    Utils,
    build_llava,
    build_te_gpt,
    nccl_skip_reason,
    on_rank_zero,
    run_chain,
    set_format,
)

pytestmark = pytest.mark.covers("identity", "real_save")
EXTENDED = os.environ.get("MCORE_FORMAT_CONVERTER_EXTENDED") == "1"
TP_CASES = [1, pytest.param(2, marks=pytest.mark.skipif(not EXTENDED, reason="extended matrix"))]
MODELS = {"gpt": build_te_gpt, "llava": build_llava}
BLOCKS = {"gpt": ("decoder",), "llava": ("language_model.decoder", "vision_model.decoder")}
"""The TransformerBlocks of each model."""


def _layouts(store, blocks) -> dict[str, set[str]]:
    """How ``store`` holds each block: ``stacked`` and/or ``per_layer`` keys."""
    keys = MetadataView.load(store / ITER).tensors
    layouts = {}
    for block in blocks:
        heads = [
            key[len(block) + len(".layers.") :]
            for key in keys
            if key.startswith(f"{block}.layers.")
        ]
        layouts[block] = {"per_layer" if head[0].isdigit() else "stacked" for head in heads}
    return layouts


@pytest.fixture(name="shared_dir")
def fixture_shared_dir(request):
    """The root conftest's directory shared by all ranks (absent under ``--confcutdir``)."""
    try:
        return request.getfixturevalue("tmp_path_dist_ckpt")
    except pytest.FixtureLookupError:
        pytest.skip("needs the root tests/unit_tests/conftest.py (run without --confcutdir)")


class TestStackedBlockChains:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("model", sorted(MODELS))
    @pytest.mark.parametrize("tp", TP_CASES)
    def test_torch_dist_fsdp_dtensor_torch_dist(self, shared_dir, model, tp):
        if Utils.world_size % tp:
            pytest.skip(f"needs a world size divisible by tp={tp}")
        Utils.initialize_model_parallel(tp, 1)
        reason = nccl_skip_reason()
        if reason is not None:
            pytest.skip(reason)
        with TempNamedDir(shared_dir / f"fc_{model}_tp{tp}") as root:
            findings = run_chain(
                root,
                tp=tp,
                build=partial(MODELS[model], tp=tp),
                configure=set_format,
                per_layer=True,
            )
            layouts = on_rank_zero(
                lambda: {
                    name: _layouts(root / name, BLOCKS[model])
                    for name in ("s1", "c2", "c2_per_layer")
                }
            )
        assert findings == {check: [] for check in CHAIN_CHECKS}
        stacked = {block: {"stacked"} for block in BLOCKS[model]}
        per_layer = {block: {"per_layer"} for block in BLOCKS[model]}
        assert layouts == {"s1": stacked, "c2": stacked, "c2_per_layer": per_layer}
