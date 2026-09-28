# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Mixture-of-experts models against Megatron's own saves, loads and expert naming.

* T3 real-save chains (``run_chain`` in ``_real_harness.py``) of a GPT whose
  every layer is a mixture-of-experts layer (``TransformerBlock`` stacked in
  torch_dist: expert tensors ``[L, E, ...]``, one chunk per (layer, expert))
  and of a hybrid ``*E*E`` model (per layer, ``[E, ...]``): Megatron's saves,
  strict loads in both formats, every parameter and router expert bias equal
  across the models, every store checked against a real save under its
  contract: grouped experts with biases at expert-parallel size 1,
  sequential experts with biases and grouped experts without them at 2
  (grouped expert biases are refused with expert parallelism, FC-R05).
* A T2 pin on the live model: every expert tensor of Megatron's torch_dist
  sharded state dict, decomposed and composed in fsdp_dtensor names by the
  converter, is the tensor Megatron-FSDP saves under that name
  (``handle_experts_in_state_dict``), at expert-parallel sizes 1 and 2.

Runs only in the Megatron container with GPUs and the root
``tests/unit_tests/conftest.py``; skipped elsewhere. Cases that need more
ranks than the world has are skipped.
"""

from __future__ import annotations

import dataclasses
from functools import partial

import pytest
import torch

from tools.checkpoint.format_converter.mcore_state.common_state import megatron_available

if not (megatron_available() and torch.cuda.is_available()):
    pytest.skip("needs Megatron and CUDA (the training container)", allow_module_level=True)

# pylint: disable=wrong-import-position
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.transformer.fsdp_dtensor_checkpoint import (
    expert_param_local_key,
    handle_experts_in_state_dict,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tools.checkpoint.format_converter.core.blocks import TRANSFORMER_BLOCK, BlockLayout
from tools.checkpoint.format_converter.core.experts import ExpertLayout
from tools.checkpoint.format_converter.core.names import (
    FSDP_DTENSOR,
    GROUPED,
    SEQUENTIAL,
    STACKED,
    TORCH_DIST,
)
from tools.checkpoint.format_converter.core.structure import compose, decompose
from tools.checkpoint.format_converter.engine.metadata import MetadataView, TensorEntry
from tools.checkpoint.format_converter.engine.regions import Region

from .._real_harness import (
    CHAIN_CHECKS,
    GPT_LAYERS,
    ITER,
    MOE_CONFIG,
    MOE_EXPERTS,
    Utils,
    build_hybrid,
    build_moe_gpt,
    moe_gpt_provider,
    nccl_skip_reason,
    on_rank_zero,
    run_chain,
    set_moe_format,
)

pytestmark = [pytest.mark.covers("identity", "real_save"), pytest.mark.covers("identity", "parity")]
GPT_CASES = [
    pytest.param(True, 1, True, id="grouped-ep1"),
    pytest.param(False, 2, True, id="sequential-ep2"),
    pytest.param(True, 2, False, id="grouped-ep2-no-bias"),
]
"""(grouped, expert-parallel size, biases): grouped expert biases are refused at EP > 1 (FC-R05)."""
FC1 = "decoder.layers.mlp.experts.experts.linear_fc1.weight"
HYBRID_PATTERN = "*E*E"


@pytest.fixture(name="shared_dir")
def fixture_shared_dir(request):
    """The root conftest's directory shared by all ranks (absent under ``--confcutdir``)."""
    try:
        return request.getfixturevalue("tmp_path_dist_ckpt")
    except pytest.FixtureLookupError:
        pytest.skip("needs the root tests/unit_tests/conftest.py (run without --confcutdir)")


def _initialize(ep: int) -> None:
    if Utils.world_size % ep:
        pytest.skip(f"needs a world size divisible by ep={ep}")
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=ep)
    reason = nccl_skip_reason()
    if reason is not None:
        pytest.skip(reason)


def _grid(store, key) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    return [(c.offsets, c.sizes) for c in MetadataView.load(store / ITER).tensors[key].chunks]


class TestMoeChains:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(("grouped", "ep", "bias"), GPT_CASES)
    def test_gpt_moe_torch_dist_fsdp_dtensor_torch_dist(self, shared_dir, grouped, ep, bias):
        _initialize(ep)
        name = f"fc_moe_{'grouped' if grouped else 'sequential'}_ep{ep}"
        with TempNamedDir(shared_dir / name) as root:
            findings = run_chain(
                root,
                tp=1,
                build=partial(build_moe_gpt, tp=1, ep=ep, grouped=grouped, add_bias_linear=bias),
                configure=partial(set_moe_format, grouped=grouped, ep=ep),
                buffers=True,
            )
            grids = on_rank_zero(lambda: {name: _grid(root / name, FC1) for name in ("s1", "c2")})
        assert findings == {check: [] for check in CHAIN_CHECKS}
        per_expert = [
            ((layer, expert, 0, 0), (1, 1, 32, 64))
            for layer in range(GPT_LAYERS)
            for expert in range(MOE_EXPERTS)
        ]
        assert grids == {"s1": per_expert, "c2": per_expert}  # Megatron's grid, and ours

    def test_hybrid_moe_torch_dist_fsdp_dtensor_torch_dist(self, shared_dir):
        _initialize(1)
        with TempNamedDir(shared_dir / "fc_moe_hybrid") as root:
            findings = run_chain(
                root,
                tp=1,
                build=partial(build_hybrid, tp=1, pattern=HYBRID_PATTERN, **MOE_CONFIG),
                configure=partial(set_moe_format, pattern=HYBRID_PATTERN, grouped=False, ep=1),
                buffers=True,
            )
        assert findings == {check: [] for check in CHAIN_CHECKS}


class TestMegatronExpertNaming:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("grouped", [True, False])
    @pytest.mark.parametrize("ep", [1, 2])
    def test_the_converter_names_experts_as_megatron_fsdp_saves_them(self, grouped, ep):
        """Every local expert tensor of Megatron's torch_dist sharded state dict (its global
        expert index on the expert axis) composes to the fsdp_dtensor name under which
        Megatron-FSDP saves the same tensor, except a grouped bias at EP > 1, which Megatron-FSDP
        keeps under its local index (FC-R05)."""
        _initialize(ep)
        model = moe_gpt_provider(seed=1, grouped=grouped, expert_model_parallel_size=ep)
        fsdp = handle_experts_in_state_dict(model.state_dict(), MOE_EXPERTS)
        form = GROUPED if grouped else SEQUENTIAL
        block = BlockLayout(
            "decoder", TRANSFORMER_BLOCK, GPT_LAYERS, STACKED, tuple(range(GPT_LAYERS)), None, ""
        )
        layout = ExpertLayout(MOE_EXPERTS, form)
        local_experts = MOE_EXPERTS // ep
        offset = parallel_state.get_expert_model_parallel_rank() * local_experts
        named = set()
        for tensor in model.sharded_state_dict().values():
            if not isinstance(tensor, ShardedTensor) or ".mlp.experts.experts." not in tensor.key:
                continue
            entry = TensorEntry(tensor.key, tuple(tensor.global_shape), tensor.dtype, ())
            leading = len(tensor.global_shape) - len(tensor.local_shape)
            held = Region(tuple(tensor.global_offset), (1,) * leading + tuple(tensor.local_shape))
            (piece,) = [
                s for s in decompose(tensor.key, entry, [block], TORCH_DIST) if s.region == held
            ]
            lid = piece.lid
            if grouped and lid.leaf == "bias" and ep > 1:  # saved under its local index
                lid = dataclasses.replace(lid, expert=lid.expert - offset)
            key = compose(lid, piece.shape, [block], FSDP_DTENSOR, layout).key
            assert torch.equal(fsdp[key], tensor.data), key
            if piece.lid.leaf == "weight":
                assert expert_param_local_key(key, MOE_EXPERTS) in model.state_dict()
            named.add(key)
        assert named == {k for k in fsdp if ".mlp.experts." in k and "_extra_state" not in k}
        assert len(named) == GPT_LAYERS * local_experts * 4  # fc1, fc2: weight and bias
