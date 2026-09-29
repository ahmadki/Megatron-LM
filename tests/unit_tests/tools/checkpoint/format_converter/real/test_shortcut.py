# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shortcut-MoE hybrid models against Megatron's own keys, saves and loads.

Megatron cannot build a Megatron-FSDP job with ``--moe-shortcut-connection``
(``validate_args``), so the fsdp_dtensor side is checked against what
Megatron-FSDP's own state-dict handlers make of the live model instead of a
real fsdp_dtensor save:

* T2: every tensor of the torch_dist sharded state dict of a live
  ``HybridModel`` with shortcut-MoE pairs (and a hybrid MTP layer with a pair
  of its own), decomposed with the layout the converter derives from the
  arguments and composed in fsdp_dtensor names, is the tensor the model's
  state dict holds under that name after ``handle_experts_in_state_dict``:
  the sparse torch_dist indices (``layer_number - 1`` of each module's first
  layer) map to the layer-list indices, and the MTP layer keeps its inner
  name (``get_mtp_inner_layer_paths`` finds no hybrid MTP layer).
* T3: S1 = a real model-only torch_dist save of a classic model A.
  Forward: C1 = convert(S1) must equal the *handler oracle*, the live state
  dict of A through the handlers Megatron-FSDP applies on save (FP8 extra
  state dropped, experts renamed, the MTP rename) under ``model.module.``,
  written as an fsdp_dtensor store (contract A: every key, dtype and value).
  Reverse: convert(oracle) must match S1 (contract B) and load strictly into a
  classic model C (another seed) whose parameters and router buffers then
  equal A's. Round trip: convert(C1) must match S1 under contract D.

Runs only in the Megatron container with GPUs and the root
``tests/unit_tests/conftest.py``; skipped elsewhere.
"""

from __future__ import annotations

import argparse
import copy
from functools import partial

import pytest
import torch

from tools.checkpoint.format_converter.mcore_state.common_state import megatron_available

if not (megatron_available() and torch.cuda.is_available()):
    pytest.skip("needs Megatron and CUDA (the training container)", allow_module_level=True)

# pylint: disable=wrong-import-position
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.transformer.fsdp_dtensor_checkpoint import (
    get_mtp_inner_layer_paths,
    handle_experts_in_state_dict,
    handle_fp8_extra_state_case,
    handle_mtp_in_state_dict,
)
from megatron.core.utils import unwrap_model
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.blocks import BlockFacts, classify_blocks
from tools.checkpoint.format_converter.core.experts import ExpertLayout
from tools.checkpoint.format_converter.core.keys import EXTRA_STATE
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, GROUPED, TORCH_DIST
from tools.checkpoint.format_converter.core.stacking import transformer_block
from tools.checkpoint.format_converter.core.structure import compose, decompose
from tools.checkpoint.format_converter.engine.metadata import TensorEntry
from tools.checkpoint.format_converter.engine.regions import Region

from .._real_harness import (
    ITER,
    ITERATION,
    MOE_CONFIG,
    MOE_EXPERTS,
    Utils,
    assert_params,
    build_hybrid,
    convert_store,
    full_params,
    hybrid_provider,
    load_weights_only,
    main_layers,
    nccl_skip_reason,
    on_rank_zero,
    persistent_buffers,
    save_weights_only,
    set_moe_format,
    training_args,
    verify_store,
)
from .._synthetic import write_fsdp_store

pytestmark = pytest.mark.covers("identity", "real_save")
PATTERN = "E*E-*EE/*EE"
"""A leading E, pairs, a dense MLP, an E after a pair, and an MTP layer holding a pair and an E."""
DENSE_TO_TD = {"decoder": (0, 1, 3, 4, 6), "mtp.layers.0.mtp_model_layer": (0, 2)}
"""The torch_dist index of each module of the hybrid layer lists (written out by hand)."""
SHORTCUT = {"moe_shortcut_connection": True, **MOE_CONFIG}
"""TransformerConfig fields of the shortcut-MoE models."""
CHECKS = ("C1 vs oracle (A)", "reverse(oracle) vs S1 (B)", "back vs S1 (D)")


@pytest.fixture(name="shared_dir")
def fixture_shared_dir(request):
    """The root conftest's directory shared by all ranks (absent under ``--confcutdir``)."""
    try:
        return request.getfixturevalue("tmp_path_dist_ckpt")
    except pytest.FixtureLookupError:
        pytest.skip("needs the root tests/unit_tests/conftest.py (run without --confcutdir)")


def _initialize() -> None:
    Utils.initialize_model_parallel(1, 1)
    reason = nccl_skip_reason()
    if reason is not None:
        pytest.skip(reason)


def _args() -> ArgsView:
    """The arguments the converter reads to lay out the model's blocks."""
    return ArgsView(
        argparse.Namespace(
            hybrid_layer_pattern=PATTERN,
            num_layers=main_layers(PATTERN),
            mtp_num_layers=PATTERN.count("/"),
            moe_shortcut_connection=True,
            num_experts=MOE_EXPERTS,
        )
    )


class TestShortcutShardedKeys:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_megatrons_sparse_keys_map_to_its_state_dict_names(self):
        _initialize()
        model = hybrid_provider(seed=1, pattern=PATTERN, **SHORTCUT)
        assert get_mtp_inner_layer_paths(model) == []  # no rename of a hybrid MTP layer
        sharded = [
            tensor
            for tensor in model.sharded_state_dict().values()
            if isinstance(tensor, ShardedTensor) and not EXTRA_STATE.search(tensor.key)
        ]
        census = {t.key: TensorEntry(t.key, tuple(t.global_shape), t.dtype, ()) for t in sharded}
        blocks = classify_blocks(BlockFacts(_args(), TORCH_DIST, census), default=transformer_block)
        assert {b.prefix: b.dense_to_td for b in blocks if b.symbols} == DENSE_TO_TD
        state = handle_experts_in_state_dict(model.state_dict(), MOE_EXPERTS)
        experts = ExpertLayout(MOE_EXPERTS, GROUPED)
        named = set()
        for tensor in sharded:
            leading = len(tensor.global_shape) - len(tensor.local_shape)
            held = Region(tuple(tensor.global_offset), (1,) * leading + tuple(tensor.local_shape))
            (piece,) = [
                s
                for s in decompose(tensor.key, census[tensor.key], blocks, TORCH_DIST)
                if s.region == held
            ]
            key = compose(piece.lid, piece.shape, blocks, FSDP_DTENSOR, experts).key
            assert torch.equal(state[key], tensor.data.reshape(piece.shape)), key
            named.add(key)
        assert named == {key for key in state if not EXTRA_STATE.search(key)}
        assert "decoder.layers.2.mlp.linear_fc1.weight" in named  # torch_dist index 3
        assert "mtp.layers.0.mtp_model_layer.layers.1.mlp.router.weight" in named  # index 2


def _configure(args, *, fsdp: bool, bf16: bool = True) -> None:
    set_moe_format(args, fsdp=fsdp, grouped=True, ep=1, pattern=PATTERN, bf16=bf16)
    args.moe_shortcut_connection = True


def _handler_oracle(model) -> dict[str, torch.Tensor]:
    """What Megatron-FSDP's save handlers make of the live model's state dict (CPU copies under
    ``model.module.``): FP8 extra state dropped, experts under global indices, the MTP rename."""
    module = unwrap_model(model[0])
    state = dict(module.state_dict())
    handle_fp8_extra_state_case(state)
    state = handle_experts_in_state_dict(state, MOE_EXPERTS)
    state, _ = handle_mtp_in_state_dict(module, state, None)
    return {f"model.module.{name}": tensor.detach().cpu().clone() for name, tensor in state.items()}


def _write_oracle(root, state: dict[str, torch.Tensor], args: argparse.Namespace) -> None:
    """The handler oracle as an fsdp_dtensor store of a Megatron-FSDP job with ``args``."""
    fsdp_args = copy.copy(args)
    fsdp_args.ckpt_format = "fsdp_dtensor"
    fsdp_args.use_megatron_fsdp = True
    tensors = {key.removeprefix("model.module."): tensor for key, tensor in state.items()}
    write_fsdp_store(root / ITER, tensors, args=fsdp_args, iteration=ITERATION)
    (root / "latest_checkpointed_iteration.txt").write_text(str(ITERATION))


class TestShortcutChain:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_torch_dist_to_the_handler_oracle_and_back(self, shared_dir):
        _initialize()
        build = partial(build_hybrid, tp=1, pattern=PATTERN, **SHORTCUT)
        with TempNamedDir(shared_dir / "fc_shortcut") as root:
            s1, c1, oracle, reverse, back = (
                root / name for name in ("s1", "c1", "oracle", "reverse", "back")
            )
            with training_args(s1, tp=1) as args:
                _configure(args, fsdp=False)
                model_a, optimizer_a = build(2)
                expected = full_params(model_a)
                buffers = persistent_buffers(model_a)
                state = _handler_oracle(model_a)
                save_weights_only(model_a, optimizer_a, args, s1, iteration=ITERATION)
                torch.distributed.barrier()
                on_rank_zero(convert_store, s1, c1, "fsdp_dtensor")
                on_rank_zero(_write_oracle, oracle, state, args)
                on_rank_zero(convert_store, oracle, reverse, "torch_dist")
                on_rank_zero(convert_store, c1, back, "torch_dist")
                model_c, optimizer_c = build(4)
                assert load_weights_only(model_c, optimizer_c, args, reverse) == ITERATION
                assert_params(full_params(model_c), expected, dtype=torch.bfloat16)
                assert_params(persistent_buffers(model_c), buffers, dtype=torch.float32)
                findings = on_rank_zero(
                    lambda: {
                        "C1 vs oracle (A)": verify_store(c1, oracle, "A"),
                        "reverse(oracle) vs S1 (B)": verify_store(reverse, s1, "B"),
                        "back vs S1 (D)": verify_store(back, s1, "D"),
                    }
                )
        assert findings == {check: [] for check in CHECKS}
