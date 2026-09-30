# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Gated DeltaProduct mixers against Megatron's own modules, saves and loads.

* T3 real-save chain (``run_chain`` in ``_real_harness.py``) of a Gated
  DeltaProduct hybrid model at TP 1: Megatron saves and strictly loads every
  store in both formats, every parameter must equal model A's after each
  load, and every store is checked against a real save under its contract.
  At TP 2 current Megatron-FSDP cannot hold the mixer's ``A_log``,
  ``dt_bias`` and ``norm.weight`` (tensor-parallel without a
  ``partition_dim``, so it builds them as replicated tensors of one rank's
  shard): the chain is pinned as a strict expected failure (extended matrix),
  and a real TP 2 torch_dist save converted with ``--fsdp-tp 2`` must hold
  the model's rank-major fused tensors, convert back, load strictly into a
  classic TP 2 model with every parameter equal, and match the source under
  contract D.
* The torch_dist keys and shapes of a tiny Gated DeltaProduct hybrid model
  (its sharded state dict, split factories applied) at tensor-parallel sizes 1
  and 2: the converter's context tells both mixers apart from the real keys
  (the ``M`` family signature, the names, the spec and the widths), their parts
  are the ones written out below, and the fsdp_dtensor names it composes are
  the model's state-dict names (what Megatron-FSDP saves).
* A real mixer at tensor-parallel size 2, with ``in_proj`` and ``conv1d``
  biases: each rank's fused weight and bias is its block of the rank-major
  pack, by the converter's layout, of the canonical parts gathered from the
  mixer's own torch_dist factory sub-tensors (not their concatenation), and
  the parameters' ``partition_sizes`` are the parts' sizes over 2.

Runs only in the Megatron container with GPUs; cases that need more ranks than
the world has are skipped.
"""

from __future__ import annotations

import argparse
import os
from functools import partial

import pytest
import torch

from tools.checkpoint.format_converter.mcore_state.common_state import megatron_available

if not (megatron_available() and torch.cuda.is_available()):
    pytest.skip("needs Megatron and CUDA (the training container)", allow_module_level=True)

gated_delta_product = pytest.importorskip("megatron.core.ssm.gated_delta_product")

# pylint: disable=wrong-import-position
from torch.distributed.checkpoint.api import CheckpointException

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensor, apply_factories
from megatron.core.models.hybrid.hybrid_layer_specs import gated_delta_product_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.context import build_arch_context
from tools.checkpoint.format_converter.core.keys import EXTRA_STATE
from tools.checkpoint.format_converter.core.logical import LogicalId
from tools.checkpoint.format_converter.core.mixers import HYBRID_SPEC_MODULES, mixer_layout
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, TORCH_DIST
from tools.checkpoint.format_converter.core.structure import compose, decompose
from tools.checkpoint.format_converter.engine.metadata import TensorEntry

from .._real_harness import (
    CHAIN_CHECKS,
    HYBRID_HIDDEN,
    ITER,
    ITERATION,
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
    randomize_parameters,
    run_chain,
    save_weights_only,
    set_format,
    training_args,
    verify_store,
)
from ..test_convert_cross import read_tensors

pytestmark = [pytest.mark.covers("gdp", "parity"), pytest.mark.covers("gdp", "real_save")]
EXTENDED = os.environ.get("MCORE_FORMAT_CONVERTER_EXTENDED") == "1"
PATTERN = "M*M-"
GDP = {
    "mamba_num_heads": 4,
    "mamba_head_dim": 8,
    "mamba_num_groups": 2,
    "mamba_state_dim": 8,
    "gdp_num_householder": 3,
}
"""TransformerConfig (and argument) fields of the tiny mixers: d = 32, GN = 16, H = 4, M = 3."""
IN_PROJ = {
    "z": 32,
    "V0": 32,
    "V1": 32,
    "V2": 32,
    "K0": 16,
    "K1": 16,
    "K2": 16,
    "Q": 16,
    "b0": 4,
    "b1": 4,
    "b2": 4,
    "a": 4,
}
CONV = {"V0": 32, "V1": 32, "V2": 32, "K0": 16, "K1": 16, "K2": 16, "Q": 16}
GDP_SPEC = [HYBRID_SPEC_MODULES[0], "gated_delta_product_stack_spec"]


def _initialize(tp: int) -> None:
    if Utils.world_size % tp:
        pytest.skip(f"needs a world size divisible by tp={tp}")
    Utils.initialize_model_parallel(tp, 1)
    reason = nccl_skip_reason()
    if reason is not None:
        pytest.skip(reason)


def _args(tp: int) -> ArgsView:
    """The arguments a job of the tiny model saves (those the converter reads)."""
    return ArgsView(
        argparse.Namespace(
            hybrid_layer_pattern=PATTERN,
            num_layers=main_layers(PATTERN),
            spec=GDP_SPEC,
            hidden_size=HYBRID_HIDDEN,
            tensor_model_parallel_size=tp,
            **GDP,
        )
    )


@pytest.fixture(name="shared_dir")
def fixture_shared_dir(request):
    """The root conftest's directory shared by all ranks (absent under ``--confcutdir``)."""
    try:
        return request.getfixturevalue("tmp_path_dist_ckpt")
    except pytest.FixtureLookupError:
        pytest.skip("needs the root tests/unit_tests/conftest.py (run without --confcutdir)")


def _configure(args, *, fsdp: bool, bf16: bool = True) -> None:
    """A job of the tiny Gated DeltaProduct hybrid model (its spec and mixer dimensions)."""
    set_format(args, fsdp=fsdp, pattern=PATTERN, bf16=bf16)
    args.spec = GDP_SPEC
    for name, value in GDP.items():
        setattr(args, name, value)


def _build(tp: int):
    """``build(seed, ...)`` of the tiny Gated DeltaProduct hybrid model at ``tp``."""
    return partial(
        build_hybrid, tp=tp, pattern=PATTERN, stack_spec=gated_delta_product_stack_spec, **GDP
    )


class TestGdpChains:
    """A Gated DeltaProduct hybrid model through Megatron's saves and strict loads."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gdp_hybrid(self, shared_dir):
        _initialize(1)
        with TempNamedDir(shared_dir / "fc_gdp_tp1") as root:
            findings = run_chain(root, tp=1, build=_build(1), configure=_configure)
        assert findings == {check: [] for check in CHAIN_CHECKS}

    @pytest.mark.skipif(not EXTENDED, reason="extended matrix")
    @pytest.mark.xfail(
        strict=True,
        raises=CheckpointException,
        reason="Megatron-FSDP builds a Gated DeltaProduct mixer's A_log, dt_bias and norm.weight "
        "(tensor_model_parallel without partition_dim) as tensor-parallel replicated DTensors of "
        "one rank's shard, so a TP 2 job cannot load their full tensors (and saves one rank's "
        "shard); when this passes, run the TP 2 chain",
    )
    def test_gdp_hybrid_through_megatron_fsdp_at_two_ranks(self, shared_dir):
        _initialize(2)
        with TempNamedDir(shared_dir / "fc_gdp_tp2") as root:
            run_chain(root, tp=2, build=_build(2), configure=_configure, options={"fsdp_tp": 2})

    def test_a_store_packed_at_two_ranks_holds_the_models_rank_major_tensors(self, shared_dir):
        """The live TP 2 model's fused tensors, its ranks' shards one after the other, are what
        Megatron-FSDP stores; converted back, the store loads strictly into a classic model."""
        _initialize(2)
        with TempNamedDir(shared_dir / "fc_gdp_pack_tp2") as root:
            s1, c1, back = (root / name for name in ("s1", "c1", "back"))
            with training_args(s1, tp=2) as args:
                _configure(args, fsdp=False)
                model_a, optimizer_a = _build(2)(2)
                expected = full_params(model_a)
                fused = {n for n in expected if n.endswith(("in_proj.weight", "conv1d.weight"))}
                assert len(fused) == 4  # of the two mixers
                save_weights_only(model_a, optimizer_a, args, s1, iteration=ITERATION)
                torch.distributed.barrier()
                on_rank_zero(convert_store, s1, c1, "fsdp_dtensor", fsdp_tp=2)
                on_rank_zero(convert_store, c1, back, "torch_dist")
                stored = on_rank_zero(
                    lambda: {
                        name: tensor
                        for name, tensor in read_tensors(c1 / ITER).items()
                        if name.removeprefix("model.module.") in fused
                    }
                )
                assert {name.removeprefix("model.module.") for name in stored} == fused
                for name, tensor in stored.items():
                    live = expected[name.removeprefix("model.module.")]
                    assert tensor.dtype == live.dtype and torch.equal(tensor, live), name
                model_c, optimizer_c = _build(2)(4)
                assert load_weights_only(model_c, optimizer_c, args, back) == ITERATION
                assert_params(full_params(model_c), expected, dtype=torch.bfloat16)
                findings = on_rank_zero(lambda: verify_store(back, s1, "D"))
        assert findings == []


def _sharded_tensors(sharded_state_dict) -> list[ShardedTensor]:
    """The sharded tensors of a sharded state dict, split factories applied (no extra state)."""
    apply_factories(sharded_state_dict)
    found = []
    for value in sharded_state_dict.values():
        for tensor in value if isinstance(value, list) else [value]:
            if isinstance(tensor, ShardedTensor) and not EXTRA_STATE.search(tensor.key):
                found.append(tensor)
    return found


class TestMegatronKeys:
    """The converter's view of a real Gated DeltaProduct hybrid model's torch_dist keys."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("tp", [1, 2])
    def test_the_mixers_of_megatrons_keys_and_their_fsdp_dtensor_names(self, tp):
        _initialize(tp)
        model = hybrid_provider(
            seed=1, pattern=PATTERN, stack_spec=gated_delta_product_stack_spec, **GDP
        )
        census = {
            t.key: TensorEntry(t.key, tuple(t.global_shape), t.dtype, ())
            for t in _sharded_tensors(model.sharded_state_dict())
        }
        assert "decoder.layers.2.mixer.in_proj.weight.V2" in census
        ctx = build_arch_context(
            source=TORCH_DIST, target=FSDP_DTENSOR, args=_args(tp), model=census, fsdp_tp=tp
        )
        assert ctx.mixers.families_json() == {"decoder": {"gdp": 2}}
        assert {
            lid: dict(zip(fused.parts.parts, fused.parts.sizes))
            for lid, fused in ctx.mixers.projections.items()
        } == {
            LogicalId("decoder", layer, f"mixer.{name}", "weight"): parts
            for layer in (0, 2)
            for name, parts in (("in_proj", IN_PROJ), ("conv1d", CONV))
        }
        names = set()
        for key, entry in census.items():
            for piece in decompose(key, entry, ctx.blocks, TORCH_DIST):
                fused = ctx.mixers.of(piece.lid)
                shape = piece.shape if fused is None else (fused.parts.fused_rows,)
                lid = piece.lid if fused is None else fused.lid
                names.add(compose(lid, shape, ctx.blocks, FSDP_DTENSOR).key)
        assert names == {name for name in model.state_dict() if not EXTRA_STATE.search(name)}


def _mixer(tp_rank_seed: int):
    """A Gated DeltaProduct mixer of :data:`GDP` with ``in_proj`` and ``conv1d`` biases and
    distinct values in every parameter (and every tensor-parallel shard)."""
    model_parallel_cuda_manual_seed(tp_rank_seed)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HYBRID_HIDDEN,
        num_attention_heads=4,
        use_cpu_initialization=True,
        **GDP,
    )
    mixer = gated_delta_product.GatedDeltaProductMixer(
        config,
        gated_delta_product_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
        config.hidden_size,
        bias=True,
        conv_bias=True,
        layer_number=1,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"]),
    ).cuda()
    randomize_parameters(mixer, tp_rank_seed)
    return mixer


def _canonical(tensors: list[ShardedTensor]) -> dict[str, torch.Tensor]:
    """Every sharded tensor of every tensor-parallel rank placed in its global tensor (a
    collective)."""
    mine = [
        (t.key, tuple(t.global_shape), tuple(t.global_offset), t.data.detach().cpu())
        for t in tensors
    ]
    group = parallel_state.get_tensor_model_parallel_group()
    everyone: list = [None] * torch.distributed.get_world_size(group)
    torch.distributed.all_gather_object(everyone, mine, group=group)
    canonical: dict[str, torch.Tensor] = {}
    for entries in everyone:
        for key, shape, offset, data in entries:
            whole = canonical.setdefault(key, torch.full(shape, float("nan"), dtype=data.dtype))
            index = tuple(slice(o, o + n) for o, n in zip(offset, data.shape))
            whole[index] = data
    return canonical


def _gathered(local: torch.Tensor) -> torch.Tensor:
    """The tensor-parallel ranks' local tensors, one after the other (a collective)."""
    group = parallel_state.get_tensor_model_parallel_group()
    shards = [torch.empty_like(local) for _ in range(torch.distributed.get_world_size(group))]
    torch.distributed.all_gather(shards, local.contiguous(), group=group)
    return torch.cat(shards).cpu()


class TestMegatronPack:
    """A real mixer's fused tensors against the pack of its own torch_dist parts."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_each_ranks_fused_projections_are_its_block_of_the_rank_major_pack(self):
        _initialize(2)
        mixer = _mixer(3)
        canonical = _canonical(_sharded_tensors(mixer.sharded_state_dict(prefix="mixer.")))
        assert not any(torch.isnan(t).any() for t in canonical.values() if t.is_floating_point())
        census = {k: TensorEntry(k, tuple(t.shape), t.dtype, ()) for k, t in canonical.items()}
        layout, _ = mixer_layout(_args(2), census, (), TORCH_DIST)
        assert len(layout.projections) == 4  # in_proj and conv1d, weight and bias
        for lid, fused in layout.projections.items():
            parts = {part: canonical[f"{lid.name}.{part}"] for part in fused.parts.parts}
            rest = next(iter(parts.values())).shape[1:]
            packed = torch.full((fused.parts.fused_rows, *rest), float("nan"))
            for run in fused.parts.rank_major_map(2):
                rows = slice(run.fused_start, run.fused_start + run.rows)
                packed[rows] = parts[run.part][run.part_start : run.part_start + run.rows]
            param = mixer.get_parameter(lid.name.removeprefix("mixer."))
            assert torch.equal(_gathered(param.detach()).float(), packed.float()), lid
            assert not torch.equal(packed, torch.cat(list(parts.values()))), lid
            assert list(param.partition_sizes) == [size // 2 for size in fused.parts.sizes], lid
