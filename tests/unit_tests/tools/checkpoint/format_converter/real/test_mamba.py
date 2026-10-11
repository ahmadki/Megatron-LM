# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Mamba-2 mixers against Megatron's own modules, saves and loads.

* T3 real-save chains (``run_chain`` in ``_real_harness.py``) of Mamba-2
  hybrid models (``M*-``, and ``M*E`` with a mixture-of-experts layer) at TP 1:
  Megatron saves and strictly loads every store in both formats, every
  parameter must equal model A's after each load, and every store is checked
  against a real save under its contract (A, B, C, D). The TP 2 chain
  (``--fsdp-tp 2``) runs in the extended matrix.
* ``MambaMixer.sharded_state_dict`` at tensor-parallel sizes 1 and 2 (split
  factories applied): the parts of ``in_proj.weight`` and of the convolution's
  weight and bias, their names, order, global and local sizes, are the
  converter's layouts, and the parameters' ``partition_sizes`` are the parts'
  sizes over the tensor-parallel size. ``MambaMixer`` asserts ``bias=False``
  and ``conv_bias=True``: no mixer has an ``in_proj.bias``.
* The torch_dist keys and shapes of a tiny Mamba-2 hybrid model (default
  stack spec) at tensor-parallel sizes 1 and 2, with ``mamba_num_heads`` set
  and unset: the converter's context finds both mixers from the real keys,
  their parts are the ones written out below, and the fsdp_dtensor names it
  composes are the model's state-dict names (what Megatron-FSDP saves:
  ``conv1d_weight``, ``conv1d_bias``).
* A real mixer at tensor-parallel size 2: each rank's fused ``in_proj``
  weight and convolution weight and bias are its block of the
  rank-major pack, by the converter's layout, of the canonical parts gathered
  from the mixer's own torch_dist factory sub-tensors (not their
  concatenation).

Runs only in the Megatron container with GPUs (and ``mamba_ssm``); cases that
need more ranks than the world has are skipped.
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

pytest.importorskip("mamba_ssm")

# pylint: disable=wrong-import-position
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensor, apply_factories
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.context import build_arch_context
from tools.checkpoint.format_converter.core.keys import EXTRA_STATE
from tools.checkpoint.format_converter.core.logical import LogicalId
from tools.checkpoint.format_converter.core.mixers import mixer_layout
from tools.checkpoint.format_converter.core.names import FSDP_DTENSOR, TORCH_DIST
from tools.checkpoint.format_converter.core.structure import compose, decompose
from tools.checkpoint.format_converter.engine.metadata import TensorEntry

from .._real_harness import (
    CHAIN_CHECKS,
    HYBRID_HIDDEN,
    MOE_CONFIG,
    Utils,
    build_hybrid,
    hybrid_provider,
    main_layers,
    nccl_skip_reason,
    randomize_parameters,
    run_chain,
    set_format,
    set_moe_format,
)

pytestmark = [pytest.mark.covers("mamba2", "parity"), pytest.mark.covers("mamba2", "real_save")]
EXTENDED = os.environ.get("MCORE_FORMAT_CONVERTER_EXTENDED") == "1"
PATTERN = "M*M-"
MAMBA = {"mamba_num_heads": 8, "mamba_head_dim": 8, "mamba_num_groups": 2, "mamba_state_dim": 16}
"""TransformerConfig (and argument) fields of the tiny mixers: d = 64, GN = 32, H = 8."""
IN_PROJ = {"z": 64, "x": 64, "B": 32, "C": 32, "dt": 8}
CONV = {"x": 64, "B": 32, "C": 32}
DEFAULT_HEADS = {"z": 128, "x": 128, "B": 32, "C": 32, "dt": 16}
"""Without mamba_num_heads: d = 2 x 64 = 128 and H = 128 / 8 = 16."""


def _initialize(tp: int) -> None:
    if Utils.world_size % tp:
        pytest.skip(f"needs a world size divisible by tp={tp}")
    Utils.initialize_model_parallel(tp, 1)
    reason = nccl_skip_reason()
    if reason is not None:
        pytest.skip(reason)


def _args(tp: int, **mamba) -> ArgsView:
    """The arguments a job of the tiny model saves (those the converter reads)."""
    return ArgsView(
        argparse.Namespace(
            hybrid_layer_pattern=PATTERN,
            num_layers=main_layers(PATTERN),
            spec=None,
            hidden_size=HYBRID_HIDDEN,
            tensor_model_parallel_size=tp,
            **{**MAMBA, **mamba},
        )
    )


@pytest.fixture(name="shared_dir")
def fixture_shared_dir(request):
    """The root conftest's directory shared by all ranks (absent under ``--confcutdir``)."""
    try:
        return request.getfixturevalue("tmp_path_dist_ckpt")
    except pytest.FixtureLookupError:
        pytest.skip("needs the root tests/unit_tests/conftest.py (run without --confcutdir)")


def _configure(args, *, pattern: str, fsdp: bool, bf16: bool = True) -> None:
    """A job of a tiny Mamba-2 hybrid model of ``pattern`` (default stack spec, :data:`MAMBA`
    mixers; one routed-expert module per ``E``)."""
    if "E" in pattern:
        set_moe_format(args, fsdp=fsdp, grouped=False, ep=1, pattern=pattern, bf16=bf16)
    else:
        set_format(args, fsdp=fsdp, pattern=pattern, bf16=bf16)
    args.spec = None
    for name, value in MAMBA.items():
        setattr(args, name, value)


def _chain(shared_dir, name: str, *, tp: int, pattern: str) -> dict[str, list[str]]:
    """:func:`run_chain` of a tiny Mamba-2 hybrid model of ``pattern`` at ``tp``."""
    _initialize(tp)
    config = {**MAMBA, **(MOE_CONFIG if "E" in pattern else {})}
    with TempNamedDir(shared_dir / name) as root:
        return run_chain(
            root,
            tp=tp,
            build=partial(build_hybrid, tp=tp, pattern=pattern, **config),
            configure=partial(_configure, pattern=pattern),
            buffers="E" in pattern,
            options={"fsdp_tp": tp},
        )


class TestMambaChains:
    """Mamba-2 hybrid models through Megatron's saves and strict loads."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("pattern", ["M*-", "M*E"])
    def test_mamba_hybrid(self, shared_dir, pattern):
        findings = _chain(shared_dir, f"fc_mamba_{pattern[-1]}", tp=1, pattern=pattern)
        assert findings == {check: [] for check in CHAIN_CHECKS}

    @pytest.mark.skipif(not EXTENDED, reason="extended matrix")
    def test_mamba_hybrid_at_two_ranks(self, shared_dir):
        findings = _chain(shared_dir, "fc_mamba_tp2", tp=2, pattern="M*-")
        assert findings == {check: [] for check in CHAIN_CHECKS}


def _sharded_tensors(sharded_state_dict) -> list[ShardedTensor]:
    """The sharded tensors of a sharded state dict, split factories applied (no extra state)."""
    apply_factories(sharded_state_dict)
    found = []
    for value in sharded_state_dict.values():
        for tensor in value if isinstance(value, list) else [value]:
            if isinstance(tensor, ShardedTensor) and not EXTRA_STATE.search(tensor.key):
                found.append(tensor)
    return found


def _mixer(seed: int) -> MambaMixer:
    """A Mamba-2 mixer of :data:`MAMBA` with distinct values in every parameter (and every
    tensor-parallel shard)."""
    model_parallel_cuda_manual_seed(seed)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HYBRID_HIDDEN,
        num_attention_heads=4,
        use_cpu_initialization=True,
        **MAMBA,
    )
    mixer = MambaMixer(
        config,
        hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
        config.hidden_size,
        layer_number=1,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"]),
    ).cuda()
    randomize_parameters(mixer, seed)
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


class TestMegatronFactories:
    """``MambaMixer.sharded_state_dict``'s parts against the converter's layouts."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("tp", [1, 2])
    def test_the_factories_cut_the_converters_parts(self, tp):
        _initialize(tp)
        mixer = _mixer(5)
        shapes = {
            t.key: (tuple(t.global_shape), tuple(t.local_shape))
            for t in _sharded_tensors(mixer.sharded_state_dict(prefix="mixer."))
        }
        census = {
            k: TensorEntry(k, global_shape, torch.bfloat16, ())
            for k, (global_shape, _) in shapes.items()
        }
        layout, _ = mixer_layout(_args(tp), census, (), TORCH_DIST)
        expected = {
            f"mixer.in_proj.weight.{part}": ((rows, HYBRID_HIDDEN), (rows // tp, HYBRID_HIDDEN))
            for part, rows in IN_PROJ.items()
        }
        for leaf, rest in (("weight", (1, 4)), ("bias", ())):
            expected.update(
                {
                    f"mixer.conv1d.{leaf}.{part}": ((rows, *rest), (rows // tp, *rest))
                    for part, rows in CONV.items()
                }
            )
        fused = ("mixer.in_proj.weight", "mixer.in_proj.bias", "mixer.conv1d.")
        assert {key: shape for key, shape in shapes.items() if key.startswith(fused)} == expected
        parts = {
            lid.name: dict(zip(fused.parts.parts, fused.parts.sizes))
            for lid, fused in layout.projections.items()
        }
        assert parts == {
            "mixer.in_proj.weight": IN_PROJ,
            "mixer.conv1d.weight": CONV,
            "mixer.conv1d.bias": CONV,
        }
        for lid, fused in layout.projections.items():
            param = mixer.get_parameter(fused.target_lid(FSDP_DTENSOR).name.removeprefix("mixer."))
            assert list(param.partition_sizes) == [size // tp for size in fused.parts.sizes], lid


class TestMegatronKeys:
    """The converter's view of a real Mamba-2 hybrid model's torch_dist keys."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("heads", [8, None])
    @pytest.mark.parametrize("tp", [1, 2])
    def test_the_mixers_of_megatrons_keys_and_their_fsdp_dtensor_names(self, tp, heads):
        _initialize(tp)
        mamba = {**MAMBA, "mamba_num_heads": heads}
        model = hybrid_provider(seed=1, pattern=PATTERN, **mamba)
        census = {
            t.key: TensorEntry(t.key, tuple(t.global_shape), t.dtype, ())
            for t in _sharded_tensors(model.sharded_state_dict())
        }
        assert "decoder.layers.2.mixer.conv1d.weight.C" in census
        ctx = build_arch_context(
            source=TORCH_DIST,
            target=FSDP_DTENSOR,
            args=_args(tp, **mamba),
            model=census,
            fsdp_tp=tp,
        )
        assert ctx.mixers.families_json() == {"decoder": {"mamba2": 2}}
        in_proj = IN_PROJ if heads else DEFAULT_HEADS
        conv = {"x": in_proj["x"], "B": 32, "C": 32}
        assert {
            lid: dict(zip(fused.parts.parts, fused.parts.sizes))
            for lid, fused in ctx.mixers.projections.items()
        } == {
            LogicalId("decoder", layer, f"mixer.{name}", leaf): parts
            for layer in (0, 2)
            for name, leaf, parts in (
                ("in_proj", "weight", in_proj),
                ("conv1d", "weight", conv),
                ("conv1d", "bias", conv),
            )
        }
        names = set()
        for key, entry in census.items():
            for piece in decompose(key, entry, ctx.blocks, TORCH_DIST):
                fused = ctx.mixers.of(piece.lid)
                shape = piece.shape if fused is None else (fused.parts.fused_rows,)
                lid = piece.lid if fused is None else fused.target_lid(FSDP_DTENSOR)
                names.add(compose(lid, shape, ctx.blocks, FSDP_DTENSOR).key)
        assert names == {name for name in model.state_dict() if not EXTRA_STATE.search(name)}
        assert "decoder.layers.0.mixer.conv1d_weight" in names


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
        assert len(layout.projections) == 3  # in_proj.weight, conv1d weight and bias
        for lid, fused in layout.projections.items():
            param = mixer.get_parameter(fused.target_lid(FSDP_DTENSOR).name.removeprefix("mixer."))
            live = _gathered(param.detach()).float()
            parts = {part: canonical[f"{lid.name}.{part}"] for part in fused.parts.parts}
            rest = next(iter(parts.values())).shape[1:]
            packed = torch.full((fused.parts.fused_rows, *rest), float("nan"))
            for run in fused.parts.rank_major_map(2):
                rows = slice(run.fused_start, run.fused_start + run.rows)
                packed[rows] = parts[run.part][run.part_start : run.part_start + run.rows]
            assert torch.equal(live, packed.float()), lid
            assert not torch.equal(packed, torch.cat(list(parts.values()))), lid
