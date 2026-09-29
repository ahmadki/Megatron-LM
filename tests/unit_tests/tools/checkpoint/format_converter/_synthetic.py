# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Torch-only builders of synthetic DCP stores for the format_converter tests.

Stores are written with ``torch.distributed.checkpoint.save(no_dist=True)``
through :class:`ChunkedSavePlanner`, a TEST-ONLY planner that emits explicit
``SHARD`` write items, so a tensor can be stored as any chunk grid. Nothing here
imports megatron; Megatron classes that stores reference are written under
their real global names by :class:`AliasPickler` without importing them.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import re
import types
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.default_planner import create_default_global_save_plan
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import (
    SavePlan,
    SavePlanner,
    TensorWriteData,
    WriteItem,
    WriteItemType,
)

from tools.checkpoint.format_converter.engine.regions import Region


class ChunkedSavePlanner(SavePlanner):
    """TEST-ONLY save planner writing each tensor as an explicit chunk grid."""

    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor],
        grids: Mapping[str, Sequence[Region]],
        byte_items: Mapping[str, bytes],
        planner_data: Mapping[str, tuple[str, ...]] | None,
    ) -> None:
        self.tensors = tensors
        self.grids = grids
        self.byte_items = byte_items
        self.planner_data = planner_data

    def set_up_planner(self, state_dict, storage_meta=None, is_coordinator=False) -> None:
        del state_dict, storage_meta, is_coordinator

    def create_local_plan(self) -> SavePlan:
        items = []
        for key, tensor in self.tensors.items():
            for index, region in enumerate(self.grids[key]):
                chunk = ChunkStorageMetadata(
                    offsets=torch.Size(region.offsets), sizes=torch.Size(region.sizes)
                )
                items.append(
                    WriteItem(
                        index=MetadataIndex(key, region.offsets, index),
                        type=WriteItemType.SHARD,
                        tensor_data=TensorWriteData(
                            chunk=chunk,
                            properties=TensorProperties.create_from_tensor(tensor),
                            size=tensor.size(),
                        ),
                    )
                )
        for key in self.byte_items:
            items.append(WriteItem(index=MetadataIndex(key), type=WriteItemType.BYTE_IO))
        return SavePlan(items)

    def create_global_plan(self, all_plans):
        plans, metadata = create_default_global_save_plan(all_plans)
        if self.planner_data is not None:
            metadata.planner_data = dict(self.planner_data)
        return plans, metadata

    def finish_plan(self, new_plan):
        return new_plan

    def resolve_data(self, write_item):
        if write_item.type == WriteItemType.BYTE_IO:
            return io.BytesIO(self.byte_items[write_item.index.fqn])
        chunk = write_item.tensor_data.chunk
        region = Region(tuple(chunk.offsets), tuple(chunk.sizes))
        return self.tensors[write_item.index.fqn][region.slices()].clone()


def write_store(
    iter_dir: Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    grids: Mapping[str, Sequence[Region]] | None = None,
    byte_items: Mapping[str, bytes] | None = None,
    planner_data: Mapping[str, tuple[str, ...]] | None = None,
    thread_count: int = 1,
) -> Path:
    """Write a DCP store; tensors without a grid are stored as one chunk."""
    grids = dict(grids or {})
    for key, tensor in tensors.items():
        grids.setdefault(key, (Region.full(tuple(tensor.shape)),))
    iter_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="torch.distributed is disabled")
        dcp.save(
            {},
            storage_writer=FileSystemWriter(
                iter_dir, thread_count=thread_count, per_thread_copy_ahead=0
            ),
            planner=ChunkedSavePlanner(tensors, grids, dict(byte_items or {}), planner_data),
            no_dist=True,
        )
    return iter_dir


def dim0_grid(shape: Sequence[int], rows: int) -> tuple[Region, ...]:
    """Chunks of ``rows`` rows along dimension 0 (the last one may be shorter)."""
    return tuple(Region.full(tuple(shape)).split_dim0(rows))


def product_grid(shape: Sequence[int], cuts: Sequence[Sequence[int]]) -> tuple[Region, ...]:
    """The grid whose per-dimension boundaries are ``cuts[d]`` (interior cut points)."""
    axes = []
    for extent, points in zip(shape, cuts):
        bounds = [0, *points, extent]
        axes.append([(lo, hi - lo) for lo, hi in zip(bounds, bounds[1:])])
    regions = [((), ())]
    for axis in axes:
        regions = [(o + (lo,), s + (n,)) for o, s in regions for lo, n in axis]
    return tuple(Region(o, s) for o, s in regions)


def torch_save_bytes(obj: Any, aliases: Mapping[Any, tuple[str, str]] | None = None) -> bytes:
    """``torch.save(obj)`` into memory, as DCP stores byte items.

    With ``aliases``, the listed classes are pickled under other global names
    (see :class:`AliasPickler`), e.g. to reference Megatron enums.
    """
    buffer = io.BytesIO()
    if aliases:
        torch.save(obj, buffer, pickle_module=_alias_pickle_module(aliases), pickle_protocol=2)
    else:
        torch.save(obj, buffer)
    return buffer.getvalue()


class AliasPickler(pickle._Pickler):  # pylint: disable=protected-access
    """Pickler that writes chosen classes or functions under another global name.

    Lets tests produce pickles that reference ``megatron.*`` (or forbidden)
    globals without importing or even having them.
    """

    def __init__(self, file, aliases: Mapping[Any, tuple[str, str]], protocol: int = 2) -> None:
        super().__init__(file, protocol=protocol)
        self.aliases = dict(aliases)

    def save_global(self, obj, name=None):
        alias = self.aliases.get(obj)
        if alias is None:
            super().save_global(obj, name)
            return
        module, qualname = alias
        self.write(pickle.GLOBAL + f"{module}\n{qualname}\n".encode())
        self.memoize(obj)


def _alias_pickle_module(aliases: Mapping[Any, tuple[str, str]]) -> types.SimpleNamespace:
    class _Pickler(AliasPickler):
        def __init__(self, file, protocol: int = 2) -> None:
            super().__init__(file, aliases, protocol=protocol)

    return types.SimpleNamespace(__name__="alias_pickle", Pickler=_Pickler, dump=pickle.dump)


def alias_dumps(obj: Any, aliases: Mapping[Any, tuple[str, str]]) -> bytes:
    """Pickle ``obj`` (protocol 2) with :class:`AliasPickler`."""
    buffer = io.BytesIO()
    AliasPickler(buffer, aliases).dump(obj)
    return buffer.getvalue()


def rewrite_metadata(
    iter_dir: Path,
    mutate: Callable[[Metadata], Any],
    aliases: Mapping[Any, tuple[str, str]] | None = None,
) -> None:
    """Load ``.metadata`` with plain pickle, let ``mutate`` edit or replace it, write it back."""
    path = iter_dir / ".metadata"
    metadata = pickle.loads(path.read_bytes())
    replacement = mutate(metadata)
    path.write_bytes(alias_dumps(metadata if replacement is None else replacement, aliases or {}))


TORCH_DIST_CONFIG = {"sharded_backend": "torch_dist", "sharded_backend_version": 1}


def default_args(**overrides: Any) -> argparse.Namespace:
    """A small training-args Namespace like the one Megatron stores."""
    values = {
        "num_layers": 2,
        "hidden_size": 8,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "bf16": True,
        "ckpt_format": "torch_dist",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def write_torch_dist_store(
    iter_dir: Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    grids: Mapping[str, Sequence[Region]] | None = None,
    common: Mapping[str, Any] | None = None,
    common_aliases: Mapping[Any, tuple[str, str]] | None = None,
    extra_byte_items: Mapping[str, bytes] | None = None,
) -> Path:
    """A torch_dist-like store: bare keys, in-DCP common state and metadata.json."""
    if common is None:
        common = {"args": default_args(), "iteration": 7, "checkpoint_version": 3.0}
    byte_items = {"common_state/shard_0_1": torch_save_bytes([dict(common)], common_aliases)}
    byte_items.update(extra_byte_items or {})
    write_store(iter_dir, tensors, grids=grids, byte_items=byte_items)
    (iter_dir / "metadata.json").write_text(json.dumps(TORCH_DIST_CONFIG))
    return iter_dir


def write_fsdp_store(
    iter_dir: Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    grids: Mapping[str, Sequence[Region]] | None = None,
    args: argparse.Namespace | None = None,
    iteration: int = 7,
    prefix: str = "model.module.",
    common_aliases: Mapping[Any, tuple[str, str]] | None = None,
    extra_byte_items: Mapping[str, bytes] | None = None,
    extra_tensors: Mapping[str, torch.Tensor] | None = None,
) -> Path:
    """An fsdp_dtensor-like store: ``prefix`` + fqn keys and top-level common byte items.

    ``extra_tensors`` are stored under their own keys (optimizer or RNG state).
    """
    common = {
        "args": args if args is not None else default_args(ckpt_format="fsdp_dtensor"),
        "iteration": iteration,
        "checkpoint_version": 3.0,
    }
    byte_items = {key: torch_save_bytes(value, common_aliases) for key, value in common.items()}
    byte_items.update(extra_byte_items or {})
    keyed = {prefix + fqn: tensor for fqn, tensor in tensors.items()}
    keyed.update(extra_tensors or {})
    keyed_grids = {prefix + fqn: grid for fqn, grid in (grids or {}).items()}
    planner_data = {key: ("model", key[len("model.") :]) for key in keyed}
    planner_data.update({key: (key,) for key in byte_items})
    return write_store(
        iter_dir, keyed, grids=keyed_grids, byte_items=byte_items, planner_data=planner_data
    )


def write_tracker(root: Path, value: int | str) -> Path:
    """Write ``latest_checkpointed_iteration.txt`` as Megatron does."""
    tracker = root / "latest_checkpointed_iteration.txt"
    tracker.write_text(f"{value}\n")
    return tracker


def hybrid_args(pattern: str = "*-|*-", **overrides: Any) -> argparse.Namespace:
    """Training arguments of a per-layer hybrid model (``*`` attention and ``-`` MLP layers)."""
    values = {
        "hybrid_layer_pattern": pattern,
        "num_layers": len(pattern.split("/")[0].replace("|", "")),
        "hidden_size": 8,
        "transformer_impl": "transformer_engine",
        "spec": None,
        "swiglu": False,
        "num_experts": None,
        "bf16": True,
        "fp16": False,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 2,
        "virtual_pipeline_model_parallel_size": None,
        "ckpt_format": "torch_dist",
        "use_megatron_fsdp": False,
        "no_save_optim": False,
        "no_save_rng": False,
        "lr": 3e-4,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def hybrid_tensors(
    pattern: str = "*-*-", hidden: int = 8, ffn: int = 16
) -> dict[str, torch.Tensor]:
    """Random model tensors of a per-layer hybrid model, by model-relative key.

    Weights are bfloat16 drawn from a fixed seed; ``decoder.final_norm.weight``
    is float32.
    """
    shapes = {"embedding.word_embeddings.weight": (32, hidden)}
    for index, symbol in enumerate(pattern):
        layer = f"decoder.layers.{index}."
        if symbol == "*":
            shapes[layer + "self_attention.linear_qkv.weight"] = (3 * hidden, hidden)
            shapes[layer + "self_attention.linear_qkv.layer_norm_weight"] = (hidden,)
            shapes[layer + "self_attention.linear_proj.weight"] = (hidden, hidden)
        else:
            shapes[layer + "mlp.linear_fc1.weight"] = (ffn, hidden)
            shapes[layer + "mlp.linear_fc1.layer_norm_weight"] = (hidden,)
            shapes[layer + "mlp.linear_fc2.weight"] = (hidden, ffn)
    shapes["output_layer.weight"] = (32, hidden)
    generator = torch.Generator().manual_seed(1234)
    tensors = {
        key: torch.randn(shape, generator=generator).to(torch.bfloat16)
        for key, shape in shapes.items()
    }
    tensors["decoder.final_norm.weight"] = torch.randn(hidden, generator=generator)
    return tensors


def gpt_args(num_layers: int = 3, **overrides: Any) -> argparse.Namespace:
    """Training arguments of a GPT model (Transformer Engine spec, no hybrid pattern)."""
    values = {
        "num_layers": num_layers,
        "hidden_size": 8,
        "hybrid_layer_pattern": None,
        "transformer_impl": "transformer_engine",
        "spec": None,
        "swiglu": False,
        "num_experts": None,
        "moe_layer_freq": 1,
        "bf16": True,
        "fp16": False,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "virtual_pipeline_model_parallel_size": None,
        "ckpt_format": "torch_dist",
        "use_megatron_fsdp": False,
        "no_save_optim": False,
        "no_save_rng": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def transformer_layers(
    block: str, layers: int, hidden: int = 8, ffn: int = 16, seed: int = 99
) -> dict[str, torch.Tensor]:
    """Random tensors of a ``TransformerBlock`` (Transformer Engine layers with biases), one
    key per layer as fsdp_dtensor stores them (``<block>.layers.<l>.<name>``), plus the block's
    final layer norm.

    Weights are bfloat16; the layer norms are float32, as frozen or fp32 tensors may be.
    """
    shapes = {
        "self_attention.linear_qkv.layer_norm_weight": (hidden,),
        "self_attention.linear_qkv.layer_norm_bias": (hidden,),
        "self_attention.linear_qkv.weight": (3 * hidden, hidden),
        "self_attention.linear_qkv.bias": (3 * hidden,),
        "self_attention.linear_proj.weight": (hidden, hidden),
        "self_attention.linear_proj.bias": (hidden,),
        "mlp.linear_fc1.layer_norm_weight": (hidden,),
        "mlp.linear_fc1.layer_norm_bias": (hidden,),
        "mlp.linear_fc1.weight": (ffn, hidden),
        "mlp.linear_fc1.bias": (ffn,),
        "mlp.linear_fc2.weight": (hidden, ffn),
        "mlp.linear_fc2.bias": (hidden,),
    }
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer in range(layers):
        for name, shape in shapes.items():
            value = torch.randn(shape, generator=generator)
            dtype = torch.float32 if "layer_norm" in name else torch.bfloat16
            tensors[f"{block}.layers.{layer}.{name}"] = value.to(dtype)
    tensors[f"{block}.final_layernorm.weight"] = torch.randn(hidden, generator=generator)
    return tensors


def gpt_tensors(layers: int = 3, hidden: int = 8, ffn: int = 16) -> dict[str, torch.Tensor]:
    """Random model tensors of a GPT model, one key per layer (see :func:`transformer_layers`)."""
    generator = torch.Generator().manual_seed(7)
    return {
        "embedding.word_embeddings.weight": torch.randn(32, hidden, generator=generator).bfloat16(),
        **transformer_layers("decoder", layers, hidden, ffn),
        "output_layer.weight": torch.randn(32, hidden, generator=generator).bfloat16(),
    }


def stack_layers(tensors: Mapping[str, torch.Tensor], block: str) -> dict[str, torch.Tensor]:
    """The tensors with ``block``'s layers stacked as torch_dist stores a homogeneous block.

    ``<block>.layers.<l>.<name>`` for ``l = 0..L-1`` become ``<block>.layers.<name>`` of shape
    ``[L, ...]`` (``torch.stack`` in layer order); other keys are kept.
    """
    head = f"{block}.layers."
    stacked: dict[str, list[torch.Tensor]] = {}
    kept = {}
    for key, tensor in tensors.items():
        if key.startswith(head):
            index, name = key[len(head) :].split(".", 1)
            layers = stacked.setdefault(f"{head}{name}", [])
            assert int(index) == len(layers), f"{key} is out of layer order"
            layers.append(tensor)
        else:
            kept[key] = tensor
    return {**kept, **{key: torch.stack(layers) for key, layers in stacked.items()}}


def layer_grids(tensors: Mapping[str, torch.Tensor], block: str) -> dict[str, tuple[Region, ...]]:
    """One chunk per layer for every tensor stacked over ``block``'s layers (as Megatron saves)."""
    head = f"{block}.layers."
    return {
        key: dim0_grid(tuple(tensor.shape), 1)
        for key, tensor in tensors.items()
        if key.startswith(head) and not key[len(head)].isdigit()
    }


def vision_language_tensors(
    language_model: Mapping[str, torch.Tensor], vision_layers: int = 2, vision_hidden: int = 4
) -> dict[str, torch.Tensor]:
    """A vision-language (LLaVA-style) model: ``language_model`` under ``language_model.``,
    a CLIP-like vision encoder (a TransformerBlock of ``vision_layers`` layers, one key per
    layer, bfloat16 as a frozen encoder is stored) and an MLP projection."""
    generator = torch.Generator().manual_seed(11)
    vision = {
        key: tensor.bfloat16()
        for key, tensor in transformer_layers(
            "vision_model.decoder", vision_layers, vision_hidden, 2 * vision_hidden, seed=5
        ).items()
    }
    hidden = language_model["embedding.word_embeddings.weight"].shape[1]
    shapes = {
        "vision_model.class_token": (1, 1, vision_hidden),
        "vision_model.conv1.weight": (vision_hidden, 3, 2, 2),
        "vision_model.position_embeddings.weight": (5, vision_hidden),
        "vision_model.ln_pre.weight": (vision_hidden,),
        "vision_projection.encoder.linear_fc1.weight": (hidden, vision_hidden),
        "vision_projection.encoder.linear_fc1.bias": (hidden,),
        "vision_projection.encoder.linear_fc2.weight": (hidden, hidden),
        "vision_projection.encoder.linear_fc2.bias": (hidden,),
    }
    extra = {
        key: torch.randn(shape, generator=generator).bfloat16() for key, shape in shapes.items()
    }
    return {
        **{f"language_model.{key}": tensor for key, tensor in language_model.items()},
        **vision,
        **extra,
    }


def moe_layer_tensors(
    layer: str, experts: int, hidden: int = 8, ffn: int = 6, *, bias: bool, seed: int
) -> dict[str, torch.Tensor]:
    """Random tensors of one mixture-of-experts layer under ``layer`` (e.g. ``decoder.layers.1``).

    The routed experts are named as fsdp_dtensor names grouped experts
    (``mlp.experts.linear_fc<n>.weight<e>``, ``bias<e>`` with ``bias``); the router has a
    bfloat16 weight and a float32 ``expert_bias``, the pre-MLP norm is float32. Each expert
    draws its own values.
    """
    generator = torch.Generator().manual_seed(seed)
    shapes = {"linear_fc1": (ffn, hidden), "linear_fc2": (hidden, ffn)}
    tensors = {
        f"{layer}.pre_mlp_layernorm.weight": torch.randn(hidden, generator=generator),
        f"{layer}.mlp.router.weight": torch.randn(experts, hidden, generator=generator).bfloat16(),
        f"{layer}.mlp.router.expert_bias": torch.randn(experts, generator=generator),
    }
    for expert in range(experts):
        for linear, shape in shapes.items():
            weight = torch.randn(shape, generator=generator).bfloat16()
            tensors[f"{layer}.mlp.experts.{linear}.weight{expert}"] = weight
            if bias:
                tensors[f"{layer}.mlp.experts.{linear}.bias{expert}"] = torch.randn(
                    shape[0], generator=generator
                ).bfloat16()
    return tensors


def gpt_moe_tensors(
    layers: int = 2, experts: int = 4, hidden: int = 8, *, moe_layers=None, bias: bool = True
) -> dict[str, torch.Tensor]:
    """A GPT model whose ``moe_layers`` (default: all) are mixture-of-experts layers.

    One tensor per layer and per expert, grouped experts named as fsdp_dtensor
    names them (see :func:`moe_layer_tensors`); other layers have a dense MLP.
    """
    dense = gpt_tensors(layers, hidden, ffn=2 * hidden)
    moe_layers = range(layers) if moe_layers is None else moe_layers
    tensors = {}
    for key, tensor in dense.items():
        head, _, name = key.partition(".layers.")
        index = name.split(".", 1)[0] if head == "decoder" else None
        if index is not None and int(index) in moe_layers and ".mlp." in name:
            continue
        tensors[key] = tensor
    for layer in moe_layers:
        tensors.update(
            moe_layer_tensors(f"decoder.layers.{layer}", experts, hidden, bias=bias, seed=layer)
        )
    return tensors


_GROUPED_EXPERT = re.compile(
    r"^(?P<head>.*\.mlp\.experts)\.(?P<linear>linear_fc\d)\.(?P<leaf>weight|bias)(?P<e>\d+)$"
)


_GROUPED_EXPERT_HALF = re.compile(
    r"^(?P<head>.*\.mlp\.experts)\.(?P<linear>linear_fc\d)\.(?P<leaf>weight|bias)(?P<e>\d+)"
    r"(?P<half>_[wv])?$"
)


def sequential_experts(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The tensors with grouped expert names (SwiGLU halves too) renamed as ``SequentialMLP``
    names them."""
    renamed = {}
    for key, tensor in tensors.items():
        match = _GROUPED_EXPERT_HALF.match(key)
        if match:
            key = (
                f"{match['head']}.local_experts.{match['e']}.{match['linear']}.{match['leaf']}"
                f"{match['half'] or ''}"
            )
        renamed[key] = tensor
    return renamed


def stack_experts(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The tensors with each layer's grouped experts stacked as torch_dist stores them.

    ``<...>.mlp.experts.linear_fc<n>.<leaf><e>`` for ``e = 0..E-1`` become
    ``<...>.mlp.experts.experts.linear_fc<n>.<leaf>`` of shape ``[E, ...]``
    (``torch.stack`` in expert order); other keys are kept.
    """
    stacked: dict[str, dict[int, torch.Tensor]] = {}
    kept = {}
    for key, tensor in tensors.items():
        match = _GROUPED_EXPERT.match(key)
        if match:
            name = f"{match['head']}.experts.{match['linear']}.{match['leaf']}"
            stacked.setdefault(name, {})[int(match["e"])] = tensor
        else:
            kept[key] = tensor
    for name, experts in stacked.items():
        assert sorted(experts) == list(range(len(experts))), name
        kept[name] = torch.stack([experts[e] for e in range(len(experts))])
    return kept


def expert_grids(tensors: Mapping[str, torch.Tensor]) -> dict[str, tuple[Region, ...]]:
    """One chunk per expert (and per layer of a stacked tensor) of every torch_dist expert
    tensor, as Megatron saves them."""
    grids = {}
    for key, tensor in tensors.items():
        if ".mlp.experts.experts." in key:
            axes = 2 if key.split(".layers.", 1)[1][0].isalpha() else 1
            cuts = [list(range(1, tensor.shape[d])) if d < axes else [] for d in range(tensor.ndim)]
            grids[key] = product_grid(tuple(tensor.shape), cuts)
    return grids


_FC1_TENSOR = re.compile(r"\.linear_fc1\.(?:weight|bias)\d*$")


def gated_units(
    tensors: Mapping[str, torch.Tensor], seed: int = 17, *, prefix: str = ""
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """``tensors`` with every ``linear_fc1`` weight and bias under ``prefix`` made a gated unit.

    Each ``[F, ...]`` tensor is replaced by a gate ``W`` and a value ``V`` of its shape and dtype,
    drawn independently from ``seed``. Returns them twice: as SwiGLU halves (``<key>_w`` and
    ``<key>_v``, as Megatron-FSDP splits them) and fused (``<key>`` = ``torch.cat([W, V])``, as
    torch_dist stores them); every other tensor is kept in both.
    """
    generator = torch.Generator().manual_seed(seed)
    halves, fused = {}, {}
    for key, tensor in tensors.items():
        if not key.startswith(prefix) or _FC1_TENSOR.search(key) is None:
            halves[key] = fused[key] = tensor
            continue
        gate, value = (
            torch.randn(tensor.shape, generator=generator).to(tensor.dtype) for _ in range(2)
        )
        halves[f"{key}_w"], halves[f"{key}_v"] = gate, value
        fused[key] = torch.cat([gate, value])
    return halves, fused


def swiglu_grids(tensors: Mapping[str, torch.Tensor]) -> dict[str, tuple[Region, ...]]:
    """Megatron's torch_dist grid of every gated ``linear_fc1`` of ``tensors`` (torch_dist keys) at
    tensor-parallel size 1: one chunk per half (the SwiGLU factory), per layer of a stacked tensor
    and per expert of an expert tensor."""
    grids = {}
    for key, tensor in tensors.items():
        if not re.search(r"\.linear_fc1\.(?:weight|bias)$", key):
            continue
        stacked = ".layers." in key and key.split(".layers.", 1)[1][0].isalpha()
        lead = int(stacked) + int(".mlp.experts.experts." in key)
        cuts = [
            list(range(1, size)) if d < lead else [size // 2] if d == lead else []
            for d, size in enumerate(tensor.shape)
        ]
        grids[key] = product_grid(tuple(tensor.shape), cuts)
    return grids


def pack_by_ranks(gate: torch.Tensor, value: torch.Tensor, tp: int) -> torch.Tensor:
    """What ``tp`` column-parallel ranks of a gated linear unit hold, concatenated rank after rank:
    rank ``r`` holds ``[W_r; V_r]``, its chunk of the gate and of the value."""
    return torch.cat([torch.cat([gate.chunk(tp)[r], value.chunk(tp)[r]]) for r in range(tp)])


def packed_units(halves: Mapping[str, torch.Tensor], sizes: Callable[[str], int]):
    """The fused tensors Megatron-FSDP saves without splitting: every pair of SwiGLU halves
    ``<key>_w``/``<key>_v`` of ``halves`` packed rank after rank (:func:`pack_by_ranks`) at
    ``sizes(key)`` ranks, as ``<key>``; every other tensor is kept."""
    packed = {}
    for key, tensor in halves.items():
        if key.endswith("_v"):
            continue
        if key.endswith("_w"):
            fused = key[: -len("_w")]
            packed[fused] = pack_by_ranks(tensor, halves[f"{fused}_v"], sizes(fused))
        else:
            packed[key] = tensor
    return packed
