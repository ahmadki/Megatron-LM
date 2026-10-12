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


def pack_by_ranks(parts: Sequence[torch.Tensor], tp: int) -> torch.Tensor:
    """What ``tp`` column-parallel ranks of a layer whose output concatenates ``parts`` hold,
    concatenated rank after rank: rank ``r`` holds its chunk of every part, in order (``[W_r; V_r]``
    for the gate and value of a gated linear unit)."""
    return torch.cat([torch.cat([part.chunk(tp)[r] for part in parts]) for r in range(tp)])


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
            packed[fused] = pack_by_ranks((tensor, halves[f"{fused}_v"]), sizes(fused))
        else:
            packed[key] = tensor
    return packed


PAIR_CHILDREN = ("attn_layer", "moe_layer")
"""The children of Megatron's ``ShortcutMoEBlock``: the layer before the MoE layer, and the MoE
layer."""


def hybrid_layer_tensors(
    layer: str, symbol: str, *, experts: int = 4, hidden: int = 8, ffn: int = 16, seed: int
) -> dict[str, torch.Tensor]:
    """Random tensors of one hybrid layer under ``layer`` (e.g. ``decoder.layers.2``).

    Attention (``*``: fused QKV with its norm, and the output projection), a
    dense MLP (``-``) or a mixture-of-experts layer (``E``: see
    :func:`moe_layer_tensors`, grouped expert names without biases); bfloat16
    weights, float32 norms and router bias.
    """
    if symbol == "E":
        return moe_layer_tensors(layer, experts, hidden, bias=False, seed=seed)
    shapes = {
        "*": {
            "self_attention.linear_qkv.weight": (3 * hidden, hidden),
            "self_attention.linear_qkv.layer_norm_weight": (hidden,),
            "self_attention.linear_proj.weight": (hidden, hidden),
        },
        "-": {
            "mlp.linear_fc1.weight": (ffn, hidden),
            "mlp.linear_fc1.layer_norm_weight": (hidden,),
            "mlp.linear_fc2.weight": (hidden, ffn),
        },
    }[symbol]
    generator = torch.Generator().manual_seed(seed)
    return {
        f"{layer}.{name}": torch.randn(shape, generator=generator).to(
            torch.float32 if name.endswith("norm_weight") else torch.bfloat16
        )
        for name, shape in shapes.items()
    }


def hybrid_module_tensors(
    module: str,
    symbols: str,
    *,
    seed: int,
    children: tuple[str, str] = PAIR_CHILDREN,
    experts: int = 4,
    hidden: int = 8,
) -> dict[str, torch.Tensor]:
    """Random tensors of one module of a hybrid layer list under ``module``: a layer
    (:func:`hybrid_layer_tensors`), or a shortcut pair (``*E``) whose layers are its ``children``,
    with the pair's own float32 pre-MLP norm (``shortcut_pre_mlp_layernorm``)."""
    if len(symbols) == 1:
        return hybrid_layer_tensors(module, symbols, experts=experts, hidden=hidden, seed=seed)
    tensors = {}
    for index, (child, symbol) in enumerate(zip(children, symbols)):
        tensors.update(
            hybrid_layer_tensors(
                f"{module}.{child}", symbol, experts=experts, hidden=hidden, seed=seed + index
            )
        )
    generator = torch.Generator().manual_seed(seed + len(symbols))
    tensors[f"{module}.shortcut_pre_mlp_layernorm.weight"] = torch.randn(
        hidden, generator=generator
    )
    return tensors


def shortcut_hybrid_tensors(
    modules: Sequence[str],
    *,
    children: tuple[str, str] = PAIR_CHILDREN,
    experts: int = 4,
    hidden: int = 8,
) -> dict[str, torch.Tensor]:
    """A hybrid model whose decoder's layer list holds ``modules`` (symbols of each, in layer-list
    order), one key per module and expert under its layer-list index, as fsdp_dtensor stores it,
    with an embedding, a final norm and an output layer; each module draws its own values."""
    generator = torch.Generator().manual_seed(21)
    tensors = {
        "embedding.word_embeddings.weight": torch.randn(32, hidden, generator=generator).bfloat16(),
        "decoder.final_norm.weight": torch.randn(hidden, generator=generator),
        "output_layer.weight": torch.randn(32, hidden, generator=generator).bfloat16(),
    }
    for dense, symbols in enumerate(modules):
        tensors.update(
            hybrid_module_tensors(
                f"decoder.layers.{dense}",
                symbols,
                seed=100 * dense,
                children=children,
                experts=experts,
                hidden=hidden,
            )
        )
    return tensors


def renumber_layers(
    tensors: Mapping[str, torch.Tensor], block: str, stored: Sequence[int]
) -> dict[str, torch.Tensor]:
    """The tensors with the layer ``d`` of ``block`` stored under the index ``stored[d]``."""
    head = f"{block}.layers."
    renamed = {}
    for key, tensor in tensors.items():
        if key.startswith(head) and key[len(head)].isdigit():
            index, name = key[len(head) :].split(".", 1)
            key = f"{head}{stored[int(index)]}.{name}"
        renamed[key] = tensor
    return renamed


def hybrid_mtp_tensors(
    modules: Sequence[str],
    *,
    layers: int,
    prefix: str = "mtp",
    children: tuple[str, str] = PAIR_CHILDREN,
    experts: int = 4,
    hidden: int = 8,
) -> dict[str, torch.Tensor]:
    """The ``layers`` multi-token prediction layers of a hybrid model under ``prefix``: each with
    its norms and projection, and its inner model's layer list of ``modules`` (symbols of each,
    in layer-list order) under ``mtp_model_layer.layers.<j>``, as fsdp_dtensor stores them."""
    tensors = {}
    for layer in range(layers):
        head = f"{prefix}.layers.{layer}"
        generator = torch.Generator().manual_seed(1000 + layer)
        for name, shape in {
            "enorm.weight": (hidden,),
            "hnorm.weight": (hidden,),
            "eh_proj.weight": (hidden, 2 * hidden),
            "final_layernorm.weight": (hidden,),
        }.items():
            value = torch.randn(shape, generator=generator)
            tensors[f"{head}.{name}"] = value if name.endswith("norm.weight") else value.bfloat16()
        for dense, symbols in enumerate(modules):
            tensors.update(
                hybrid_module_tensors(
                    f"{head}.mtp_model_layer.layers.{dense}",
                    symbols,
                    seed=1000 + 100 * layer + 10 * dense,
                    children=children,
                    experts=experts,
                    hidden=hidden,
                )
            )
    return tensors


def gpt_mtp_tensors(
    depths: int,
    *,
    inner: str = "transformer_layer",
    prefix: str = "mtp",
    experts: int = 0,
    hidden: int = 8,
    ffn: int = 16,
) -> dict[str, torch.Tensor]:
    """The ``depths`` multi-token prediction layers of a GPT model under ``prefix``: each with its
    norms and projection, and its inner ``TransformerLayer`` (see :func:`transformer_layers`)
    under ``inner``; with ``experts``, its MLP is a mixture-of-experts layer (grouped
    fsdp_dtensor expert names, see :func:`moe_layer_tensors`)."""
    tensors = {}
    layers = transformer_layers(prefix, depths, hidden, ffn, seed=55)
    del layers[f"{prefix}.final_layernorm.weight"]
    for key, tensor in layers.items():
        depth, name = key[len(f"{prefix}.layers.") :].split(".", 1)
        if experts and name.startswith("mlp."):
            continue
        tensors[f"{prefix}.layers.{depth}.{inner}.{name}"] = tensor
    for depth in range(depths):
        head = f"{prefix}.layers.{depth}"
        generator = torch.Generator().manual_seed(2000 + depth)
        for name, shape in {
            "enorm.weight": (hidden,),
            "hnorm.weight": (hidden,),
            "eh_proj.weight": (hidden, 2 * hidden),
            "final_layernorm.weight": (hidden,),
        }.items():
            value = torch.randn(shape, generator=generator)
            tensors[f"{head}.{name}"] = value if name.endswith("norm.weight") else value.bfloat16()
        if experts:
            tensors.update(
                moe_layer_tensors(f"{head}.{inner}", experts, hidden, bias=True, seed=3000 + depth)
            )
    return tensors


def gdp_mixer_tensors(
    mixer: str,
    *,
    d_inner: int,
    group_state: int,
    heads: int,
    householders: int,
    hidden: int = 8,
    bias: bool = False,
    seed: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Random tensors of one Gated DeltaProduct mixer under ``mixer`` (e.g. ``decoder.layers.0.mixer``).

    Returns its fused projections by part, ``{"<mixer>.in_proj.weight": {"z": ..., "V0": ...}}``,
    each in the order ``GatedDeltaProductMixer`` concatenates them (written out here from its
    definition: ``in_proj`` is ``z, V0..V(M-1), K0..K(M-1), Q, b0..b(M-1), a`` of sizes ``d``,
    ``d``, ``GN``, ``GN``, ``H``, ``H``; ``conv1d`` is ``V*, K*, Q``), with their biases if ``bias``,
    and its other tensors (``in_proj``'s layer norm, ``A_log`` in float32, ``dt_bias``, the gated
    norm and ``out_proj``). Weights are bfloat16; every part draws its own values.
    """
    generator = torch.Generator().manual_seed(seed)
    steps = range(householders)
    in_proj = {
        "z": d_inner,
        **{f"V{i}": d_inner for i in steps},
        **{f"K{i}": group_state for i in steps},
        "Q": group_state,
        **{f"b{i}": heads for i in steps},
        "a": heads,
    }
    conv = {**{f"V{i}": d_inner for i in steps}, **{f"K{i}": group_state for i in steps}}
    conv["Q"] = group_state
    fused = {}
    for name, parts, rest in (("in_proj", in_proj, (hidden,)), ("conv1d", conv, (1, 4))):
        for leaf, trailing in (("weight", rest), ("bias", ())) if bias else (("weight", rest),):
            fused[f"{mixer}.{name}.{leaf}"] = {
                part: torch.randn(rows, *trailing, generator=generator).bfloat16()
                for part, rows in parts.items()
            }
    others = {
        f"{mixer}.in_proj.layer_norm_weight": torch.randn(hidden, generator=generator).bfloat16(),
        f"{mixer}.A_log": torch.randn(heads, generator=generator),
        f"{mixer}.dt_bias": torch.randn(heads, generator=generator).bfloat16(),
        f"{mixer}.norm.weight": torch.randn(d_inner, generator=generator).bfloat16(),
        f"{mixer}.out_proj.weight": torch.randn(hidden, d_inner, generator=generator).bfloat16(),
    }
    return fused, others


def torch_dist_parts(fused: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Fused projections by part as torch_dist stores them: ``<tensor>.<part>`` each."""
    return {
        f"{name}.{part}": value for name, parts in fused.items() for part, value in parts.items()
    }


def packed_projections(
    fused: Mapping[str, Mapping[str, torch.Tensor]], tp: int
) -> dict[str, torch.Tensor]:
    """Fused projections by part as Megatron-FSDP stores them at tensor-parallel size ``tp``: each
    whole, packed rank after rank (:func:`pack_by_ranks`)."""
    return {name: pack_by_ranks(tuple(parts.values()), tp) for name, parts in fused.items()}


def gdp_hybrid_tensors(
    pattern: str, mixer: Mapping[str, int], *, hidden: int = 8, bias: bool = False
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """A per-layer hybrid model of ``pattern`` whose ``M`` layers are Gated DeltaProduct mixers of
    the dimensions ``mixer`` (:func:`gdp_mixer_tensors`), with an embedding, a final norm and an
    output layer: its mixers' fused projections by part, and every other tensor."""
    return mixer_hybrid_tensors(
        pattern, {"M": mixer_builder(gdp_mixer_tensors, mixer, hidden=hidden, bias=bias)}, hidden
    )


def mamba_mixer_tensors(
    mixer: str,
    *,
    d_inner: int,
    group_state: int,
    heads: int,
    hidden: int = 8,
    bias: bool = False,
    seed: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Random tensors of one Mamba-2 mixer under ``mixer`` (e.g. ``decoder.layers.0.mixer``).

    Returns its fused projections by part, ``{"<mixer>.in_proj.weight": {"z": ..., "x": ...}}``,
    each in the order ``MambaMixer`` concatenates them (written out here from its definition:
    ``in_proj`` is ``z, x, B, C, dt`` of sizes ``d``, ``d``, ``GN``, ``GN``, ``H``; ``conv1d`` is
    ``x, B, C``), named by the ``conv1d`` module as torch_dist names them; the convolution always
    has its bias (``conv_bias=True``) and ``in_proj`` has one if ``bias``. Also its other tensors:
    ``in_proj``'s layer norm, ``A_log`` and ``D`` in float32, ``dt_bias``, the gated norm and
    ``out_proj``. Weights are bfloat16; every part draws its own values.
    """
    generator = torch.Generator().manual_seed(seed)
    in_proj = {"z": d_inner, "x": d_inner, "B": group_state, "C": group_state, "dt": heads}
    conv = {"x": d_inner, "B": group_state, "C": group_state}
    leaves = {"in_proj": (("weight", (hidden,)),) + ((("bias", ()),) if bias else ())}
    leaves["conv1d"] = (("weight", (1, 4)), ("bias", ()))
    fused = {}
    for name, parts in (("in_proj", in_proj), ("conv1d", conv)):
        for leaf, trailing in leaves[name]:
            fused[f"{mixer}.{name}.{leaf}"] = {
                part: torch.randn(rows, *trailing, generator=generator).bfloat16()
                for part, rows in parts.items()
            }
    others = {
        f"{mixer}.in_proj.layer_norm_weight": torch.randn(hidden, generator=generator).bfloat16(),
        f"{mixer}.A_log": torch.randn(heads, generator=generator),
        f"{mixer}.dt_bias": torch.randn(heads, generator=generator).bfloat16(),
        f"{mixer}.D": torch.randn(heads, generator=generator),
        f"{mixer}.norm.weight": torch.randn(d_inner, generator=generator).bfloat16(),
        f"{mixer}.out_proj.weight": torch.randn(hidden, d_inner, generator=generator).bfloat16(),
    }
    return fused, others


def mamba_hybrid_tensors(
    pattern: str, mixer: Mapping[str, int], *, hidden: int = 8, bias: bool = False
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """:func:`gdp_hybrid_tensors` with Mamba-2 mixers (:func:`mamba_mixer_tensors`)."""
    return mixer_hybrid_tensors(
        pattern, {"M": mixer_builder(mamba_mixer_tensors, mixer, hidden=hidden, bias=bias)}, hidden
    )


MAMBA_CONV_PARAMETERS = {
    ".mixer.conv1d.weight": ".mixer.conv1d_weight",
    ".mixer.conv1d.bias": ".mixer.conv1d_bias",
}
"""``MambaMixer``'s convolution as torch_dist names it (its ``sharded_state_dict`` keeps the names
of the old ``conv1d`` module) and as Megatron-FSDP saves it (the mixer's own parameters)."""


def mamba_fsdp_names(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``tensors`` with every Mamba-2 convolution renamed as Megatron-FSDP saves it."""
    renamed = {}
    for key, tensor in tensors.items():
        for module_name, parameter in MAMBA_CONV_PARAMETERS.items():
            if key.endswith(module_name):
                key = key[: -len(module_name)] + parameter
        renamed[key] = tensor
    return renamed


MixerBuilder = Callable[
    [str, int], tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]
]
"""Builds the tensors of one mixer layer under a layer prefix with a seed: its fused projections by
part and its other tensors."""


def mixer_builder(
    mixer_tensors: Callable[..., tuple[dict, dict]],
    dims: Mapping[str, Any],
    *,
    module: str = "mixer",
    hidden: int,
    bias: bool,
) -> MixerBuilder:
    """A :data:`MixerBuilder` of ``mixer_tensors`` mixers of the dimensions ``dims`` under the
    layer's ``module`` (``mixer``, or a Gated DeltaNet's ``self_attention``)."""

    def build(layer: str, seed: int) -> tuple[dict, dict]:
        return mixer_tensors(f"{layer}.{module}", hidden=hidden, bias=bias, seed=seed, **dims)

    return build


def mixer_hybrid_tensors(
    pattern: str, builders: Mapping[str, MixerBuilder], hidden: int
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """A per-layer hybrid model of ``pattern`` whose layers of the symbols of ``builders`` are their
    mixers, with an embedding, a final norm and an output layer: its mixers' fused projections by
    part, and every other tensor."""
    generator = torch.Generator().manual_seed(31)
    fused: dict[str, dict[str, torch.Tensor]] = {}
    others = {
        "embedding.word_embeddings.weight": torch.randn(32, hidden, generator=generator).bfloat16(),
        "decoder.final_norm.weight": torch.randn(hidden, generator=generator),
        "output_layer.weight": torch.randn(32, hidden, generator=generator).bfloat16(),
    }
    for index, symbol in enumerate(pattern):
        layer = f"decoder.layers.{index}"
        if symbol in builders:
            projections, rest = builders[symbol](layer, index)
            fused.update(projections)
            others.update(rest)
        else:
            others.update(hybrid_layer_tensors(layer, symbol, hidden=hidden, seed=index))
    return fused, others


GDN_IN_PROJ = {
    "gdn": (
        ("query", "qk"),
        ("key", "qk"),
        ("value", "v"),
        ("z", "v"),
        ("beta", "nvh"),
        ("alpha", "nvh"),
    ),
    "gdn2": (
        ("query", "qk"),
        ("key", "qk"),
        ("value", "v"),
        ("z", "v"),
        ("f", "qk"),
        ("b", "qk"),
        ("w", "v"),
    ),
}
"""The in_proj components of each Gated DeltaNet, in order, and the size of each, written out from
``in_proj_split_names``/``in_proj_split_sections`` of ``GatedDeltaNet`` and ``GatedDeltaNet2``:
``qk`` = key head dim x key heads, ``v`` = value head dim x value heads, ``nvh`` = value heads."""
GDN_VECTORS = {
    "gdn": (("A_log", "nvh"), ("dt_bias", "nvh")),
    "gdn2": (("A_log", "nkh"), ("dt_bias", "qk")),
}
"""The rows of each Gated DeltaNet's ``A_log`` and ``dt_bias`` (``a_log_dim``, ``dt_bias_dim``):
a value head each (GDN), a key head and a key channel (GDN2)."""


def gdn_tensors(
    module: str,
    *,
    variant: str = "gdn",
    key_heads: int,
    key_dim: int,
    value_heads: int,
    value_dim: int,
    hidden: int = 8,
    bias: bool = False,
    seed: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Random tensors of one Gated DeltaNet (``variant`` ``gdn``) or Gated DeltaNet 2 (``gdn2``)
    under ``module`` (e.g. ``decoder.layers.0.self_attention``).

    Returns its fused projections by part (:data:`GDN_IN_PROJ`; ``conv1d`` is ``query, key,
    value``), named as torch_dist names them; the convolution has its bias (``conv_bias=True``) and
    ``in_proj`` has one if ``bias``. Also its other tensors: ``in_proj``'s layer norm, ``A_log`` in
    float32 and ``dt_bias`` (:data:`GDN_VECTORS`), the output norm (``value_dim`` rows) and
    ``out_proj``. Weights are bfloat16; every part draws its own values.
    """
    sizes = {
        "qk": key_heads * key_dim,
        "v": value_heads * value_dim,
        "nvh": value_heads,
        "nkh": key_heads,
    }
    generator = torch.Generator().manual_seed(seed)
    in_proj = {part: sizes[size] for part, size in GDN_IN_PROJ[variant]}
    conv = {"query": sizes["qk"], "key": sizes["qk"], "value": sizes["v"]}
    leaves = {"in_proj": (("weight", (hidden,)),) + ((("bias", ()),) if bias else ())}
    leaves["conv1d"] = (("weight", (1, 4)), ("bias", ()))
    fused = {}
    for name, parts in (("in_proj", in_proj), ("conv1d", conv)):
        for leaf, trailing in leaves[name]:
            fused[f"{module}.{name}.{leaf}"] = {
                part: torch.randn(rows, *trailing, generator=generator).bfloat16()
                for part, rows in parts.items()
            }
    vectors = dict(GDN_VECTORS[variant])
    others = {
        f"{module}.in_proj.layer_norm_weight": torch.randn(hidden, generator=generator).bfloat16(),
        f"{module}.A_log": torch.randn(sizes[vectors["A_log"]], generator=generator),
        f"{module}.dt_bias": torch.randn(sizes[vectors["dt_bias"]], generator=generator).bfloat16(),
        f"{module}.out_norm.weight": torch.randn(value_dim, generator=generator).bfloat16(),
        f"{module}.out_proj.weight": torch.randn(
            hidden, sizes["v"], generator=generator
        ).bfloat16(),
    }
    return fused, others


def gdn_hybrid_tensors(
    pattern: str, gdn: Mapping[str, Any], *, hidden: int = 8, bias: bool = False
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """:func:`gdp_hybrid_tensors` with Gated DeltaNet ``G`` layers (:func:`gdn_tensors` of ``gdn``)
    under ``self_attention``."""
    build = mixer_builder(gdn_tensors, gdn, module="self_attention", hidden=hidden, bias=bias)
    return mixer_hybrid_tensors(pattern, {"G": build}, hidden)
