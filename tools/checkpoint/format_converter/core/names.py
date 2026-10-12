# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Names of the supported checkpoint formats, their dialects, the fsdp_dtensor key envelope,
the torch_dist block layouts, the fsdp_dtensor names of routed experts, the SwiGLU choice and the
names of a multi-token prediction layer's inner layer.

This module imports nothing, so the command line can validate format flags
without importing torch.
"""

from __future__ import annotations

TORCH_DIST = "torch_dist"
FSDP_DTENSOR = "fsdp_dtensor"
FORMATS = (TORCH_DIST, FSDP_DTENSOR)
DIALECTS = {TORCH_DIST: "mcore", FSDP_DTENSOR: "mfsdp_v1"}
"""The dialect of each format this tool reads: Megatron Core and Megatron-FSDP v1."""

FSDP_MODEL_PREFIX = "model.module."
"""Prefix of model keys in Megatron-FSDP v1 stores (one ``module.`` level for every precision)."""

FSDP_OPTIMIZER_STATE = "optimizer.state."
"""Prefix of optimizer-state keys in Megatron-FSDP v1 stores (``module.`` levels follow)."""

STACKED = "stacked"
"""torch_dist block layout: one key per tensor, the layers along a new first axis."""

PER_LAYER = "per_layer"
"""torch_dist block layout: one key per layer (``<block>.layers.<index>.<name>``)."""

TD_LAYOUTS = (STACKED, PER_LAYER)
"""How torch_dist stores a block's layers."""

NATIVE_LAYOUT = "native"
"""``--torch-dist-layout``: stack each TransformerBlock as Megatron would (the default)."""

PER_LAYER_LAYOUT = "per-layer"
"""``--torch-dist-layout``: every TransformerBlock per layer, marked in ``content_metadata``."""

TORCH_DIST_LAYOUTS = (NATIVE_LAYOUT, PER_LAYER_LAYOUT)
"""Values of ``--torch-dist-layout``."""

GROUPED = "grouped"
"""fsdp_dtensor expert naming of ``TEGroupedMLP``: ``<...>.mlp.experts.linear_fc<n>.weight<e>``."""

SEQUENTIAL = "sequential"
"""fsdp_dtensor expert naming of ``SequentialMLP``: ``<...>.mlp.experts.local_experts.<e>.*``."""

EXPERT_FORMS = (GROUPED, SEQUENTIAL)
"""How fsdp_dtensor names the tensors of routed experts (as the model's experts module does)."""

AUTO_EXPERT_NAMING = "auto"
"""``--expert-naming``: the naming the model's builder gives its experts (the default)."""

EXPERT_NAMINGS = (AUTO_EXPERT_NAMING, *EXPERT_FORMS)
"""Values of ``--expert-naming``."""

AUTO_SWIGLU, SWIGLU_ON, SWIGLU_OFF = "auto", "on", "off"
TARGET_SWIGLU = (AUTO_SWIGLU, SWIGLU_ON, SWIGLU_OFF)
"""Values of ``--target-swiglu``: whether the job that loads an fsdp_dtensor target sets
``--swiglu`` (``auto``: as the source's arguments say)."""

GPT_MTP_INNER_LAYER = "transformer_layer"
"""The name both formats store the inner layer of a GPT multi-token prediction layer under
(``MTP_INNER_LAYER_CHECKPOINT_NAME`` of ``fsdp_dtensor_checkpoint.py``, pinned by a test)."""

MTP_INNER_LAYER = "mtp_model_layer"
"""The live name of a multi-token prediction layer's inner layer (``MTP_INNER_LAYER`` of
``fsdp_dtensor_checkpoint.py``, pinned by a test): a hybrid one's is stored under it in both
formats, a GPT one's only by Megatron-FSDP saves older than ``handle_mtp_in_state_dict``."""

MTP_INNER_LAYERS = (GPT_MTP_INNER_LAYER, MTP_INNER_LAYER)
"""Every name a store of either format holds a multi-token prediction layer's inner layer
under."""

FSDP_GPT_MTP_INNER_NAMES = MTP_INNER_LAYERS
"""Values of ``--fsdp-gpt-mtp-inner-name``: what an fsdp_dtensor target names a GPT multi-token
prediction layer's inner layer (``transformer_layer`` by default, as current Megatron does)."""
