<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Checkpoint Format Conversion

`tools/checkpoint/format_converter` converts Megatron checkpoints between the
two PyTorch Distributed Checkpoint (DCP) formats Megatron trains with:
`torch_dist` (Megatron Core distributed checkpointing) and `fsdp_dtensor`
(Megatron-FSDP). It also inspects stores and verifies them bit for bit.

The tool **streams**: it reads and writes one bounded unit of work at a time,
so the host memory it needs depends on its budget and on the largest stored
chunk, never on the checkpoint size. It **never uses a GPU or a process
group**; a conversion is a CPU job, run as a single task. It is **fail-closed**:
every model part is converted by a rule verified for it, and anything else is
refused with a stable code before any tensor is read.

This version converts **model weights only**. Optimizer, scheduler, RNG and
rerun state are dropped and reported; resume from a converted store with
`--no-load-optim --no-load-rng` (see [Resume Recipes](#resume-recipes)).

```bash
python -m tools.checkpoint.format_converter inspect SRC
python -m tools.checkpoint.format_converter plan SRC --to fsdp_dtensor
python -m tools.checkpoint.format_converter convert SRC DST --to fsdp_dtensor
python -m tools.checkpoint.format_converter verify A B
```

Run from the repository root (or with it on `PYTHONPATH`). `inspect`, `plan`
and `verify` run on any host with PyTorch 2.8 or newer, without Megatron,
Transformer Engine or a GPU. `convert` decodes and re-encodes the common state
with Megatron's classes, so it runs where Megatron is importable (the training
container); elsewhere it is refused (`FC-E01`).

## Supported Formats and Dialects

| Format | Dialect | What a store holds |
|---|---|---|
| `torch_dist` | Megatron Core | model tensors under their `ShardedTensor` keys (`decoder.layers.0.mlp.linear_fc1.weight`), a `metadata.json` naming the backend, and the common state as the DCP byte item `common_state/shard_0_1` (a legacy `common.pt` is read and never written) |
| `fsdp_dtensor` | Megatron-FSDP v1 | model tensors under `model.module.<name>` (one `module.` level at every precision), the byte items `args`, `iteration`, `checkpoint_version` and `num_floating_point_operations_so_far`, and no `metadata.json` (Megatron would read it as `torch_dist`) |

`SRC`, `A` and `B` name a store directory, or a checkpoint root whose
`latest_checkpointed_iteration.txt` names it; `--iteration N` picks another
iteration under a root. The format is detected from the store; `--from`
names it for a store without format markers.

Megatron-FSDP v2 stores (`model.<name>` keys) are refused (`FC-R01`), and so
are stores with virtual-pipeline sections (`FC-R02`) and legacy `mp_rank_*`
checkpoints (`FC-R13`).

## Support Matrix

| Model family | `torch_dist` to `fsdp_dtensor` | `fsdp_dtensor` to `torch_dist` | Optimizer state | Tensor parallelism | Tested by |
|---|---|---|---|---|---|
| Hybrid attention and MLP layers (`--hybrid-layer-pattern` with `*` and `-`) | yes | yes | dropped | any source size; the output holds full tensors | synthetic stores; real-save chain at TP 1 and 2 |
| GPT with dense MLP layers (plain or gated, see gated MLPs below; mixture-of-experts, multi-token prediction and multi-latent attention below; `TransformerBlock` decoder, stacked or per layer in `torch_dist`) | yes | yes | dropped | as above | synthetic stores; stacking predicate pinned to Megatron; real-save chain at TP 1 and 2 |
| Vision-language models in the LLaVA layout (a GPT or hybrid language model above, a vision encoder `vision_model.decoder`, a projection) | yes | yes | dropped | as above | synthetic stores; real-save chain of a tiny LLaVA (GPT language model, CLIP encoder) at TP 1 and 2; hybrid language model on synthetic stores only |
| Mixture-of-experts layers of the models above (GPT `TransformerBlock` layers, hybrid symbol `E`): routed experts, plain or gated, grouped (`TEGroupedMLP`) or sequential (`SequentialMLP`), with the router and its buffers, shared experts and latent projections | yes | yes | dropped | as above; any expert-parallel size, except grouped expert biases (`FC-R05`) | synthetic stores; expert naming pinned to Megatron-FSDP's; real-save chains of a GPT mixture-of-experts model at expert-parallel sizes 1 and 2 and of a hybrid `*E*E` model |
| Gated MLPs of the models above (`--swiglu`, `--quick-geglu`, a gated vision encoder or projection): the `linear_fc1` of dense MLPs, shared experts and routed experts, split into SwiGLU halves or kept fused as Megatron-FSDP stores them | yes | yes | dropped | SwiGLU halves: as above; an `fsdp_dtensor` store with fused gated `linear_fc1` is packed per tensor-parallel rank and loads only at that size (`--fsdp-tp`, `--fsdp-etp`; see [Tensor parallelism](#tensor-parallelism)) | synthetic stores against halves and rank-major packs built by hand (tensor-parallel sizes 1, 2 and 4, expert tensor-parallel size 2); the SwiGLU split predicate pinned to Megatron-FSDP's; the rank-major pack pinned to live Megatron-FSDP models at TP 2 and expert TP 2; real-save chains of a GEGLU GPT at TP 2 and of a LLaVA whose vision encoder is gated under a plain language model (TP 1 and 2); real saves of a SwiGLU GPT (TP 1 and 2) and a SwiGLU mixture-of-experts GPT with shared experts at expert-parallel size 2, whose converted stores hold a Megatron-FSDP save's keys and whose Megatron-FSDP saves convert back and load strictly (see [Limitations](#limitations-and-non-claims)) |
| Shortcut-connected mixture-of-experts layers of hybrid models (`--moe-shortcut-connection`: each `E` paired with the layer before it into one module, see [Hybrid layer lists](#hybrid-layer-lists)) | yes | yes | dropped | as above | synthetic stores; the grouping pinned to Megatron's `group_layers_into_shortcut_blocks` on edge patterns; a live shortcut model's `torch_dist` keys pinned to its state-dict names; a real `torch_dist` save converted and compared with what Megatron-FSDP's save handlers make of the live model; that handler output converted back, loaded strictly and compared parameter for parameter; the round trip against the source. Current Megatron refuses `--moe-shortcut-connection` with Megatron-FSDP, so no current Megatron job loads such an `fsdp_dtensor` store (see [Limitations](#limitations-and-non-claims)) |
| Hybrid multi-token prediction (the `/` parts of `--hybrid-layer-pattern`, or the legacy `--mtp-hybrid-override-pattern`; with `--mtp-use-repeated-layer` too), including its mixture-of-experts and shortcut-connected layers | yes | yes | dropped | as above | synthetic stores; the inner layer name pinned to Megatron's; real-save chains of hybrid models with two MTP depths, MTP layers with experts, and one repeated MTP layer |
| GPT multi-token prediction (`--mtp-num-layers` without a hybrid layer pattern; with `--mtp-use-repeated-layer` too), including MTP layers with experts or multi-latent attention; also legacy Megatron-FSDP stores that name the inner layer `mtp_model_layer`, see [GPT multi-token prediction](#gpt-multi-token-prediction) | yes | yes | dropped | as above | synthetic stores (both names, routed experts in the MTP layer, contract D round trips); the inner-layer names pinned to Megatron's (source scan); real-save chains of a GPT with two MTP layers at TP 1 (TP 2 extended); one repeated GPT MTP layer on a synthetic store; a real `torch_dist` save converted with `--fsdp-gpt-mtp-inner-name mtp_model_layer`, whose store holds a real Megatron-FSDP save's keys renamed by hand, converted back and loaded strictly |
| Multi-latent attention (`--multi-latent-attention`, with or without `--mla-down-proj-fusion`): GPT layers and hybrid symbol `+` (in shortcut-connected pairs and multi-token prediction layers too), see [Multi-latent attention](#multi-latent-attention) | yes | yes | dropped | as above | synthetic stores (a stacked GPT with and without fusion, `+-*+`, a `+E-+/+E` shortcut-MoE hybrid); the unfused names Megatron saves in both formats pinned (source scan); a real-save chain of an MLA GPT at TP 1 (TP 2 extended); real saves of a fused MLA GPT with MTP layers, whose converted store holds a bf16 Megatron-FSDP save's keys and whose Megatron-FSDP save converts back and loads strictly (see [Limitations](#limitations-and-non-claims)) |
| Gated DeltaProduct mixers of hybrid models (symbol `M` with `--spec megatron.core.models.hybrid.hybrid_layer_specs gated_delta_product_stack_spec`, or its alias `gdp_stack_spec`): the fused `in_proj` and `conv1d` weights and biases of every mixer, in shortcut-connected pairs and multi-token prediction layers too, see [Gated DeltaProduct mixers](#gated-deltaproduct-mixers) | yes | yes | dropped | an `fsdp_dtensor` store is packed per tensor-parallel rank and loads only at that size (`--fsdp-tp`; see [Tensor parallelism](#tensor-parallelism)); current Megatron-FSDP jobs hold these mixers at TP 1 only (see [Limitations](#limitations-and-non-claims)) | synthetic stores against parts and rank-major packs built by hand (tensor-parallel sizes 1, 2 and 4, with biases) and a vision-language store of every family above; the parts pinned to Megatron's split helpers for 1 to 3 Householder steps; a real TP 2 mixer's fused tensors pinned to the rank-major pack of its own `torch_dist` parts; a real model's `torch_dist` keys and state-dict names at TP 1 and 2; a real-save chain at TP 1; a real TP 2 `torch_dist` save converted with `--fsdp-tp 2`, whose store holds the model's rank-major tensors, converted back and loaded strictly |
| Mamba-2 mixers of hybrid models (symbol `M` with the default stack spec, `hybrid_stack_spec` or its alias `mamba_stack_spec`): the fused `in_proj` weight and the convolution's weight and bias of every mixer, in shortcut-connected pairs and multi-token prediction layers too, see [Mamba-2 mixers](#mamba-2-mixers) | yes | yes | dropped | an `fsdp_dtensor` store is packed per tensor-parallel rank and loads only at that size (`--fsdp-tp`; see [Tensor parallelism](#tensor-parallelism)) | synthetic stores against parts and rank-major packs built by hand (tensor-parallel sizes 1, 2 and 4; heads from `mamba_num_heads` or from `hidden_size`; a store saved before the mixer owned its convolution) and a vision-language store of every family above, each failing under converters that swap two parts, slice the parts' concatenation by rank or misname the convolution; the parts and names pinned to `MambaMixer.sharded_state_dict` (source scan) and to a real mixer's split factories at TP 1 and 2; a real TP 2 mixer's fused tensors pinned to the rank-major pack of its own `torch_dist` parts; a real model's `torch_dist` keys and state-dict names at TP 1 and 2; real-save chains of `M*-` and `M*E` at TP 1 (and `M*-` at TP 2 with `--fsdp-tp 2`, extended) |
| Gated DeltaNet layers (`GatedDeltaNet`, and `GatedDeltaNet2` with `--experimental-attention-variant gdn2`): symbol `G` of hybrid models (with either stack spec; in shortcut-connected pairs and multi-token prediction layers too), and the linear-attention layers of GPT models (`--experimental-attention-variant gdn` or `gdn2` with `--linear-attention-freq`): the fused `in_proj` weight and the convolution's weight and bias of every layer, see [Gated DeltaNet layers](#gated-deltanet-layers) | yes | yes | dropped | an `fsdp_dtensor` store is packed per tensor-parallel rank and loads only at that size (`--fsdp-tp`; see [Tensor parallelism](#tensor-parallelism)) | synthetic stores against parts and rank-major packs built by hand (tensor-parallel sizes 1, 2 and 4; both variants; a hybrid of Gated DeltaProduct mixers and Gated DeltaNets; GPT linear attention; a whole `in_proj.bias` at equal and other pack sizes) and a vision-language store of every family above, failing under converters that swap two parts or slice the parts' concatenation by rank, and refused under one that takes either variant for the other; the parts, per-head sizes and variant choice pinned to `GatedDeltaNet`, `GatedDeltaNet2` and the hybrid stack (source scan) and to real modules' split factories at TP 1 and 2; real TP 2 modules' fused tensors pinned to the rank-major pack of their own `torch_dist` parts, and a real whole `in_proj.bias` to their ranks' local biases; real hybrid and GPT models' `torch_dist` keys and state-dict names at TP 1 and 2; real-save chains of `G*-` (both variants) and a linear-attention GPT at TP 1 (at TP 2 with `--fsdp-tp 2`, `G*E` and a Gated DeltaNet 2 GPT, extended) |

Every model family above is converted; what later versions add is refused
before anything is read, with a pending code:

| Code | Family |
|---|---|
| `FC-P11`, `FC-P12` | optimizer state (`--optimizer convert`) |

A `torch_dist` source may have been saved with any tensor-, pipeline- or
data-parallel layout: its keys are global, and every stored chunk is read into
its place in the full tensor. An `fsdp_dtensor` source must have been saved
without pipeline parallelism (`FC-R02`), as Megatron-FSDP stores are.

## How It Runs

1. **Plan** from the metadata and the common state only: detect the format,
   classify every key, run the refusal fence, derive the architecture (which
   blocks the model has, how each format stores their layers) and plan every
   output chunk as copies from stored source chunks. The plan checks that
   every source element is read exactly once and every output chunk is filled
   exactly. `plan` stops here and prints the derivations, the rules, the
   sections kept and dropped, and the memory and disk estimates.
2. **Pack** the output chunks into units under the unit budget (`--unit-bytes`,
   4 GiB by default), keeping the chunks that read one stored item in one
   unit, so every stored item is read once.
3. **Execute** the units in worker processes (`--workers`; 0 runs them in the
   driver). Workers start with no visible GPU and without launcher variables,
   never create a process group, read with one `dcp.load` per unit and write
   each unit as a DCP part store.
4. **Merge** the part stores (hard links) into one store, check its metadata
   against the plan (keys, shapes, dtypes, chunk grids) and spot-check chunks
   against digests the workers recorded.
5. **Publish**: rename the store into place and write the tracker last.

The peak host memory of a worker is about the largest unit's buffers plus
the largest stored item it decodes plus a fixed baseline; the conversion's
total is the number of workers times that plus the driver. `plan` and
`convert` refuse (exit 8) before reading anything when the estimate exceeds
`--max-host-bytes` (default 0.8 of the available memory), and `convert` when
the destination lacks the disk space.

## Recommended Workflow

1. `inspect SRC`: the format, the sections, the dtypes and the arguments.
2. `plan SRC --to FORMAT`: every refusal (with its code and a hint), the
   derived architecture, and the memory and disk estimates. Nothing is written.
3. `convert SRC DST --to FORMAT`: the conversion, resumable if interrupted.
4. `verify`: compare the converted store with an independent reference,
   under the expected differences of the scenario (`--expected-diff`). A
   round trip compares with its source under contract `D`:

   ```bash
   python -m tools.checkpoint.format_converter convert SRC FSDP --to fsdp_dtensor
   python -m tools.checkpoint.format_converter convert FSDP BACK --to torch_dist
   python -m tools.checkpoint.format_converter verify BACK SRC --expected-diff D
   ```

   Contract `D` compares the common state field by field (whatever `--common`
   says): a converted store's common state is re-encoded, and a re-encoded
   common state is never byte-identical to the one Megatron wrote, so a plain
   `verify BACK SRC` always reports `common_state/shard_0_1`. It also accepts
   the rerun state that a job with Megatron's default
   `--rerun-mode validate_results` saves and a conversion drops.

   `verify SRC DST --map convert` replays the conversion of `SRC` and compares
   it with `DST`; it detects faults of the engine or the storage, not of the
   conversion rules, and the report says it is not independent.
5. Resume training from the converted store with strict loads (see
   [Resume Recipes](#resume-recipes)).

## Conversion Semantics

### Keys

`fsdp_dtensor` model keys are `model.module.<name>`, and the store's planner
data records `('model', 'module.<name>')` for each; `torch_dist` keys are the
bare names. A key that still starts with `module.` after `model.module.` is
refused (`FC-R13`): no Megatron-FSDP v1 save writes it. Top-level modules
other than those of the supported families are refused (`FC-R13`) unless
`--allow-root PREFIX` converts their keys unchanged.

### Layer lists: stacked or per layer

`fsdp_dtensor` stores one key per layer (`decoder.layers.<l>.<name>`). A
hybrid decoder is stored per layer in `torch_dist` too (see
[Hybrid layer lists](#hybrid-layer-lists) for how the formats number it). A `TransformerBlock`
(a GPT decoder, a vision encoder) is stored by `torch_dist` either:

* **stacked**: one key per tensor with the layers along a new first axis
  (`decoder.layers.<name>`, shape `[L, ...]`), each layer its own chunk; or
* **per layer**: `decoder.layers.<l>.<name>`, where `l` is the global layer
  index.

Converting from `torch_dist`, the layout is observed from the keys, and each
layer of a stacked tensor is written as a tensor of its own, read straight
from its chunk. Converting to `torch_dist`, each block gets the layout the job
that loads it will ask for, because Megatron's loader requests the layout its
own model builds:

* The language model's decoder is per layer if Megatron builds it per layer:
  the `content_metadata` the store gets has `non_homogeneous_layers` (or
  `singleton_local_shards`), or the arguments set `moe_layer_freq` or
  `linear_attention_freq` to an integer above 1 or a list. Otherwise it is
  stacked, one chunk per layer, and its layers must hold the same tensors
  with the same shapes and dtypes (`FC-R12` otherwise: the layout is never
  switched silently). Heterogeneous layer specs and `dsv4_hybrid` attention,
  which Megatron also builds per layer, are refused (`FC-R03`, `FC-R20`).
* A vision encoder is decided the same way: Megatron's vision-language
  builders (`pretrain_vlm.py`, `examples/multimodal`) make its configuration
  a copy of the language model's.
* A layer list of any other kind needs `--block-layout` (`FC-A02`).

Two flags choose the layout instead:

* `--torch-dist-layout per-layer` stores every `TransformerBlock` one layer
  per key and sets `non_homogeneous_layers` in the store's `content_metadata`,
  so Megatron loads every block per layer.
* `--block-layout PREFIX=stacked|per_layer` (repeatable) chooses for one
  block, for example `--block-layout vision_model.decoder=per_layer` for a
  job that builds its vision encoder from a configuration of its own. It does
  not change the `content_metadata`, so the job that loads the store must
  build the block that way; a layout of the language model's decoder or the
  vision encoder other than the one Megatron builds from the arguments is
  written with a warning. To store the language model's decoder per layer,
  use `--torch-dist-layout per-layer`.

When the source carries the manifest of the conversion that produced it and
the manifest records another layout of a block than the one decided, the
conversion is refused (`FC-A06`) unless a flag chooses that block's layout or
`--ignore-manifest` is given, so a round trip restores the source's layout or
fails.

### Hybrid layer lists

A hybrid decoder (`--hybrid-layer-pattern`) is stored one key per layer in
both formats: `torch_dist` under each layer's position in the pattern
(`layer_number - 1`, without the `|` stage separators, so the indices are
global at any pipeline-parallel size) and `fsdp_dtensor` under its index in
the layer list.

With `--moe-shortcut-connection`, Megatron pairs each mixture-of-experts layer
(`E`) with the layer before it (a Mamba, Gated DeltaNet or attention layer)
into one module, whose children hold the two layers (`attn_layer` and
`moe_layer`) next to the pair's own norms (`shortcut_pre_mlp_layernorm`,
optionally `shortcut_post_norm`); an `E` at the start of the pattern or right
after a pair stays a module of its own. `torch_dist` stores each module under
its first layer's position and `fsdp_dtensor` under its layer-list index, so
the `torch_dist` indices skip the `E` of every pair. For `E*E-*EE`:

| Module (layer-list index) | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Layers | `E` | `*E` | `-` | `*E` | `E` |
| `torch_dist` index | 0 | 1 | 3 | 4 | 6 |

The converter mirrors Megatron's grouping, requires the source to hold exactly
the layer indices it gives, and checks that every module holds the tensors of
its layers (an attention's `linear_qkv`, a dense MLP's `linear_fc1`, a
router's weight; the names of a pair's children are not assumed). A mismatch,
or a pattern Megatron cannot group, is refused (`FC-R11`) with the whole index
table: an `E` whose preceding layer is neither in a pair already nor a Mamba,
Gated DeltaNet or attention layer (the `E` of `-E`, the last `E` of `*EEE`),
or a pipeline stage after the first that begins with an `E` (`*E|E*`). `plan`
shows the map (`blocks.decoder`: `dense_to_td` and `symbols`).

Multi-token prediction (MTP) layers are stored under `mtp.layers.<d>`, one per
depth (the `/` parts of the pattern; a single layer serves every depth with
`--mtp-use-repeated-layer`), numbered alike in both formats. The inner model
of a hybrid MTP layer is a hybrid layer list of its part of the pattern,
stored under `mtp.layers.<d>.mtp_model_layer.layers.<j>` in both formats
(Megatron renames the inner model to `transformer_layer` for GPT MTP layers
only) and numbered like the decoder from position 0, shortcut pairs included.
`args.mtp_num_layers` must agree with the pattern, and MTP layers that the
arguments do not have are refused (`FC-R11`). MTP layers are built with the
language model's configuration, so their MLPs must be gated exactly as the
decoder's are. A `transformer_layer` key in a hybrid model's MTP layers is
refused (`FC-R09`).

### Mixture-of-experts layers

A mixture-of-experts layer's routed experts (`<...>.mlp.experts`) have one
`linear_fc1` and one `linear_fc2` weight, and bias, per expert. `torch_dist`
stores each as one tensor over all experts,
`<...>.mlp.experts.experts.linear_fc<n>.weight` of shape `[E, ...]` (or
`[L, E, ...]` in a stacked decoder), expert `e` at index `e` of its expert
axis and each (layer, expert) as a chunk of its own; `TEGroupedMLP` and
`SequentialMLP` experts write the same keys. `fsdp_dtensor` stores one
tensor per expert under its global index, named as the experts module names
its parameters:

* grouped (`TEGroupedMLP`): `<...>.mlp.experts.linear_fc1.weight<e>`;
* sequential (`SequentialMLP`): `<...>.mlp.experts.local_experts.<e>.linear_fc1.weight`.

Converting to `fsdp_dtensor`, each expert is read straight from its chunk
and named as the loading job's experts module names it: grouped for a hybrid
model (its stack spec always builds `TEGroupedMLP` experts), otherwise
grouped if the arguments set `--moe-grouped-gemm` and sequential if not.
For a GPT model, `--expert-naming grouped|sequential` chooses instead, for a
job built otherwise, and sets `moe_grouped_gemm` in the store's arguments to
match. A hybrid model's experts cannot be named otherwise (exit 2): no
argument changes the experts module its stack spec builds.
Converting to `torch_dist`, each expert is written as its own chunk of the
expert tensor, which is never assembled whole in memory. An `fsdp_dtensor`
source must name its experts as its arguments build them (`FC-A04`
otherwise; `--expert-naming` with the store's naming accepts it).
`args.num_experts` must be the length of every expert axis, and every
`fsdp_dtensor` expert tensor must be stored for exactly the experts
`0..E-1` (`FC-R12` for indices that are not contiguous from 0, `FC-R13` for
another number of experts).

The router (`router.weight`, the fp32 buffers `expert_bias` and `qb_beta`,
the int32 hash-routing table `tid2eid`), shared experts and latent
projections (`fc1_latent_proj`, `fc2_latent_proj`, `fc2_norm`) are stored
alike in both formats and keep their dtypes.

Megatron-FSDP names the biases of grouped experts by their
expert-parallel-local index (it renumbers only the weights), so a store with
grouped expert biases is right only at expert-parallel size 1: converting a
checkpoint saved with expert parallelism and grouped expert biases is
refused in both directions (`FC-R05`; for a GPT model,
`--expert-naming sequential` writes sequential experts, which name their
biases by global index), and a conversion that writes grouped expert biases
warns that the store loads correctly only into a job with
`--expert-model-parallel-size 1`.

Refused as well: single grouped expert parameters
(`--moe-single-grouped-weight`, `--moe-single-grouped-bias`, which
Megatron-FSDP does not support; `FC-R04`), the legacy `GroupedMLP`
(`weight1`/`weight2`; `FC-R13`) and checkpoints saved with
`singleton_local_shards` (`FC-R17`).

### Gated MLPs (SwiGLU)

A gated MLP (`--swiglu`, or GEGLU with `--quick-geglu`) holds the gate `W` and
the value `V` projections in its `linear_fc1`, which then has twice the rows
its `linear_fc2` has inputs. Every `linear_fc1` is classified from these
shapes: of dense MLPs, shared experts, each routed expert, and MLPs outside
the layer lists such as a vision-language projection. The MLPs of one layer
must agree, and the language model is gated exactly when `--swiglu` or
`--quick-geglu` is set (`FC-R12` otherwise); a vision encoder or a projection
may have a configuration of its own. `plan` shows the counts (`gated_fc1`). A
`linear_fc1` outside every layer with no `linear_fc2` beside it (the plain
linear layer of a vision encoder's patch merger) is no MLP's and converts
unchanged.

`torch_dist` stores a gated `linear_fc1` whole, `[W; V]`, whatever the
tensor-parallel size; Megatron saves each half as chunks of its own, one per
tensor-parallel rank. Megatron-FSDP stores it in one of two ways:

* **Split** (`handle_swiglu_in_state_dict`): when the job sets `--swiglu`, the
  `linear_fc1` weight and bias of dense MLPs, shared experts and routed experts
  in a gated layer are stored as the halves `<name>_w` (`W`) and `<name>_v`
  (`V`), for grouped experts `linear_fc1.weight<e>_w` under the global expert
  index. The halves hold full tensors.
* **Fused**: every other gated `linear_fc1` (no `--swiglu`, a gated vision
  encoder under a plain language model, a vision-language projection's
  `encoder.linear_fc1`) is stored whole as the tensor-parallel ranks' local
  tensors one after the other, `[W_0; V_0; W_1; V_1; ...]`: *packed* at the
  job's tensor-parallel size (its expert tensor-parallel size for routed
  experts), which is `[W; V]` only at size 1.

Converting to `fsdp_dtensor` writes each `linear_fc1` as the loading job's
Megatron-FSDP stores it: split under the source's `--swiglu`, otherwise
packed (see [Tensor parallelism](#tensor-parallelism)); `plan` shows which
(`swiglu_split`). `--target-swiglu on|off` converts for a job that sets
`--swiglu` or not. `--swiglu` also makes the MLPs SiLU-gated, and a job
without it gates them only with `--quick-geglu` (quick-GELU), so for a
language model with MLPs the flag must say what the source's arguments say:
another value describes a job that builds another model and is refused (to
load a GEGLU store at another tensor-parallel size, convert it with that
`--fsdp-tp`). For a language model without MLPs, `on`
splits a gated vision encoder's `linear_fc1` as a job with `--swiglu` does,
`off` keeps it fused, and the store's arguments get the job's `swiglu` (and
`quick_geglu`), which a conversion back restores. Converting to `torch_dist`
merges the halves, or unpacks a fused tensor, into `[W; V]`, one chunk per
half (as Megatron saves it at tensor-parallel size 1). An `fsdp_dtensor` source must be split
exactly as Megatron-FSDP splits (`FC-R12`), and every half needs its pair
(`FC-R20`).

### Gated DeltaProduct mixers

A hybrid model built with the Gated DeltaProduct stack spec
(`gated_delta_product_stack_spec`, alias `gdp_stack_spec`) has a Gated
DeltaProduct mixer in every `M` layer. Its input projection `in_proj`
concatenates the components `z, V0..V(M-1), K0..K(M-1), Q, b0..b(M-1), a`
(sizes `d`, `d` each, `GN` each, `GN`, `H` each, `H`, with
`d = mamba_num_heads x mamba_head_dim`, `GN = mamba_num_groups x
mamba_state_dim`, `H = mamba_num_heads` and `M = gdp_num_householder`
Householder steps), and its convolution `conv1d` the components
`V0..V(M-1), K0..K(M-1), Q`; every tensor-parallel rank holds its shard of
each component, in this order.

`torch_dist` stores each component of the weight (and bias) as a tensor of
its own, `<...>.mixer.in_proj.weight.V0` (Megatron's split factory), whatever
the tensor-parallel size. Megatron-FSDP stores each whole, as the ranks'
local tensors one after the other: packed per tensor-parallel rank like a
fused gated `linear_fc1` (see [Tensor parallelism](#tensor-parallelism)),
the components' concatenation only at size 1. Converting to `fsdp_dtensor`
packs the components at `--fsdp-tp`; converting to `torch_dist` cuts them out
of the store packed at its own tensor-parallel size, one tensor per
component.

The mixer family is never told from the fused widths alone: the `torch_dist`
component names name one family, the spec says which family the job builds
under `mixer` (Gated DeltaProduct with its spec, Mamba-2 with any other), and
of the widths the arguments give the families the names leave, the stored
widths must match exactly the spec's. The family is decided for every mixer
by its module, so a model may hold Gated DeltaProduct mixers and Gated
DeltaNet layers (see [Gated DeltaNet layers](#gated-deltanet-layers)). The mixer
needs `mamba_num_heads`, and as many Householder steps as
`gdp_num_householder` (3 for a store saved before the argument existed, as
Megatron assumes). Any contradiction is refused (`FC-R10`); `plan` shows the
family of every mixer with its dimensions and widths (`mixers`). A mixer
store without the Gated DeltaProduct spec and without its components'
names (an `fsdp_dtensor` store's fused tensors are those of either family) is
a Mamba-2 model (see [Mamba-2 mixers](#mamba-2-mixers)).

The mixer's `A_log` and `dt_bias` (`H` rows) and `norm.weight` (`d` rows)
are copied unchanged, and must have their full sizes (`FC-R10`): a
Megatron-FSDP job above TP 1 saves one tensor-parallel rank's shard of each
(see [Limitations](#limitations-and-non-claims)), and such a store is
refused rather than converted to an incomplete `torch_dist` store. Converting
to `fsdp_dtensor` with `--fsdp-tp` above 1 warns that no current Megatron-FSDP
job loads the store. The inference stack specs
(`gated_delta_product_inference_stack_spec`, alias `gdp_inference_stack_spec`)
are not on the spec allowlist and are refused (`FC-R03`).

### Mamba-2 mixers

A hybrid model built with the default stack spec (`hybrid_stack_spec`, alias
`mamba_stack_spec`, or no `--spec`) has a Mamba-2 mixer (`MambaMixer`) in
every `M` layer. Its input projection `in_proj` concatenates the components
`z, x, B, C, dt` (sizes `d`, `d`, `GN`, `GN`, `H`) and its convolution the
components `x, B, C`, with `H = mamba_num_heads` and
`d = H x mamba_head_dim`, or without `mamba_num_heads`, `d = 2 x hidden_size`
and `H = d / mamba_head_dim` (as `MambaMixer` derives them), and
`GN = mamba_num_groups x mamba_state_dim`; every tensor-parallel rank holds
its shard of each component, in this order. The mixer has no `in_proj` bias,
and always a convolution bias.

`torch_dist` stores each component of the `in_proj` weight and of the
convolution's weight and bias as a tensor of its own,
`<...>.mixer.in_proj.weight.z` and `<...>.mixer.conv1d.weight.x`. The mixer
holds its convolution as its own parameters, which Megatron-FSDP stores
under those names, `<...>.mixer.conv1d_weight` and `<...>.mixer.conv1d_bias`,
each whole and packed per tensor-parallel rank like `in_proj.weight` (see
[Tensor parallelism](#tensor-parallelism)). Converting to `fsdp_dtensor`
packs the components at `--fsdp-tp` and writes these names; converting to
`torch_dist` cuts the components out of the store packed at its own
tensor-parallel size. An `fsdp_dtensor` store saved before `MambaMixer` owned
its convolution names it `<...>.mixer.conv1d.weight` and `.bias`, as a Gated
DeltaProduct mixer's are named: it is read as Mamba-2 unless the spec is the
Gated DeltaProduct one. A convolution stored under both names is refused
(`FC-R10`).

The family is told as for [Gated DeltaProduct mixers](#gated-deltaproduct-mixers)
(names, spec, the unique width match; `FC-R10` for any contradiction). The
mixer's `A_log`, `dt_bias` and `D` (`H` rows) and `norm.weight` (`d` rows)
are copied unchanged and must have their full sizes (`FC-R10`); unlike a
Gated DeltaProduct mixer's, Megatron-FSDP shards them over the
tensor-parallel ranks, so Megatron-FSDP jobs above TP 1 hold Mamba-2 mixers
whole.

If an `in_proj.bias` is present (a mixer built otherwise than `MambaMixer`,
which asserts it has none), Megatron's split factory would not split it:
`torch_dist` then holds the ranks' local biases one after the other, as
`fsdp_dtensor` does. It is copied as is when `--fsdp-tp` is the source's
tensor-parallel size, and refused otherwise (`FC-R10`).

### Gated DeltaNet layers

A Gated DeltaNet layer holds a `GatedDeltaNet`, or a `GatedDeltaNet2` when
`--experimental-attention-variant` is `gdn2`, under `self_attention`: the `G`
layers of a hybrid model (with the default or the Gated DeltaProduct stack
spec), and the linear-attention layers of a GPT model built with
`--experimental-attention-variant gdn` (or its deprecated spelling
`gated_delta_net`) or `gdn2` and `--linear-attention-freq` (which makes
Megatron save the decoder one layer per key). With
`qk = linear_key_head_dim x linear_num_key_heads` and
`v = linear_value_head_dim x linear_num_value_heads`, a Gated DeltaNet's
input projection `in_proj` concatenates the components
`query, key, value, z, beta, alpha` (sizes `qk`, `qk`, `v`, `v` and a row per
value head each), a Gated DeltaNet 2's `query, key, value, z, f, b, w` (`qk`,
`qk`, `v`, `v`, `qk`, `qk`, `v`), and both convolutions `query, key, value`;
every tensor-parallel rank holds its shard of each component, in this order.

`torch_dist` stores each component of the `in_proj` weight and of the
convolution's weight and bias as a tensor of its own,
`<...>.self_attention.in_proj.weight.beta` and
`<...>.self_attention.conv1d.weight.query`; Megatron-FSDP stores each whole
under the same module names, packed per tensor-parallel rank (see
[Tensor parallelism](#tensor-parallelism)). Converting to `fsdp_dtensor`
packs the components at `--fsdp-tp`; converting to `torch_dist` cuts them out
of the store packed at its own tensor-parallel size.

The variant is told as for the mixers: `torch_dist` component names name one
(`beta, alpha` or `f, b, w`), `--experimental-attention-variant` says which
one the job builds (`gdn2` a Gated DeltaNet 2, any other value a Gated
DeltaNet, as Megatron decides), and an `fsdp_dtensor` store's widths must
match exactly that variant's of the two (`FC-R10` for any contradiction). A
GPT decoder holds a Gated DeltaNet only where Megatron builds one: with
`--experimental-attention-variant gdn`, `gated_delta_net` or `gdn2` and
`--linear-attention-freq`, at a layer it makes linear attention (an int `N`
makes every `N`-th layer standard attention); any other store is refused
(`FC-R10`).
`A_log` and `dt_bias` (a row per value head each; a Gated DeltaNet 2's `A_log`
a row per key head and its `dt_bias` one per key channel) are copied unchanged
and must have their full sizes (`FC-R10`); Megatron-FSDP shards them over the
tensor-parallel ranks, so Megatron-FSDP jobs above TP 1 hold these layers
whole. `out_norm`, `out_proj` and the layer norm fused into `in_proj` are
copied unchanged.

The layer specs build no `in_proj` bias, but a Gated DeltaNet built with one
has a real one, which Megatron's split factory does not split: `torch_dist`
then holds the ranks' local biases one after the other, as `fsdp_dtensor`
does. It is copied as is when `--fsdp-tp` is the source's tensor-parallel
size, and refused otherwise (`FC-R10`).

### GPT multi-token prediction

A GPT model's MTP layers (`--mtp-num-layers` without a hybrid layer pattern)
are stored under `mtp.layers.<d>`, one per depth (one with
`--mtp-use-repeated-layer`), never stacked, each with its `enorm`, `hnorm`,
`eh_proj` and `final_layernorm` and one inner `TransformerLayer`. The live
module of that layer is `mtp_model_layer`, but Megatron stores it as
`transformer_layer` in both formats (`MultiTokenPredictionLayer.sharded_state_dict`
and Megatron-FSDP's `handle_mtp_in_state_dict` rename it on save and load), so
it converts unchanged, its experts and gated MLPs as the decoder's are.

Megatron-FSDP saved the live name before `handle_mtp_in_state_dict` existed.
Such a store is read under it only when it is detected: the arguments have no
hybrid layer pattern, its MTP layers' inner keys are all under
`mtp_model_layer` (none under `transformer_layer`) and hold a single layer.
`torch_dist` then gets `transformer_layer`; `plan` and `convert` warn, and the
`gpt_mtp_inner_layer` derivation (also in the manifest) records each format's
name. `--fsdp-gpt-mtp-inner-name mtp_model_layer` writes that legacy name, for
Megatron-FSDP versions older than the handler only (current ones load only
`transformer_layer`); without GPT MTP layers it has no effect. A store with
both names, `mtp_model_layer` in a GPT `torch_dist` store, or an inner layer
list under GPT arguments is refused (`FC-R09`).

### Multi-latent attention

Multi-latent attention (`--multi-latent-attention`; hybrid symbol `+`) is
stored under the same names in both formats: `input_layernorm`, and under
`self_attention` the `linear_q_down_proj` (or `linear_q_proj` without a q LoRA
rank), `linear_q_up_proj`, `linear_kv_down_proj`, `linear_kv_up_proj` and
`linear_proj` weights (the up-projections' `layer_norm_weight` with
`--qk-layernorm`), all full tensors, so it converts unchanged. With
`--mla-down-proj-fusion` the live model holds one `linear_qkv_down_proj` that
also absorbs the input layer norm, but both formats store it unfused
(`FusedMLASelfAttention.sharded_state_dict` and the fused layer specs' key
maps for `torch_dist`, `handle_mla_down_proj_in_state_dict` for
Megatron-FSDP): a store converts the same whether the job that saved it or
the one that loads it fuses. A store holding the fused live name (a
Megatron-FSDP store saved before that handler, or a `torch_dist` store, which
Megatron never writes so) is refused (`FC-R08`). To convert such a legacy
Megatron-FSDP store, load it with the Megatron version that saved it, in a
job with `--mla-down-proj-fusion`, save it as `torch_dist` and convert that.

### Vision-language models

A model whose keys are under `language_model.`, `vision_model.` and
`vision_projection.` is converted as a vision-language model in the LLaVA
layout: `language_model.decoder` is its language model's decoder (a hybrid
decoder or a GPT `TransformerBlock`, whose rules and arguments apply to it),
`vision_model.decoder` its vision encoder, and every other tensor (patch and
position embeddings, the projection, layer norms) is converted unchanged.
Keys under both these roots and the single-model roots (`embedding`,
`decoder`, `output_layer`, `mtp`) are refused (`FC-R13`).

### Values and dtypes

Every tensor is copied bit for bit with its dtype. Megatron's loaders cast on
load: bfloat16 weights load exactly into Megatron-FSDP's fp32 main weights,
and fp32 main weights round to a `torch_dist` job's parameters as a training
step would. Tensors a job keeps in another dtype (a frozen encoder, fp32
buffers such as a router's `expert_bias`, integer buffers such as
`tid2eid`) keep it.

### Tensor parallelism

Converted stores hold full tensors (tensor-parallel shards of the source are
read into place), so they load at any tensor-parallel size the model allows,
with one exception.

```{warning}
An `fsdp_dtensor` store that holds fused gated `linear_fc1` (see
[Gated MLPs](#gated-mlps-swiglu)), Gated DeltaProduct mixers (see
[Gated DeltaProduct mixers](#gated-deltaproduct-mixers)), Mamba-2 mixers
(see [Mamba-2 mixers](#mamba-2-mixers)) or Gated DeltaNet layers (see
[Gated DeltaNet layers](#gated-deltanet-layers)) is
**tensor-parallel-bound**: each such tensor is packed per tensor-parallel
rank (per expert tensor-parallel rank for routed experts) and loads right only
at the size it was packed at. The Megatron-FSDP loader does not check the
tensor-parallel size, so a job at another size loads it without an error and
mixes the rows of different parts.
```

The converter packs at `--fsdp-tp` and `--fsdp-etp`, the sizes of the job
that will load the store. They default to the source's sizes only when those
are 1; a source saved at a larger size needs the flag (`FC-A01`). Routed
experts need `--fsdp-etp` also when `--fsdp-tp` is not the source's
tensor-parallel size (`FC-A01`): a job's expert tensor-parallel size follows
its tensor-parallel size unless it sets `--expert-tensor-parallel-size`. The
store's arguments record the sizes (`tensor_model_parallel_size`,
`expert_tensor_parallel_size`) and `plan` shows them (`pack_tp`, `pack_etp`).
When `--fsdp-tp` changes the tensor-parallel size, the store keeps the
source's padded vocabulary, which must divide by it (`FC-R10`); a job at that
size that pads its vocabulary itself may pad it otherwise (a warning says
when), so load with `--use-checkpoint-args`, which takes `padded_vocab_size`
from the store.
Load such a store with `--use-checkpoint-args --use-mp-args-from-checkpoint-args`,
which take the tensor-parallel size from the store (pass
`--expert-tensor-parallel-size` yourself: Megatron does not take it from a
checkpoint), or with the same `--tensor-model-parallel-size` and
`--expert-tensor-parallel-size`. Converting such a store back to `torch_dist`
unpacks it at the sizes its arguments give; a conversion manifest that records
other sizes is refused (`FC-R10`). A store without fused gated MLPs and
without mixers loads at any size.

### Common state and arguments

The common state is carried over: `args`, `iteration`, `checkpoint_version`
and `num_floating_point_operations_so_far` (an `fsdp_dtensor` source without
one of the first three is refused, `FC-R13`). The arguments are sanitized so
they describe the converted store, changing only existing fields:
`no_save_optim` and `no_save_rng` are set, `ckpt_format` and
`use_megatron_fsdp` name the target format, and an `fsdp_dtensor` store gets
one pipeline stage (the pipeline and virtual-pipeline fields reset, `|`
removed from the hybrid pattern), the tensor-parallel sizes it is packed at
if it packs tensors, and the `swiglu` and `quick_geglu` that `--target-swiglu`
asks for (a language model without MLPs only). Transformer Engine `_extra_state` is dropped: `fsdp_dtensor` does
not store it.

### Conversion manifest

A converted store gets `conversion_manifest.json`: the source, the options,
the derived architecture, the original values of the sanitized arguments, the
source's `content_metadata` (in its key order), the dropped sections and the
digest of the store's `.metadata`. Converting the store back restores the
original arguments and `content_metadata` from it (`--ignore-manifest` skips
it), so a round trip returns the source except the dropped state. Tensors
never depend on the manifest.

## Resume Recipes

Resume with the default strict loads. Never validate a conversion with
`--finetune`, `--dist-ckpt-strictness log_all` or
`--no-strict-fsdp-dtensor-load`: they skip or relax the checks that would
reveal a missing or misplaced tensor.

A `torch_dist` target (the store has no `_extra_state`):

```bash
--ckpt-format torch_dist --ckpt-drop-redundant-extra-state --no-load-optim --no-load-rng
```

An `fsdp_dtensor` target, loaded by a Megatron-FSDP job with fp32 parameters
(without `--bf16` or `--fp16`; on current Megatron a bf16 or fp16
Megatron-FSDP job cannot load a store without optimizer state, see
[Limitations](#limitations-and-non-claims)):

```bash
--use-megatron-fsdp --ckpt-format fsdp_dtensor --no-load-optim --no-load-rng
```

A store packed per tensor-parallel rank also needs the tensor-parallel sizes
it was packed at (see [Tensor parallelism](#tensor-parallelism)).

## Fail-Closed Validation

Every refusal prints one line, `FC-xxx: <message> [keys...] hint: <what to do>`,
and every refusal found is printed at once. The package README lists every
condition of every code.

| Exit | Meaning |
|---|---|
| 0 | ok |
| 1 | internal error |
| 2 | usage error |
| 3 | refused (`FC-R`, `FC-P`, `FC-E` codes) |
| 4 | ambiguous, needs a flag (`FC-A` codes) |
| 5 | verification mismatch |
| 6 | integrity or I/O error |
| 7 | the staging directory holds another conversion (`--restart` discards it) |
| 8 | memory or disk budget exceeded |
| 130 | interrupted (rerun to resume) |

| Code | Refused input |
|---|---|
| `FC-R01` | Megatron-FSDP v2 store or arguments |
| `FC-R02` | virtual-pipeline sections, or an `fsdp_dtensor` source saved with pipeline parallelism |
| `FC-R03` | a spec or implementation whose keys no rule knows (not Transformer Engine, a custom `--spec`, heterogeneous specs, the operation-fuser MLP) |
| `FC-R04` | single grouped expert parameters (`--moe-single-grouped-weight`, `--moe-single-grouped-bias`), which Megatron-FSDP does not support |
| `FC-R05` | grouped expert biases in `fsdp_dtensor` with expert parallelism (Megatron-FSDP keeps their expert-parallel-local indices) |
| `FC-R06` | grouped tensor parallelism (weight rematerialization, or more weight shards than tensor-parallel ranks) |
| `FC-R07` | an FP8 delayed-scaling model converted to `torch_dist` (`--allow-missing-fp8-extra-state` writes it without the scaling state) |
| `FC-R08` | multi-latent attention's fused down-projection stored under its live name `linear_qkv_down_proj` (a Megatron-FSDP store saved before Megatron stored it unfused, or a `torch_dist` store, which Megatron never writes so) |
| `FC-R09` | the inner layers of multi-token prediction layers under names no save of the model gives them (both `transformer_layer` and `mtp_model_layer` for GPT MTP layers, `mtp_model_layer` in a GPT `torch_dist` store, `transformer_layer` in a hybrid model) |
| `FC-R10` | a tensor packed per tensor-parallel rank whose parts do not divide by the tensor-parallel size, or whose pack size the manifest records otherwise; a padded vocabulary that does not divide by `--fsdp-tp`; a state-space mixer whose family the names, the spec and the widths do not tell alike, whose dimensions the arguments do not give, whose `torch_dist` parts are not its family's, whose `A_log`, `dt_bias`, `D` or `norm.weight` is one rank's shard, or whose convolution is stored under both its names; a Gated DeltaNet layer whose variant the names, `--experimental-attention-variant` and the widths do not tell alike, or whose `A_log` or `dt_bias` is one rank's shard; a Gated DeltaNet in a GPT decoder where `--experimental-attention-variant` and `--linear-attention-freq` build none; a Mamba-2 or Gated DeltaNet `in_proj.bias` stored whole in `torch_dist` converted with `--fsdp-tp` other than the source's tensor-parallel size |
| `FC-R11` | layer indices or layer tensors that contradict the hybrid layer pattern, or a pattern whose shortcut-MoE layers Megatron cannot group |
| `FC-R12` | inconsistently stacked layers, non-contiguous layer indices (or expert indices of an `fsdp_dtensor` store), layers that differ where they would be stacked, or gated MLPs that contradict the arguments (`--swiglu`, `--quick-geglu`) or one another |
| `FC-R13` | an unknown layout, key, top-level module, byte item or common-state key |
| `FC-R14` | the legacy N-D reformulated layout (load with current Megatron and save again) |
| `FC-R16` | `--optimizer convert` of optimizer state that is not Adam's (moments other than `exp_avg` and `exp_avg_sq`) |
| `FC-R17` | a store saved with `singleton_local_shards` (its `content_metadata` says so, or its keys show it: experts saved one key per expert, gated `linear_fc1` saved as SwiGLU halves) |
| `FC-R18` | a ModelOpt state |
| `FC-R19` | an `fsdp_dtensor` store with split fused-projection keys |
| `FC-R20` | hybrid layer symbols or attention variants no rule is verified for, or a SwiGLU half without its pair |
| `FC-R21` | `--optimizer convert` of an `fsdp_dtensor` store whose model section holds a trained parameter in a narrower type than fp32 (`--allow-bf16-masters` takes it as the fp32 master) |
| `FC-A01` | an `fsdp_dtensor` target packs tensors per tensor-parallel rank, and the source's tensor-parallel size is above 1 (`--fsdp-tp`, `--fsdp-etp`), or it packs routed experts' tensors and `--fsdp-tp` is not the source's size (`--fsdp-etp`) |
| `FC-A02` | a layer list of unknown kind converted to `torch_dist` (`--block-layout`) |
| `FC-A04` | an `fsdp_dtensor` source names its routed experts otherwise than its arguments build them (`--expert-naming`) |
| `FC-A05` | `--from` disagrees with the detected format |
| `FC-A06` | the source's manifest describes another store, or records another block layout |
| `FC-E01` | `convert` without Megatron (run in the training container) |
| `FC-E02` | the staging directory is not on the destination's file system |

## Output Publication and Resume

`DST` is a checkpoint root, which gets `iter_NNNNNNN` and the tracker, or,
when named `iter_NNNNNNN` or `release` (or with `--dst-is-iter-dir`), the
iteration directory itself. The conversion runs in a
staging directory `.fc-staging-<name>` beside the target (or in
`--staging-dir`, on the same file system), holding a lock for its lifetime.
An interrupted or failed conversion resumes when the same command runs again:
completed units are kept, and the output is byte-identical to an uninterrupted
run, whatever the worker count. The staging of another conversion is not
reused (exit 7; `--restart` discards it).

An existing target iteration is refused unless `--overwrite`, which replaces
that iteration directory only (and only if it holds a single store). The
tracker is written last and only moved forward. Nothing but the staging
directory, the target iteration and the tracker is ever changed.

## Limitations and Non-Claims

- **Weights only.** Optimizer state is not converted yet (`FC-P11`,
  `FC-P12`); resume with `--no-load-optim`.
- **Fused multi-latent attention and Megatron-FSDP.** On current Megatron, a
  Megatron-FSDP job with `--mla-down-proj-fusion` and fp32 parameters cannot
  build its checkpoint state dict (`handle_mla_down_proj_in_state_dict` looks
  its parameters up under one `module.` level too many without
  `Float16Module`), as for SwiGLU below: its `fsdp_dtensor` stores are
  checked against a bf16 job's real saves (keys, shapes, dtypes) and its
  saves convert to `torch_dist` and load strictly.
- **Legacy names.** Megatron-FSDP stores that hold GPT MTP layers under
  `mtp_model_layer` are converted (see
  [GPT multi-token prediction](#gpt-multi-token-prediction)); those that hold
  multi-latent attention's fused `linear_qkv_down_proj` are refused
  (`FC-R08`), as no such store is available to check against.
- **Multi-latent attention in hybrid models.** Hybrid `+` layers (also `+E`
  shortcut pairs) are checked on synthetic stores and against the names in
  Megatron's hybrid layer specs, but not yet against a real save.
- **Tensor-parallel-bound `fsdp_dtensor` stores.** A store with fused gated
  `linear_fc1`, Gated DeltaProduct or Mamba-2 mixers or Gated DeltaNet layers
  loads right only at the tensor-parallel sizes it was packed at, and the
  Megatron-FSDP loader does not check them (see
  [Tensor parallelism](#tensor-parallelism)).
- **Gated DeltaProduct and Megatron-FSDP at TP > 1.** On current Megatron, a
  Gated DeltaProduct mixer's `A_log`, `dt_bias` and `norm.weight` are
  tensor-parallel without a partition dimension, which Megatron-FSDP takes
  for replicated parameters: a Megatron-FSDP job at a tensor-parallel size
  above 1 holds one rank's shard of each, so it cannot load a converted store
  (converting to such a store warns), and a store it saves is refused
  (`FC-R10`). Stores packed for TP 1 load, and the TP 2 layout
  is checked against real modules and saves without Megatron-FSDP.
- **SwiGLU and Megatron-FSDP.** On current Megatron, a Megatron-FSDP job
  with `--swiglu` and fp32 parameters cannot build its checkpoint state dict
  (`handle_swiglu_in_state_dict` looks its parameters up under one `module.`
  level too many without `Float16Module`), and a bf16 or fp16 one cannot load
  a store without optimizer state (below). A weights-only `fsdp_dtensor`
  store of a SwiGLU model is written as a bf16 job saves it (its keys are
  checked against real saves), but loads only once either is fixed; stores
  from Megatron-FSDP jobs convert to `torch_dist` as usual.
- **Grouped expert biases.** Megatron-FSDP keeps them under
  expert-parallel-local indices: an `fsdp_dtensor` store with them loads
  correctly only at expert-parallel size 1 (see
  [Mixture-of-experts layers](#mixture-of-experts-layers)).
- **Shortcut-connected mixture-of-experts layers and Megatron-FSDP.** Current
  Megatron refuses `--moe-shortcut-connection` with `--use-megatron-fsdp` (the
  shortcut module bypasses Megatron-FSDP's per-layer parameter all-gather), so
  no current Megatron job loads the `fsdp_dtensor` store of such a model, and
  no real Megatron-FSDP save exists to compare it with: the conversion writes
  the layout a Megatron-FSDP v1 job uses and is checked against what
  Megatron-FSDP's own save handlers make of the live model. `plan` and
  `convert` warn when they write such a store.
- **Megatron-FSDP v1 only.** Megatron-FSDP v2 stores are refused.
- **bf16 and fp16 Megatron-FSDP jobs.** On current Megatron, a Megatron-FSDP
  job with bf16 or fp16 parameters cannot load any `fsdp_dtensor` store
  without optimizer state: after loading, `load_checkpoint` asks the
  distributed optimizer to copy the model parameters into its main
  parameters, which Megatron-FSDP does not implement. Jobs with fp32
  parameters load converted stores; the resume recipe above assumes one.
- **FP8 delayed scaling.** Its scaling history (`_extra_state`) is not in
  `fsdp_dtensor` stores and restarts after a conversion.
- **No model is built.** Conversion follows the stored keys, the training
  arguments and the rules; validate the result with `verify` and a strict
  load in the target job.
- **Local file systems.** Stores are read and written through the local file
  system; object storage is not supported.

## Examples

Convert a `torch_dist` checkpoint for a Megatron-FSDP job, then back:

```bash
python -m tools.checkpoint.format_converter convert \
    /checkpoints/run_a /checkpoints/run_a_fsdp --to fsdp_dtensor
python -m tools.checkpoint.format_converter convert \
    /checkpoints/run_a_fsdp /checkpoints/run_a_back --to torch_dist
```

Convert a mixture-of-experts GPT checkpoint trained with grouped experts
(`--moe-grouped-gemm`) for a Megatron-FSDP job that builds sequential experts
(a hybrid model's experts are always grouped):

```bash
python -m tools.checkpoint.format_converter convert \
    /checkpoints/moe_run /checkpoints/moe_run_fsdp --to fsdp_dtensor --expert-naming sequential
```

Convert a GEGLU checkpoint (`--quick-geglu`, whose MLPs Megatron-FSDP keeps
fused) saved at tensor-parallel size 4 for a Megatron-FSDP job at size 2:

```bash
python -m tools.checkpoint.format_converter convert \
    /checkpoints/geglu_run /checkpoints/geglu_run_fsdp --to fsdp_dtensor --fsdp-tp 2
```

Convert a Megatron-FSDP save to `torch_dist` with every block per layer, and
check what would happen first:

```bash
python -m tools.checkpoint.format_converter plan \
    /checkpoints/fsdp_run --to torch_dist --torch-dist-layout per-layer
python -m tools.checkpoint.format_converter convert \
    /checkpoints/fsdp_run /checkpoints/fsdp_run_td --to torch_dist --torch-dist-layout per-layer
```

A single-task CPU job on a Slurm cluster (the converter never needs a GPU;
`--workers` sets the worker processes):

```bash
srun -N1 -n1 --cpus-per-task=32 python -m tools.checkpoint.format_converter convert \
    /checkpoints/run_a /checkpoints/run_a_fsdp --to fsdp_dtensor --workers 4
```

## Library API

The commands are available from Python:

```python
from pathlib import Path

from tools.checkpoint.format_converter.api import convert_checkpoint, plan_conversion
from tools.checkpoint.format_converter.options import ConvertOptions, PlanOptions

plan = plan_conversion(Path("/checkpoints/run_a"), PlanOptions(to_format="fsdp_dtensor"))
print(plan.to_json()["derivations"])

report = convert_checkpoint(
    Path("/checkpoints/run_a"),
    Path("/checkpoints/run_a_fsdp"),
    ConvertOptions(to_format="fsdp_dtensor", workers=4),
)
print(report.render())
```

Refusals raise `ConverterError` subclasses whose `exit_code` is the CLI's
exit status.

## Extending

Each model family is one rule module with its derivations, its refusal
codes, hand-built ground-truth tests, a parity pin against Megatron or a
real-save chain, and a row in the support matrix. The package README
(`tools/checkpoint/format_converter/README.md`) describes the logical key
space, the rule model and the extension contract.

## Validation

The tool is tested at four tiers: pure logic against hand-built expectations
(key grammar, layouts, refusals, plans); synthetic stores converted and
verified end to end, including interrupted and resumed runs and planted
corruptions; parity pins against Megatron's own code (such as the stacking
predicate against `TransformerBlock.sharded_state_dict`); and real-save chains
in the training container: Megatron's own save, converted, loaded strictly by
a model of the other format and compared parameter for parameter, saved again,
converted back, loaded strictly again, and every store checked against a real
save under its expected differences.
