<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# format_converter

`tools/checkpoint/format_converter` is a streaming tool for Megatron checkpoint
stores written with PyTorch Distributed Checkpoint (DCP): `torch_dist` stores
and Megatron-FSDP v1 `fsdp_dtensor` stores. It reads stores chunk by chunk
with bounded host memory, never uses a GPU or a process group, and runs on a
host without Megatron, Transformer Engine or nvidia-resiliency-ext installed.

This version inspects and verifies stores, rewrites `torch_dist` stores with
model weights only, and converts model weights between `torch_dist` and
`fsdp_dtensor` in both directions: hybrid models (attention `*`, MLP `-`,
mixture-of-experts `E`, Mamba-2 or Gated DeltaProduct mixer `M`, Gated
DeltaNet `G` and multi-latent attention `+` layers, shortcut-connected ones
included, and hybrid multi-token prediction layers), GPT models
(`TransformerBlock` decoders, stacked or per layer in `torch_dist`, with dense
or mixture-of-experts MLPs, plain or gated, standard, multi-latent or Gated
DeltaNet linear attention, and multi-token prediction layers), and
vision-language models in the LLaVA layout whose language model is one of
those. Optimizer state is dropped by default; converting it (`--optimizer
convert`) is refused, before anything is read, with a pending code until the
change that supports it lands (see [Roadmap](#roadmap)). This file is the
living design document of the package; the user guide is
`docs/user-guide/features/checkpoint-conversion.md`.

## Commands

Run from the repository root (or with it on `PYTHONPATH`):

```bash
python -m tools.checkpoint.format_converter inspect SRC
python -m tools.checkpoint.format_converter verify A B
python -m tools.checkpoint.format_converter verify A --oracle 'digests_rank*.json' --oracle-key-map map.py
python -m tools.checkpoint.format_converter plan SRC --to fsdp_dtensor
python -m tools.checkpoint.format_converter convert SRC DST --to fsdp_dtensor
python -m tools.checkpoint.format_converter convert FSDP_SRC DST --to torch_dist
python -m tools.checkpoint.format_converter convert SRC DST --to torch_dist
```

`SRC`, `A` and `B` are either a store directory (the one holding `.metadata`
and, for `torch_dist`, `metadata.json`) or a checkpoint root whose
`latest_checkpointed_iteration.txt` names the store (`iter_%07d` or `release`).
`--iteration N` opens `iter_%07d` under a root instead of the tracked iteration.

Global flags: `--log-level {DEBUG,INFO,WARNING,ERROR}` (`INFO` logs the resolved
store, the detected format and its evidence, the memory budget and read
statistics) and `--version`.

### `inspect`

Summarizes a store from its metadata: format, dialect and the evidence for
them, key counts and tensor bytes per section (model, optimizer.state,
optimizer other, scheduler, common, rng, rerun, extra_state, other), dtype and chunks-per-tensor
histograms, the largest stored chunk, DCP metadata facts, and the common state
with selected training arguments. The common state is read from the byte item
`common_state/shard_0_1` or a legacy `common.pt` file (which wins) for
`torch_dist`, and from the `args`, `iteration`, `checkpoint_version` (and
`num_floating_point_operations_so_far`) byte items for `fsdp_dtensor`. It is
decoded with every class outside a small allowlist replaced by an inert stub,
so Megatron enums appear as `<stub megatron....AttnBackend(5)>` and nothing
from the checkpoint is imported or executed.

| Flag | Meaning |
|---|---|
| `--from {torch_dist,fsdp_dtensor}` | format to assume when the store has no format markers |
| `--keys RE` | list the keys matching `RE` (`re.search`) |
| `--chunks` | list keys with their stored chunk grids |
| `--json` | print the report as JSON |
| `--digests OUT.json` | write SHA-256 digests of the selected (default: all) tensors |
| `--iteration N` | iteration to open under a checkpoint root |
| `--max-item-bytes SIZE` | per-read-unit memory budget, e.g. `8GiB` (default: a quarter of `MemAvailable`) |
| `--workers N` | reader threads (default 4) |

### `verify`

Compares store `A` with store `B` key by key, or checks `A` against digest
files with `--oracle`. Any difference exits 5; `--report PATH.jsonl` writes one
JSON line per finding followed by a summary line.

With `B`, the checks are: key sets (after `--ignore-key`), shape and dtype per
key, optionally identical chunk grids (`--layout`), and values bit for bit.
With `--common semantic`, common-state differences are reported per field as
`args.<field>` or `<key>`; `--ignore-key` patterns apply to these names too, for
example `--ignore-key '^args\.no_save_(optim|rng)$'` to accept the arguments a
weights-only conversion sets.
Values are compared through `uint8` views, so `-0.0` versus `0.0` and different
NaN payloads count as differences. The two stores may use different chunk
grids unless `--layout` is given.

| Flag | Meaning |
|---|---|
| `--mode MODE` | `meta` (keys, shapes, dtypes), `full` (all values, default) or `sample:N` (values of `N` keys drawn with `--seed`) |
| `--seed SEED` | seed of `sample:N` (default 0) |
| `--layout` | also require identical chunk grids (not with `--oracle`) |
| `--common {skip,bytes,semantic}` | `bytes` (default): byte items must be byte-identical; `semantic`: decode both common states (stubbing unknown classes) and compare them, `args` field by field, wherever each side stores them (in-DCP item, `common.pt`, or `fsdp_dtensor` top-level items); other byte items byte for byte; `skip`: ignore byte items |
| `--ignore-key RE` | ignore keys matching `RE`, repeatable |
| `--expected-diff {A,B,C,D,E}` | compare `A` (a converted store) with `B` under an [expected-difference contract](#expected-difference-contracts); it sets the common-state comparison, overriding `--common` |
| `--map {identity,convert}` | `identity` (default) compares key by key; `convert` compares `B` with the replayed conversion of `A` (see below) |
| `--oracle GLOB` | verify `A` against these digest files instead of `B` |
| `--oracle-key-map PY` | Python file defining `to_store_key(oracle_key, file_index)` |
| `--report PATH.jsonl` | write every finding as JSON lines |
| `--iteration N`, `--max-item-bytes SIZE`, `--workers N` | as for `inspect` |

### Expected-difference contracts

A converted store differs from its references in known ways: a conversion
drops optimizer and RNG state, may write another chunk grid, keeps bfloat16
where Megatron-FSDP holds fp32 main weights, and sanitizes some arguments.
`verify A B --expected-diff X` reports the differences contract `X` allows as
expected (they are counted, and listed in `--report` with `"expected": true`)
and every other difference as a finding (exit 5):

| Id | `A` vs `B` | Allowed differences |
|---|---|---|
| `A` | a `torch_dist` -> `fsdp_dtensor` conversion vs a real `fsdp_dtensor` save | optimizer, scheduler, RNG, rerun and extra state only in `B`; `A`'s narrower floats equal `B`'s after the exact upcast; any chunk grid; the common state is not compared |
| `B` | an `fsdp_dtensor` -> `torch_dist` conversion vs a real `torch_dist` save | that state only in `B`; `A`'s wider floats round (RNE) to `B`'s; any chunk grid; the common state is not compared |
| `C` | two model-only saves of one step of a conversion chain in one format (real vs converted, or real vs real) | floats equal after the exact upcast; any chunk grid; the sanitized argument fields |
| `D` | `torch_dist` -> `fsdp_dtensor` -> `torch_dist` vs the model-only source | only `args.no_save_optim` and `args.no_save_rng`, and rerun state only in `B`; `A`'s chunk grid must merge whole chunks of `B`'s (tensor-parallel splits merged), and never layers of a tensor stacked over layers or experts of a routed-expert tensor; dtypes exact |
| `E` | `fsdp_dtensor` -> `torch_dist` -> `fsdp_dtensor` vs the source | that state only in `B`; any chunk grid; the sanitized argument fields |

Keys, shapes, values and every other argument or common-state field must
agree. The contract sets the common-state comparison, whatever `--common`
says: `C`, `D` and `E` compare the common state field by field
(`--common semantic`), and `A` and `B` skip it. A converted store's common
state is re-encoded, and a re-encoded common state is never byte-identical to
the one Megatron wrote, even when every field is equal, so a plain `verify`
(`--common bytes`) of a round trip always reports `common_state/shard_0_1`.
State a contract lets `A` lack may also be a common-state key only `B` holds.
Rerun state (the `rerun_state_machine_state/*` items and the
`rerun_state_machine` common-state key) is in every model-only save of a job
that runs with Megatron's default `--rerun-mode validate_results`, and a
conversion drops it. The sanitized argument fields are `no_save_optim`,
`no_save_rng`, `ckpt_format`, `use_megatron_fsdp`, the pipeline and
virtual-pipeline fields and the hybrid layer pattern.

### Plan replay (`--map convert`)

`verify SRC OUT --map convert` rebuilds the plan that converted `SRC` into
`OUT` (with the options `OUT`'s conversion manifest records), fills every
planned chunk from `SRC` and compares it bit for bit with `OUT`, and compares
the key sets, shapes and dtypes with the plan's. The reference is the
converter's own plan, so this is **not an independent check** (the report says
so): it detects faults of the engine, the workers, the merge or the storage,
never a wrong rule. Byte items are compared by key only.

### Digest files and key maps

A digest file maps keys to records:

```json
{"decoder.final_layernorm.weight": {"shape": [4096], "dtype": "torch.bfloat16", "sha256": "..."}}
```

The digest is SHA-256 over the tensor's elements in row-major order, each in
its native in-memory encoding. It does not depend on the chunk grid or on the
`torch.save` framing, so the same tensor has the same digest in every store and
format. `inspect --digests` writes this schema; any other tool can too.

`--oracle GLOB` takes the matching files in sorted order; a file's position is
its `file_index` (zero-pad numbers in file names so that sorted order is the
intended order). A key map is a plain Python file:

```python
import re

LOCAL_EXPERT = re.compile(r"^(?P<stem>.*\.experts\.linear_fc[12]\.weight)(?P<local>\d+)$")


def to_store_key(oracle_key: str, file_index: int) -> str | None:
    """Map an oracle key written by rank `file_index` to a store key (None skips it)."""
    match = LOCAL_EXPERT.match(oracle_key)
    if match:
        oracle_key = match["stem"] + str(int(match["local"]) + 8 * file_index)
    return "model.module." + oracle_key
```

After mapping, a store key contributed by several entries (for example a
tensor replicated on every rank) must get the same record from each; otherwise
it is reported as `oracle_inconsistent`. The store is then checked for keys
missing from either side, shapes and dtypes (every mode), and SHA-256 values
(`full` and `sample:N`).

### `plan` and `convert`

`convert` writes model weights only; optimizer, scheduler, RNG and rerun state
are dropped and reported (converting optimizer state lands later:
`--optimizer convert` is refused with `FC-P11`/`FC-P12`).

#### Between `torch_dist` and `fsdp_dtensor`

`convert SRC DST --to fsdp_dtensor` converts a `torch_dist` store into a
Megatron-FSDP v1 store, and `convert SRC DST --to torch_dist` an
`fsdp_dtensor` store into a `torch_dist` one. Before anything is read, the
training arguments and the model keys (names and shapes) go through the
refusal fence: a model part no rule is verified for is refused with an `FC-R`
(never supported) or `FC-P` (pending family) code, and every refusal found is
printed. This version converts hybrid models (`--hybrid-layer-pattern`) built
from attention (`*`), multi-latent attention (`+`), MLP (`-`),
mixture-of-experts (`E`), mixer (`M`) layers (Mamba-2 mixers with the default
stack spec, Gated DeltaProduct ones with the Gated DeltaProduct stack spec)
and Gated DeltaNet (`G`) layers (shortcut-connected ones included,
`--moe-shortcut-connection`) with or without multi-token prediction layers
(the pattern's `/` parts), and GPT models (no hybrid pattern;
`TransformerBlock` layers of attention, multi-latent attention or Gated
DeltaNet linear attention, and a dense MLP or mixture-of-experts layer, with
or without multi-token prediction layers), with gated (SwiGLU, GEGLU) or
plain MLPs, in both directions, with a single model root (`embedding`,
`decoder`, `output_layer`, `mtp`)
or as the language model of a vision-language model (LLaVA layout:
`language_model.`, `vision_model.` with a `TransformerBlock` encoder
`vision_model.decoder`, `vision_projection.`); stage separators (`|`) in the
pattern are fine, and a `torch_dist` source may have been saved with pipeline
parallelism (its keys are global). Keys under other top-level modules are
refused (`FC-R13`) unless `--allow-root PREFIX` converts them unchanged.

* **Keys.** `fsdp_dtensor` model keys are `model.module.<name>` (exactly one
  `module.` level; the store's planner data records `('model', 'module.<name>')`);
  `torch_dist` keys are the bare names.
* **Hybrid layer numbering.** A hybrid decoder is stored per layer in both
  formats: `torch_dist` under each module's first layer's position in the
  pattern (`layer_number - 1`, `|` removed), `fsdp_dtensor` under its
  layer-list index. Without `--moe-shortcut-connection` both are the layer's
  position. With it, Megatron pairs each `E` with the layer before it into one
  `ShortcutMoEBlock` (`group_layers_into_shortcut_blocks`; a leading `E` and an
  `E` after a pair stay modules of their own), so the `torch_dist` indices
  skip the `E` of every pair while the `fsdp_dtensor` ones stay dense
  (`E*E-*EE`: modules `E`, `*E`, `-`, `*E`, `E` at `torch_dist` 0, 1, 3, 4, 6
  and `fsdp_dtensor` 0-4). The grouping is mirrored exactly, including the
  refusal of a `|` stage that begins with an `E` (`core/hybrid_pattern.py`);
  the source must hold exactly the indices it gives (`FC-R11`, with the full
  index table otherwise), and every module must hold the tensors of its
  layers, by family signature: presence-based, the names of a pair's children
  not assumed (`FC-R11`). Multi-token prediction
  layers (`mtp.layers.<d>`, one per depth, or one with
  `--mtp-use-repeated-layer`) keep their numbering, and the inner model of a
  hybrid MTP layer, `mtp.layers.<d>.mtp_model_layer` (the same name in both
  formats: Megatron renames it to `transformer_layer` for GPT MTP layers
  only), is a hybrid layer list of its part of the pattern numbered like the
  decoder from position 0 (`transformer_layer` keys there are refused,
  `FC-R09`). Megatron refuses `--moe-shortcut-connection` with
  Megatron-FSDP, so no current Megatron job loads the `fsdp_dtensor` store of
  a shortcut-MoE model (`plan` and `convert` warn): it is written in the
  layout a Megatron-FSDP v1 job uses, and tested against Megatron-FSDP's own
  save handlers applied to the live model.
* **Stacked layers.** `torch_dist` stores a `TransformerBlock` (a GPT decoder)
  either stacked, one key per tensor with the layers along a new first axis
  (`decoder.layers.<name>`, shape `[L, ...]`, one chunk per layer), or per
  layer (`decoder.layers.<l>.<name>`, `l` the global layer index);
  `fsdp_dtensor` always stores one key per layer. Converting from `torch_dist`,
  the layout is observed from the keys: layer `l` of a stacked tensor
  (`[l:l+1, ...]`) is written as its own tensor, read straight from its
  stored chunk. Stacked tensors must agree on the number of layers, a
  per-layer block must hold layers `0..L-1`, a block holds one form only, and
  the decoder must have `args.num_layers` layers (`args.decoder_num_layers` in
  a vision-language model that sets it, as `pretrain_vlm.py` does; `FC-R12`
  otherwise).
  Converting to `torch_dist`, each block gets the layout the loading job will
  ask for, since Megatron's loader requests the layout its own model builds:
  the decoder is per layer iff `TransformerBlock.sharded_state_dict` builds it
  so (the `content_metadata` the store gets has `non_homogeneous_layers` or
  `singleton_local_shards`, or the arguments set `moe_layer_freq` or
  `linear_attention_freq` to an int above 1 or a list), and stacked
  otherwise, written as one chunk per layer (Megatron's own grid). Megatron
  also builds it per layer for heterogeneous layer specs and `dsv4_hybrid`
  attention, whose arguments are refused (`FC-R03`, `FC-R20`). Arguments
  that make it stacked while its layers differ in names, shapes or dtypes are
  refused (`FC-R12`); the layout is never switched silently. A vision
  encoder (`vision_model.decoder`) is decided the same way: Megatron's
  vision-language builders (`pretrain_vlm.py`, `examples/multimodal`) make its
  configuration a copy of the language model's. A job that builds it from a
  configuration of its own needs `--block-layout`. A layer list of an unknown
  kind needs `--block-layout` (`FC-A02`).
  `--torch-dist-layout per-layer` stores every `TransformerBlock` one layer
  per key and sets `non_homogeneous_layers` in the store's `content_metadata`
  (from the one the manifest records, or Megatron's defaults
  `singleton_local_shards=False`, `chained_optim_avoid_prefix=True`), so
  Megatron loads every block per layer; it cannot be combined with
  `--drop-common-key content_metadata`.
  `--block-layout PREFIX=stacked|per_layer` (repeatable) decides for the
  `TransformerBlock` with that prefix (stacking needs layers that are alike);
  it wins over the other rules. It does not change the `content_metadata`, so
  the job that loads the store must build the block that way: a layout of the
  decoder or the vision encoder other than the one Megatron builds from the
  arguments and the `content_metadata` is written with a warning, since a job
  built from them cannot load it strictly (`--torch-dist-layout per-layer` is
  the loadable way to store the decoder per layer). Both flags are refused
  (exit 2) unless an `fsdp_dtensor` store is converted to `torch_dist`.
  When the source's manifest records another `torch_dist` layout of a block
  than the one decided, the conversion is refused (`FC-A06`) unless a flag
  asks for the block's layout or `--ignore-manifest` is given, so a round trip
  restores the source's layout or fails.
* **Routed experts.** `torch_dist` stores each linear layer of a
  mixture-of-experts layer's routed experts as one tensor over all experts,
  `<...>.mlp.experts.experts.linear_fc<n>.<weight|bias>` of shape `[E, ...]`
  (`[L, E, ...]` in a stacked block), expert `e` at the global index `e` of
  its expert axis and saved as a chunk of its own; `TEGroupedMLP` and
  `SequentialMLP` write the same keys. `fsdp_dtensor` stores one tensor per
  expert under its global index, named as the experts module names its
  parameters: *grouped* (`TEGroupedMLP`), `<...>.mlp.experts.linear_fc<n>.<leaf><e>`,
  or *sequential* (`SequentialMLP`),
  `<...>.mlp.experts.local_experts.<e>.linear_fc<n>.<leaf>`. Converting to
  `fsdp_dtensor`, each expert (of each layer) is read straight from its
  stored chunk, and the naming is the one of the experts module the loading
  job builds: grouped for a hybrid model (its stack spec always builds
  `TEGroupedMLP`), else grouped iff `args.moe_grouped_gemm`. For a GPT model,
  `--expert-naming grouped|sequential` chooses it instead, and then sets
  `moe_grouped_gemm` in the store's arguments to match (the GPT builders read
  it); a hybrid model's experts cannot be named otherwise (exit 2), since no
  argument changes the module its stack spec builds. Converting to
  `torch_dist`, each expert is written as its own chunk of the expert tensor,
  which is never materialized whole; an `fsdp_dtensor` source must name its
  experts as its arguments (or `--expert-naming`) build them (`FC-A04`
  otherwise). The number of experts is `args.num_experts`: every expert axis
  has that length, and every `fsdp_dtensor` expert tensor is stored for
  exactly the experts `0..E-1` (`FC-R12` for indices that are not contiguous
  from 0, `FC-R13` for another number).
* **SwiGLU.** A gated MLP's `linear_fc1` (`--swiglu`, or GEGLU with
  `--quick-geglu`) holds the gate `W` and value `V` projections, twice the
  rows its `linear_fc2` has inputs; every first projection is classified from
  these shapes, per layer and per routed expert (the MLPs of a layer agree,
  and the language model is gated iff `swiglu` or `quick_geglu` is set;
  `FC-R12` otherwise). `torch_dist` stores it whole, `[W; V]`, each half
  saved as chunks of its own (one per tensor-parallel rank). Megatron-FSDP splits it into `<name>_w` and `<name>_v`, both
  TP-canonical, exactly where `handle_swiglu_in_state_dict` does: with
  `args.swiglu`, for the `linear_fc1` of dense MLPs, shared experts and
  routed experts (`linear_fc1.weight<e>_w`, `local_experts.<e>.linear_fc1.weight_w`
  under the global index), in a gated layer. Converting to `fsdp_dtensor`
  splits them, and converting back writes each half as its rows of the whole
  tensor, one chunk per half. An `fsdp_dtensor` source must be split exactly
  so (`FC-R12`), with every half paired (`FC-R20`). On current Megatron a
  weights-only SwiGLU store loads into no Megatron-FSDP job: an fp32 job
  cannot build its SwiGLU state dict (`handle_swiglu_in_state_dict` looks
  parameters up under one `module.` level too many without `Float16Module`)
  and a bf16 one cannot load without optimizer state. Megatron-FSDP keeps every
  other gated `linear_fc1` fused (without `--swiglu`, e.g. GEGLU, a vision
  encoder of its own configuration under a plain language model, or a vision
  projection's `encoder.linear_fc1`) and stores the concatenation of the
  tensor-parallel ranks' local tensors, `[W_0; V_0; W_1; V_1; ...]`: *packed*
  at the tensor-parallel size of the job (the expert tensor-parallel size for
  routed experts), which is the canonical `[W; V]` only at size 1. Converting
  to `fsdp_dtensor` packs these at `--fsdp-tp` and `--fsdp-etp` (default: the
  source's size, only if it is 1, else `FC-A01`; routed experts need
  `--fsdp-etp` also when `--fsdp-tp` is not the source's tensor-parallel
  size, as a job's expert size follows its tensor-parallel size unless it sets
  `--expert-tensor-parallel-size`), and converting back unpacks
  them at the `fsdp_dtensor` store's own `tensor_model_parallel_size` and
  `expert_tensor_parallel_size` into one chunk per half (`FC-R10` for halves
  that do not divide by the size, or a manifest that records another size).
  Such a store is *TP-bound*: it loads right only at the size it was packed
  at, and Megatron-FSDP's loader does not check it, so load it with
  `--use-checkpoint-args --use-mp-args-from-checkpoint-args` (the store's
  arguments record the pack sizes; pass the expert size, which is not taken
  from a checkpoint) or
  with the same `--tensor-model-parallel-size` and
  `--expert-tensor-parallel-size`. `--target-swiglu on|off` converts for a job
  that sets `--swiglu` or not. `--swiglu` also makes a job's MLPs SiLU-gated,
  and a job without it gates them only with `--quick-geglu` (quick-GELU), so
  for a language model with MLPs it must say what the source's arguments say
  (anything else is a usage error: such a job builds another model). For a
  language model without MLPs, `on` splits the other gated `linear_fc1` that
  Megatron-FSDP splits under `--swiglu` (a vision encoder's), `off` packs
  them, and the store's arguments get the job's `swiglu` (and `quick_geglu`).
* **Gated DeltaProduct mixers.** With the Gated DeltaProduct stack spec
  (`gated_delta_product_stack_spec` or `gdp_stack_spec` of the hybrid spec
  modules), every `M` layer holds a mixer whose `in_proj` concatenates
  `z, V0..V(M-1), K0..K(M-1), Q, b0..b(M-1), a` and whose `conv1d`
  concatenates `V0..V(M-1), K0..K(M-1), Q`, each rank holding its shard of
  every component in this order. `torch_dist` stores every component of the
  weight and bias as a TP-canonical tensor of its own
  (`<...>.mixer.in_proj.weight.V0`, Megatron's split factory);
  `fsdp_dtensor` stores each whole, packed per tensor-parallel rank. The rule
  packs the components at `--fsdp-tp` (like a fused gated `linear_fc1`; the
  store is TP-bound) and cuts them out at an `fsdp_dtensor` store's own
  tensor-parallel size, along one `PartLayout` entry per component. The
  family of every mixer (`core/mixers.py`) is never decided from its widths
  alone: the `torch_dist` part names name a family (Mamba-2's `z, x, B, C, dt`,
  Gated DeltaProduct's with `M` the number of `V` parts, Gated DeltaNet's),
  `fsdp_dtensor`'s `conv1d_weight` is Mamba-2's and its fused `conv1d.weight`
  either family's, the spec decides between them (for every mixer by its
  module: a model may hold Gated DeltaProduct mixers and Gated DeltaNet
  layers, `built_family`), and of the fused widths the
  arguments give the families the names leave (`d = heads x mamba_head_dim`,
  `GN = mamba_num_groups x mamba_state_dim`, `H` heads; Mamba-2's heads
  default to `2 x hidden_size / mamba_head_dim`) the stored ones must match
  exactly the spec's family. A family the names exclude does not count, so a
  Mamba-2 mixer whose widths are Gated DeltaNet's under the default `linear_*`
  arguments still converts.
  Gated DeltaProduct needs `mamba_num_heads` and as many Householder steps as
  `gdp_num_householder` (3 when unset, as Megatron back-fills it); the
  `torch_dist` parts must have their family's names and sizes (`FC-R10` for
  any contradiction). A mixer store with neither the spec nor Gated
  DeltaProduct parts is a Mamba-2 model (below). On current
  Megatron, Megatron-FSDP takes the mixer's `A_log`, `dt_bias` and
  `norm.weight` (tensor-parallel without a `partition_dim`) for replicated
  parameters, so no Megatron-FSDP job above TP 1 holds these mixers: these
  identity tensors must have `H`, `H` and `d` rows (`FC-R10` for a store of
  one rank's shards), and packing for `--fsdp-tp` above 1 warns. The
  inference stack specs (`gated_delta_product_inference_stack_spec`,
  `gdp_inference_stack_spec`) are not allowlisted (`FC-R03`).
* **Mamba-2 mixers.** With the default stack spec (`hybrid_stack_spec`,
  `mamba_stack_spec`) every `M` layer holds a `MambaMixer`, whose `in_proj`
  concatenates `z, x, B, C, dt` (`d, d, GN, GN, H`, with `H` from
  `mamba_num_heads` or `2 x hidden_size / mamba_head_dim`, as
  `MambaMixer.__init__` derives it) and whose convolution concatenates
  `x, B, C`. The `mamba2` rule packs and cuts them like the Gated DeltaProduct
  components, with one difference in names: the mixer owns its convolution as
  `conv1d_weight` and `conv1d_bias`, which Megatron-FSDP stores, while
  `MambaMixer.sharded_state_dict` keeps the `conv1d` module's `torch_dist`
  names (`<...>.mixer.conv1d.weight.x`). Each rule's outputs take the target
  format's names (`FusedProjection.target_lid`). An `fsdp_dtensor` store saved
  before the mixer owned its convolution (`mixer.conv1d.weight`) is read as
  Mamba-2 unless the spec is the Gated DeltaProduct one; both names at once are
  refused (`FC-R10`). `A_log`, `dt_bias`, `D` (`H`, or `d` with
  `D_has_hdim`) and `norm.weight` (`d`) must be whole (`FC-R10`); Megatron-FSDP
  shards them along their `partition_dim`, so jobs above TP 1 hold these
  mixers (the TP 2 chain passes). *TP-bound identity* (`check_tp_bound` in
  `core/mixers.py`): a fused tensor `torch_dist` stores whole
  (`TORCH_DIST_WHOLE`: a Mamba-2 `in_proj.bias`, which the factory would not
  split; `MambaMixer` asserts `bias=False`, so only a mixer built otherwise has
  one) holds the ranks' local tensors one after the other in both formats, so
  it converts as is when `--fsdp-tp` is the source's tensor-parallel size and
  is refused otherwise (`FC-R10`).
* **GPT multi-token prediction.** A GPT MTP layer (`mtp_num_layers` without a
  hybrid pattern) holds one `TransformerLayer`, whose live name
  `mtp_model_layer` both formats store as `transformer_layer`
  (`MultiTokenPredictionLayer.sharded_state_dict`, `handle_mtp_in_state_dict`),
  so it converts unchanged. A Megatron-FSDP store saved before that handler
  holds the live name; it is accepted only when detected (no hybrid pattern,
  every inner-layer key under `mtp_model_layer`, a single layer): the MTP
  block's `BlockLayout` then carries a `ModuleRename` of its `fsdp_dtensor`
  side, which `core/structure.py` applies when it decomposes and composes
  keys, so the `LogicalId` is the same and `torch_dist` gets
  `transformer_layer`, with a warning and the `gpt_mtp_inner_layer`
  derivation (manifest). `--fsdp-gpt-mtp-inner-name mtp_model_layer` writes
  the legacy name for older runtimes (off by default; main's forward
  auto-rename is not ported). Both names in one store, `mtp_model_layer` in
  a `torch_dist` GPT store and an inner layer list under GPT arguments are
  refused (`FC-R09`).
* **Multi-latent attention.** Both formats store it under the unfused names
  (`input_layernorm`, `self_attention.linear_q_down_proj`, `linear_q_up_proj`,
  `linear_kv_down_proj`, `linear_kv_up_proj`, `linear_proj`), also with
  `--mla-down-proj-fusion`: `FusedMLASelfAttention.sharded_state_dict` and the
  fused layer specs' key maps (`torch_dist`) and
  `handle_mla_down_proj_in_state_dict` (Megatron-FSDP) save the fused
  `linear_qkv_down_proj` as the two down-projections and its absorbed layer
  norm as `input_layernorm`, so it converts as identity (GPT layers and hybrid
  `+`). A store under the fused live name is refused (`FC-R08`). Hybrid `+`
  layers (also `+E` shortcut pairs) are tested on synthetic stores and pinned
  to the hybrid layer specs' names by source scan, but not yet against a real
  save: the real-save harness builds no hybrid model with an
  `MLATransformerConfig` (a follow-up).
* **Gated DeltaNet layers.** A `G` layer of a hybrid model (with either stack
  spec), or a linear-attention layer of a GPT model
  (`experimental_attention_variant` `gdn`, `gated_delta_net` or `gdn2` with
  `linear_attention_freq`, which makes Megatron save the decoder per layer),
  holds a `GatedDeltaNet`, or a `GatedDeltaNet2` when the variant is `gdn2`,
  under `self_attention`; the mixer key grammar (`MIXER_TENSOR`,
  `split_mixer_tensor` in `core/keys.py`) takes `in_proj` and `conv1d` under
  `<...>.self_attention` as well as `<...>.mixer`, which standard and
  multi-latent attention never have. With `qk = linear_key_head_dim x
  linear_num_key_heads` and `v = linear_value_head_dim x
  linear_num_value_heads`, a Gated DeltaNet's `in_proj` concatenates
  `query, key, value, z, beta, alpha` (`qk, qk, v, v` and a row per value head
  each), a Gated DeltaNet 2's `query, key, value, z, f, b, w`
  (`qk, qk, v, v, qk, qk, v`), and both convolutions `query, key, value`. The
  `gdn` and `gdn2` rules pack and cut them like the mixers' components, under
  the same module names in both formats. The variant is told per layer
  (`built_family`): `torch_dist` parts name it, the variant argument decides
  (`gdn2` builds a Gated DeltaNet 2, any other value a Gated DeltaNet, as
  `HybridBlock` and the GPT linear-attention spec do; a `TransformerBlock`
  holds one only for a Gated DeltaNet variant, at a layer
  `linear_attention_freq` makes linear attention), and an `fsdp_dtensor`
  `self_attention.in_proj` leaves both, whose widths must then leave exactly
  the variant's (`FC-R10` for any contradiction). `A_log` and `dt_bias` must
  be whole (a row per value head; a Gated DeltaNet 2's `A_log` a row per key
  head and its `dt_bias` one per key channel; `FamilyLayout.vectors`):
  Megatron-FSDP shards them along their `partition_dim`, so jobs above TP 1
  hold these layers (the TP 2 chains pass). The layer specs build no `in_proj`
  bias, but `_GDNBase.sharded_state_dict` would leave one whole, so a module
  built with `bias=True` has a real TP-bound `in_proj.bias`
  (`TORCH_DIST_WHOLE`, pinned on a real TP 2 module).
* **Values and dtypes.** Every tensor is copied bit for bit with its dtype:
  Megatron's loaders cast on load, so bfloat16 weights load exactly into
  Megatron-FSDP's fp32 main weights, and the fp32 main weights of an
  `fsdp_dtensor` store round to the parameters of a `torch_dist` job as the
  training step would. Buffers keep their dtype too, whatever the model's:
  the router's `expert_bias` and `qb_beta` stay fp32 and its hash-routing
  table `tid2eid` stays int32 (the router's buffers, shared experts and the
  latent projections `fc1_latent_proj`, `fc2_latent_proj` and `fc2_norm` of a
  mixture-of-experts layer are stored alike in both formats).
* **Chunk grids.** `fsdp_dtensor` tensors are written as one chunk split along
  dimension 0 into chunks of at most `--max-output-chunk-bytes` (1 GiB);
  `torch_dist` tensors as one chunk per index of their leading layer and
  expert axes (per layer of a tensor stacked over layers, per expert of an
  expert tensor, per (layer, expert) of a stacked expert tensor) and other
  tensors as one chunk. A chunk that reads several stored chunks and is larger
  than half a unit is split along its first axis longer than 1. Both loaders
  read any grid.
* **Common state.** `fsdp_dtensor` stores get the byte items `args`,
  `iteration`, `checkpoint_version` and `num_floating_point_operations_so_far`
  (`content_metadata`, which they cannot hold, is recorded in the manifest; a
  source without one of the first three, which the loader requires, is
  refused with `FC-R13`);
  `torch_dist` stores get `common_state/shard_0_1` and `metadata.json`.
  `fsdp_dtensor` stores never get a `metadata.json` (Megatron would read them as
  `torch_dist`). TE `_extra_state` is dropped (`fsdp_dtensor` does not store
  it).
* **Arguments.** `no_save_optim` and `no_save_rng` are set; `fsdp_dtensor`
  stores get `ckpt_format='fsdp_dtensor'`, `use_megatron_fsdp=True`, one
  pipeline stage (the pipeline and virtual-pipeline fields are reset and `|` is
  removed from the hybrid pattern), the pack sizes as
  `tensor_model_parallel_size` and `expert_tensor_parallel_size` (with the
  weight-shard counts `tensor_parallel_num_weight_shards` and
  `expert_tensor_parallel_num_weight_shards` a job saves one per rank) if they
  pack tensors, and `swiglu`/`quick_geglu` as `--target-swiglu` asks for a
  language model without MLPs;
  `torch_dist` stores get `ckpt_format='torch_dist'` and
  `use_megatron_fsdp=False`. Only existing fields change; every change is
  reported.
* **Conversion manifest.** The output gets `conversion_manifest.json`: the
  source, the options, the derived architecture, the sanitized argument fields
  with their original values, the source's `content_metadata` (in its key
  order), the dropped sections, and the digest of the output's `.metadata`.
  Converting an `fsdp_dtensor` store back to `torch_dist` restores the
  original arguments and `content_metadata` from it (never clearing
  `no_save_optim` or `no_save_rng`; `--drop-common-key content_metadata` leaves
  that out), so a round trip gives the source back except for the dropped
  state. A manifest whose digest does not match its store is refused
  (`FC-A06`); `--ignore-manifest` converts without restoring. Tensors never
  depend on the manifest.

Resume from a converted store without its optimizer and RNG state, and with
the default strict loads (never `--finetune`, `--dist-ckpt-strictness log_all`
or `--no-strict-fsdp-dtensor-load` to validate a conversion):

* `fsdp_dtensor`: `--use-megatron-fsdp --ckpt-format fsdp_dtensor --no-load-optim --no-load-rng`;
* `torch_dist`: `--ckpt-format torch_dist --ckpt-drop-redundant-extra-state --no-load-optim --no-load-rng`
  (the store has no TE `_extra_state`).

Megatron-FSDP limitation: on current Megatron a bf16 or fp16 Megatron-FSDP job
cannot load any store without optimizer state (after loading, `load_checkpoint`
asks the distributed optimizer to copy the model parameters into its main
parameters, which Megatron-FSDP does not implement and raises
`NotImplementedError`). fp32 Megatron-FSDP jobs load converted stores; a test
pins the limitation so that its fix shows up.

`plan SRC --to FORMAT` prints the derived architecture (every field with its
evidence) and the number of model tensors each rule claims.

#### `torch_dist` to `torch_dist`

`convert SRC DST --to torch_dist` on a `torch_dist` store rewrites it as a
`torch_dist` store holding the model weights only:

* model tensors and TE `_extra_state` (tensors and byte items) are copied
  chunk for chunk, so the output has the source's chunk grid and dtypes;
* optimizer state (`optimizer.*`, `chained_<i>.optimizer.*`), RNG state and
  rerun state are dropped and reported;
* the common state keeps `args`, `iteration`, `checkpoint_version`,
  `num_floating_point_operations_so_far` and `content_metadata`, drops
  `optimizer`, `opt_param_scheduler` and `rerun_state_machine`, sets the
  existing `args.no_save_optim` and `args.no_save_rng` to `True` (so Megatron
  loads it without looking for optimizer or RNG state), and is written as the
  in-DCP item `common_state/shard_0_1` (a legacy `common.pt` is migrated);
* `run_config.yaml` and `tokenizer/` are copied, `train_dataloader_dprank*.pt`
  and a `conversion_manifest.json` (which describes the source's files) are
  dropped, other files are listed but not copied;
* a Megatron integrity manifest (`integrity.json`, checked on load with
  `--verify-integrity`) is written anew over the converted store's files.

Load the result with `--no-load-optim --no-load-rng` (a training job that
resumes from it starts a fresh optimizer). To check the result against its
source, ignore what the rewrite drops or sets:

```bash
python -m tools.checkpoint.format_converter verify DST SRC --layout --common semantic \
    --ignore-key '^(chained_\d+\.)?optimizer' --ignore-key '^rng_state' \
    --ignore-key '^rerun_state_machine' --ignore-key '^opt_param_scheduler$' \
    --ignore-key '^args\.no_save_(optim|rng)$'
```

`DST` is a checkpoint root, which gets `iter_%07d` (or `release`) and then
`latest_checkpointed_iteration.txt`, written last; or, if its name is
`iter_<N>` or `release` (or with `--dst-is-iter-dir`), the store directory
itself. Only that iteration directory is ever created or, with `--overwrite`,
replaced, and only if it is a single store (a directory holding `.metadata` and
no other store); other iterations in the root are never touched. The tracker
only moves forward: it is left alone (and the report says so) when it names a
newer iteration, `release` or something unreadable. The target, the source store
and the staging directory may not contain one another (exit 2). An interrupted
run resumes when the same command is run again.

`plan SRC --to torch_dist` prints the same decisions (sections kept and dropped,
the common state, sidecars, units, the memory and disk estimates) from the
metadata and the common state (byte items are sized from the metadata, not
read). It exits 3 when `convert` would be refused and 8 when the conversion
exceeds the memory budgets, without writing anything; free disk space is
checked by `convert`, which knows the destination.

The common state is decoded with Megatron's own safe globals when Megatron can
be imported (the training container). Elsewhere a plain decoder accepts only
`argparse.Namespace`, containers, numbers, strings and torch dtypes; a common
state that references Megatron classes then makes `convert` exit 3 with
`FC-E01`, while `plan` decodes it with inert stubs (reported as the `stub`
codec) and still reports everything.

| Flag (`plan` and `convert`) | Meaning |
|---|---|
| `--to {torch_dist,fsdp_dtensor}` | target format (required) |
| `--from {torch_dist,fsdp_dtensor}` | format to assume when the store has no format markers |
| `--iteration N` | iteration to open under a checkpoint root |
| `--drop-common-key KEY` | drop this common-state key: an unknown one instead of refusing, or a kept one that is not required, such as `content_metadata` (repeatable) |
| `--workers N` | worker processes; `0` runs the units in the driver (default `min(4, cores/8)`) |
| `--unit-bytes SIZE` | target peak memory of one unit (default `4GiB`) |
| `--max-unit-bytes-hard SIZE` | largest memory one unit may need, else exit 8 (default 0.6 x `MemAvailable` / workers) |
| `--writer-threads N` | data files written per unit (default 4) |
| `--max-host-bytes SIZE` | host-memory budget of the whole conversion, else exit 8 (default 0.8 x `MemAvailable`) |
| `--optimizer {drop,convert}` | `drop` optimizer state (default); `convert` is not supported yet (`FC-P11`, `FC-P12`) |
| `--allow-root PREFIX` | convert model keys under this unknown top-level prefix unchanged (repeatable; cross-format) |
| `--block-layout PREFIX=stacked\|per_layer` | store the block `PREFIX` of a `torch_dist` target stacked over its layers or one layer per key (repeatable; `fsdp_dtensor` source only) |
| `--torch-dist-layout {native,per-layer}` | `per-layer` stores every `TransformerBlock` of a `torch_dist` target one layer per key and marks it in the `content_metadata`; `native` as Megatron would (default; `fsdp_dtensor` source only) |
| `--expert-naming {auto,grouped,sequential}` | how `fsdp_dtensor` names routed experts: `auto` as the model's builder does (default), or as given, for a source or a GPT target (whose `moe_grouped_gemm` is set to match; a hybrid target is always `grouped`) |
| `--target-swiglu {auto,on,off}` | whether the job that loads an `fsdp_dtensor` target sets `--swiglu`, under which Megatron-FSDP splits gated `linear_fc1` into SwiGLU halves: `auto` as the source's arguments say (default), `on` or `off`; for a language model with MLPs only as its arguments say (`--swiglu` also sets the MLPs' activation), for one without MLPs as the job sets it, which the store's arguments then record (`torch_dist` source only) |
| `--fsdp-gpt-mtp-inner-name {transformer_layer,mtp_model_layer}` | the name an `fsdp_dtensor` target gives the inner layer of GPT multi-token prediction layers: `transformer_layer`, which current Megatron-FSDP saves and loads (default), or the live name `mtp_model_layer`, only for Megatron-FSDP versions older than `handle_mtp_in_state_dict` (an `fsdp_dtensor` source is read under the name it holds; `torch_dist` source only) |
| `--fsdp-tp N`, `--fsdp-etp N` | the tensor-parallel and expert tensor-parallel sizes of the job that loads an `fsdp_dtensor` target, which packs its fused tensors per rank at them and records them in its arguments (default: the source's, if it is 1; `FC-A01` without it when the source's is above 1 and there are such tensors, and without `--fsdp-etp` when routed experts are packed and `--fsdp-tp` is not the source's tensor-parallel size; a padded vocabulary that `--fsdp-tp` does not divide is refused, `FC-R10`, and one a job at that size would pad otherwise is warned about; `torch_dist` source only) |
| `--max-output-chunk-bytes SIZE` | largest chunk of an `fsdp_dtensor` output tensor (default `1GiB`) |
| `--allow-missing-fp8-extra-state` | write a `torch_dist` store for an FP8 delayed-scaling model without its scaling state (`FC-R07` otherwise) |
| `--ignore-manifest` | do not restore what the source's `conversion_manifest.json` records |
| `--json` | print the report as JSON |

| Flag (`convert` only) | Meaning |
|---|---|
| `--staging-dir DIR` | directory in which the staging directory `.fc-staging-<name>` is created (only that subdirectory is ever deleted), on the destination's file system (default: the target's parent) |
| `--resume {auto,never}` | `auto` continues staging of the same conversion (default); `never` exits 6 if staging exists |
| `--restart` | discard existing staging first |
| `--keep-staging` | keep the staging directory (and its part stores) afterwards |
| `--overwrite` | replace an existing target iteration directory (only that directory, and only if it is a single store) |
| `--dst-is-iter-dir` | treat `DST` as the store directory (no tracker) |
| `--no-tracker` | do not write `latest_checkpointed_iteration.txt` |
| `--no-copy-sidecars` | do not copy `run_config.yaml` and `tokenizer/` |
| `--report PATH.json` | also write the report as JSON |

## Exit codes and refusal codes

| Exit | Meaning |
|---|---|
| 0 | ok |
| 1 | internal error |
| 2 | usage error |
| 3 | refused (`FC-R`, `FC-P`, `FC-E` codes) |
| 4 | ambiguous, needs a flag (`FC-A` codes) |
| 5 | verification mismatch |
| 6 | integrity or I/O error (corrupt or truncated store, bad tracker or digest file, existing target, locked or foreign staging, failed worker) |
| 7 | staging fingerprint mismatch: the staging directory holds another conversion (`--restart` discards it) |
| 8 | memory or disk budget exceeded |
| 130 | interrupted |

Refusals print one line each, `FC-xxx: <message> [keys...] hint: <what to do>`;
every refusal found is printed, permanent ones (`FC-R`) first. `FC-P` codes
are pending families: supported configurations whose conversion rules land in
a later version (see [Roadmap](#roadmap)); they are detected from the training
arguments and the model keys before any tensor is read.

| Code | Refused input |
|---|---|
| `FC-R01` | Megatron-FSDP v2 store (`model.<fqn>` keys without `module.`) or arguments (`megatron_fsdp_version` 2) |
| `FC-R02` | virtual-pipeline sections (`model<i>.` or `optimizer.<i>.` keys), or an `fsdp_dtensor` source saved with pipeline or virtual-pipeline parallelism |
| `FC-R03` | a spec or implementation whose keys the rules do not know: `transformer_impl` other than `transformer_engine`, a `--spec` outside the verified list, heterogeneous layer specs, Transformer Engine operation-fuser MLPs |
| `FC-R04` | single grouped expert parameters (`moe_single_grouped_weight` or `moe_single_grouped_bias`, or an `fsdp_dtensor` `<...>.mlp.experts.linear_fc<n>.weight` without an expert index): every local expert of a linear layer in one tensor, which Megatron-FSDP does not support |
| `FC-R05` | grouped expert biases (`linear_fc<n>.bias<e>`) of an `fsdp_dtensor` store with expert parallelism (`expert_model_parallel_size` above 1), in either direction: Megatron-FSDP names them by their expert-parallel-local index (it renumbers only the weights), so the expert-parallel ranks save and load one another's; sequential experts (`--expert-naming sequential` for a GPT job without `--moe-grouped-gemm`; a hybrid model always builds grouped experts) name them by global index |
| `FC-R06` | grouped tensor parallelism (a weight rematerialization size above 1, or more weight shards than tensor-parallel ranks, for the dense or the expert layers), which may pad stored tensors; as many shards as tensor-parallel ranks, which every Megatron job saves, is plain tensor parallelism |
| `FC-R07` | an `fsdp_dtensor` source of an FP8 delayed-scaling model converted to `torch_dist`, whose loader needs the scaling state (`--allow-missing-fp8-extra-state` writes the store without it) |
| `FC-R08` | multi-latent attention's fused down-projection under its live name `linear_qkv_down_proj` (and its absorbed `layer_norm_*`): a Megatron-FSDP store saved before `handle_mla_down_proj_in_state_dict`, or a `torch_dist` store, neither of which current Megatron writes or loads (both formats store `linear_q_down_proj`, `linear_kv_down_proj` and `input_layernorm`) |
| `FC-R09` | the inner layers of multi-token prediction layers are stored under names no save of the model gives them: a store holding GPT MTP layers under both `transformer_layer` and `mtp_model_layer`, a `torch_dist` store holding `mtp_model_layer` under arguments without a hybrid layer pattern (GPT MTP), an inner layer list (`<name>.layers.`) under such arguments, or `transformer_layer` in a hybrid model |
| `FC-R10` | a tensor packed per tensor-parallel rank (rank-major) cannot be laid out: a part of the fused tensor does not divide by the tensor-parallel size it is packed at, or an `fsdp_dtensor` source's `conversion_manifest.json` records another pack size than its arguments' tensor-parallel size; or converting to `fsdp_dtensor`, `args.padded_vocab_size` does not divide by `--fsdp-tp`; or a state-space mixer's family cannot be told: its `torch_dist` part names are no family's or not the family Megatron builds at its module (the spec's under `mixer`, the `experimental_attention_variant`'s under `self_attention`), its widths match another family's or two families' as the arguments give them, its dimensions are not in the arguments (`mamba_num_heads` for Gated DeltaProduct), its Householder steps are not `gdp_num_householder`, its parts have other sizes than its family's, its `A_log`, `dt_bias`, `D` or `norm.weight` is not whole (one rank's shard, as a Megatron-FSDP job above TP 1 saves a Gated DeltaProduct mixer's), `torch_dist` stores its `in_proj` or `conv1d` whole (but a Mamba-2 `in_proj.bias`) or a Mamba-2 `in_proj.bias` as parts, or its convolution is stored under both its names (and likewise for a Gated DeltaNet layer: its `A_log` or `dt_bias` is one rank's shard, its `in_proj.bias` is stored as parts, or its widths are the other variant's); or a GPT decoder (`TransformerBlock`) holds a Gated DeltaNet where the arguments build none (`experimental_attention_variant` no Gated DeltaNet variant, `linear_attention_freq` unset, or a layer it makes standard attention); or a fused tensor `torch_dist` stores whole (a Mamba-2 or Gated DeltaNet `in_proj.bias`) is converted with `--fsdp-tp` other than the source's tensor-parallel size |
| `FC-R11` | the layer indices a store holds contradict the hybrid layer pattern (or `args.num_layers`), or a module holds the tensors of other layer types than its pattern symbols (family signatures), or the pattern cannot be grouped into shortcut-MoE modules (an `E` whose preceding layer is neither in a pair already nor `M`, `G` or an attention layer, such as the `E` of `-E` or the last `E` of `*EEE`, or a pipeline stage after the first that begins with an `E`); the message lists the expected and stored indices and every module's dense index, `torch_dist` index and symbols |
| `FC-R12` | a block's layers are stored inconsistently: tensors stacked over its layers disagree on the number of layers (or it contradicts `args.num_layers`, or `args.decoder_num_layers` of a vision-language model, for the decoder), a block holds both stacked and per-layer keys, per-layer indices are not `0..L-1`, an `fsdp_dtensor` store or a hybrid decoder holds stacked tensors; or layers that differ would be stacked (the arguments make Megatron stack the decoder or the vision encoder, or `--block-layout PREFIX=stacked`); or an `fsdp_dtensor` expert tensor is stored for expert indices that are not `0..n-1` (such as `0-1, 3`); or gated linear units contradict one another or the arguments: the `linear_fc1` weights of one layer are both gated (twice the rows `linear_fc2` has inputs) and plain, the language model's are gated while neither `swiglu` nor `quick_geglu` is set or plain while one is, or a `linear_fc1` has no `linear_fc2` beside it (one outside every layer under a name Megatron-FSDP never splits, such as a vision encoder's patch merger's, is a plain linear layer and converts unchanged); or an `fsdp_dtensor` store splits a `linear_fc1` into SwiGLU halves where Megatron-FSDP keeps it whole or the other way round, or splits its weight but not its bias |
| `FC-R13` | unknown layout: another `sharded_backend`, no format markers (unless `--from`), mixed model prefixes, `model.module.module.` keys, a legacy `mp_rank_*` checkpoint, an unknown byte item, an unknown common-state key, model keys under an unknown top-level module (`--allow-root`) or under both single-model and vision-language roots, model keys that do not fit the key grammar or the layer layout, model tensors no rule handles (the message names the reason, e.g. a parameter stored as parts or a fused mixer projection, and the rule or pending code that owns it), `bf16` and `fp16` both set, a `content_metadata` the conversion manifest cannot record (`--drop-common-key content_metadata`), a common state without `args`, `iteration` or `checkpoint_version` converted to `fsdp_dtensor`, parameters of the legacy `GroupedMLP` (`<...>.mlp.experts.weight1`/`weight2`: every local expert in one tensor), routed-expert tensors while `args.num_experts` is unset, or expert tensors of another number of experts than `args.num_experts` |
| `FC-R14` | model tensors stored in the legacy N-D reformulated layout (`mcore_data`); load with current Megatron and save again |
| `FC-R17` | a store saved with `singleton_local_shards` (its `content_metadata` says so, or its experts are saved one key per expert, `<...>.mlp.experts.experts.<e>.<name>`, or its gated `linear_fc1` as SwiGLU halves `<key>_w`/`<key>_v`) |
| `FC-R18` | a ModelOpt state (`modelopt_state` in the common state or next to the store) |
| `FC-R19` | `fsdp_dtensor` store with split fused-projection sub-keys (`in_proj`/`conv1d` `.weight.<part>`) |
| `FC-R20` | layer types no rule is verified for: hybrid layer symbols `D`, `C`, `H`, `W` (or unknown ones), or an `experimental_attention_variant` other than the Gated DeltaNet ones; or an `fsdp_dtensor` SwiGLU half (`linear_fc1.weight_w`) without its pair, of another shape than its pair, or stored beside the whole tensor |
| `FC-P11` | pending: converting optimizer state from `fsdp_dtensor` to `torch_dist` |
| `FC-P12` | pending: converting optimizer state from `torch_dist` to `fsdp_dtensor` |
| `FC-A01` | converting to `fsdp_dtensor`, the store packs fused tensors per tensor-parallel rank (gated `linear_fc1` it keeps fused, per expert tensor-parallel rank for routed experts; Gated DeltaProduct, Mamba-2 and Gated DeltaNet `in_proj` and `conv1d`), and the source was saved at a (expert) tensor-parallel size above 1, or the store packs routed experts' tensors and `--fsdp-tp` is not the source's tensor-parallel size (a job at that size runs its experts at it unless it sets `--expert-tensor-parallel-size`): pass `--fsdp-tp N` (`--fsdp-etp N`) with the size of the job that will load it |
| `FC-A02` | converting to `torch_dist`, a layer list of an unknown kind (not the language model's decoder or `vision_model.decoder`), whose layout depends on its module: pass `--block-layout PREFIX=stacked` or `=per_layer` |
| `FC-A04` | an `fsdp_dtensor` source names its routed experts otherwise (grouped or sequential) than the experts module its arguments build (`args.moe_grouped_gemm`; hybrid models always build grouped experts): pass `--expert-naming` with the store's naming if the store is right |
| `FC-A05` | `--from` disagrees with the detected format |
| `FC-A06` | the source's `conversion_manifest.json` describes another store (its `.metadata` digest differs), or records another `torch_dist` layout of a block than the one decided (the hint names the flag that restores it); `--ignore-manifest` converts without restoring what it records |
| `FC-E01` | the common state references Megatron (or other non-plain) classes and Megatron cannot be imported, or it was decoded for display only; run inside the Megatron container |
| `FC-E02` | the staging directory is not on the destination's filesystem (part files are hard-linked into the output) |

## Format detection

| Evidence | Result |
|---|---|
| `metadata.json` with `sharded_backend == "torch_dist"` | `torch_dist` / `mcore` |
| `metadata.json` with another backend | `FC-R13` |
| `model<i>.` or `optimizer.<i>.` keys | `FC-R02` |
| `model.<fqn>` keys, none under `model.module.` | `FC-R01` |
| `model.module.*` keys and other `model.*` keys | `FC-R13` |
| `model.module.*` keys with split `in_proj`/`conv1d` sub-keys | `FC-R19` |
| `model.module.*` keys and byte items `args`, `iteration`, `checkpoint_version` | `fsdp_dtensor` / `mfsdp_v1` |
| none of the above | `FC-R13`, or the format given with `--from` |
| `mp_rank_*` directories instead of a DCP store | `FC-R13` |

`verify` compares any two DCP stores and does not require a detected format.

## Memory model

### Reading (`inspect`, `verify`)

DCP decodes a stored item with one whole-item `torch.load` per read request,
however little of it is needed. The reader therefore asks for exactly one box
per touched stored chunk per batch (the bounding box of what the batch needs
from that chunk). The box is copied straight into the destination when one
request covers it exactly, and staged otherwise, so the transient cost of a
read is at most twice the stored chunk (`C + |box|`). Every touched stored
chunk is decoded once (`ReadStats.amplification == 1.0`).

`--max-item-bytes` bounds every single read unit, and every unit is checked
before anything is read (exit 8 with the size needed):

* a stored chunk (its decoded size);
* in `verify A B`, a compare unit per side: the bounding box of a connected
  component of the overlap graph between A's and B's chunks (identical grids
  give one unit per chunk);
* for digests, a whole tensor whose chunk grid splits more than dimension 0.
  Tensors split along dimension 0 only are streamed in row slabs.

Small reads are grouped into batches of up to 256 MiB (capped by
`--max-item-bytes`) to amortize the per-call overhead of `dcp.load`. Per worker
thread the resident set is about one batch or unit plus one decoded stored
chunk for `inspect --digests`, and two (one per side) plus one decoded stored
chunk for `verify A B`. The total is `--workers` times that plus the parsed
metadata of each store (roughly one to two times the size of its `.metadata`
file). Page cache is not counted.

### Converting (`plan`, `convert`)

A conversion runs as units of work. A unit's pieces (output chunks) are all
held in memory while it runs, because DCP's writer keeps every tensor of a data
file referenced until the file is closed. Per unit `u`:

| Term | Meaning |
|---|---|
| `P_u` | the unit's piece buffers (at most the unit budget `U`, `--unit-bytes`) |
| `S_u` | read staging: the largest stored item DCP decodes whole (`C`), plus a staging box for every stored chunk that several copies read |
| `M_w` | a worker's baseline: interpreter, torch and the unit's metadata (768 MiB assumed) |
| `M_driver` | the driver: its resident set after planning, plus about 2 KiB per output chunk for the merge |

```
worker_peak = max_u (P_u + S_u) + M_w
total       = N * worker_peak + M_driver       (N = --workers; with 0 the driver runs the units)
```

`plan` and `convert` refuse (exit 8), before any tensor is read, when:

* `total` exceeds `--max-host-bytes` (default 0.8 x `MemAvailable`);
* the pieces that read one stored item cannot be held together within
  `--max-unit-bytes-hard` (default 0.6 x `MemAvailable` / workers);
* a piece that reads several stored items has a single row larger than `U/2`.

`convert` also refuses (exit 8) when the destination's file system has less
than 1.02 x the estimated output free.

Packing keeps every stored item in exactly one unit, so each is read once: the
report's `torch_load_calls` equals `items_touched`. For a weights-only rewrite
every piece copies one stored chunk `C`, which costs `2C` while it is read (the
piece plus DCP's whole-item decode), so a unit needs at most `max(U, 2C)` for
the largest stored chunk `C`: the resident set depends on `U` and the largest
stored chunk, never on the store size. The report
records the measured peaks: the driver's `VmHWM`, the largest worker's
`ru_maxrss` and each unit's `VmHWM` in the journal.

## Design

### Layering and import boundary

| Layer | Package | May import | Runs on |
|---|---|---|---|
| L0 | `engine/` | stdlib, torch, `torch.distributed.checkpoint` | host Python 3.10 with torch 2.8 and newer, containers, CI |
| L1 | `core/` | L0; pure, no I/O | same |
| L2 | `formats/`, `verify/`, `rules/` | L0, L1 | same |
| L3 | `mcore_state/` | L0 to L2; megatron only inside functions | same; Megatron's codec needs the container |
| L4 | `cli.py`, `api.py`, `options.py`, `planning.py`, `report.py`, `sidecars.py` | L0 to L3 | same as the layers it uses |

L0 to L2 avoid `typing.Self`, `StrEnum` and PEP 695 syntax so they import on
Python 3.10. `test_import_boundary.py` imports every module, and runs
`inspect`, `verify`, `plan` and `convert`, in a fresh interpreter whose import
system refuses `megatron`, `transformer_engine` and `nvidia_resiliency_ext`.
Importing the package itself does not import torch; the Python API
(`inspect_checkpoint`, `verify_checkpoint`, `plan_conversion`,
`convert_checkpoint`) is exported lazily.

### Conversion engine

A conversion is planned from the metadata and the common state, and then
executed by the engine, which knows nothing about formats:

1. **Plan** (`planning.py`). Detect the source, classify every key, apply the
   refusals, decode the common state and encode the output one, and describe
   the output as *pieces*: boxes of output tensors (`engine/pieces.py`), each
   filled by copies from boxes of stored source chunks and backed at run time
   by one standalone buffer. Byte items are encoded in the driver. A
   cross-format plan gets its pieces from the rule pipeline (see
   [Rule model](#rule-model)).
2. **Pack** (`engine/scheduler.py`). Group pieces into units under the memory
   budget (see [Memory model](#memory-model)); pieces that read the same stored
   item share a unit.
3. **Stage** (`engine/staging.py`). Write each unit (pieces, byte items and the
   source metadata tables it needs; engine data classes only, loaded back with
   an allowlist unpickler) and `PLAN.json` into a staging directory beside the
   target.
4. **Execute** (`engine/worker.py`, `engine/workers.py`). Workers claim units,
   read every copy with one batched `dcp.load` (each stored item decoded once),
   and write the unit's buffers as a complete DCP *part store* with one
   `dcp.save` through `PieceSavePlanner` (`engine/part_writer.py`). The planner
   writes each piece as one `SHARD` item, so a tensor may be spread over
   several parts, and hands the writer the buffer itself (no copy). Workers
   are fresh interpreters started with `CUDA_VISIBLE_DEVICES=''` and without
   launcher variables, so a conversion started under `torchrun` never joins a
   process group or touches a GPU.
5. **Merge** (`engine/merge.py`). Hard-link every part's data files into the
   final store as `u<uid>_<file>`, union the parts' metadata, and check it
   against the plan: exact key sets, shapes, dtypes, a chunk grid that tiles
   each tensor and equals the planned grid, data files long enough. Then write
   a fresh `.metadata` atomically and re-read it: the installed torch must
   parse it, and a sample of pieces and every byte item are read back and
   compared with SHA-256 digests the workers recorded.
6. **Publish**. Add the format's marker files (`metadata.json`) and sidecars,
   rename the merged store into place and write the tracker last.

### Rule model

Cross-format conversion never translates one format's keys into the other's.
Both formats are mapped onto one **logical key space**, and rules work there
only (`core/`, `rules/`):

* A **model-relative key** is a model tensor's key without its envelope
  (`fsdp_dtensor` stores prefix `model.module.`; `torch_dist` keys are bare).
  `core/keys.py` holds the whole key grammar: `<block>.layers.<index>.<path>.<leaf>`
  inside a layered block (`<block>.layers.<path>.<leaf>` for a tensor stacked
  over the block's layers), `<path>.<leaf>` elsewhere, and every regular
  expression over keys. Blocks nest (a multi-token prediction depth holds the
  layer list of its inner model): a key belongs to the innermost block whose
  layer list holds it, the longest block prefix, and only that block's layer
  segment is interpreted, so `vision_model.decoder`, `language_model.decoder`
  and `mtp` keys never mix.
* A **`LogicalId`** (`core/logical.py`) names a tensor as the live model does:
  its block (e.g. `decoder`), its *dense* layer index (the `nn.ModuleList`
  position), the module path inside the layer and the leaf name. The two
  formats may number the layers of a block differently; each block's
  `BlockLayout` (`core/blocks.py`) maps dense indices to each format's stored
  indices, and `core/structure.py` decomposes model-relative keys into ids and
  composes them back; a module of a block's layers that one format names
  otherwise (a legacy GPT MTP store's `mtp_model_layer`) is a `ModuleRename`
  of the block, applied there. Format adapters (`formats/`) own only the envelope
  (`relative_key` strips it, `stored_key` adds it), the planner paths and the
  output chunk grid.
* **Refusal fence** (`core/refusals.py`). Detectors registered with
  `@REGISTRY.detector` read the training arguments and the model keys and
  shapes, before any tensor is read, and report every refusal: permanent
  `FC-R` codes for configurations no rule will support, and one pending
  `FC-P` code per family whose rules have not landed yet (now the optimizer
  state in each direction). A family's change lifts exactly its code, so no
  version of the tool converts a family it does not know.
* **Architecture context** (`core/context.py`). Everything rules may depend on
  (direction, formats, parameter dtype, the parallel sizes the source was
  saved with, the blocks, the routed experts, which MLP first projections are
  gated linear units, the family and fused projections of every state-space
  mixer, the dtype policy, the source's conversion manifest)
  is derived from the arguments, cross-checked against the store where the
  store can tell, and logged as `(field, value, evidence)`; `plan` prints the
  log and the manifest keeps it.
* **Rules** (`core/rule.py`, `rules/`). The planner (`core/planner.py`) turns
  every source tensor into a `LogicalSlice` (a region of a stored key holding
  one logical tensor), all in one `LogicalPool`. Rules registered with `@REGISTRY.rule`
  run in `Stage` order (`FUSED`, `GLU`, `OPTIM`, then `IDENTITY`) and take the
  slices they handle; each returns `FamilyInstance`s naming the slices it
  consumed and the `LogicalOutput`s (output tensors as `LogicalCopy` boxes of
  source slices) it builds. Every slice is claimed and consumed exactly once;
  anything else is an internal error.
* **Fused parts** (`core/part_layout.py`). A parameter that concatenates
  several parts along its first axis (a gated `linear_fc1`'s `W` and `V`, a
  mixer's `in_proj` components) is sharded part by part over the
  tensor-parallel ranks, so a store that concatenates the ranks' local tensors
  holds it *rank-major*, `[W_0; V_0; W_1; V_1; ...]`. A `PartLayout` lists the
  parts (one entry per component, never an aggregate of components) with their
  global sizes; `rank_major_map(tp)` gives where each rank's shard of each part
  sits in the tensor packed at `tp` (the parts' concatenation at `tp = 1`), and
  `repack_map` moves rows between two pack sizes. A part that does not divide
  by the pack size is refused (`FC-R10`).
* **Sentinels** (`Sentinel` in `core/registry.py`, registered in
  `core/refusals.py`, which the identity rule imports so that it never runs
  without them) name source keys the generic identity rule must never convert
  (a parameter stored as parts, SwiGLU halves, fused mixer projections). Each
  is owned by the rule, or while it is pending the `FC-P` code, that handles
  such keys; a reserved slice nobody claims is refused (`FC-R13`) with the
  sentinel's reason. (A tensor stacked over layers needs no sentinel: every
  stacked key belongs to a block of the census, whose layout is checked.)
* **Compile** (`core/planner.py`). `core/structure.py` composes every
  output's model-relative key, and the target adapter adds its envelope,
  planner path and chunk grid; each output chunk becomes an engine piece whose
  copies read boxes of the source's stored chunks. The plan must read every element
  of every consumed source tensor exactly once, tile every piece and every
  output tensor, and write each key once.

This version has six rules, `gdp`, `mamba2`, `gdn` and `gdn2`
(`rules/mixers.py`: the weights and biases of the `in_proj` and `conv1d` of a
Gated DeltaProduct or Mamba-2 mixer or of a Gated DeltaNet of either
variant: `torch_dist` stores one tensor per component, which decomposes to
the whole tensor's `LogicalId` with the component as its `part` and composes
back to `<name>.<part>`; the rule packs the parts rank-major at `pack_tp` into
the whole `fsdp_dtensor` tensor, and cuts them back out along the
projection's `PartLayout`, naming the output as the target format does), `glu` (the weight and bias of a gated
`linear_fc1` that the formats store differently, as `core/glu.py` decides:
one that `fsdp_dtensor` stores as SwiGLU halves has each half as an output of
its own, a `LogicalId` with `part` `w` or `v`, which `fsdp_dtensor` names
`<name>_w`/`<name>_v` and `torch_dist` places at its rows of the whole tensor;
one that `fsdp_dtensor` keeps fused is packed rank-major, canonical `[W; V]`
to `[W_0; V_0; W_1; V_1; ...]` and back along its halves' `PartLayout`, at
`pack_tp`, or `pack_etp` for a routed expert) and `identity` (tensors stored
the same way in both formats), and four
kinds of blocks. The blocks a store holds are found from its
keys (each key's innermost layer list), and the model's roots from the keys'
top-level modules: the language model's decoder is `decoder`, or
`language_model.decoder` in a vision-language model, and its multi-token
prediction block `mtp` (`language_model.mtp`). The registered block
classifiers recognize the `HybridStack` decoder of a hybrid model and the inner
layer list of each hybrid MTP layer (both stored per layer; their modules, a
layer or a shortcut-MoE pair, numbered by their first layer's pattern position
in `torch_dist` and densely in `fsdp_dtensor`; each `BlockLayout` records the
map and every module's symbols, and every module's tensors are checked against
its symbols' family signatures), and the MTP block (one layer per depth, never
stacked; a GPT MTP block records how each format names its layers' inner
layer); every other block is a `TransformerBlock` (`core/stacking.py`: the
language model's decoder, a vision encoder, or a layer list of unknown kind),
numbered identically in both formats and stored stacked or per layer in
`torch_dist`. The layer map applies through the dense layer index of every
`LogicalId`, so every key (and every namespace that decomposes keys the same
way) is renumbered alike. A stacked `torch_dist` tensor decomposes into
one logical slice per layer: the box `[l:l+1, ...]` holding a logical tensor
of the layer's shape (the slice records the squeezed leading axis), so rules
never see the stacking; engine copies may add or drop leading size-1 axes
(`CopyOp` boxes are equal up to them). A routed-expert tensor of `torch_dist`
(`<...>.mlp.experts.experts.linear_fc<n>.<leaf>`) has an expert axis after
the layer axis, and decomposes into one slice per (layer, expert),
`[l:l+1, e:e+1, ...]`; its `LogicalId` carries the global expert index and
the module path `<...>.mlp.experts.linear_fc<n>`, which is also what both
`fsdp_dtensor` namings (grouped `linear_fc<n>.<leaf><e>`, sequential
`local_experts.<e>.linear_fc<n>.<leaf>`) parse to. The expert layout (number
of experts and `fsdp_dtensor` naming, `core/experts.py`) is derived with the
context. The verified hybrid symbols are `*`, `-`, `E`, `M` (Mamba-2 or,
with its stack spec, Gated DeltaProduct mixers), `G` (Gated DeltaNets) and
`+` (multi-latent attention, converted by the identity rule), with
shortcut-MoE grouping and hybrid multi-token prediction layers, and the
verified attention variants the Gated DeltaNet ones. GPT multi-token
prediction and multi-latent attention need no rule of their own: both
formats store them under the same names (but a legacy GPT MTP store's,
which the MTP block renames).

### Extension contract

Supporting another model family is one change that:

1. adds one rule module `rules/<family>.py` (or, for a family that shares
   another's machinery, a rule beside it: `rules/mixers.py` holds one rule per
   mixer family) with a `LeafRule` subclass registered with `@REGISTRY.rule`, declaring its `stage` and `enabled(ctx)`
   and working on logical ids only (key syntax, including its sentinel
   patterns, goes into `core/keys.py`), and takes over the sentinels its
   family owns (their owners in `core/refusals.py`);
2. derives and cross-checks what it needs in `core/context.py` (a derivation
   with evidence; a contradiction is a refusal naming the field, the argument
   value and the observed value), and adds a block classifier to
   `core/blocks.py` if the family has its own layer list;
3. removes its `FC-P` code from the fence, or adds its layer symbol, spec or
   attention variant to the registry's allowlists;
4. adds tests: hand-built expected values that the rule does not produce
   (`@pytest.mark.covers("<rule>", "ground_truth")`), a parity pin against
   Megatron or a real-save chain (`covers(..., "parity")` or `"real_save"`),
   and a synthetic store of the family;
5. documents the family here and in the support matrix of the user guide.

`test_registry_contract.py` enforces the checkable parts: every rule module
registers a rule, every rule has both kinds of evidence, every sentinel has a
registered owner, every pending code has a triggering test, and rule modules
neither import `re` nor spell key syntax.

### Crash safety and resume

* Staging lives in `.fc-staging-<name>`, beside the target or inside
  `--staging-dir`; the tool deletes only that directory, never the one it is
  created in.
* The driver holds an `flock` on the staging directory and its workers inherit
  it, so a second conversion into the same target exits 6, also while workers
  of a killed driver still run (they stop before their next unit once the
  driver is gone).
* A unit is committed by renaming its complete part store (with a `DONE`
  record of digests) into place; a torn part (`*.tmp`) is deleted on resume.
* The fingerprint in `PLAN.json` covers the source store's content (its
  `.metadata` digest and every file's size and mtime), the target, the options
  that shape the output, the tool version and the torch minor version. Running
  the same command again resumes: complete units are kept, and every claim is
  treated as stale while the lock is held (process ids are never compared; a
  container may have reused them). A different fingerprint exits 7 unless
  `--restart`; `--resume never` refuses existing staging (exit 6).
* Staging must be on the target's file system (`FC-E02` otherwise), so part
  files are hard-linked into the merged store and the store is renamed into
  place without copying. Where hard links are refused (EPERM, ENOTSUP) the
  merge renames files instead, journaling each move first; a link that fails
  with EXDEV is refused (`FC-E02`), never copied.
* Publishing works on one iteration directory. An existing one is refused
  (exit 6) or, with `--overwrite` and only if it is a single store, renamed to
  a trash directory beside it and deleted after success; nothing else in the
  destination root is touched. The tracker is replaced atomically after the
  store is in place, and only if that moves it forward; `publish.json` makes
  every step restartable.
* The merged store's `storage_meta.save_id` derives from the fingerprint, so an
  interrupted and resumed run, or a run with any number of workers, writes the
  same bytes as an uninterrupted one.
* A worker that fails stops the others and fails the run (exit 6); Ctrl-C exits
  130. Staging is kept in both cases, and the same command resumes.
* Tests kill conversions at each commit point through the test-only
  `_FC_TEST_FAULT` hook (`engine/faults.py`), resume them and compare the
  stores byte for byte.

### Private DCP APIs

`engine/torch_compat.py` is the only module that touches DCP internals: the
private `_StorageInfo` entries of `Metadata.storage_data`, the optional
`Metadata.version` field and saver attachments (`mcore_data`,
`all_local_plans`), the checkpointable-tensor protocol
(`__create_chunk_list__`, `__get_tensor_shard__`) used for multi-box reads,
`ReadItem` indices, the no-process-group warning and the `CheckpointException`
wrapping of load and save failures. On the write side it builds `SHARD` and
`BYTE_IO` write items, the global plan (`create_default_global_save_plan`,
which unlike `DefaultSavePlanner` does not validate the chunk grid), relocated
storage entries and new `Metadata` objects (recording `version` only where torch
has that field). `test_torch_contract.py` pins each behaviour the engine relies
on, including the writer's retention of tensors until a data file closes, its
copy of views of larger storages, and the CUDA-stream loader that
`per_thread_copy_ahead=0` avoids; an AST scan keeps other modules off these
internals. The suite passes with torch 2.8, 2.10, 2.12 and 2.14.

### Invariants

* `.metadata` is unpickled once per process by `SafeMetadataUnpickler`. It
  allows an explicit list of globals: the DCP metadata and save-plan classes
  (data classes and enums only, listed in `torch_compat.PICKLED_DCP_CLASSES`),
  a few standard-library types and torch's tensor rebuild helpers, each by its
  exact name (never a dotted path). `megatron.*` and `nvidia_resiliency_ext.*`
  classes become inert stubs, and any other global, including every DCP
  function, is refused (exit 6), never imported or called. Later loads reuse
  the parsed object through `CachedMetadataFileSystemReader`.
* Byte items are read as raw bytes. Common state is decoded for display
  (`inspect`, `verify --common semantic`) with every unknown class stubbed, and
  such objects are never written back; `convert` decodes it with
  `torch.load(weights_only=True)` (Megatron's safe globals in the container, a
  plain allowlist elsewhere) and checks that what it encodes reloads the same
  way.
* Every chunk grid is checked to tile its tensor exactly when the metadata is
  loaded (exit 6 otherwise).
* Reads and writes use `dcp.load`/`dcp.save` with `no_dist=True`: no process
  group is created or used, and nothing touches CUDA (`plan` and `convert` hide
  the GPUs from their own process; workers start with none visible). The first
  read or write adds one process-wide `warnings` filter for torch's
  "torch.distributed is disabled" warning and leaves it in place; that is the
  only process-wide state the Python API changes. A Ctrl-C exits 130.
* `verify` compares bytes, not numbers; digests cover logical bytes, not files.
* Budgets and refusals are checked before any tensor data is read.
* Outputs are deterministic: the same source, target and options give the same
  bytes, whatever the worker count or how often the run was interrupted.

## Tests

Tests live in `tests/unit_tests/tools/checkpoint/format_converter/` and build
their own synthetic stores with a test-only chunked DCP save planner
(`_synthetic.py`); only the T2 and T3 tests import megatron. Expected values
come from independent computations (hand-built tensors, `hashlib` over numpy
bytes, planted corruptions, Megatron's own saves and loads), never only from
the code under test.

| Tier | Content |
|---|---|
| T0 | region algebra (checked exhaustively on small shapes), error and refusal formats, options, detection table, key sections, args view and sanitation, key grammar (layers, experts and SwiGLU halves), logical ids and parts, pool and sentinels, one case per refusal code, architecture derivations, block and expert layouts, gatedness and the SwiGLU split predicate on hand-built key tables, pack sizes (given, defaulted, required, the expert size a job at another tensor-parallel size needs, cross-checked with the manifest, and the padded vocabulary under them), `PartLayout` rank-major maps at tp 1, 2 and 4 against packs built with explicit loops, plan compilation against hand-written pieces, manifest schema, the registry contract, the shortcut-MoE grouping walk on edge patterns (leading and trailing `E`, every predecessor, stage boundaries, refusals), family signatures (presence-based, opaque child names), hybrid and MTP block layouts and their index and family cross-checks, `torch_dist` factory parts in the key grammar, the state-space mixer family (names, spec and the unique width match over the families the names leave, including a Gated DeltaProduct width that solves the Mamba-2 formula with other heads, which is refused without the Gated DeltaProduct spec), Mamba-2 conv names per format (legacy name, both names refused), the TP-bound identity of a whole `torch_dist` tensor at equal and unequal pack sizes, Gated DeltaNet layers under `self_attention` (the variant Megatron builds, the variant's parts and widths, the widths both variants share refused, per-head sizes, a whole `in_proj.bias`, a model of Gated DeltaProduct mixers and Gated DeltaNets, a family under the other module refused, Mamba-2 widths never counted for an `fsdp_dtensor` Gated DeltaNet, Gated DeltaNets under a shortcut pair's child and in an MTP layer, a GPT decoder's Gated DeltaNets only where `experimental_attention_variant` and `linear_attention_freq` place them), and its pack sizes, the names of MTP inner layers (the GPT MTP predicate and live name, every `FC-R09` case, GPT MTP in both formats and a vision-language model, the legacy store's `ModuleRename` through decompose and compose, hybrid MTP never renamed), and multi-latent attention passing the fence fused or not (`FC-R08` in both formats) |
| T1 | synthetic GPT, GPT mixture-of-experts (grouped and sequential experts, stacked and per layer, router buffers in fp32 and int32, latent projections), gated MLPs (SwiGLU split into halves and GEGLU kept fused, packed at tensor-parallel sizes 1, 2 and 4 and routed experts at expert tensor-parallel size 2, against gates and values drawn by hand and packed with explicit loops over ranks; routed experts packed for a job at another tensor-parallel size; `--target-swiglu` refused where it would change the language model's activation, and splitting a GLU vision encoder under a language model without MLPs), vision-language and hybrid vision-language stores (stacked and per-layer blocks, a patch merger's lone `linear_fc1`, tensor-parallel chunk grids) converted both ways against tensors drawn one layer per key and stacked with `torch.stack`, metadata view and safe unpickling, read-once region reads (counting real `torch.load` calls), torch contract pins, digests, `inspect`, `verify` planted corruptions and semantic common state, expected-difference contracts and plan replay, the partial-tensor planner (no copies), every merge failure, unit packing and budgets, staging and resume after crashes at every commit point (byte-identical stores), in-process versus two-worker byte identity, no CUDA in driver or workers, bounded peak RSS, `plan`/`convert` on synthetic training saves in every direction, shortcut-MoE hybrid stores (sparse `torch_dist` indices, pipeline stages, other child names, refusals) and hybrid MTP stores (shortcut pairs in a repeated MTP layer, the legacy MTP pattern argument, gated MTP MLPs split and checked as the language model's) converted both ways against tensors drawn under hand-written indices, Gated DeltaProduct hybrid stores (components drawn one by one, in the order written out from the mixer, against packs built with explicit loops over ranks at tensor-parallel sizes 1, 2 and 4; biases; fp32 Megatron-FSDP saves; stores without the spec or of other dimensions refused), Mamba-2 hybrid stores likewise (heads from `hidden_size`, a legacy convolution name, a whole `in_proj.bias` at equal and other pack sizes; converters that swap `z` and `x`, slice the parts' concatenation by rank or misname the convolution fail them), Gated DeltaNet hybrid and GPT linear-attention stores likewise (both variants, a hybrid of Gated DeltaProduct mixers and Gated DeltaNets, a whole `in_proj.bias`; converters that swap `value` and `z` or slice the parts' concatenation by rank fail them, and one that takes either variant for the other is refused) and a vision-language store of every family with either mixer or Gated DeltaNets (mixers under a shortcut pair's child, sparse indices, experts, a stacked gated vision encoder, all packed at TP 1 and 2) converted both ways, GPT MTP stores (a stacked GPT with two MTP layers both ways, one repeated MTP layer (`--mtp-use-repeated-layer`) both ways, a legacy `mtp_model_layer` store, also with routed experts in its MTP layer, against hand-renamed tensors and by plan replay, the `--fsdp-gpt-mtp-inner-name` store read back, a mixed store refused) and multi-latent attention stores (a stacked GPT with and without `--mla-down-proj-fusion`, `+-*+`, a `+E-+/+E` shortcut-MoE hybrid, an `fsdp_dtensor` store, a legacy fused store refused) converted both ways, CLI exit codes, import boundary |
| T2 | the common-state codec against Megatron's `load_common_state_dict`, the rewritten integrity manifest against Megatron's verifier, the pattern parser and the verified spec names against Megatron's objects, the grouped-tensor-parallel refusal against the weight-shard fields Megatron's `validate_args` stores, the stacking predicate against `TransformerBlock.sharded_state_dict` itself on int and list layer frequencies and `content_metadata` flags, the expert naming against `get_expert_index_from_key` and the experts modules Megatron's builders build, every expert tensor of a live MoE model's torch_dist sharded state dict against the name and value `handle_experts_in_state_dict` gives it at expert-parallel sizes 1 and 2, the SwiGLU split predicate against the state dict `preprocess_fsdp_dtensor_state_dict` builds for live Megatron-FSDP models (a SwiGLU MoE GPT with shared experts at expert-parallel sizes 1 and 2; a LLaVA with a SwiGLU language model, a plain vision encoder and a gated projection), the fused gated `linear_fc1` of live Megatron-FSDP models at TP 2 and expert TP 2 against the rank-major pack of the halves Megatron's own split gives, the shortcut-MoE grouping against `group_layers_into_shortcut_blocks` stage by stage and `SUPPORTED_SHORTCUT_PREDECESSORS`, and every tensor of a live shortcut-MoE hybrid model with an MTP layer (its torch_dist sharded state dict, sparse indices) against the name and value its state dict gives it, the Gated DeltaProduct parts against `_get_in_proj_checkpoint_split_layout` and `_get_conv_checkpoint_split_layout` for 1 to 3 Householder steps at tensor-parallel sizes 1, 2 and 4 (and Megatron's Householder and Mamba-2 `expand` defaults), a live Gated DeltaProduct hybrid model's `torch_dist` keys (split factories applied) against the converter's mixers and the model's state-dict names at TP 1 and 2, and a real TP 2 mixer's fused `in_proj` and `conv1d` weights and biases against the rank-major pack of the canonical parts gathered from its own factory sub-tensors, and the same for Mamba-2 (`MambaMixer`'s split factories at TP 1 and 2 against the layouts and `partition_sizes`, a live hybrid model's keys with `mamba_num_heads` set and unset, a real TP 2 mixer's `in_proj.weight`, `conv1d_weight` and `conv1d_bias`), and for Gated DeltaNets of both variants (`GatedDeltaNet` and `GatedDeltaNet2` split factories at TP 1 and 2 with both biases, a live hybrid model's and a live linear-attention GPT model's keys at TP 1 and 2, a real TP 2 module's fused tensors and whole `in_proj.bias`) (container); `MambaMixer.sharded_state_dict`'s split parts, unsplit `in_proj.bias` and convolution names, the Gated DeltaNets' split names and sections, per-head sizes, unsplit `in_proj.bias`, hybrid `self_attention` module and `gdn2` choice, the `fsdp_dtensor` preprocess handler set, the hybrid layer symbols, the Gated DeltaNet variant names, the SwiGLU names and gate of `handle_swiglu_in_state_dict`, the arguments that gate the language model, the MTP inner layer names (`MTP_INNER_LAYER`, `MTP_INNER_LAYER_CHECKPOINT_NAME`) with the conditions under which Megatron renames them, model and optimizer state alike, and multi-latent attention's unfused names in both formats (`MLA_FUSED_DOWN_PROJ`, `MLA_UNFUSED_DOWN_PROJS`, the handler applied unconditionally, the absorbed layer norm moved to `input_layernorm`, `FusedMLASelfAttention.sharded_state_dict` and the GPT and hybrid fused layer specs' key maps) scanned from Megatron's source (`ast`, host) |
| T3 | a real Megatron save rewritten and loaded strictly into a fresh model, bit-exact at TP 1 and 2; a GPT mixture-of-experts model (grouped experts at expert-parallel size 1, sequential and grouped at 2; router expert biases compared too) and a hybrid `*E*E` model; a hybrid model, a dense GPT and a tiny LLaVA (GPT language model, CLIP vision encoder, MLP projection; both `TransformerBlock`s stacked in `torch_dist`) through `torch_dist` -> `fsdp_dtensor` -> `torch_dist` with Megatron's own saves and strict loads, every store checked against a real save under its contract, and the GPT and LLaVA `fsdp_dtensor` saves also converted with `--torch-dist-layout per-layer` and loaded strictly (TP 1; TP 2 with `MCORE_FORMAT_CONVERTER_EXTENDED=1`); a GEGLU GPT at TP 2 (packed with `--fsdp-tp 2`) and a LLaVA whose vision encoder is gated under a plain language model (TP 1; TP 2 extended) through the same chain; for SwiGLU models, whose Megatron-FSDP state dicts only a bf16 job builds on current Megatron (the fp32 failure is a strict expected failure) and a bf16 job cannot load without optimizer state, a chain of what that permits (the converted store against a real bf16 Megatron-FSDP save's keys, shapes and dtypes; that save converted back, loaded strictly and compared parameter for parameter; the round trip against the source): a SwiGLU GPT (TP 1; TP 2 extended) and a SwiGLU MoE GPT at expert-parallel size 2 (grouped with shared experts, sequential with biases); a shortcut-MoE hybrid model with an MTP layer: a real `torch_dist` save converted and checked against Megatron-FSDP's save handlers applied to the live model, that handler output converted back, loaded strictly and compared parameter for parameter, and the round trip against the source (contract `D`); hybrid MTP models (two depths, MTP layers with experts, one repeated MTP layer) through the whole chain; a Gated DeltaProduct hybrid model through the whole chain at TP 1 (at TP 2 a strict expected failure: Megatron-FSDP cannot hold the mixer's `A_log`, `dt_bias` and `norm.weight`), and a real TP 2 `torch_dist` save converted with `--fsdp-tp 2`, whose store must hold the live model's rank-major fused tensors, converted back, loaded strictly and compared parameter for parameter; Mamba-2 hybrid models (`M*-`, `M*E`) through the whole chain at TP 1 (`M*-` at TP 2 with `--fsdp-tp 2` extended); Gated DeltaNet hybrid models (`G*-`, both variants) and a linear-attention GPT model through the whole chain at TP 1 (at TP 2 with `--fsdp-tp 2`, `G*E`, and a Gated DeltaNet 2 GPT model, extended); a GPT with two MTP layers and a GPT with multi-latent attention through the whole chain (TP 1; TP 2 extended); a GPT with `--mla-down-proj-fusion` and MTP layers through the bf16 chain (the fp32 Megatron-FSDP failure of `handle_mla_down_proj_in_state_dict` is an expected failure, not strict, so that a fix in Megatron does not fail this suite); a real `torch_dist` save converted with `--fsdp-gpt-mtp-inner-name mtp_model_layer` against a real Megatron-FSDP save's keys renamed by hand, converted back, loaded strictly and compared parameter for parameter (container with GPUs) |

Host run (skips the megatron-importing root `conftest.py`):

```bash
python3 -m pytest --confcutdir=tests/unit_tests/tools/checkpoint/format_converter \
    -p no:cacheprovider -q tests/unit_tests/tools/checkpoint/format_converter
```

In CI the package runs in the multi-rank unit-test bucket. Apart from the
real-save tests (T3), which use the process groups of the root test harness
and skip without it, the tests never create or destroy a process group; they
use per-test temporary directories and run subprocess tests (with launcher
variables removed and no visible GPU) on local rank 0 only, so they pass at 1,
2 and 8 ranks.

## Roadmap

The tool grows in small, independently useful steps; each keeps every
unsupported input refused with a stable code.

1. Streaming DCP `inspect` and `verify` (done).
2. Streaming DCP writer engine (partial-chunk save planner, validated metadata
   merge, staging, resume and atomic publish, subprocess workers) with a
   weights-only `torch_dist` rewrite as its first user; `plan` and `convert`
   (done).
3. Cross-format core: logical key space, rule registry, refusal fence, the
   Megatron-FSDP v1 adapter, per-layer hybrid models, expected-difference
   contracts and `verify --map convert` (done).
4. Stacked `TransformerBlock` layers and multi-root vision-language models,
   with the user guide (done).
5. Mixture-of-experts layers: routed experts in grouped and sequential
   naming, router buffers (done).
6. GLU `linear_fc1`: SwiGLU split and unsplit, rank-major packs (done).
7. Shortcut-MoE layer numbering and hybrid multi-token prediction (done).
8. Gated DeltaProduct fused projections (done).
9. Mamba-2 fused projections (done).
10. Gated DeltaNet fused projections (done).
11. GPT multi-token prediction and multi-latent attention (this version).
12. Optimizer state from `fsdp_dtensor` to `torch_dist`.
13. Optimizer state from `torch_dist` to `fsdp_dtensor`.
14. A byte-range reader for stored chunks larger than memory.
15. Route `checkpoint_inspector.py convert-torch-dist-to-fsdp-dtensor` to this tool.
16. An opt-in train, convert and resume test suite.
