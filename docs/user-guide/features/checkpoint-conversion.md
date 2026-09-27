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
| GPT with dense, non-gated MLP layers (no MoE, SwiGLU, MTP or MLA; `TransformerBlock` decoder, stacked or per layer in `torch_dist`) | yes | yes | dropped | as above | synthetic stores; stacking predicate pinned to Megatron; real-save chain at TP 1 and 2 |
| Vision-language models in the LLaVA layout (a GPT or hybrid language model above, a vision encoder `vision_model.decoder`, a projection) | yes | yes | dropped | as above | synthetic stores; real-save chain of a tiny LLaVA (GPT language model, CLIP encoder) at TP 1 and 2; hybrid language model on synthetic stores only |

Model families that later versions add are refused before anything is read,
with a pending-family code:

| Code | Family |
|---|---|
| `FC-P03` | mixture-of-experts layers |
| `FC-P04` | shortcut-MoE layer numbering and hybrid multi-token prediction |
| `FC-P05` | Gated DeltaProduct mixers |
| `FC-P06`, `FC-P07` | gated `linear_fc1` (SwiGLU), kept fused or split into halves |
| `FC-P08` | Mamba-2 mixers |
| `FC-P09` | Gated DeltaNet and linear attention |
| `FC-P10` | GPT multi-token prediction and multi-latent attention |
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
hybrid decoder is stored per layer in `torch_dist` too. A `TransformerBlock`
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
buffers) keep it.

### Tensor parallelism

Converted stores hold full tensors (tensor-parallel shards of the source are
read into place), so they load at any tensor-parallel size the model allows.

### Common state and arguments

The common state is carried over: `args`, `iteration`, `checkpoint_version`
and `num_floating_point_operations_so_far` (an `fsdp_dtensor` source without
one of the first three is refused, `FC-R13`). The arguments are sanitized so
they describe the converted store, changing only existing fields:
`no_save_optim` and `no_save_rng` are set, `ckpt_format` and
`use_megatron_fsdp` name the target format, and an `fsdp_dtensor` store gets
one pipeline stage (the pipeline and virtual-pipeline fields reset, `|`
removed from the hybrid pattern). Transformer Engine `_extra_state` is
dropped: `fsdp_dtensor` does not store it.

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
| `FC-R06` | grouped tensor parallelism (weight rematerialization, or more weight shards than tensor-parallel ranks) |
| `FC-R07` | an FP8 delayed-scaling model converted to `torch_dist` (`--allow-missing-fp8-extra-state` writes it without the scaling state) |
| `FC-R11` | layer indices that contradict the hybrid layer pattern |
| `FC-R12` | inconsistently stacked layers, non-contiguous layer indices, or layers that differ where they would be stacked |
| `FC-R13` | an unknown layout, key, top-level module, byte item or common-state key |
| `FC-R14` | the legacy N-D reformulated layout (load with current Megatron and save again) |
| `FC-R17` | a store saved with `singleton_local_shards` |
| `FC-R18` | a ModelOpt state |
| `FC-R19` | an `fsdp_dtensor` store with split fused-projection keys |
| `FC-R20` | hybrid layer symbols or attention variants no rule is verified for |
| `FC-A02` | a layer list of unknown kind converted to `torch_dist` (`--block-layout`) |
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
- **Pending families.** Mixture-of-experts, SwiGLU, state-space mixers,
  Gated DeltaNet, multi-token prediction and multi-latent attention are
  refused until the versions that add them (see the
  [Support Matrix](#support-matrix)).
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
