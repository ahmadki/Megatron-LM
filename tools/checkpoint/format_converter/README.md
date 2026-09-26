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

This version inspects and verifies stores and rewrites `torch_dist` stores
with model weights only. Conversion between the two formats lands in later
changes (see [Roadmap](#roadmap)); this file is the living design document of
the package.

## Commands

Run from the repository root (or with it on `PYTHONPATH`):

```bash
python -m tools.checkpoint.format_converter inspect SRC
python -m tools.checkpoint.format_converter verify A B
python -m tools.checkpoint.format_converter verify A --oracle 'digests_rank*.json' --oracle-key-map map.py
python -m tools.checkpoint.format_converter plan SRC --to torch_dist
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
| `--oracle GLOB` | verify `A` against these digest files instead of `B` |
| `--oracle-key-map PY` | Python file defining `to_store_key(oracle_key, file_index)` |
| `--report PATH.jsonl` | write every finding as JSON lines |
| `--iteration N`, `--max-item-bytes SIZE`, `--workers N` | as for `inspect` |

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

`convert SRC DST --to torch_dist` rewrites a `torch_dist` store as a
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
  is dropped, other files are listed but not copied;
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
| `--to torch_dist` | target format (required) |
| `--from {torch_dist,fsdp_dtensor}` | format to assume when the store has no format markers |
| `--iteration N` | iteration to open under a checkpoint root |
| `--drop-common-key KEY` | drop this unknown common-state key instead of refusing (repeatable) |
| `--workers N` | worker processes; `0` runs the units in the driver (default `min(4, cores/8)`) |
| `--unit-bytes SIZE` | target peak memory of one unit (default `4GiB`) |
| `--max-unit-bytes-hard SIZE` | largest memory one unit may need, else exit 8 (default 0.6 x `MemAvailable` / workers) |
| `--writer-threads N` | data files written per unit (default 4) |
| `--max-host-bytes SIZE` | host-memory budget of the whole conversion, else exit 8 (default 0.8 x `MemAvailable`) |
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
| `FC-R06` | grouped tensor parallelism (a weight rematerialization size above 1, or more weight shards than tensor-parallel ranks, for the dense or the expert layers), which may pad stored tensors; as many shards as tensor-parallel ranks, which every Megatron job saves, is plain tensor parallelism |
| `FC-R07` | an `fsdp_dtensor` source of an FP8 delayed-scaling model converted to `torch_dist`, whose loader needs the scaling state (`--allow-missing-fp8-extra-state` writes the store without it) |
| `FC-R11` | the layer indices a store holds contradict the hybrid layer pattern (or `args.num_layers`); the message lists the expected and stored indices |
| `FC-R13` | unknown layout: another `sharded_backend`, no format markers (unless `--from`), mixed model prefixes, a legacy `mp_rank_*` checkpoint, an unknown byte item, an unknown common-state key, or model keys under an unknown top-level module |
| `FC-R14` | model tensors stored in the legacy N-D reformulated layout (`mcore_data`); load with current Megatron and save again |
| `FC-R17` | a store saved with `singleton_local_shards` |
| `FC-R18` | a ModelOpt state (`modelopt_state` in the common state or next to the store) |
| `FC-R19` | `fsdp_dtensor` store with split fused-projection sub-keys (`in_proj`/`conv1d` `.weight.<part>`) |
| `FC-R20` | layer types no rule is verified for: hybrid layer symbols `D`, `C`, `H`, `W` (or unknown ones), or an `experimental_attention_variant` other than the Gated DeltaNet ones |
| `FC-P01` | pending: non-hybrid `TransformerBlock` layers (GPT or vision decoders, stacked keys) |
| `FC-P02` | pending: models with several roots (`language_model.`, `vision_model.`, `vision_projection.`) |
| `FC-P03` | pending: mixture-of-experts layers (`num_experts`, symbol `E`, expert or router keys) |
| `FC-P04` | pending: shortcut-MoE layer numbering (`moe_shortcut_connection`) and hybrid multi-token prediction (`/` in the pattern) |
| `FC-P05` | pending: Gated DeltaProduct mixers (its spec or its `in_proj`/`conv1d` parts) |
| `FC-P06` | pending: gated `linear_fc1` that `fsdp_dtensor` keeps fused (`swiglu` off) |
| `FC-P07` | pending: `linear_fc1` split into SwiGLU halves (`swiglu` on) |
| `FC-P08` | pending: Mamba-2 mixers (symbol `M` without the Gated DeltaProduct spec) |
| `FC-P09` | pending: Gated DeltaNet and linear attention (symbol `G`, a Gated DeltaNet `experimental_attention_variant`) |
| `FC-P10` | pending: GPT multi-token prediction and multi-latent attention (symbol `+`) |
| `FC-P11` | pending: converting optimizer state from `fsdp_dtensor` to `torch_dist` |
| `FC-P12` | pending: converting optimizer state from `torch_dist` to `fsdp_dtensor` |
| `FC-A05` | `--from` disagrees with the detected format |
| `FC-A06` | the source's `conversion_manifest.json` describes another store (its `.metadata` digest differs); `--ignore-manifest` converts without restoring what it records |
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
| L2 | `formats/`, `verify/`, `rules/` (later) | L0, L1 | same |
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
   by one standalone buffer. Byte items are encoded in the driver.
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
| T0 | region algebra (checked exhaustively on small shapes), error and refusal formats, options, detection table, key sections, args view and sanitation |
| T1 | metadata view and safe unpickling, read-once region reads (counting real `torch.load` calls), torch contract pins, digests, `inspect`, `verify` planted corruptions and semantic common state, the partial-tensor planner (no copies), every merge failure, unit packing and budgets, staging and resume after crashes at every commit point (byte-identical stores), in-process versus two-worker byte identity, no CUDA in driver or workers, bounded peak RSS, `plan`/`convert` on a synthetic training save, CLI exit codes, import boundary |
| T2 | the common-state codec against Megatron's `load_common_state_dict`, and the rewritten integrity manifest against Megatron's verifier (container) |
| T3 | a real Megatron save rewritten and loaded strictly into a fresh model, bit-exact at TP 1 and 2 (container with GPUs) |

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
   (this version).
3. Cross-format core: logical key space, rule registry, refusal fence, the
   Megatron-FSDP v1 adapter, per-layer hybrid models, `verify --map convert`.
4. Stacked `TransformerBlock` layers and multi-root vision-language models,
   with the user guide.
5. MoE experts (grouped and sequential naming, router buffers).
6. GLU `linear_fc1`: SwiGLU split and unsplit, rank-major packs.
7. Shortcut-MoE layer numbering and hybrid multi-token prediction.
8. Gated DeltaProduct fused projections.
9. Mamba-2 fused projections.
10. Gated DeltaNet fused projections.
11. GPT multi-token prediction and multi-latent attention.
12. Optimizer state from `fsdp_dtensor` to `torch_dist`.
13. Optimizer state from `torch_dist` to `fsdp_dtensor`.
14. A byte-range reader for stored chunks larger than memory.
15. Route `checkpoint_inspector.py convert-torch-dist-to-fsdp-dtensor` to this tool.
16. An opt-in train, convert and resume test suite.
