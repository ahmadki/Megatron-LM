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

This first version inspects and verifies stores. Conversion between the two
formats lands in later changes (see [Roadmap](#roadmap)); this file is the
living design document of the package.

## Commands

Run from the repository root (or with it on `PYTHONPATH`):

```bash
python -m tools.checkpoint.format_converter inspect SRC
python -m tools.checkpoint.format_converter verify A B
python -m tools.checkpoint.format_converter verify A --oracle 'digests_rank*.json' --oracle-key-map map.py
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
optimizer other, common, rng, extra_state, other), dtype and chunks-per-tensor
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
Values are compared through `uint8` views, so `-0.0` versus `0.0` and different
NaN payloads count as differences. The two stores may use different chunk
grids unless `--layout` is given.

| Flag | Meaning |
|---|---|
| `--mode MODE` | `meta` (keys, shapes, dtypes), `full` (all values, default) or `sample:N` (values of `N` keys drawn with `--seed`) |
| `--seed SEED` | seed of `sample:N` (default 0) |
| `--layout` | also require identical chunk grids (not with `--oracle`) |
| `--common {skip,bytes}` | `bytes` (default): byte items must be byte-identical; `skip`: ignore them |
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

## Exit codes and refusal codes

| Exit | Meaning |
|---|---|
| 0 | ok |
| 1 | internal error |
| 2 | usage error |
| 3 | refused (`FC-R`, `FC-P`, `FC-E` codes) |
| 4 | ambiguous, needs a flag (`FC-A` codes) |
| 5 | verification mismatch |
| 6 | integrity or I/O error (corrupt or truncated store, bad tracker or digest file) |
| 7 | staging fingerprint mismatch (conversion; not used yet) |
| 8 | memory budget exceeded |
| 130 | interrupted |

Refusals print one line, `FC-xxx: <message> [keys...] hint: <what to do>`:

| Code | Refused input |
|---|---|
| `FC-R01` | Megatron-FSDP v2 store (`model.<fqn>` keys without `module.`) |
| `FC-R02` | virtual-pipeline sections (`model<i>.` or `optimizer.<i>.` keys) |
| `FC-R13` | unknown layout: another `sharded_backend`, no format markers (unless `--from`), mixed model prefixes, or a legacy `mp_rank_*` checkpoint |
| `FC-R19` | `fsdp_dtensor` store with split fused-projection sub-keys (`in_proj`/`conv1d` `.weight.<part>`) |
| `FC-A05` | `--from` disagrees with the detected format |
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

## Design

### Layering and import boundary

| Layer | Package | May import | Runs on |
|---|---|---|---|
| L0 | `engine/` | stdlib, torch, `torch.distributed.checkpoint` | host Python 3.10 with torch 2.8 and newer, containers, CI |
| L1 | `core/` (later) | L0; pure, no I/O | same |
| L2 | `formats/`, `verify/`, `rules/` (later) | L0, L1 | same |
| L3 | `mcore_state/` (later) | megatron, imported inside functions | container, CI |
| L4 | `cli.py`, `api.py`, `options.py` | L0 to L2 eagerly, L3 lazily | same as the layers it uses |

L0 to L2 avoid `typing.Self`, `StrEnum` and PEP 695 syntax so they import on
Python 3.10. `test_import_boundary.py` imports every module, and runs
`inspect` and `verify`, in a fresh interpreter whose import system refuses
`megatron`, `transformer_engine` and `nvidia_resiliency_ext`. Importing the
package itself does not import torch; the Python API (`inspect_checkpoint`,
`verify_checkpoint`) is exported lazily.

### Private DCP APIs

`engine/torch_compat.py` is the only module that touches DCP internals: the
private `_StorageInfo` entries of `Metadata.storage_data`, the optional
`Metadata.version` field and saver attachments (`mcore_data`,
`all_local_plans`), the checkpointable-tensor protocol
(`__create_chunk_list__`, `__get_tensor_shard__`) used for multi-box reads,
`ReadItem` indices, the no-process-group warning and the `CheckpointException`
wrapping of load failures. `test_torch_contract.py` pins each behaviour the
engine relies on and an AST scan keeps other modules off these internals. The
suite passes with torch 2.8, 2.10 and 2.12.

### Invariants

* `.metadata` is unpickled once per process by `SafeMetadataUnpickler`. It
  allows an explicit list of globals: the DCP metadata and save-plan classes
  (data classes and enums only, listed in `torch_compat.PICKLED_DCP_CLASSES`),
  a few standard-library types and torch's tensor rebuild helpers, each by its
  exact name (never a dotted path). `megatron.*` and `nvidia_resiliency_ext.*`
  classes become inert stubs, and any other global, including every DCP
  function, is refused (exit 6), never imported or called. Later loads reuse
  the parsed object through `CachedMetadataFileSystemReader`.
* Byte items are read as raw bytes. They are decoded only for display
  (`inspect`), with every unknown class stubbed, and such objects are never
  written back.
* Every chunk grid is checked to tile its tensor exactly when the metadata is
  loaded (exit 6 otherwise).
* Reads use `dcp.load(no_dist=True)`: no process group is created or used, and
  nothing touches CUDA. The first read adds one process-wide `warnings` filter
  for torch's "torch.distributed is disabled" warning and leaves it in place;
  that is the only process-wide state the Python API changes. A Ctrl-C during
  a read exits 130.
* `verify` compares bytes, not numbers; digests cover logical bytes, not files.
* Budgets are checked before any tensor data is read.

## Tests

Tests live in `tests/unit_tests/tools/checkpoint/format_converter/` and build
their own synthetic stores with a test-only chunked DCP save planner
(`_synthetic.py`); they import no megatron. Expected values come from
independent computations (hand-built tensors, `hashlib` over numpy bytes,
planted corruptions), never only from the code under test.

| Tier | Content |
|---|---|
| T0 | region algebra (checked exhaustively on small shapes), error and refusal formats, options, detection table |
| T1 | metadata view and safe unpickling, read-once region reads (counting real `torch.load` calls), torch contract pins, digests, `inspect`, `verify` planted corruptions, CLI exit codes, import boundary |

Host run (skips the megatron-importing root `conftest.py`):

```bash
python3 -m pytest --confcutdir=tests/unit_tests/tools/checkpoint/format_converter \
    -p no:cacheprovider -q tests/unit_tests/tools/checkpoint/format_converter
```

In CI the package runs in the multi-rank unit-test bucket. The tests never
create or destroy a process group, use per-test temporary directories, and run
subprocess tests (with launcher variables removed and no visible GPU) on local
rank 0 only, so they pass at 1, 2 and 8 ranks.

## Roadmap

The tool grows in small, independently useful steps; each keeps every
unsupported input refused with a stable code.

1. Streaming DCP `inspect` and `verify` (this version).
2. Streaming DCP writer engine (partial-chunk save planner, validated metadata
   merge, staging, resume and atomic publish, subprocess workers) with a
   weights-only `torch_dist` rewrite as its first user; `plan` and `convert`.
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
