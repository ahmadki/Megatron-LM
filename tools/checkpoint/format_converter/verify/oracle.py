# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Verification of a store against SHA-256 digest files produced elsewhere.

An oracle is one or more digest files (``{key: {shape, dtype, sha256}}``, the
format ``inspect --digests`` writes) matched by a glob pattern. Files are
taken in sorted order; a file's position is its ``file_index``. Oracle keys
are mapped to store keys by an optional Python file that defines::

    def to_store_key(oracle_key: str, file_index: int) -> str | None: ...

(``None`` skips the key). Without it, keys map to themselves. This lets one
oracle written per rank with rank-local names (for example local expert
numbers) be checked against a store with global names.

After mapping, a store key contributed by several oracle entries must get the
same record from each ("oracle inconsistent" otherwise). The store is then
checked for missing and extra keys, shapes and dtypes (every mode), and
SHA-256 of the logical bytes (``full`` and ``sample:N``).
"""

from __future__ import annotations

import glob
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..engine.hashing import TensorDigest, compute_digests, read_digest_file
from ..engine.reader import SourceReader
from ..errors import InternalError, UsageError
from ..formats.detect import Store
from ..options import VerifyOptions
from .report import Finding, FindingKind, VerifyReport, sort_findings

KeyMap = Callable[[str, int], str | None]
KEY_MAP_FUNCTION = "to_store_key"


@dataclass(frozen=True, slots=True)
class OracleEntry:
    """An expected digest and where it came from.

    Attributes:
        digest: Expected shape, dtype and SHA-256.
        source: ``<file name>:<oracle key>`` of the entry.
    """

    digest: TensorDigest
    source: str


def oracle_files(pattern: str) -> list[Path]:
    """The digest files matching ``pattern``, sorted (position = file index).

    Raises:
        UsageError: Nothing matches.
    """
    files = sorted(Path(p) for p in glob.glob(pattern))
    if not files:
        raise UsageError(f"--oracle {pattern!r} matches no file")
    return files


def load_key_map(path: Path | None) -> KeyMap:
    """The ``to_store_key`` function of the Python file ``path`` (identity if ``None``).

    Raises:
        UsageError: The file cannot be loaded or defines no callable ``to_store_key``.
    """
    if path is None:
        return lambda oracle_key, file_index: oracle_key
    spec = importlib.util.spec_from_file_location("format_converter_oracle_key_map", path)
    if spec is None or spec.loader is None:
        raise UsageError(f"--oracle-key-map {path} is not a loadable Python file")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as err:  # the map is user code; report any failure as a usage error
        raise UsageError(f"--oracle-key-map {path} failed to load: {err!r}") from err
    key_map = getattr(module, KEY_MAP_FUNCTION, None)
    if not callable(key_map):
        raise UsageError(f"--oracle-key-map {path} must define {KEY_MAP_FUNCTION}(key, file_index)")
    return key_map


def build_oracle(
    files: list[Path], key_map: KeyMap
) -> tuple[dict[str, OracleEntry], list[Finding]]:
    """Merge digest files into expected digests by store key.

    Returns:
        The expected digests and one "oracle inconsistent" finding per store
        key whose contributing entries disagree.
    """
    oracle: dict[str, OracleEntry] = {}
    conflicts: dict[str, list[str]] = {}
    for file_index, path in enumerate(files):
        for oracle_key, digest in read_digest_file(path).items():
            store_key = _map_key(key_map, oracle_key, file_index)
            if store_key is None:
                continue
            entry = OracleEntry(digest, f"{path.name}:{oracle_key}")
            prior = oracle.setdefault(store_key, entry)
            if prior.digest != digest:
                conflicts.setdefault(store_key, [prior.source]).append(entry.source)
    findings = [
        Finding(
            FindingKind.ORACLE_INCONSISTENT, key, f"oracle entries disagree: {', '.join(sources)}"
        )
        for key, sources in conflicts.items()
    ]
    return oracle, findings


def verify_against_oracle(store: Store, opts: VerifyOptions) -> VerifyReport:
    """Check ``store`` against the digest files matching ``opts.oracle``.

    Raises:
        UsageError: The pattern matches nothing, or the key map is unusable or
            fails on an oracle key.
        IntegrityError: A digest file is malformed.
        BudgetExceeded: Before any read, if a tensor cannot be hashed within
            ``opts.max_item_bytes``.
    """
    if opts.oracle is None:
        raise InternalError("verify_against_oracle() called without an oracle pattern")
    pattern = opts.oracle
    files = oracle_files(pattern)
    oracle, findings = build_oracle(files, load_key_map(opts.oracle_key_map))
    expected = {k: e for k, e in oracle.items() if not opts.ignores(k)}
    tensors = {k: e for k, e in store.view.tensors.items() if not opts.ignores(k)}
    findings += [
        Finding(FindingKind.ONLY_IN_A, k, "not in the oracle") for k in tensors.keys() - expected
    ]
    findings += [
        Finding(FindingKind.ONLY_IN_B, k, f"missing from the store ({expected[k].source})")
        for k in expected.keys() - tensors.keys()
    ]
    comparable = []
    for key in sorted(tensors.keys() & expected.keys()):
        entry, want = tensors[key], expected[key]
        if entry.shape != want.digest.shape:
            findings.append(
                Finding(
                    FindingKind.SHAPE,
                    key,
                    f"store {entry.shape} vs oracle {want.digest.shape} ({want.source})",
                )
            )
        elif str(entry.dtype) != want.digest.dtype:
            findings.append(
                Finding(
                    FindingKind.DTYPE,
                    key,
                    f"store {entry.dtype} vs oracle {want.digest.dtype} ({want.source})",
                )
            )
        else:
            comparable.append(key)
    value_keys = opts.value_keys(comparable)
    reader = SourceReader(store.view, max_item_bytes=opts.max_item_bytes)
    digests, stats = compute_digests(reader, value_keys, workers=opts.workers)
    findings += [
        Finding(
            FindingKind.VALUE,
            key,
            f"sha256 {digest.sha256} != oracle {expected[key].digest.sha256} "
            f"({expected[key].source})",
        )
        for key, digest in digests.items()
        if digest.sha256 != expected[key].digest.sha256
    ]
    return VerifyReport(
        a=store.iter_dir,
        reference=f"oracle {pattern} ({len(files)} files, {len(oracle)} keys)",
        mode=str(opts.mode),
        findings=sort_findings(findings),
        keys_compared=len(tensors.keys() & expected.keys()),
        values_compared=len(value_keys),
        byte_items_compared=0,
        read_stats=stats,
    )


def _map_key(key_map: KeyMap, oracle_key: str, file_index: int) -> str | None:
    try:
        store_key = key_map(oracle_key, file_index)
    except Exception as err:  # the map is user code; report any failure as a usage error
        raise UsageError(
            f"{KEY_MAP_FUNCTION}({oracle_key!r}, {file_index}) raised {err!r}",
            hint="fix the --oracle-key-map file",
        ) from err
    if store_key is not None and not isinstance(store_key, str):
        raise UsageError(
            f"{KEY_MAP_FUNCTION}({oracle_key!r}, {file_index}) returned "
            f"{type(store_key).__name__}; expected str or None"
        )
    return store_key
