# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The common-state codec in plain, stub and (in the Megatron container) megatron mode.

The plain codec's FC-E01 refusal of Megatron classes is tested in
``test_import_boundary.py``, in an interpreter where Megatron cannot be imported.
"""

from __future__ import annotations

import argparse
import enum
import io
import json
import sys

import pytest
import torch

from tools.checkpoint.format_converter.engine.metadata import Stub
from tools.checkpoint.format_converter.engine.part_writer import write_part
from tools.checkpoint.format_converter.engine.pieces import BytePiece
from tools.checkpoint.format_converter.errors import (
    ExitCode,
    IntegrityError,
    InternalError,
    RefusedError,
)
from tools.checkpoint.format_converter.formats.base import WHOLE, CommonLayout, RawCommon
from tools.checkpoint.format_converter.mcore_state import common_state
from tools.checkpoint.format_converter.mcore_state.common_state import CodecMode, decode, encode

from .._synthetic import TORCH_DIST_CONFIG, default_args, torch_save_bytes

ENUMS = "megatron.core.transformer.enums"


class FakeAttnBackend(enum.Enum):
    """Pickled under Megatron's name ``AttnBackend``."""

    auto = 5


class Opaque:
    """A class no codec allow-lists."""


COMMON = {
    "args": default_args(params_dtype=torch.bfloat16, no_save_optim=False),
    "iteration": 7,
    "checkpoint_version": 3.0,
    "num_floating_point_operations_so_far": 1.5e12,
}


def _raw(obj, layout=CommonLayout.WRAPPED, aliases=None) -> RawCommon:
    return RawCommon("test payload", layout, {WHOLE: torch_save_bytes(obj, aliases)})


def _loaded(payloads):
    return {key: torch.load(io.BytesIO(p), weights_only=False) for key, p in payloads.items()}


def test_plain_round_trip_in_every_layout():
    for layout, wrapped in ((CommonLayout.WRAPPED, [COMMON]), (CommonLayout.DICT, COMMON)):
        decoded = decode(_raw(wrapped, layout), CodecMode.PLAIN)
        assert decoded.state == COMMON and decoded.mode is CodecMode.PLAIN
        payloads = encode(decoded.state, layout, mode=CodecMode.PLAIN, label="out")
        assert _loaded(payloads) == {WHOLE: wrapped}
    per_key = RawCommon(
        "items", CommonLayout.PER_KEY, {k: torch_save_bytes(v) for k, v in COMMON.items()}
    )
    decoded = decode(per_key, CodecMode.PLAIN)
    assert decoded.state == COMMON
    payloads = encode(decoded.state, CommonLayout.PER_KEY, mode=CodecMode.PLAIN, label="out")
    assert _loaded(payloads) == COMMON


def test_encoding_is_deterministic():
    first = encode(COMMON, CommonLayout.WRAPPED, mode=CodecMode.PLAIN, label="a")
    assert encode(dict(COMMON), CommonLayout.WRAPPED, mode=CodecMode.PLAIN, label="b") == first


def test_layout_mismatches_are_integrity_errors():
    with pytest.raises(IntegrityError, match="must hold a one-element list, found dict"):
        decode(_raw(COMMON), CodecMode.PLAIN)
    with pytest.raises(IntegrityError, match="does not hold a common-state dict"):
        decode(_raw([[1]]), CodecMode.PLAIN)
    with pytest.raises(IntegrityError, match="cannot decode test payload"):
        decode(RawCommon("test payload", CommonLayout.WRAPPED, {WHOLE: b"junk"}), CodecMode.PLAIN)


@pytest.mark.parametrize(
    "layout, keys",
    [
        (CommonLayout.WRAPPED, {"args"}),
        (CommonLayout.DICT, {WHOLE, "args"}),
        (CommonLayout.PER_KEY, {WHOLE}),
        (CommonLayout.PER_KEY, set()),
    ],
)
def test_payload_keys_must_fit_the_layout(layout, keys):
    with pytest.raises(InternalError, match="do not fit"):
        RawCommon("test payload", layout, dict.fromkeys(keys, b""))


def test_stub_mode_decodes_anything_and_can_never_encode():
    args = default_args(attention_backend=FakeAttnBackend.auto)
    raw = _raw([{"args": args}], aliases={FakeAttnBackend: (ENUMS, "AttnBackend")})
    decoded = decode(raw, CodecMode.STUB)
    backend = decoded.state["args"].attention_backend
    assert isinstance(backend, Stub) and backend.qualified_name == f"{ENUMS}.AttnBackend"
    assert backend.args == (5,)
    with pytest.raises(RefusedError, match="decoded with stubs for display"):
        encode(decoded.state, CommonLayout.WRAPPED, mode=CodecMode.STUB, label="out")


def test_encode_refuses_state_that_does_not_reload_safely():
    with pytest.raises(RefusedError, match=r"FC-E01: out references .*Opaque"):
        encode({"thing": Opaque()}, CommonLayout.WRAPPED, mode=CodecMode.PLAIN, label="out")


def test_auto_mode_follows_megatron_availability():
    expected = CodecMode.MEGATRON if common_state.megatron_available() else CodecMode.PLAIN
    assert common_state.auto_mode() is expected
    if sys.version_info < common_state.MEGATRON_MIN_PYTHON:
        assert not common_state.megatron_available()
        assert "megatron" not in sys.modules  # never even attempted on old interpreters


def _megatron_common():
    from megatron.core.transformer.enums import (  # pylint: disable=import-outside-toplevel
        AttnBackend,
    )

    args = argparse.Namespace(**vars(COMMON["args"]), attention_backend=AttnBackend.auto)
    return {**COMMON, "args": args}


@pytest.mark.skipif(
    not common_state.megatron_available(), reason="needs Megatron (the training container)"
)
def test_megatron_loader_reads_the_encoded_common_state(tmp_path):
    """The real load_common_state_dict reads what the codec writes into a store."""
    from megatron.core.dist_checkpointing.serialization import (  # pylint: disable=C0415
        load_common_state_dict,
    )

    common = _megatron_common()
    payload = encode(common, CommonLayout.WRAPPED, mode=CodecMode.MEGATRON, label="common")[WHOLE]
    store = tmp_path / "store"
    write_part(store, [], [BytePiece("common_state/shard_0_1", payload)], threads=1)
    (store / "metadata.json").write_text(json.dumps(TORCH_DIST_CONFIG))
    loaded = load_common_state_dict(str(store))
    assert set(loaded) == set(common)
    assert vars(loaded["args"]) == vars(common["args"])
    assert {k: v for k, v in loaded.items() if k != "args"} == {
        k: v for k, v in common.items() if k != "args"
    }
    decoded = decode(RawCommon("store", CommonLayout.WRAPPED, {WHOLE: payload}), CodecMode.MEGATRON)
    assert vars(decoded.state["args"]) == vars(common["args"])
