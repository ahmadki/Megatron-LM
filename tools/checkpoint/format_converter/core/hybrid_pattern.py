# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Hybrid layer patterns (``--hybrid-layer-pattern``), parsed as Megatron parses them.

A pattern such as ``M*-M*-|M*-/*-/*-`` lists one symbol per decoder layer;
``|`` marks pipeline-stage boundaries (dropped here: stored layer indices are
global) and ``/`` separates the main decoder from one identical pattern per
multi-token-prediction depth. Older checkpoints store the pattern as
``hybrid_override_pattern`` with the MTP part in ``mtp_hybrid_override_pattern``;
:func:`hybrid_pattern_of` applies Megatron's backward-compatible merge
(``megatron/training/arguments.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .args_view import ArgsView

PIPE = "|"
MTP_SEPARATOR = "/"
LAYER_SYMBOLS = frozenset("MG*DCH+W-E")
"""Every layer symbol Megatron defines (``models/hybrid/layers/utils.py``, pinned by a test)."""


@dataclass(frozen=True, slots=True)
class HybridPattern:
    """A parsed hybrid layer pattern.

    Attributes:
        text: The pattern as stored in the arguments.
        main: The main decoder's symbols, one per layer, without ``|``.
        mtp: The symbols of each multi-token-prediction depth.
    """

    text: str
    main: str
    mtp: tuple[str, ...]

    @property
    def symbols(self) -> frozenset[str]:
        """Every symbol of the main decoder and the MTP depths."""
        return frozenset(self.main).union(*self.mtp)


def parse_hybrid_pattern(text: str) -> HybridPattern:
    """Split ``text`` into the main decoder symbols and the MTP depths.

    Symbols are not validated here (the refusal fence reports unknown ones).
    As in Megatron, a pattern whose MTP parts are all empty has no MTP.
    """
    main, *depths = text.split(MTP_SEPARATOR)
    if all(not depth for depth in depths):
        depths = []
    return HybridPattern(
        text, main.replace(PIPE, ""), tuple(depth.replace(PIPE, "") for depth in depths)
    )


def hybrid_pattern_of(args: ArgsView) -> HybridPattern | None:
    """The hybrid layer pattern the checkpoint's model was built with, if it is a hybrid model.

    ``hybrid_layer_pattern`` wins; an older ``hybrid_override_pattern`` is used
    otherwise, and a legacy ``mtp_hybrid_override_pattern`` is appended once per
    ``mtp_num_layers`` when the pattern has no ``/`` part, as Megatron does.
    """
    text = args.get("hybrid_layer_pattern", None) or args.get("hybrid_override_pattern", None)
    legacy_mtp = args.get("mtp_hybrid_override_pattern", None)
    depths = args.get("mtp_num_layers", None) or 0
    if legacy_mtp is not None and depths > 0 and (text is None or MTP_SEPARATOR not in text):
        text = (text or "") + (MTP_SEPARATOR + legacy_mtp) * depths
    return None if text is None else parse_hybrid_pattern(text)
