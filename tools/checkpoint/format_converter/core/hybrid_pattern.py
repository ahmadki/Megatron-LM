# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Hybrid layer patterns (``--hybrid-layer-pattern``), parsed as Megatron parses them.

A pattern such as ``M*-M*-|M*-/*-/*-`` lists one symbol per decoder layer;
``|`` marks pipeline-stage boundaries (dropped here: stored layer indices are
global) and ``/`` separates the main decoder from one identical pattern per
multi-token-prediction depth. Older checkpoints store the pattern as
``hybrid_override_pattern`` with the MTP part in ``mtp_hybrid_override_pattern``;
:func:`hybrid_pattern_of` applies Megatron's backward-compatible merge
(``megatron/training/arguments.py``).

With ``--moe-shortcut-connection``, Megatron's ``HybridStack`` groups each
mixture-of-experts layer (``E``) with the layer before it into one module
(``group_layers_into_shortcut_blocks``, ``models/hybrid/shortcut_block.py``),
so its layer list has fewer modules than the pattern has layers:
:func:`group_modules` mirrors that grouping, and :func:`family_mismatch`
checks that the tensors a module holds are those of its layers' symbols.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .. import errors
from ..errors import InternalError, Refusal, RefusalError
from .args_view import ArgsView
from .keys import LAYER_FAMILIES
from .names import FSDP_DTENSOR

PIPE = "|"
MTP_SEPARATOR = "/"
LAYER_SYMBOLS = frozenset("MG*DCH+W-E")
"""Every layer symbol Megatron defines (``models/hybrid/layers/utils.py``, pinned by a test)."""

MOE = "E"
"""The symbol of a mixture-of-experts layer."""

SHORTCUT_PREDECESSORS = frozenset("MG*DCH+W")
"""The layer symbols a shortcut-connected mixture-of-experts layer may follow: Mamba, Gated
DeltaNet and the attention layers (``SUPPORTED_SHORTCUT_PREDECESSORS``, pinned by a test)."""

HybridModule = tuple[int, str]
"""One module of a hybrid layer list: the pattern position of its first layer (the torch_dist
layer index) and its layers' symbols (one symbol, or a predecessor and ``E``)."""


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

    @property
    def stage_starts(self) -> frozenset[int]:
        """The positions in :attr:`main` at which a ``|`` begins another pipeline stage."""
        segments = self.text.split(MTP_SEPARATOR)[0].split(PIPE)
        starts, position = set(), 0
        for segment in segments[:-1]:
            position += len(segment)
            starts.add(position)
        return frozenset(starts)


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


def group_modules(
    symbols: str, *, shortcut: bool, stage_starts: frozenset[int] = frozenset()
) -> tuple[HybridModule, ...]:
    """The modules of the hybrid layer list of ``symbols``, in layer-list (dense) order.

    Without ``shortcut`` every layer is a module of its own. With it, this is
    ``group_layers_into_shortcut_blocks`` applied to each pipeline stage: a
    layer followed by an ``E`` forms one module with it, and every other layer
    (a leading ``E``, an ``E`` after a pair) stays a module of its own; the
    module's torch_dist index is its first layer's position.

    Args:
        symbols: One symbol per layer, without ``|`` (a main decoder or an MTP depth).
        shortcut: ``args.moe_shortcut_connection``.
        stage_starts: The positions that begin a pipeline stage (``|``).

    Raises:
        RefusedError: With ``shortcut``, a stage after the first begins with an
            ``E`` (its pair would cross the stage boundary) or an ``E`` follows a
            layer that cannot precede a shortcut-connected MoE layer (FC-R11):
            Megatron cannot build such a model.
    """
    if not shortcut:
        return tuple(enumerate(symbols))
    for start in sorted(stage_starts):
        if 0 < start < len(symbols) and symbols[start] == MOE:
            raise _shortcut_refusal(
                symbols,
                f"layer {start} ({MOE!r}) begins a pipeline stage, so its shortcut pair would "
                "cross the stage boundary",
            )
    modules: list[HybridModule] = []
    position = 0
    while position < len(symbols):
        if position + 1 < len(symbols) and symbols[position + 1] == MOE:
            if symbols[position] not in SHORTCUT_PREDECESSORS:
                raise _shortcut_refusal(
                    symbols,
                    f"layer {position + 1} ({MOE!r}) follows layer {position} "
                    f"({symbols[position]!r}), which cannot precede a shortcut-connected "
                    f"mixture-of-experts layer (only {''.join(sorted(SHORTCUT_PREDECESSORS))} can)",
                )
            modules.append((position, symbols[position : position + 2]))
            position += 2
        else:
            modules.append((position, symbols[position]))
            position += 1
    return tuple(modules)


def shortcut_warnings(args: ArgsView, target: str) -> list[str]:
    """Why no current Megatron job loads an fsdp_dtensor ``target`` of a shortcut-MoE model:
    Megatron refuses ``--moe-shortcut-connection`` with Megatron-FSDP (``validate_args``)."""
    if target != FSDP_DTENSOR or not args.get("moe_shortcut_connection", False):
        return []
    return [
        "args.moe_shortcut_connection=True: Megatron refuses --moe-shortcut-connection with "
        "Megatron-FSDP, so no current Megatron job loads this fsdp_dtensor store (it is written "
        "in the layout a Megatron-FSDP v1 job uses)"
    ]


def _shortcut_refusal(symbols: str, reason: str) -> RefusalError:
    return Refusal(
        errors.FC_R11,
        f"with args.moe_shortcut_connection=True, Megatron cannot group the hybrid layers "
        f"{symbols!r}: {reason}",
        hint="check that the arguments belong to this checkpoint",
    ).to_error()


def family_mismatch(symbols: str, names: Iterable[str]) -> str | None:
    """Why a module of a hybrid layer list with layers ``symbols`` cannot hold the tensors
    ``names`` (the names inside the module), or ``None`` when it can.

    The check is presence-based, by family signature (:data:`~.keys.LAYER_FAMILIES`):

    * a single layer holds its symbol's family and no other;
    * a shortcut pair ``XE`` holds ``X``'s family and ``E``'s, each inside one
      child module of its own (the children's names are not assumed), and no
      other family.

    Any other tensor (a layer norm, the pair's own norms) may be anywhere.

    Raises:
        InternalError: A symbol has no family signature (the refusal fence
            refuses its layers before any module is checked).
    """
    unknown = set(symbols) - LAYER_FAMILIES.keys()
    if unknown:
        raise InternalError(f"no family signature for the hybrid layer symbols {sorted(unknown)}")
    holders: dict[str, set[str | None]] = {}
    for name in names:
        for symbol, family in LAYER_FAMILIES.items():
            match = family.pattern.search(name)
            if match is not None:  # held by the child it follows, or by the module itself
                child = name.partition(".")[0] if match.start() else None
                holders.setdefault(symbol, set()).add(child)
    if set(holders) != set(symbols):
        return (
            f"its tensors are those of {_families(holders) or 'no layer type'}, but {symbols!r} "
            f"means {_families(symbols)}"
        )
    children = [holders[symbol] for symbol in symbols]
    if len(symbols) == 1 or (
        all(len(held) == 1 and None not in held for held in children) and children[0] != children[1]
    ):
        return None
    where = " and ".join(
        f"its {symbol!r} tensors in {', '.join(sorted(c or 'the pair itself' for c in held))}"
        for symbol, held in zip(symbols, children)
    )
    return f"a shortcut pair holds each layer in a child module of its own, but it has {where}"


def _families(symbols: Iterable[str]) -> str:
    """``'*' (self_attention.linear_qkv), ...`` in the order of :data:`~.keys.LAYER_FAMILIES`."""
    present = set(symbols)
    return ", ".join(
        f"{symbol!r} ({family.tensors})"
        for symbol, family in LAYER_FAMILIES.items()
        if symbol in present
    )
