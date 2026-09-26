# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Drift pins read from Megatron's source with ``ast`` (host-safe: nothing is imported).

The converter mirrors what Megatron writes. These scans turn red when Megatron
changes a mirrored fact, so the mirror is revisited instead of silently going
stale; each failure message names what to review.
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

import pytest

from tools.checkpoint.format_converter.core.hybrid_pattern import LAYER_SYMBOLS, MTP_SEPARATOR, PIPE
from tools.checkpoint.format_converter.core.refusals import GDN_VARIANTS

from .._harness import REPO_ROOT

CHECKPOINTING = REPO_ROOT / "megatron" / "training" / "checkpointing.py"
LAYER_UTILS = REPO_ROOT / "megatron" / "core" / "models" / "hybrid" / "layers" / "utils.py"
TRANSFORMER_CONFIG = REPO_ROOT / "megatron" / "core" / "transformer" / "transformer_config.py"

FSDP_DTENSOR_HANDLERS = frozenset(
    {
        "handle_fp8_extra_state_case",
        "handle_swiglu_in_state_dict",
        "handle_mla_down_proj_in_state_dict",
        "handle_experts_in_state_dict",
        "handle_mtp_in_state_dict",
        "preprocess_state_dict_for_uneven_dtensor",
    }
)
"""The key transforms Megatron applies to fsdp_dtensor state dicts on save and on load."""

STATE_DICT_METHODS = frozenset({"copy", "get"})
"""Methods ``preprocess_fsdp_dtensor_state_dict`` calls on its state dicts (not handlers)."""


def _module(path: Path) -> ast.Module:
    if not path.is_file():
        pytest.skip(f"{path} is not in this checkout")
    return ast.parse(path.read_text(), str(path))


def _definition(tree: ast.Module, name: str) -> ast.AST:
    (node,) = [n for n in ast.walk(tree) if getattr(n, "name", None) == name]
    return node


def _bound_names(function: ast.AST) -> set[str]:
    """Names ``function`` binds itself: parameters, assignments and nested definitions."""
    bound = set()
    for node in ast.walk(function):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = node.args
            every = (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
            bound.update(argument.arg for argument in every)
            bound.update(a.arg for a in (arguments.vararg, arguments.kwarg) if a is not None)
            if node is not function and not isinstance(node, ast.Lambda):
                bound.add(node.name)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bound.add(node.id)
    return bound


def _external_references(function: ast.AST) -> set[str]:
    """What ``function`` uses from outside itself, whatever the names.

    Module-level names it reads (``module.attribute`` when it reads an
    attribute of one), and methods it calls on its own objects besides
    :data:`STATE_DICT_METHODS`. Every handler the function applies, called or
    passed to a helper such as ``apply``, is one of these.
    """
    bound = _bound_names(function)
    references, qualified = set(), set()
    for node in ast.walk(function):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id not in bound:
                references.add(f"{node.value.id}.{node.attr}")
                qualified.add(id(node.value))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            local = not isinstance(owner, ast.Name) or owner.id in bound
            if local and node.func.attr not in STATE_DICT_METHODS:
                references.add(node.func.attr)
    references.update(
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id not in bound
        and id(node) not in qualified
        and node.id not in vars(builtins)
    )
    return references


def test_the_handler_scan_sees_handlers_of_any_name_or_module():
    source = """
def preprocess(args, raw_state_dict, model):
    state_dict = raw_state_dict.copy()
    handle_a(state_dict['model'])

    def apply(handler):
        state_dict['model'] = handler(model, state_dict['model'], state_dict.get('optimizer'))

    if args.swiglu:
        apply(handle_b)
    apply(split_c)
    apply(utils.handle_d)
    state_dict['model'] = helpers.convert_e(state_dict['model'], len(args.x))
    return state_dict
"""
    function = _definition(ast.parse(source), "preprocess")
    assert _external_references(function) == {
        "handle_a",
        "handle_b",
        "split_c",
        "utils.handle_d",
        "helpers.convert_e",
    }


def test_the_fsdp_dtensor_preprocess_applies_exactly_the_mirrored_handlers():
    function = _definition(_module(CHECKPOINTING), "preprocess_fsdp_dtensor_state_dict")
    handlers = _external_references(function)
    assert handlers == FSDP_DTENSOR_HANDLERS, (
        "preprocess_fsdp_dtensor_state_dict changed its key transforms; review the "
        "fsdp_dtensor rules of format_converter (a new handler, e.g. for Gated DeltaNet, "
        "changes the fsdp_dtensor key layout)"
    )


def test_the_hybrid_layer_symbols_are_the_mirrored_ones():
    symbols = _definition(_module(LAYER_UTILS), "Symbols")
    values = {
        node.value.value
        for node in symbols.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    assert values == LAYER_SYMBOLS | {PIPE, MTP_SEPARATOR}, (
        "Megatron's hybrid layer symbols changed; classify the new symbol in "
        "core/refusals.py (verified, pending or refused)"
    )


def test_the_gated_deltanet_variants_exist():
    config = _definition(_module(TRANSFORMER_CONFIG), "TransformerConfig")
    (annotation,) = [
        node.annotation
        for node in config.body
        if isinstance(node, ast.AnnAssign)
        and getattr(node.target, "id", None) == "experimental_attention_variant"
    ]
    variants = {
        node.value
        for node in ast.walk(annotation)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert GDN_VARIANTS <= variants, "experimental_attention_variant no longer names these"
