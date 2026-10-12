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

from tools.checkpoint.format_converter.core.dtype_policy import KEEP_FP32
from tools.checkpoint.format_converter.core.format_args import GLU_ARGS
from tools.checkpoint.format_converter.core.hybrid_pattern import LAYER_SYMBOLS, MTP_SEPARATOR, PIPE
from tools.checkpoint.format_converter.core.keys import (
    FUSED_MLA_DOWN_PROJ_KEY,
    LAYER_FAMILIES,
    MLA_FUSED_DOWN_PROJ,
    SWIGLU_KEYS,
)
from tools.checkpoint.format_converter.core.names import GPT_MTP_INNER_LAYER, MTP_INNER_LAYER
from tools.checkpoint.format_converter.core.param_groups import ALIAS_PREFIX, IDENTIFIER_KEYS
from tools.checkpoint.format_converter.core.refusals import GDN_VARIANTS

from .._harness import REPO_ROOT

CHECKPOINTING = REPO_ROOT / "megatron" / "training" / "checkpointing.py"
MEGATRON_CORE = REPO_ROOT / "megatron" / "core"
OPTIMIZER = MEGATRON_CORE / "optimizer" / "optimizer.py"
DISTRIB_OPTIMIZER = MEGATRON_CORE / "optimizer" / "distrib_optimizer.py"
PARAM_SCHEDULER = MEGATRON_CORE / "optimizer_param_scheduler.py"
TRANSFORMER_MODULE = MEGATRON_CORE / "transformer" / "module.py"
KEEP_FP32_MARK = "mark_keep_in_fp32"
ARGUMENT_UTILS = REPO_ROOT / "megatron" / "training" / "argument_utils.py"
FSDP_DTENSOR_CHECKPOINT = (
    REPO_ROOT / "megatron" / "core" / "transformer" / "fsdp_dtensor_checkpoint.py"
)
LAYER_UTILS = REPO_ROOT / "megatron" / "core" / "models" / "hybrid" / "layers" / "utils.py"
TRANSFORMER_CONFIG = REPO_ROOT / "megatron" / "core" / "transformer" / "transformer_config.py"
MULTI_TOKEN_PREDICTION = (
    REPO_ROOT / "megatron" / "core" / "transformer" / "multi_token_prediction.py"
)
MULTI_LATENT_ATTENTION = (
    REPO_ROOT / "megatron" / "core" / "transformer" / "multi_latent_attention.py"
)
MLA_LAYER_SPECS = (
    REPO_ROOT / "megatron" / "core" / "models" / "gpt" / "gpt_layer_specs.py",
    REPO_ROOT / "megatron" / "core" / "models" / "hybrid" / "hybrid_layer_specs.py",
)
"""The layer specs that build ``FusedMLASelfAttention`` (GPT, and the hybrid ``+`` layers)."""
MLA_DOWN_PROJS = ("linear_q_down_proj", "linear_kv_down_proj")
"""The unfused down-projections of multi-latent attention, as both formats store them."""

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


def _is_args_field(node: ast.AST, name: str | None = None) -> bool:
    """Whether ``node`` reads ``args.<name>`` (any field if ``name`` is ``None``)."""
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
        and (name is None or node.attr == name)
    )


@pytest.mark.covers("glu", "parity")
def test_the_swiglu_names_are_the_ones_megatron_fsdp_splits():
    """``is_swiglu_key`` of ``handle_swiglu_in_state_dict``: its patterns, searched (``re.search``)."""
    function = _definition(_module(FSDP_DTENSOR_CHECKPOINT), "is_swiglu_key")
    patterns = [
        element.value
        for node in ast.walk(function)
        if isinstance(node, ast.List)
        for element in node.elts
        if isinstance(element, ast.Constant)
    ]
    assert patterns == [
        pattern.pattern for pattern in SWIGLU_KEYS
    ], "Megatron-FSDP splits other linear_fc1 names into SwiGLU halves; update SWIGLU_KEYS"
    calls = {ast.unparse(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)}
    assert "re.search" in calls and "re.match" not in calls


@pytest.mark.covers("glu", "parity")
def test_megatron_fsdp_splits_only_with_swiglu_and_in_gated_or_ownerless_layers():
    """The handler runs only under ``args.swiglu``, and a key without an owning
    TransformerLayer counts as gated (``_key_in_glu_layer``)."""
    preprocess = _definition(_module(CHECKPOINTING), "preprocess_fsdp_dtensor_state_dict")
    (gate,) = [
        node
        for node in ast.walk(preprocess)
        if isinstance(node, ast.If) and "handle_swiglu_in_state_dict" in ast.unparse(node)
    ]
    assert _is_args_field(gate.test, "swiglu") and not gate.orelse
    owner = _definition(_module(FSDP_DTENSOR_CHECKPOINT), "_key_in_glu_layer")
    (ownerless,) = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "best_glu is None"
    ]
    assert [ast.unparse(statement) for statement in ownerless.body] == ["return True"]


@pytest.mark.covers("glu", "parity")
def test_the_arguments_that_gate_the_language_model_are_the_mirrored_ones():
    """``core_transformer_config_from_args`` sets ``gated_linear_unit`` under exactly these."""
    function = _definition(_module(ARGUMENT_UTILS), "core_transformer_config_from_args")
    gating = set()
    for node in ast.walk(function):
        if isinstance(node, ast.If) and _is_args_field(node.test):
            sets_glu = any(
                isinstance(statement, ast.Assign)
                and "'gated_linear_unit'" in ast.unparse(statement.targets[0])
                for statement in node.body
            )
            if sets_glu:
                gating.add(node.test.attr)
    assert gating == set(GLU_ARGS), "other arguments gate the MLPs now; update GLU_ARGS"


def _string_constants(tree: ast.AST) -> dict[str, str]:
    """Module-level ``NAME = '<string>'`` assignments."""
    return {
        target.id: node.value.value
        for node in getattr(tree, "body", ())
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name) and isinstance(node.value.value, str)
    }


def test_hybrid_mtp_layers_keep_the_inner_layer_name_in_both_formats():
    """torch_dist renames a GPT MTP layer's inner model (``mtp_model_layer`` to
    ``transformer_layer``) only without an MTP layer pattern, and Megatron-FSDP only for the
    layers ``get_mtp_inner_layer_paths`` finds, which excludes those with a pattern: a hybrid MTP
    layer keeps the name in both formats."""
    constants = _string_constants(_module(FSDP_DTENSOR_CHECKPOINT))
    assert constants["MTP_INNER_LAYER"] == MTP_INNER_LAYER
    paths = _definition(_module(FSDP_DTENSOR_CHECKPOINT), "get_mtp_inner_layer_paths")
    assert "getattr(mod, 'mtp_layer_pattern', None) is None" in ast.unparse(paths)
    layer = _definition(_module(MULTI_TOKEN_PREDICTION), "MultiTokenPredictionLayer")
    (rename,) = [
        node
        for node in ast.walk(_definition(layer, "sharded_state_dict"))
        if isinstance(node, ast.If) and "apply_prefix_mapping" in ast.unparse(node)
    ]
    assert ast.unparse(rename.test) == "self.mtp_layer_pattern is None"
    assert f"{MTP_INNER_LAYER}." in ast.unparse(rename.body[0])


@pytest.mark.covers("identity", "parity")
def test_gpt_mtp_layers_store_their_inner_layer_as_transformer_layer_in_both_formats():
    """torch_dist (``MultiTokenPredictionLayer.sharded_state_dict``) and Megatron-FSDP
    (``handle_mtp_in_state_dict``) both store a GPT MTP layer's ``mtp_model_layer`` as
    ``transformer_layer``, model and optimizer state alike."""
    tree = _module(FSDP_DTENSOR_CHECKPOINT)
    constants = _string_constants(tree)
    assert constants["MTP_INNER_LAYER_CHECKPOINT_NAME"] == GPT_MTP_INNER_LAYER
    assert constants["MTP_INNER_LAYER"] == MTP_INNER_LAYER
    rename = ast.unparse(_definition(tree, "rename_mtp_inner_layer_keys"))
    assert "{MTP_INNER_LAYER}." in rename and "{MTP_INNER_LAYER_CHECKPOINT_NAME}." in rename
    handler = ast.unparse(_definition(tree, "handle_mtp_in_state_dict"))
    assert "rename_mtp_inner_layer_keys(model_state_dict, mtp_layer_paths)" in handler
    layer = _definition(_module(MULTI_TOKEN_PREDICTION), "MultiTokenPredictionLayer")
    mapping = ast.unparse(_definition(layer, "sharded_state_dict"))
    assert f"{{prefix}}{MTP_INNER_LAYER}.': f'{{prefix}}{GPT_MTP_INNER_LAYER}." in mapping


def _top_level_calls(function: ast.AST) -> set[str]:
    """The calls ``function`` makes as statements of its own body (not under a condition)."""
    return {
        ast.unparse(statement.value)
        for statement in getattr(function, "body", ())
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    }


@pytest.mark.covers("identity", "parity")
def test_both_formats_store_fused_mla_unfused():
    """Megatron-FSDP always unfuses ``linear_qkv_down_proj`` (``handle_mla_down_proj_in_state_dict``
    runs unconditionally) into the two down-projections, and moves the input layer norm it
    absorbed to the layer's ``input_layernorm``; torch_dist does the same
    (``FusedMLASelfAttention.sharded_state_dict`` and the fused layer specs' key maps), so MLA
    converts as identity and only the fused live name is refused (FC-R08)."""
    tree = _module(FSDP_DTENSOR_CHECKPOINT)
    assert _string_constants(tree)["MLA_FUSED_DOWN_PROJ"] == MLA_FUSED_DOWN_PROJ
    (unfused,) = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "MLA_UNFUSED_DOWN_PROJS" for t in node.targets)
    ]
    assert ast.literal_eval(unfused) == MLA_DOWN_PROJS
    absorbed = ast.unparse(_definition(tree, "absorbed_input_layernorm_key"))
    assert ".input_layernorm." in absorbed
    preprocess = _definition(_module(CHECKPOINTING), "preprocess_fsdp_dtensor_state_dict")
    assert "apply(handle_mla_down_proj_in_state_dict)" in _top_level_calls(preprocess)
    fused = _definition(_module(MULTI_LATENT_ATTENTION), "FusedMLASelfAttention")
    torch_dist = ast.unparse(_definition(fused, "sharded_state_dict"))
    for name in MLA_DOWN_PROJS:
        assert f"{{prefix}}{name}.weight" in torch_dist
    for path in MLA_LAYER_SPECS:
        maps = {
            key.value: value.value
            for node in ast.walk(_module(path))
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and "_down_proj.layer_norm_" in str(key.value)
        }
        assert maps == {
            f"self_attention.{name}.layer_norm_": "input_layernorm."
            for name in (*MLA_DOWN_PROJS, MLA_FUSED_DOWN_PROJ)
        }, f"{path.name} maps the fused MLA's keys otherwise; review the MLA conversion"
    for name in MLA_DOWN_PROJS:
        key = f"decoder.layers.0.self_attention.{name}.weight"
        assert LAYER_FAMILIES["+"].pattern.search(key) and not FUSED_MLA_DOWN_PROJ_KEY.search(key)


def _self_attribute(node: ast.AST) -> str | None:
    """``a.b`` for the expression ``self.a.b``, else ``None``."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id == "self" and parts:
        return ".".join(reversed(parts))
    return None


def _marked(owner: str, node: ast.AST) -> tuple[str, str] | None:
    """``(owner, attribute)`` of a parameter ``node`` marks to keep in fp32, if it marks one:
    ``mark_keep_in_fp32(self.<attribute>)``, or ``self.<attribute> = mark_keep_in_fp32(...)``.
    An unresolvable mark gives its source text as the attribute (so the pin fails)."""
    value = node.value if isinstance(node, (ast.Assign, ast.Expr)) else None
    if not (isinstance(value, ast.Call) and getattr(value.func, "id", None) == KEEP_FP32_MARK):
        return None
    targets = [_self_attribute(t) for t in node.targets] if isinstance(node, ast.Assign) else []
    attribute = next(filter(None, targets), None) or _self_attribute(value.args[0])
    return owner, attribute or ast.unparse(node)


def test_the_keep_fp32_registry_is_every_parameter_megatron_keeps_in_fp32():
    """Every ``mark_keep_in_fp32`` target in Megatron Core is in the registry (and only those), and
    nothing but ``mark_keep_in_fp32`` sets the ``keep_in_fp32`` flag ``Float16Module`` honors."""
    _module(TRANSFORMER_MODULE)  # skips when Megatron's source is not in this checkout
    marked, flagged = set(), []
    for path in sorted(MEGATRON_CORE.rglob("*.py")):
        source = path.read_text()
        if "keep_in_fp32" not in source:
            continue
        tree = ast.parse(source, str(path))
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        for cls in classes:
            marked.update(filter(None, (_marked(cls.name, node) for node in ast.walk(cls))))
        flagged += [
            f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(getattr(t, "attr", None) == "keep_in_fp32" for t in node.targets)
        ]
        loose = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == KEEP_FP32_MARK
        ]
        in_classes = [
            node
            for cls in classes
            for node in ast.walk(cls)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == KEEP_FP32_MARK
        ]
        assert len(loose) == len(in_classes), f"{path}: a mark outside a class; review KEEP_FP32"
    assert marked == {(entry.owner, entry.attribute) for entry in KEEP_FP32}, (
        "Megatron keeps other parameters in fp32 (mark_keep_in_fp32); review KEEP_FP32 in "
        "core/dtype_policy.py"
    )
    assert flagged == [
        f"{TRANSFORMER_MODULE.relative_to(REPO_ROOT)}:{_mark_line()}"
    ], "something but mark_keep_in_fp32 sets keep_in_fp32; review KEEP_FP32"


def _mark_line() -> int:
    """The line of ``mark_keep_in_fp32``'s ``tensor.keep_in_fp32 = True``."""
    (node,) = [
        node
        for node in ast.walk(_definition(_module(TRANSFORMER_MODULE), KEEP_FP32_MARK))
        if isinstance(node, ast.Assign)
    ]
    return node.lineno


def test_the_parameter_group_identifier_keys_are_the_mirrored_ones():
    """``ParamGroupOverride``'s fields, sorted, then the structural flags; read with a ``pre_``
    alias by the classic loader."""
    override = _definition(_module(PARAM_SCHEDULER), "ParamGroupOverride")
    fields = sorted(node.target.id for node in override.body if isinstance(node, ast.AnnAssign))
    tree = _module(OPTIMIZER)
    (assigned,) = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "param_group_identifier_keys" for t in node.targets)
    ]
    starred, *flags = assigned.elts
    assert ast.unparse(starred) == "*_param_group_override_keys()"
    overrides = ast.unparse(_definition(tree, "_param_group_override_keys"))
    assert "sorted(_ParamGroupOverride.__annotations__.keys())" in overrides
    assert IDENTIFIER_KEYS == (
        *fields,
        *(ast.literal_eval(flag) for flag in flags),
    ), "Megatron identifies parameter groups by other keys; review core/param_groups.py"
    make_needed = ast.unparse(_definition(_module(DISTRIB_OPTIMIZER), "make_needed_groups"))
    assert f"f'{ALIAS_PREFIX}{{key}}'" in make_needed and "value = None" in make_needed
