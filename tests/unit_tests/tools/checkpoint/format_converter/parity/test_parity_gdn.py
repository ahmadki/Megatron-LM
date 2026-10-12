# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Gated DeltaNet layouts pinned to Megatron's GDN modules, read with ``ast`` (host-safe).

``GatedDeltaNet`` and ``GatedDeltaNet2`` set the parts of their fused
``in_proj`` (``in_proj_split_names`` and ``in_proj_split_sections``, local to a
tensor-parallel rank) and the sizes of ``dt_bias`` and ``A_log`` in
``_setup_variant_attrs``; ``_GDNBase.sharded_state_dict`` cuts ``in_proj.weight``
by those and the convolution's weight and bias into ``query, key, value``, and
leaves an ``in_proj.bias`` whole. The hybrid stack builds them under
``self_attention``, the ``gdn2`` variant choosing ``GatedDeltaNet2``. The
converter's layouts, its tensors kept whole, its per-head sizes and its choice
of family must be those. Real modules are pinned in the training container
(``real/test_gdn.py``).
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest

from tools.checkpoint.format_converter.core.args_view import ArgsView
from tools.checkpoint.format_converter.core.keys import (
    LINEAR_ATTENTION,
    MIXER_TENSOR,
    is_linear_attention,
    split_mixer_tensor,
)
from tools.checkpoint.format_converter.core.mixers import (
    GDN,
    GDN2,
    GDN2_VARIANT,
    TORCH_DIST_WHOLE,
    FamilyLayout,
    family_layouts,
)

from .._harness import REPO_ROOT

pytestmark = [pytest.mark.covers("gdn", "parity"), pytest.mark.covers("gdn2", "parity")]
GDN_PACKAGE = REPO_ROOT / "megatron" / "core" / "ssm" / "gated_delta_net"
MODULES = {GDN: ("gdn.py", "GatedDeltaNet"), GDN2: ("gdn2.py", "GatedDeltaNet2")}
"""Where each family's class is, and its name."""
HYBRID_SPECS = REPO_ROOT / "megatron" / "core" / "models" / "hybrid" / "hybrid_layer_specs.py"
HYBRID_BLOCK = REPO_ROOT / "megatron" / "core" / "models" / "hybrid" / "hybrid_block.py"
GPT_SPECS = (
    REPO_ROOT
    / "megatron"
    / "core"
    / "models"
    / "gpt"
    / "experimental_attention_variant_module_specs.py"
)
SYMBOLS = {
    "self.qk_dim_local_tp": "qk",
    "self.v_dim_local_tp": "v",
    "self.num_value_heads // self.tp_size": "nvh",
    "self.num_v_heads_local_tp": "nvh",
    "self.num_k_heads_local_tp": "nkh",
}
"""Megatron's per-rank sizes by the global size each is a rank's share of."""
ARGS = {
    "linear_key_head_dim": 3,
    "linear_num_key_heads": 2,
    "linear_value_head_dim": 2,
    "linear_num_value_heads": 5,
}
"""qk = 6, v = 10, 2 key heads, 5 value heads: every size differs, so each names its symbol."""
SIZES = {6: "qk", 10: "v", 5: "nvh", 2: "nkh"}


def _parse(path: Path) -> ast.Module:
    if not path.is_file():
        pytest.skip(f"{path} is not in this checkout")
    return ast.parse(path.read_text(), str(path))


def _method(path: Path, cls: str, name: str) -> ast.FunctionDef:
    (node,) = [n for n in _parse(path).body if isinstance(n, ast.ClassDef) and n.name == cls]
    (method,) = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return method


def _self_assignments(method: ast.FunctionDef) -> dict[str, ast.expr]:
    """``self.<name> = <value>`` in ``method``, by name."""
    found = {}
    for node in ast.walk(method):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and getattr(target.value, "id", "") == "self":
                    found[target.attr] = node.value
    return found


def _layout(family: str) -> FamilyLayout:
    layout = family_layouts(ArgsView(argparse.Namespace(**ARGS)))[family]
    assert isinstance(layout, FamilyLayout)
    return layout


@pytest.mark.parametrize("family", [GDN, GDN2])
def test_the_in_proj_parts_are_the_variants_split_names_and_sections(family):
    path, cls = MODULES[family]
    attrs = _self_assignments(_method(GDN_PACKAGE / path, cls, "_setup_variant_attrs"))
    names = ast.literal_eval(attrs["in_proj_split_names"])
    sections = attrs["in_proj_split_sections"]
    assert isinstance(sections, ast.Tuple)
    symbols = [SYMBOLS[ast.unparse(size)] for size in sections.elts]
    in_proj = _layout(family).in_proj
    assert list(zip(names, symbols)) == [
        (part, SIZES[size]) for part, size in zip(in_proj.parts, in_proj.sizes)
    ]


@pytest.mark.parametrize("family", [GDN, GDN2])
def test_the_per_head_parameters_have_the_variants_sizes(family):
    path, cls = MODULES[family]
    attrs = _self_assignments(_method(GDN_PACKAGE / path, cls, "_setup_variant_attrs"))
    megatron = {
        "dt_bias": SYMBOLS[ast.unparse(attrs["dt_bias_dim"])],
        "A_log": SYMBOLS[ast.unparse(attrs["a_log_dim"])],
    }
    vectors = _layout(family).vectors
    assert {name: [SIZES[rows] for rows in vectors[name]] for name in vectors} == {
        name: [symbol] for name, symbol in megatron.items()
    }


@pytest.fixture(name="sharded_state_dict", scope="module")
def fixture_sharded_state_dict() -> ast.FunctionDef:
    """``_GDNBase.sharded_state_dict``."""
    return _method(GDN_PACKAGE / "common.py", "_GDNBase", "sharded_state_dict")


def test_the_split_factory_cuts_the_in_proj_weight_and_the_convolution(sharded_state_dict):
    """``in_proj.weight`` by the variant's split names; ``conv1d.weight`` and ``conv1d.bias`` (the
    names of the loop over ``conv_layer_name_list``) into ``query, key, value``."""
    calls = [
        node
        for node in ast.walk(sharded_state_dict)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_split_tensor_factory"
    ]
    targets = {}
    for call in calls:
        target, sections, names, axis = call.args
        assert ast.literal_eval(axis) == 0
        tensor = "".join(
            v.value if isinstance(v, ast.Constant) else f"{{{ast.unparse(v.value)}}}"
            for v in target.slice.values
        )
        targets[tensor] = (ast.unparse(sections), ast.unparse(names))
    conv = _layout(GDN).conv1d
    conv_names = repr(list(conv.parts))
    conv_sizes = f"[{', '.join(f'self.{SIZES[s]}_dim_local_tp' for s in conv.sizes)}]"
    assert targets == {
        "{prefix}in_proj.weight": ("list(self.in_proj_split_sections)", "self.in_proj_split_names"),
        "{prefix}{conv_layer_name}": (conv_sizes, conv_names),
    }
    names = {
        ast.literal_eval(node)
        for node in ast.walk(sharded_state_dict)
        if isinstance(node, ast.Constant) and str(node.value).startswith("conv1d.")
    }
    assert names == {"conv1d.weight", "conv1d.bias"}


def test_only_the_in_proj_bias_is_left_whole(sharded_state_dict):
    strings = {
        value.value
        for node in ast.walk(sharded_state_dict)
        if isinstance(node, ast.JoinedStr)
        for value in node.values
        if isinstance(value, ast.Constant)
    }
    assert "in_proj.weight" in strings and "in_proj.bias" not in strings
    assert TORCH_DIST_WHOLE[GDN] == TORCH_DIST_WHOLE[GDN2] == {("in_proj", "bias")}


def test_the_hybrid_stack_builds_both_under_self_attention():
    """``gdn_layer`` and ``gdn2_layer`` of every hybrid stack spec are transformer layers whose
    ``self_attention`` is the family's class."""
    found = set()
    for node in ast.walk(_parse(HYBRID_SPECS)):
        if isinstance(node, ast.keyword) and node.arg in ("gdn_layer", "gdn2_layer"):
            if not isinstance(node.value, ast.Call):
                continue  # a reference to another spec's layer
            (attention,) = [
                k.value
                for k in ast.walk(node.value)
                if isinstance(k, ast.keyword) and k.arg == LINEAR_ATTENTION
            ]
            (module,) = [k.value.id for k in attention.keywords if k.arg == "module"]
            found.add((node.arg, module))
    assert found == {("gdn_layer", MODULES[GDN][1]), ("gdn2_layer", MODULES[GDN2][1])}
    # The key grammar matches the module Megatron names, so the pin covers what is matched.
    module = f"attn_layer.{LINEAR_ATTENTION}"
    assert is_linear_attention(LINEAR_ATTENTION) and is_linear_attention(module)
    assert MIXER_TENSOR.search(f"decoder.layers.0.{module}.in_proj.weight.beta") is not None
    tensor = split_mixer_tensor(f"{module}.conv1d", "weight")
    assert tensor is not None and tensor.mixer == module


def _variants_compared(tree: ast.AST) -> set[str]:
    """Every constant compared with an ``experimental_attention_variant`` attribute."""
    return {
        comparator.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and getattr(node.left, "attr", None) == "experimental_attention_variant"
        for comparator in node.comparators
        if isinstance(comparator, ast.Constant)
    }


def test_the_gdn2_variant_alone_chooses_gated_deltanet_2():
    """``HybridBlock`` builds ``gdn2_layer`` iff the variant is ``gdn2``; the GPT spec builds
    ``GatedDeltaNet2 if ... == "gdn2" else GatedDeltaNet``."""
    assert _variants_compared(_parse(HYBRID_BLOCK)) == {GDN2_VARIANT}
    (spec,) = [
        node
        for node in _parse(GPT_SPECS).body
        if isinstance(node, ast.FunctionDef) and node.name == "get_gated_delta_net_module_spec"
    ]
    (choice,) = [node for node in ast.walk(spec) if isinstance(node, ast.IfExp)]
    assert (choice.body.id, choice.orelse.id) == (MODULES[GDN2][1], MODULES[GDN][1])
    assert _variants_compared(choice.test) == {GDN2_VARIANT}
