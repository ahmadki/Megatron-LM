# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The extension contract of the rule registry (see the package README).

* every rule module registers a rule, and every registered rule has evidence
  of two kinds: a test against hand-built expectations and a parity pin or a
  real-save case (``@pytest.mark.covers(rule, evidence)``);
* every sentinel is owned by a registered rule or a pending-family code;
* every pending-family code has a test that triggers exactly it;
* rules never spell physical key syntax (that lives in ``core/keys.py``).
"""

from __future__ import annotations

import ast
import importlib
import pkgutil

from tools.checkpoint.format_converter import errors, rules
from tools.checkpoint.format_converter.core.refusals import PENDING_FAMILIES
from tools.checkpoint.format_converter.core.registry import REGISTRY
from tools.checkpoint.format_converter.core.rule import LeafRule

from ._harness import REPO_ROOT
from .core.test_refusals import CASES

TESTS = REPO_ROOT / "tests" / "unit_tests" / "tools" / "checkpoint" / "format_converter"
RULES = REPO_ROOT / "tools" / "checkpoint" / "format_converter" / "rules"
EVIDENCE = ("ground_truth", "parity", "real_save")
KEY_SYNTAX = ("layers.", "module.", "optimizer.", "_extra_state", "/shard_")
"""Fragments of physical keys that rules must take from core/keys.py instead of spelling."""


def _covers() -> dict[str, set[str]]:
    """``{rule: {evidence kinds}}`` from every ``covers(rule, evidence)`` in the test tree."""
    found: dict[str, set[str]] = {}
    for path in TESTS.rglob("test_*.py"):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "covers"
                and len(node.args) == 2
            ):
                rule, evidence = (ast.literal_eval(arg) for arg in node.args)
                assert evidence in EVIDENCE, f"{path}: unknown evidence {evidence!r}"
                found.setdefault(rule, set()).add(evidence)
    return found


def test_every_rule_module_registers_a_rule():
    registered_modules = {type(rule).__module__ for rule in REGISTRY.rules.values()}
    for info in pkgutil.iter_modules(rules.__path__):
        module = importlib.import_module(f"{rules.__name__}.{info.name}")
        assert module.__name__ in registered_modules, f"{info.name} registers no rule"
        assert any(
            isinstance(value, type) and issubclass(value, LeafRule) and value is not LeafRule
            for value in vars(module).values()
        )


def test_every_rule_has_ground_truth_and_megatron_evidence():
    covers = _covers()
    assert set(covers) <= set(REGISTRY.rules), "covers() names an unknown rule"
    for name in REGISTRY.rules:
        evidence = covers.get(name, set())
        assert "ground_truth" in evidence, f"rule {name} has no hand-built ground-truth test"
        assert evidence & {"parity", "real_save"}, f"rule {name} has no parity or real-save test"


def test_every_sentinel_has_an_owner():
    owners = set(REGISTRY.rules) | set(PENDING_FAMILIES)
    assert REGISTRY.sentinels
    for sentinel in REGISTRY.sentinels:
        assert sentinel.owners and set(sentinel.owners) <= owners, sentinel.name


def test_every_pending_family_has_a_triggering_case():
    triggered = {code for code, _ in CASES.values()}
    assert set(PENDING_FAMILIES) <= triggered
    declared = {v for k, v in vars(errors).items() if k.startswith("FC_P")}
    assert declared == set(PENDING_FAMILIES)


def test_rules_do_not_spell_key_syntax():
    for path in RULES.glob("*.py"):
        tree = ast.parse(path.read_text(), str(path))
        docstrings = {
            id(body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
            and (body := node.body)
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names] + [getattr(node, "module", "")]
                assert "re" not in names, f"{path.name} imports re; key patterns go in core/keys.py"
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) in docstrings:
                    continue
                spelled = [f for f in KEY_SYNTAX if f in node.value]
                assert not spelled, f"{path.name} spells key syntax {spelled} in {node.value!r}"
