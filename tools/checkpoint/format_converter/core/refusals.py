# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The refusal fence: permanent refusals and pending families, detected before any tensor is read.

Cross-format conversion only runs on models whose every part a rule has been
verified for. Everything else is refused from the checkpoint's training
arguments and its model keys (names and shapes from the metadata), before any
tensor data is read:

* ``FC-R`` codes are permanent: the input is not a supported configuration.
* ``FC-P`` codes are *pending families*: a supported configuration whose rules
  are not implemented yet. Each code is lifted by the change that adds its
  rules (:data:`PENDING_FAMILIES`), so no intermediate version of the tool can
  silently mis-convert a family it does not know.

Every detector runs and every refusal is reported together, so ``plan`` shows
the whole list at once. Detectors are pure functions of :class:`Facts`
registered with ``@REGISTRY.detector``.

The layer symbols and specs verified so far are registered here too, and the
sentinels owned by pending families: a pending family's keys that slip past
the detectors are still refused by the identity rule (see ``core/registry.py``).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from .. import errors
from ..engine.metadata import TensorEntry
from ..errors import Refusal
from .args_view import ArgsView
from .blocks import LANGUAGE_MODEL, MTP, SINGLE_ROOTS, VLM_ROOTS
from .experts import built_expert_form
from .hybrid_pattern import LAYER_SYMBOLS, hybrid_pattern_of
from .keys import (
    FACTORY_PART,
    FUSED_MIXER,
    GDN_PARAM,
    GDP_PART,
    GLU_HALF,
    GROUPED_EXPERT_BIAS,
    LAYER_SEGMENT,
    LEGACY_GROUPED_EXPERTS,
    MIXER_PARAM,
    MLA_PARAM,
    SINGLE_GROUPED_EXPERTS,
    SINGLETON_EXPERT_SHARD,
    TE_OP_FUSER_MLP,
    TORCH_DIST_EXPERT_BIAS,
    root_of,
)
from .mixers import GDP_SPEC_NAMES, HYBRID_SPEC_MODULES, is_gdp_spec, spec_name
from .names import AUTO_EXPERT_NAMING, FSDP_DTENSOR, GROUPED, TORCH_DIST
from .registry import REGISTRY, Allowlist, Sentinel

PENDING_FAMILIES: Mapping[str, str] = {
    errors.FC_P08: "Mamba-2 mixers",
    errors.FC_P09: "Gated DeltaNet and linear-attention layers",
    errors.FC_P10: "GPT multi-token prediction and multi-latent attention",
    errors.FC_P11: "optimizer state from fsdp_dtensor to torch_dist",
    errors.FC_P12: "optimizer state from torch_dist to fsdp_dtensor",
}
"""Every pending-family code and the family it stands for."""

MTP_BLOCKS = (MTP, f"{LANGUAGE_MODEL}.{MTP}")
"""Block prefixes of multi-token prediction."""

PENDING_SYMBOLS = frozenset("MG+")
"""Hybrid layer symbols of pending families (mixers, Gated DeltaNet, MLA)."""

GDN_VARIANTS = frozenset({"gdn", "gdn2", "gated_delta_net"})
"""``experimental_attention_variant`` values of the Gated DeltaNet family."""

GTP_REMAT_ARGS = ("gtp_weight_remat_size", "expert_gtp_weight_remat_size")
"""Rematerialization sizes that enable grouped tensor parallelism when above 1."""

GTP_SHARD_ARGS = (
    ("tensor_parallel_num_weight_shards", "tensor_model_parallel_size"),
    ("expert_tensor_parallel_num_weight_shards", "expert_tensor_parallel_size"),
)
"""Weight-shard counts that enable grouped tensor parallelism when above their tensor-parallel
size (the expert size defaults to ``tensor_model_parallel_size`` when unset)."""

HETEROGENEOUS_ARGS = (
    "heterogeneous_layers_config_path",
    "heterogeneous_layers_config_encoded_json",
)

SINGLE_GROUPED_ARGS = ("moe_single_grouped_weight", "moe_single_grouped_bias")
"""Arguments that store every local expert of a grouped linear layer in one parameter."""

SINGLETON_LOCAL_SHARDS = "singleton_local_shards"
"""The ``content_metadata`` flag of a checkpoint saved one local shard per key."""

REGISTRY.allow(Allowlist.HYBRID_SYMBOLS, "*", "-", "E")
REGISTRY.allow(
    Allowlist.STACK_SPECS,
    *(
        f"{module}.{name}"
        for module in HYBRID_SPEC_MODULES
        for name in ("hybrid_stack_spec", "mamba_stack_spec", *sorted(GDP_SPEC_NAMES))
    ),
)

REGISTRY.sentinel(
    Sentinel(
        "parameter_parts",
        TORCH_DIST,
        FACTORY_PART,
        ("gdp", errors.FC_P08, errors.FC_P09),
        "a parameter stored as parts can only be reassembled by its family's rule",
    )
)
REGISTRY.sentinel(
    Sentinel(
        "swiglu_halves",
        FSDP_DTENSOR,
        GLU_HALF,
        ("glu",),
        "SwiGLU halves must be merged back into linear_fc1 by the GLU rule",
    )
)
REGISTRY.sentinel(
    Sentinel(
        "fused_mixer_projections",
        FSDP_DTENSOR,
        FUSED_MIXER,
        ("gdp", errors.FC_P08, errors.FC_P09),
        "a fused mixer projection is packed per tensor-parallel rank and needs its mixer rule",
    )
)


@dataclass(frozen=True)
class Facts:
    """What the refusal detectors see of a conversion (no tensor data).

    Attributes:
        source: Source format.
        target: Target format.
        args: The checkpoint's training arguments.
        model: The source's model tensors by model-relative key.
        optimizer: ``drop`` or ``convert`` (``--optimizer``).
        allowed_roots: Key prefixes accepted besides the known roots (``--allow-root``).
        allow_missing_fp8_extra_state: ``--allow-missing-fp8-extra-state``.
        content_metadata: The source's ``content_metadata`` (torch_dist common state).
        expert_naming: ``--expert-naming``.
    """

    source: str
    target: str
    args: ArgsView
    model: Mapping[str, TensorEntry]
    optimizer: str = "drop"
    allowed_roots: tuple[str, ...] = ()
    allow_missing_fp8_extra_state: bool = False
    content_metadata: Any = None
    expert_naming: str = AUTO_EXPERT_NAMING

    def keys_matching(self, pattern: re.Pattern[str]) -> tuple[str, ...]:
        """The model-relative keys ``pattern`` finds a match in (sorted)."""
        return tuple(key for key in sorted(self.model) if pattern.search(key))


def detect_refusals(facts: Facts) -> list[Refusal]:
    """Every refusal the registered detectors find for ``facts``."""
    return [refusal for detector in REGISTRY.detectors.values() for refusal in detector(facts)]


def _pending(code: str, evidence: list[str], keys: tuple[str, ...] = ()) -> Iterator[Refusal]:
    """One pending-family refusal naming its evidence, if there is any."""
    if evidence or keys:
        if keys:
            evidence.append(f"{len(keys)} model key(s) such as {keys[0]}")
        yield Refusal(
            code,
            f"not supported yet: {PENDING_FAMILIES[code]} ({'; '.join(evidence)})",
            keys=keys,
            hint="a later version of format_converter adds this family",
        )


def _symbols(facts: Facts) -> frozenset[str]:
    pattern = hybrid_pattern_of(facts.args)
    return frozenset() if pattern is None else pattern.symbols


@REGISTRY.detector
def megatron_fsdp_v2(facts: Facts) -> Iterator[Refusal]:
    """FC-R01: the arguments select Megatron-FSDP v2, whose key prefix differs."""
    version = facts.args.get("megatron_fsdp_version", 1)
    if version == 2:
        yield Refusal(
            errors.FC_R01,
            f"args.megatron_fsdp_version={version}: only Megatron-FSDP v1 stores are supported",
            hint="use a checkpoint of a --megatron-fsdp-version 1 run",
        )


@REGISTRY.detector
def fsdp_pipeline_parallel(facts: Facts) -> Iterator[Refusal]:
    """FC-R02: an fsdp_dtensor source saved with pipeline or virtual-pipeline parallelism."""
    if facts.source != FSDP_DTENSOR:
        return
    pp = facts.args.get("pipeline_model_parallel_size", 1) or 1
    vpp = facts.args.get("virtual_pipeline_model_parallel_size", None)
    if pp > 1 or (vpp or 1) > 1:
        yield Refusal(
            errors.FC_R02,
            f"args.pipeline_model_parallel_size={pp}, args.virtual_pipeline_model_parallel_size="
            f"{vpp}: fsdp_dtensor stores hold a single pipeline stage",
            hint="use a checkpoint saved without pipeline parallelism",
        )


@REGISTRY.detector
def unsupported_spec(facts: Facts) -> Iterator[Refusal]:
    """FC-R03: a non-Transformer-Engine implementation, a custom or heterogeneous spec."""
    impl = facts.args.get("transformer_impl", "transformer_engine")
    if impl != "transformer_engine":
        yield Refusal(
            errors.FC_R03,
            f"args.transformer_impl={impl!r}: only Transformer Engine layer specs are supported "
            "(other implementations rename keys in torch_dist only)",
            hint="convert a checkpoint of a --transformer-impl transformer_engine run",
        )
    spec = spec_name(facts.args)
    if spec is not None and spec not in REGISTRY.allowed(Allowlist.STACK_SPECS):
        yield Refusal(
            errors.FC_R03,
            f"args.spec={spec!r} is not a spec whose keys the rules know",
            hint="convert a checkpoint built with the default hybrid stack spec or the Gated "
            "DeltaProduct stack spec",
        )
    heterogeneous = [
        f"args.{name}={facts.args.get(name, None)!r}"
        for name in HETEROGENEOUS_ARGS
        if facts.args.get(name, None) is not None
    ]
    if facts.args.get("heterogeneous_block_specs", False):
        heterogeneous.append("args.heterogeneous_block_specs=True")
    if heterogeneous:
        yield Refusal(
            errors.FC_R03,
            f"heterogeneous layer specs are not supported ({', '.join(heterogeneous)})",
            hint="convert a checkpoint of a homogeneous-spec model",
        )
    fused = facts.keys_matching(TE_OP_FUSER_MLP)
    if fused:
        yield Refusal(
            errors.FC_R03,
            "MLPs built with the Transformer Engine operation fuser store other keys per format",
            keys=fused,
            hint="convert a checkpoint of a model without the operation-fuser MLP",
        )


def _gtp_evidence(args: ArgsView) -> list[str]:
    """The arguments that enable grouped tensor parallelism (none: it is off).

    Megatron's ``validate_args`` sets an unset ``tensor_parallel_num_weight_shards``
    to ``tensor_model_parallel_size * gtp_weight_remat_size`` (the expert pair
    likewise, against the expert tensor-parallel size), so as many shards as
    the tensor-parallel size is plain tensor parallelism: only more shards, or
    a rematerialization size above 1, is grouped tensor parallelism.
    """
    evidence = [
        f"args.{name}={args.get(name, None)}"
        for name in GTP_REMAT_ARGS
        if (args.get(name, None) or 1) > 1
    ]
    tp = args.get("tensor_model_parallel_size", None) or 1
    expert_tp = args.get("expert_tensor_parallel_size", None) or tp
    for (shards_name, size_name), size in zip(GTP_SHARD_ARGS, (tp, expert_tp)):
        shards = args.get(shards_name, None)
        if shards is not None and shards > size:
            evidence.append(f"args.{shards_name}={shards} > {size_name}={size}")
    return evidence


@REGISTRY.detector
def grouped_tensor_parallel(facts: Facts) -> Iterator[Refusal]:
    """FC-R06: grouped tensor parallelism (weight shards beyond tensor parallelism, remat > 1)."""
    enabled = _gtp_evidence(facts.args)
    if enabled:
        yield Refusal(
            errors.FC_R06,
            f"grouped tensor parallelism may pad stored tensors ({', '.join(enabled)})",
            hint="convert a checkpoint saved without grouped tensor parallelism",
        )


@REGISTRY.detector
def fp8_delayed_scaling(facts: Facts) -> Iterator[Refusal]:
    """FC-R07: a torch_dist target of an FP8 delayed-scaling model (its amax history is lost)."""
    if facts.source != FSDP_DTENSOR or facts.target != TORCH_DIST:
        return
    fp8 = facts.args.get("fp8", None)
    recipe = facts.args.get("fp8_recipe", "delayed")
    if not fp8 or recipe != "delayed":
        return
    if not facts.allow_missing_fp8_extra_state:
        yield Refusal(
            errors.FC_R07,
            f"args.fp8={fp8!r} with args.fp8_recipe={recipe!r}: the torch_dist loader requires "
            "the FP8 scaling state (_extra_state), which fsdp_dtensor stores do not hold",
            hint="pass --allow-missing-fp8-extra-state to write the store without it "
            "(the scaling history restarts)",
        )


@REGISTRY.detector
def unknown_roots(facts: Facts) -> Iterator[Refusal]:
    """FC-R13: model keys under a top-level module the rules do not know."""
    known = SINGLE_ROOTS | VLM_ROOTS
    unknown = tuple(
        key
        for key in sorted(facts.model)
        if root_of(key) not in known
        and not any(key == p or key.startswith(p + ".") for p in facts.allowed_roots)
    )
    if unknown:
        roots = sorted({root_of(key) for key in unknown})
        yield Refusal(
            errors.FC_R13,
            f"model keys under unknown top-level modules {', '.join(roots)}",
            keys=unknown,
            hint="if they are stored identically in both formats, pass --allow-root "
            + " --allow-root ".join(roots),
        )


@REGISTRY.detector
def unverified_layers(facts: Facts) -> Iterator[Refusal]:
    """FC-R20: hybrid symbols or attention variants no rule has been verified for."""
    verified = REGISTRY.allowed(Allowlist.HYBRID_SYMBOLS)
    unverified = sorted(_symbols(facts) - verified - PENDING_SYMBOLS)
    if unverified:
        unknown = [s for s in unverified if s not in LAYER_SYMBOLS]
        yield Refusal(
            errors.FC_R20,
            f"hybrid layer symbols {''.join(unverified)} have no verified conversion rules"
            + (f" ({''.join(unknown)} is not a Megatron layer symbol)" if unknown else ""),
            hint="convert a model built from verified layer types only",
        )
    variant = facts.args.get("experimental_attention_variant", None)
    allowed = REGISTRY.allowed(Allowlist.ATTENTION_VARIANTS)
    if variant is not None and variant not in allowed | GDN_VARIANTS:
        yield Refusal(
            errors.FC_R20,
            f"args.experimental_attention_variant={variant!r} has no verified conversion rules",
            hint="convert a model without this attention variant",
        )


def _block_heads(blocks: tuple[str, ...]) -> tuple[str, ...]:
    """The key prefixes of the layer lists of ``blocks``."""
    return tuple(block + LAYER_SEGMENT for block in blocks)


def _mtp_keys(facts: Facts) -> tuple[str, ...]:
    """Keys of multi-token prediction layers."""
    return tuple(key for key in sorted(facts.model) if key.startswith(_block_heads(MTP_BLOCKS)))


def _key_evidence(keys: tuple[str, ...]) -> list[str]:
    return [f"{len(keys)} model key(s) such as {keys[0]}"] if keys else []


@REGISTRY.detector
def single_grouped_experts(facts: Facts) -> Iterator[Refusal]:
    """FC-R04: single grouped expert parameters (every local expert of a layer in one tensor).

    Megatron-FSDP does not support them (``validate_args``), and their
    fsdp_dtensor keys are not per expert.
    """
    flags = [f"args.{name}=True" for name in SINGLE_GROUPED_ARGS if facts.args.get(name, False)]
    keys = facts.keys_matching(SINGLE_GROUPED_EXPERTS)
    evidence = flags + _key_evidence(keys)
    if evidence:
        yield Refusal(
            errors.FC_R04,
            "single grouped expert parameters hold every local expert of a linear layer in one "
            f"tensor, which Megatron-FSDP does not support ({'; '.join(evidence)})",
            keys=keys,
            hint="convert a checkpoint of a run without --moe-single-grouped-weight and "
            "--moe-single-grouped-bias",
        )


@REGISTRY.detector
def ep_local_grouped_biases(facts: Facts) -> Iterator[Refusal]:
    """FC-R05: grouped expert biases in fsdp_dtensor with expert parallelism.

    Megatron-FSDP renumbers the weights of a grouped experts module from their
    expert-parallel-local index to the global one, but not the biases
    (``get_expert_index_from_key`` matches ``weight<e>`` only), so every
    expert-parallel rank saves and loads its local experts' biases under the
    same keys.
    """
    ep = facts.args.get("expert_model_parallel_size", None) or 1
    if ep <= 1:
        return
    if facts.source == FSDP_DTENSOR:
        keys = facts.keys_matching(GROUPED_EXPERT_BIAS)
        hint = (
            "the store holds the biases of one expert-parallel rank only: convert a checkpoint "
            "saved with --expert-model-parallel-size 1, or with sequential experts (no "
            "--moe-grouped-gemm), which name their biases by global index"
        )
    elif _fsdp_expert_form(facts) == GROUPED:
        keys = facts.keys_matching(TORCH_DIST_EXPERT_BIAS)
        hint = (
            "for a GPT model, pass --expert-naming sequential for a job built without "
            "--moe-grouped-gemm (SequentialMLP experts name their biases by global index); the "
            "hybrid stack spec always builds grouped experts"
        )
    else:
        return
    if keys:
        yield Refusal(
            errors.FC_R05,
            f"grouped expert biases with args.expert_model_parallel_size={ep}: Megatron-FSDP "
            "names them by their expert-parallel-local index, so the expert-parallel ranks "
            "overwrite one another's",
            keys=keys,
            hint=hint,
        )


def _fsdp_expert_form(facts: Facts) -> str:
    """How the fsdp_dtensor target names routed experts (``--expert-naming``, or as built).

    A hybrid model's are named as its stack spec builds them whatever the flag
    says (the expert layout refuses another naming as a usage error).
    """
    if facts.expert_naming != AUTO_EXPERT_NAMING and hybrid_pattern_of(facts.args) is None:
        return facts.expert_naming
    return built_expert_form(facts.args)[0]


@REGISTRY.detector
def legacy_grouped_experts(facts: Facts) -> Iterator[Refusal]:
    """FC-R13: parameters of the legacy ``GroupedMLP`` (``weight1``/``weight2``)."""
    keys = facts.keys_matching(LEGACY_GROUPED_EXPERTS)
    if keys:
        yield Refusal(
            errors.FC_R13,
            "parameters of the legacy GroupedMLP, whose weight1 and weight2 hold every local "
            "expert of a layer in one tensor",
            keys=keys,
            hint="load the checkpoint into a current Megatron model (TEGroupedMLP experts) and "
            "save it again",
        )


def singleton_local_shards_refusal(
    content_metadata: Any, keys: tuple[str, ...] = ()
) -> Refusal | None:
    """FC-R17 for a checkpoint saved with ``singleton_local_shards``, if it was.

    Args:
        content_metadata: The checkpoint's ``content_metadata``, whose flag says so.
        keys: Model keys in the singleton form (experts saved as
            ``experts.experts.<e>.<name>``, gated ``linear_fc1`` saved as
            SwiGLU halves ``<key>_w``/``<key>_v``), which say so too.
    """
    flagged = isinstance(content_metadata, Mapping) and bool(
        content_metadata.get(SINGLETON_LOCAL_SHARDS)
    )
    if not flagged and not keys:
        return None
    forms = [
        form
        for pattern, form in (
            (SINGLETON_EXPERT_SHARD, "its experts are saved one key per expert"),
            (GLU_HALF, "its gated linear_fc1 are saved as SwiGLU halves <key>_w and <key>_v"),
        )
        if any(pattern.search(key) for key in keys)
    ]
    return Refusal(
        errors.FC_R17,
        f"the checkpoint was saved with {SINGLETON_LOCAL_SHARDS}, which this tool does not read"
        + ("" if flagged else f" ({'; '.join(forms)})"),
        keys=(("content_metadata",) if flagged else ()) + keys,
        hint="load it with current Megatron and save it again without that option",
    )


@REGISTRY.detector
def singleton_local_shards(facts: Facts) -> Iterator[Refusal]:
    """FC-R17: a torch_dist source saved with ``singleton_local_shards``, as its
    ``content_metadata`` says or keys only that option writes show: experts saved one key per
    expert, or gated ``linear_fc1`` saved as SwiGLU halves (Megatron's SwiGLU factory names them
    ``<key>_w``/``<key>_v`` only then; torch_dist otherwise stores them whole)."""
    if facts.source != TORCH_DIST:
        return
    keys = facts.keys_matching(SINGLETON_EXPERT_SHARD) + facts.keys_matching(GLU_HALF)
    refusal = singleton_local_shards_refusal(facts.content_metadata, tuple(sorted(set(keys))))
    if refusal is not None:
        yield refusal


@REGISTRY.detector
def pending_mamba(facts: Facts) -> Iterator[Refusal]:
    """FC-P08: state-space mixers that are no Gated DeltaProduct mixers (Mamba-2).

    A model built with the Gated DeltaProduct spec, or whose torch_dist
    mixers are stored as its parts, passes: the architecture context tells
    its mixers' family from their names, the spec and their widths, and
    refuses any contradiction (``core/mixers.py``).
    """
    if is_gdp_spec(facts.args) or facts.keys_matching(GDP_PART):
        return
    evidence = ["hybrid layer symbol M"] if "M" in _symbols(facts) else []
    yield from _pending(errors.FC_P08, evidence, facts.keys_matching(MIXER_PARAM))


@REGISTRY.detector
def pending_linear_attention(facts: Facts) -> Iterator[Refusal]:
    """FC-P09: Gated DeltaNet (hybrid symbol G) and linear attention."""
    evidence = []
    if "G" in _symbols(facts):
        evidence.append("hybrid layer symbol G")
    variant = facts.args.get("experimental_attention_variant", None)
    if variant in GDN_VARIANTS:
        evidence.append(f"args.experimental_attention_variant={variant!r}")
    yield from _pending(errors.FC_P09, evidence, facts.keys_matching(GDN_PARAM))


@REGISTRY.detector
def pending_gpt_mtp_and_mla(facts: Facts) -> Iterator[Refusal]:
    """FC-P10: GPT multi-token prediction and multi-latent attention."""
    evidence = []
    keys = facts.keys_matching(MLA_PARAM)
    if hybrid_pattern_of(facts.args) is None:
        depths = facts.args.get("mtp_num_layers", None)
        if depths:
            evidence.append(f"args.mtp_num_layers={depths} without a hybrid layer pattern")
        keys += _mtp_keys(facts)
    if facts.args.get("multi_latent_attention", False):
        evidence.append("args.multi_latent_attention=True")
    if "+" in _symbols(facts):
        evidence.append("hybrid layer symbol +")
    yield from _pending(errors.FC_P10, evidence, tuple(sorted(set(keys))))


@REGISTRY.detector
def pending_optimizer(facts: Facts) -> Iterator[Refusal]:
    """FC-P11 / FC-P12: converting optimizer state (``--optimizer convert``)."""
    if facts.optimizer == "convert":
        code = errors.FC_P11 if facts.source == FSDP_DTENSOR else errors.FC_P12
        yield from _pending(code, ["--optimizer convert"])
