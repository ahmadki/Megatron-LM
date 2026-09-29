# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the fsdp_dtensor -> torch_dist reverse converter helpers.

These exercise the pure key/tensor transforms of
``tools/checkpoint/checkpoint_inspector.py`` (the inverse of
``convert_checkpoint``) without a distributed environment: prefix stripping,
optimizer-key reversal, SwiGLU merge, expert re-stacking, layer stacking, MTP
rename, homogeneity detection, and common-state unflattening.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

_INSPECTOR_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "tools", "checkpoint"
)
sys.path.insert(0, _INSPECTOR_DIR)

from checkpoint_inspector import (  # noqa: E402  (import after sys.path tweak)
    _assert_supported_scope,
    _gdp_heads_from_conv_width,
    _gdp_split_names,
    _homogeneous_layer_prefixes,
    _is_keep_fp32_key,
    _layers_are_homogeneous,
    _merge_swiglu,
    _model_param_dtype,
    _output_group_id,
    _plan_model_downcast,
    _rebuild_param_groups_from_meta,
    _restack_experts,
    _reverse_mtp_keys,
    _reverse_optimizer_state_key,
    _split_deltaproduct_projections,
    _split_gdn_projections,
    _split_mamba_projections,
    _stack_layers,
    _strip_fsdp_model_prefix,
    _unflatten,
)


class TestStripModelPrefix:
    def test_default_prefix(self):
        assert (
            _strip_fsdp_model_prefix("model.module.decoder.layers.0.mlp.linear_fc1.weight")
            == "decoder.layers.0.mlp.linear_fc1.weight"
        )

    def test_deeper_wrapper_run(self):
        assert (
            _strip_fsdp_model_prefix("model.module.module.embedding.word_embeddings.weight")
            == "embedding.word_embeddings.weight"
        )

    def test_bare_model_prefix(self):
        assert _strip_fsdp_model_prefix("model.output_layer.weight") == "output_layer.weight"

    def test_non_model_key_returns_none(self):
        assert _strip_fsdp_model_prefix("optimizer.state.module.module.module.x.exp_avg") is None
        assert _strip_fsdp_model_prefix("checkpoint_version") is None


class TestReverseOptimizerKey:
    def test_exp_avg(self):
        fsdp = "optimizer.state.module.module.module.decoder.layers.0.mlp.linear_fc2.weight.exp_avg"
        assert (
            _reverse_optimizer_state_key(fsdp)
            == "optimizer.state.exp_avg.decoder.layers.0.mlp.linear_fc2.weight"
        )

    def test_exp_avg_sq(self):
        fsdp = "optimizer.state.module.module.module.embedding.word_embeddings.weight.exp_avg_sq"
        assert (
            _reverse_optimizer_state_key(fsdp)
            == "optimizer.state.exp_avg_sq.embedding.word_embeddings.weight"
        )

    def test_step(self):
        fsdp = "optimizer.state.module.module.module.output_layer.weight.step"
        assert _reverse_optimizer_state_key(fsdp) == "optimizer.state.step.output_layer.weight"

    def test_wrapper_depth_agnostic(self):
        # Any run of ``module.`` wrappers is stripped.
        fsdp = "optimizer.state.module.decoder.final_norm.weight.exp_avg"
        assert (
            _reverse_optimizer_state_key(fsdp)
            == "optimizer.state.exp_avg.decoder.final_norm.weight"
        )


class TestReverseMTP:
    def test_top_level_mtp(self):
        out, n = _reverse_mtp_keys(
            {"mtp.layers.0.mtp_model_layer.mlp.linear_fc1.weight": torch.zeros(1)}
        )
        assert n == 1
        assert "mtp.layers.0.transformer_layer.mlp.linear_fc1.weight" in out

    def test_nested_mtp(self):
        out, n = _reverse_mtp_keys(
            {"language_model.mtp.layers.1.mtp_model_layer.self_attention.w": torch.zeros(1)}
        )
        assert n == 1
        assert "language_model.mtp.layers.1.transformer_layer.self_attention.w" in out

    def test_non_mtp_untouched(self):
        out, n = _reverse_mtp_keys({"decoder.layers.0.mlp.linear_fc1.weight": torch.zeros(1)})
        assert n == 0
        assert "decoder.layers.0.mlp.linear_fc1.weight" in out

    def test_hybrid_model_mtp_keeps_native_name(self):
        # HybridModel MTP layers store ``mtp_model_layer`` natively (no GPT rename).
        key = "mtp.layers.0.mtp_model_layer.layers.0.self_attention.linear_qkv.weight"
        args = SimpleNamespace(hybrid_layer_pattern="M*-M*E/*E")
        out, n = _reverse_mtp_keys({key: torch.zeros(1)}, args)
        assert n == 0
        assert key in out

    def test_gpt_args_still_renamed(self):
        args = SimpleNamespace(hybrid_layer_pattern=None)
        out, n = _reverse_mtp_keys({"mtp.layers.0.mtp_model_layer.w": torch.zeros(1)}, args)
        assert n == 1
        assert "mtp.layers.0.transformer_layer.w" in out


class TestMergeSwiglu:
    def test_dense_weight_merge(self):
        w = torch.arange(6.0).reshape(3, 2)
        v = torch.arange(6.0, 12.0).reshape(3, 2)
        out, n = _merge_swiglu(
            {
                "decoder.layers.0.mlp.linear_fc1.weight_w": w,
                "decoder.layers.0.mlp.linear_fc1.weight_v": v,
            }
        )
        assert n == 1
        merged = out["decoder.layers.0.mlp.linear_fc1.weight"]
        assert torch.equal(merged, torch.cat([w, v], dim=0))
        assert "decoder.layers.0.mlp.linear_fc1.weight_w" not in out

    def test_optimizer_swiglu_merge(self):
        w = torch.ones(2, 2)
        v = torch.zeros(2, 2)
        key_w = "optimizer.state.exp_avg.decoder.layers.0.mlp.linear_fc1.weight_w"
        key_v = "optimizer.state.exp_avg.decoder.layers.0.mlp.linear_fc1.weight_v"
        out, n = _merge_swiglu({key_w: w, key_v: v})
        assert n == 1
        assert "optimizer.state.exp_avg.decoder.layers.0.mlp.linear_fc1.weight" in out

    def test_indexed_expert_weight(self):
        out, n = _merge_swiglu(
            {
                "decoder.layers.0.mlp.experts.linear_fc1.weight3_w": torch.ones(1, 2),
                "decoder.layers.0.mlp.experts.linear_fc1.weight3_v": torch.ones(1, 2),
            }
        )
        assert n == 1
        assert "decoder.layers.0.mlp.experts.linear_fc1.weight3" in out

    def test_module_scope_filter(self):
        keys = {
            "language_model.decoder.layers.0.mlp.linear_fc1.weight_w": torch.ones(1, 1),
            "language_model.decoder.layers.0.mlp.linear_fc1.weight_v": torch.ones(1, 1),
            "vision_model.decoder.layers.0.mlp.linear_fc1.weight_w": torch.ones(1, 1),
            "vision_model.decoder.layers.0.mlp.linear_fc1.weight_v": torch.ones(1, 1),
        }
        out, n = _merge_swiglu(keys, swiglu_modules=["language_model"])
        assert n == 1
        assert "language_model.decoder.layers.0.mlp.linear_fc1.weight" in out
        assert "vision_model.decoder.layers.0.mlp.linear_fc1.weight_w" in out

    def test_fc2_not_merged(self):
        out, n = _merge_swiglu({"decoder.layers.0.mlp.linear_fc2.weight": torch.ones(2, 2)})
        assert n == 0


class TestRestackExperts:
    def test_stack_grouped_experts(self):
        t = {
            "decoder.layers.0.mlp.experts.linear_fc1.weight0": torch.zeros(4, 2),
            "decoder.layers.0.mlp.experts.linear_fc1.weight1": torch.ones(4, 2),
        }
        out, n = _restack_experts(t)
        assert n == 1
        key = "decoder.layers.0.mlp.experts.experts.linear_fc1.weight"
        assert out[key].shape == (2, 4, 2)
        assert torch.equal(out[key][1], torch.ones(4, 2))

    def test_optimizer_experts(self):
        t = {
            "optimizer.state.exp_avg.decoder.layers.0.mlp.experts.linear_fc2.weight0": torch.zeros(
                2
            ),
            "optimizer.state.exp_avg.decoder.layers.0.mlp.experts.linear_fc2.weight1": torch.ones(
                2
            ),
        }
        out, n = _restack_experts(t)
        assert n == 1
        assert (
            "optimizer.state.exp_avg.decoder.layers.0.mlp.experts.experts.linear_fc2.weight" in out
        )

    def test_non_contiguous_raises(self):
        t = {
            "decoder.layers.0.mlp.experts.linear_fc1.weight0": torch.zeros(1),
            "decoder.layers.0.mlp.experts.linear_fc1.weight2": torch.zeros(1),
        }
        with pytest.raises(AssertionError):
            _restack_experts(t)

    def test_shared_experts_untouched(self):
        t = {"decoder.layers.0.mlp.shared_experts.linear_fc1.weight": torch.zeros(2)}
        out, n = _restack_experts(t)
        assert n == 0
        assert "decoder.layers.0.mlp.shared_experts.linear_fc1.weight" in out

    def test_stack_non_grouped_local_experts(self):
        # SequentialMLP (no --moe-grouped-gemm) stores experts per local index;
        # mcore's sharded_state_dict still re-stacks them into the grouped key.
        t = {
            "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight": torch.zeros(4, 2),
            "decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight": torch.ones(4, 2),
        }
        out, n = _restack_experts(t)
        assert n == 1
        key = "decoder.layers.0.mlp.experts.experts.linear_fc1.weight"
        assert out[key].shape == (2, 4, 2)
        assert torch.equal(out[key][1], torch.ones(4, 2))

    def test_local_experts_optimizer(self):
        t = {
            "optimizer.state.exp_avg.decoder.layers.3.mlp.experts.local_experts.0.linear_fc2.weight": torch.zeros(
                2
            ),  # noqa: E501
            "optimizer.state.exp_avg.decoder.layers.3.mlp.experts.local_experts.1.linear_fc2.weight": torch.ones(
                2
            ),  # noqa: E501
        }
        out, n = _restack_experts(t)
        assert n == 1
        assert (
            "optimizer.state.exp_avg.decoder.layers.3.mlp.experts.experts.linear_fc2.weight" in out
        )

    def test_shared_experts_not_treated_as_local(self):
        # shared_experts must not match the local_experts pattern.
        t = {
            "decoder.layers.0.mlp.shared_experts.local_experts.0.linear_fc1.weight": torch.zeros(2)
        }
        out, n = _restack_experts(t)
        assert n == 0


class TestSplitGdnProjections:
    @staticmethod
    def _args():
        from types import SimpleNamespace

        # qk_dim = 4*64 = 256, v_dim = 8*64 = 512, num_value_heads = 8
        return SimpleNamespace(
            experimental_attention_variant="gated_delta_net",
            linear_num_key_heads=4,
            linear_key_head_dim=64,
            linear_num_value_heads=8,
            linear_value_head_dim=64,
        )

    def test_in_proj_split_names_sizes_and_order(self):
        qk, v, nvh = 256, 512, 8
        rows = 2 * qk + 2 * v + 2 * nvh  # 1552
        key = "decoder.layers.0.self_attention.in_proj.weight"
        blob = torch.arange(rows * 3).float().reshape(rows, 3)
        out, n = _split_gdn_projections({key: blob}, self._args())
        assert n == 1
        assert key not in out
        for name, size in [
            ("query", qk),
            ("key", qk),
            ("value", v),
            ("z", v),
            ("beta", nvh),
            ("alpha", nvh),
        ]:
            assert out[f"{key}.{name}"].shape == (size, 3)
        # concatenating the parts in factory order reproduces the fused blob.
        cat = torch.cat(
            [out[f"{key}.{name}"] for name in ["query", "key", "value", "z", "beta", "alpha"]],
            dim=0,
        )
        assert torch.equal(cat, blob)

    def test_conv1d_split(self):
        qk, v = 256, 512
        key = "decoder.layers.0.self_attention.conv1d.weight"
        out, n = _split_gdn_projections({key: torch.zeros(2 * qk + v, 1, 4)}, self._args())
        assert n == 1
        assert out[f"{key}.query"].shape == (qk, 1, 4)
        assert out[f"{key}.value"].shape == (v, 1, 4)

    def test_optimizer_state_split(self):
        key = "optimizer.state.exp_avg.decoder.layers.0.self_attention.in_proj.weight"
        out, n = _split_gdn_projections({key: torch.zeros(1552, 3)}, self._args())
        assert n == 1
        assert f"{key}.alpha" in out and f"{key}.query" in out

    def test_stacked_block_splits_second_dim(self):
        # A homogeneous (all-GDN) block stacks layers on axis 0, so the projection
        # dim is axis 1 and must be split there.
        key = "decoder.layers.self_attention.in_proj.weight"  # no explicit layer index
        out, n = _split_gdn_projections({key: torch.zeros(3, 1552, 5)}, self._args())
        assert n == 1
        assert out[f"{key}.query"].shape == (3, 256, 5)

    def test_noop_without_gdn_variant(self):
        from types import SimpleNamespace

        key = "decoder.layers.0.self_attention.in_proj.weight"
        out, n = _split_gdn_projections({key: torch.zeros(1552, 3)}, SimpleNamespace())
        assert n == 0 and key in out

    def test_noop_when_args_none(self):
        key = "decoder.layers.0.self_attention.in_proj.weight"
        out, n = _split_gdn_projections({key: torch.zeros(1552, 3)}, None)
        assert n == 0 and key in out


class TestStackLayers:
    def test_dense_stack(self):
        t = {
            "decoder.layers.0.mlp.linear_fc2.weight": torch.zeros(2, 3),
            "decoder.layers.1.mlp.linear_fc2.weight": torch.ones(2, 3),
            "embedding.word_embeddings.weight": torch.zeros(5, 3),
        }
        out, n = _stack_layers(t)
        assert n == 1
        assert out["decoder.layers.mlp.linear_fc2.weight"].shape == (2, 2, 3)
        # non-layer keys pass through unchanged
        assert out["embedding.word_embeddings.weight"].shape == (5, 3)

    def test_optimizer_and_model_stack_together(self):
        t = {
            "decoder.layers.0.w": torch.zeros(2),
            "decoder.layers.1.w": torch.zeros(2),
            "optimizer.state.exp_avg.decoder.layers.0.w": torch.zeros(2),
            "optimizer.state.exp_avg.decoder.layers.1.w": torch.zeros(2),
        }
        out, n = _stack_layers(t)
        assert n == 2
        assert out["decoder.layers.w"].shape == (2, 2)
        assert out["optimizer.state.exp_avg.decoder.layers.w"].shape == (2, 2)

    def test_heterogeneous_layers_raise(self):
        # 'a' present in both layers, 'b' only in layer 0 -> heterogeneous.
        t = {
            "decoder.layers.0.a": torch.zeros(1),
            "decoder.layers.1.a": torch.zeros(1),
            "decoder.layers.0.b": torch.zeros(1),
        }
        with pytest.raises(ValueError, match="Heterogeneous"):
            _stack_layers(t)


class TestLayersAreHomogeneous:
    def _uniform(self, per_layer_suffixes, n_layers, extra=()):
        keys = list(extra)
        for i in range(n_layers):
            for s in per_layer_suffixes:
                keys.append(f"decoder.layers.{i}.{s}")
        return keys

    def test_plain_dense_multi_layer_is_homogeneous(self):
        keys = self._uniform(
            ["self_attention.linear_qkv.weight", "mlp.linear_fc1.weight"],
            4,
            extra=["embedding.word_embeddings.weight"],
        )
        assert _layers_are_homogeneous(keys)

    def test_uniform_all_moe_is_homogeneous(self):
        # Every layer is MoE (moe_layer_freq == 1) -> mcore stacks the block.
        keys = self._uniform(
            [
                "self_attention.linear_qkv.weight",
                "mlp.router.weight",
                "mlp.experts.experts.linear_fc1.weight",
                "mlp.experts.experts.linear_fc2.weight",
            ],
            8,
        )
        assert _layers_are_homogeneous(keys)

    def test_interleaved_moe_and_dense_is_non_homogeneous(self):
        keys = [
            "decoder.layers.0.mlp.linear_fc1.weight",  # dense layer
            "decoder.layers.1.mlp.experts.experts.linear_fc1.weight",  # MoE layer
        ]
        assert not _layers_are_homogeneous(keys)

    def test_interleaved_linear_attention_is_non_homogeneous(self):
        keys = [
            "decoder.layers.0.self_attention.linear_qkv.weight",
            "decoder.layers.2.self_attention.in_proj.weight",  # GDN layer differs
            "decoder.layers.2.self_attention.conv1d.weight",
        ]
        assert not _layers_are_homogeneous(keys)

    def test_mtp_layers_do_not_block_decoder_stacking(self):
        keys = self._uniform(
            ["mlp.experts.experts.linear_fc1.weight"],
            4,
            extra=["mtp.layers.0.transformer_layer.mlp.linear_fc1.weight"],
        )
        assert _layers_are_homogeneous(keys)

    def test_no_per_layer_keys_returns_false(self):
        assert not _layers_are_homogeneous(["embedding.word_embeddings.weight"])


class TestUnflatten:
    def test_scalars_and_namespace_leaf(self):
        from types import SimpleNamespace

        ns = SimpleNamespace(x=1)
        flat = {
            "args": ns,
            "checkpoint_version": 3.0,
            "iteration": 100,
            "optimizer.param_groups.0.lr": 0.001,
            "optimizer.param_groups.0.weight_decay": 0.1,
        }
        out = _unflatten(flat)
        assert out["args"] is ns
        assert out["checkpoint_version"] == 3.0
        assert out["optimizer"]["param_groups"][0]["lr"] == 0.001
        assert out["optimizer"]["param_groups"][0]["weight_decay"] == 0.1

    def test_int_keyed_dict_becomes_list(self):
        out = _unflatten({"g.0": "a", "g.1": "b", "g.2": "c"})
        assert out["g"] == ["a", "b", "c"]


class TestRebuildParamGroupsFromMeta:
    _PREFIX = "optimizer.param_to_group_meta."

    def _meta(self, fqn, wd_mult, weight_decay, lr=1e-4, expert=False):
        # one flat entry per attribute, mirroring the fsdp on-disk layout
        attrs = {
            "wd_mult": wd_mult,
            "lr_mult": 1.0,
            "is_expert_parallel": expert,
            "is_decoupled_lr": False,
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "step": 25,
        }
        return {f"{self._PREFIX}module.module.module.{fqn}.{a}": v for a, v in attrs.items()}

    def test_groups_by_identifier_tuple(self):
        flat = {}
        # two decay params, one no-decay param, one expert param -> 3 distinct groups
        flat.update(self._meta("decoder.layers.0.self_attention.linear_qkv.weight", 1.0, 0.1))
        flat.update(self._meta("decoder.layers.1.self_attention.linear_qkv.weight", 1.0, 0.1))
        flat.update(self._meta("decoder.layers.0.self_attention.linear_qkv.bias", 0.0, 0.0))
        flat.update(
            self._meta("decoder.layers.0.mlp.experts.linear_fc1.weight0", 1.0, 0.1, expert=True)
        )

        groups = _rebuild_param_groups_from_meta(flat, self._PREFIX)
        idents = {
            (g["wd_mult"], g["lr_mult"], g["is_expert_parallel"], g["is_decoupled_lr"])
            for g in groups
        }
        assert len(groups) == 3
        assert idents == {
            (1.0, 1.0, False, False),
            (0.0, 1.0, False, False),
            (1.0, 1.0, True, False),
        }
        # full hyperparameters are carried; params are contiguous integer indices
        decay = next(g for g in groups if g["wd_mult"] == 1.0 and not g["is_expert_parallel"])
        assert decay["weight_decay"] == 0.1 and decay["betas"] == (0.9, 0.999)
        assert decay["step"] == 25 and "params" in decay
        allparams = sorted(i for g in groups for i in g["params"])
        assert allparams == list(range(4))  # one index per parameter, no gaps/dupes


class TestModelParamDtype:
    def test_bf16_fp16_fp32_and_none(self):
        from types import SimpleNamespace

        assert _model_param_dtype(SimpleNamespace(bf16=True, fp16=False)) == torch.bfloat16
        assert _model_param_dtype(SimpleNamespace(bf16=False, fp16=True)) == torch.float16
        assert _model_param_dtype(SimpleNamespace(bf16=False, fp16=False)) == torch.float32
        assert _model_param_dtype(None) is None


class TestSupportedScope:
    """The scope fence raises on architectures the converter cannot invert."""

    def test_unindexed_stacked_layer_buffer_raises(self):
        with pytest.raises(NotImplementedError, match="un-indexed"):
            _assert_supported_scope({"decoder.layers.norm.weight": None})

    def test_mamba_and_gdn_conv1d_are_not_fenced(self):
        # Mamba's fused ``mixer.conv1d_weight`` and GatedDeltaNet's dotted
        # ``self_attention.conv1d.weight`` are both handled by dedicated splits,
        # so neither may trip the fence.
        _assert_supported_scope(
            {
                "decoder.layers.0.mixer.conv1d_weight": None,
                "decoder.layers.0.mixer.conv1d_bias": None,
                "decoder.layers.0.mixer.in_proj.weight": None,
                "decoder.layers.0.self_attention.conv1d.weight": None,
                "optimizer.state.exp_avg.decoder.layers.0.mixer.conv1d_weight": None,
            }
        )

    def test_indexed_layers_and_plain_keys_pass(self):
        _assert_supported_scope(
            {
                "embedding.word_embeddings.weight": None,
                "decoder.layers.0.self_attention.linear_qkv.weight": None,
                "decoder.final_layernorm.weight": None,
                "mtp.layers.0.transformer_layer.self_attention.linear_qkv.weight": None,
                "optimizer.state.exp_avg.decoder.layers.3.mlp.linear_fc1.weight": None,
            }
        )


class TestKeepFp32Keys:
    """Persistent fp32 buffers (e.g. router.expert_bias) are protected from downcast."""

    def test_expert_bias_is_kept(self):
        assert _is_keep_fp32_key("decoder.layers.0.mlp.router.expert_bias")
        assert _is_keep_fp32_key("decoder.layers.mlp.router.expert_bias")  # stacked form

    def test_ordinary_weights_are_not_kept(self):
        assert not _is_keep_fp32_key("decoder.layers.0.mlp.linear_fc1.weight")
        assert not _is_keep_fp32_key("decoder.layers.0.mlp.router.weight")
        assert not _is_keep_fp32_key("output_layer.weight")

    def test_substring_expert_bias_not_falsely_matched(self):
        # only a whole trailing ``.expert_bias`` segment matches, not a substring.
        assert not _is_keep_fp32_key("decoder.layers.0.mlp.router.expert_bias_extra")


class TestPlanModelDowncast:
    """fp32 model params are preserved unless the source is an FSDP master store.

    Regression: a name allow-list (``expert_bias`` only) downcast every other fp32
    model tensor to bf16 on a mixed-precision source, silently losing precision on
    ``A_log``, retention/mixing ``logit`` and — worst — the router's
    ``qb_bin_bounds``, whose perturbation changes routing decisions.
    """

    # A mixed-precision source: bf16 weights next to genuinely-fp32 params.
    MIXED = {
        "decoder.layers.0.self_attention.linear_qkv.weight": torch.bfloat16,
        "decoder.layers.0.mixer.A_log": torch.float32,
        "decoder.layers.0.mlp.router.qb_bin_bounds": torch.float32,
        "decoder.layers.0.mlp.router.expert_bias": torch.float32,
    }
    # A native Megatron-FSDP store: every trainable param is fp32 (it *is* the
    # master); ``expert_bias`` is a buffer and carries no optimizer moments.
    MASTERS = {
        "decoder.layers.0.self_attention.linear_qkv.weight": torch.float32,
        "decoder.layers.0.mixer.A_log": torch.float32,
        "decoder.layers.0.mlp.router.expert_bias": torch.float32,
    }
    MASTER_MOMENTS = {
        "decoder.layers.0.self_attention.linear_qkv.weight",
        "decoder.layers.0.mixer.A_log",
    }

    def test_mixed_precision_source_downcasts_nothing(self):
        assert _plan_model_downcast(self.MIXED, set(), torch.bfloat16) == set()

    def test_fp32_model_param_survives_the_round_trip(self):
        # Apply the plan the way the converter does and check the dtypes out.
        tensors = {k: torch.zeros(2, dtype=d) for k, d in self.MIXED.items()}
        for key in _plan_model_downcast(self.MIXED, set(), torch.bfloat16):
            tensors[key] = tensors[key].to(torch.bfloat16)
        assert {k: v.dtype for k, v in tensors.items()} == self.MIXED

    def test_master_store_downcasts_trainable_params_only(self):
        assert (
            _plan_model_downcast(self.MASTERS, self.MASTER_MOMENTS, torch.bfloat16)
            == self.MASTER_MOMENTS
        )

    def test_master_store_without_optimizer_falls_back_to_allow_list(self):
        # No moments at all (weights-only FSDP store) -> no trainability signal.
        assert _plan_model_downcast(self.MASTERS, set(), torch.bfloat16) == {
            "decoder.layers.0.self_attention.linear_qkv.weight",
            "decoder.layers.0.mixer.A_log",
        }

    def test_fp32_training_and_unknown_dtype_downcast_nothing(self):
        assert _plan_model_downcast(self.MASTERS, self.MASTER_MOMENTS, torch.float32) == set()
        assert _plan_model_downcast(self.MASTERS, self.MASTER_MOMENTS, None) == set()

    def test_all_half_source_downcasts_nothing(self):
        halves = {"decoder.layers.0.mlp.linear_fc1.weight": torch.bfloat16}
        assert _plan_model_downcast(halves, set(), torch.bfloat16) == set()


class TestPerNamespaceStacking:
    """Stacking is decided per ``...layers`` namespace, not globally.

    Regression: a VLM pairs a uniform vision block (stored *stacked* by
    ``TransformerBlock.sharded_state_dict``) with a hybrid language decoder (stored
    per-layer). A single global decision emitted the vision block per-layer, so the
    model could not find the stacked key it asks for.
    """

    def _mixed_store_keys(self):
        keys = []
        for i in range(3):  # homogeneous vision block
            keys.append(f"vision_model.decoder.layers.{i}.self_attention.linear_qkv.weight")
            keys.append(f"vision_model.decoder.layers.{i}.mlp.linear_fc1.weight")
        # heterogeneous language decoder: layer 1 is a linear-attention layer
        keys.append("language_model.decoder.layers.0.self_attention.linear_qkv.weight")
        keys.append("language_model.decoder.layers.1.self_attention.in_proj.weight")
        return keys

    def test_only_the_homogeneous_namespace_is_selected(self):
        assert _homogeneous_layer_prefixes(self._mixed_store_keys()) == {
            "vision_model.decoder.layers"
        }

    def test_global_detector_still_reports_the_mixed_store_as_non_homogeneous(self):
        assert not _layers_are_homogeneous(self._mixed_store_keys())

    def test_stack_layers_stacks_only_the_selected_namespace(self):
        tensors = {k: torch.zeros(2) for k in self._mixed_store_keys()}
        out, n = _stack_layers(tensors, stack_prefixes={"vision_model.decoder.layers"})
        assert n == 2  # the two vision params, stacked over 3 layers
        assert out["vision_model.decoder.layers.self_attention.linear_qkv.weight"].shape == (3, 2)
        assert out["vision_model.decoder.layers.mlp.linear_fc1.weight"].shape == (3, 2)
        # the heterogeneous decoder is untouched (and does not raise)
        assert out["language_model.decoder.layers.0.self_attention.linear_qkv.weight"].shape == (2,)
        assert "language_model.decoder.layers.1.self_attention.in_proj.weight" in out

    def test_optimizer_state_follows_its_model_namespace(self):
        # ``stack_prefixes`` is keyed by bare model namespace, but optimizer keys
        # carry an ``optimizer.state.<subkey>.`` prefix. They must stack with the
        # param they mirror, not silently fall through to per-layer.
        tensors = {}
        for i in range(2):
            for sub in ("exp_avg", "exp_avg_sq"):
                tensors[f"optimizer.state.{sub}.decoder.layers.{i}.mlp.linear_fc1.weight"] = (
                    torch.zeros(2)
                )
            tensors[f"decoder.layers.{i}.mlp.linear_fc1.weight"] = torch.zeros(2)
        out, n = _stack_layers(tensors, stack_prefixes={"decoder.layers"})
        assert n == 3  # the model param plus both optimizer moments
        assert out["decoder.layers.mlp.linear_fc1.weight"].shape == (2, 2)
        for sub in ("exp_avg", "exp_avg_sq"):
            stacked = out[f"optimizer.state.{sub}.decoder.layers.mlp.linear_fc1.weight"]
            assert stacked.shape == (2, 2)

    def test_empty_prefix_set_keeps_everything_per_layer(self):
        tensors = {k: torch.zeros(2) for k in self._mixed_store_keys()}
        out, n = _stack_layers(tensors, stack_prefixes=set())
        assert n == 0 and set(out) == set(tensors)

    def test_default_still_stacks_every_namespace(self):
        # Back-compat with the global ``--stack-layers`` override.
        keys = [f"decoder.layers.{i}.mlp.linear_fc1.weight" for i in range(2)]
        out, n = _stack_layers({k: torch.zeros(2) for k in keys})
        assert n == 1 and out["decoder.layers.mlp.linear_fc1.weight"].shape == (2, 2)

    def test_sharding_group_id_collapses_only_selected_namespaces(self):
        prefixes = {"vision_model.decoder.layers"}
        v0 = "vision_model.decoder.layers.0.mlp.linear_fc1.weight"
        v1 = "vision_model.decoder.layers.1.mlp.linear_fc1.weight"
        l0 = "language_model.decoder.layers.0.mlp.linear_fc1.weight"
        l1 = "language_model.decoder.layers.1.mlp.linear_fc1.weight"
        # stacked namespace: every layer must land on one rank
        assert _output_group_id(v0, prefixes) == _output_group_id(v1, prefixes)
        # per-layer namespace: layers stay independent
        assert _output_group_id(l0, prefixes) != _output_group_id(l1, prefixes)


class TestSplitMambaProjections:
    """Reverse split of fused Mamba-2 in_proj/conv1d into named factory sub-keys."""

    # d_state=16, ngroups=2, head_dim=8 -> nheads=4, d_inner=32, gds=32.
    # in_proj width = 2*32 + 2*32 + 4 = 132 ; conv width = 32 + 2*32 = 96.
    ARGS = SimpleNamespace(mamba_state_dim=16, mamba_num_groups=2, mamba_head_dim=8)

    def test_in_proj_split_names_sizes_and_reproduce(self):
        blob = torch.randn(132, 24)
        out, n = _split_mamba_projections(
            {"decoder.layers.0.mixer.in_proj.weight": blob}, self.ARGS
        )
        assert n == 1
        base = "decoder.layers.0.mixer.in_proj.weight"
        names = ("z", "x", "B", "C", "dt")
        assert set(out) == {f"{base}.{s}" for s in names}
        assert [out[f"{base}.{s}"].shape[0] for s in names] == [32, 32, 32, 32, 4]
        # concatenation reproduces the original fused blob (order z,x,B,C,dt)
        assert torch.equal(torch.cat([out[f"{base}.{s}"] for s in names], dim=0), blob)

    def test_conv1d_weight_renamed_and_split(self):
        out, n = _split_mamba_projections(
            {"decoder.layers.1.mixer.conv1d_weight": torch.randn(96, 1, 4)}, self.ARGS
        )
        assert n == 1
        # conv1d_weight -> conv1d.weight (dotted on-disk key), split x/B/C.
        base = "decoder.layers.1.mixer.conv1d.weight"
        assert set(out) == {f"{base}.{s}" for s in ("x", "B", "C")}
        assert [out[f"{base}.{s}"].shape[0] for s in ("x", "B", "C")] == [32, 32, 32]

    def test_conv1d_bias_renamed_and_split(self):
        out, n = _split_mamba_projections(
            {"decoder.layers.0.mixer.conv1d_bias": torch.randn(96)}, self.ARGS
        )
        assert n == 1
        assert set(out) == {f"decoder.layers.0.mixer.conv1d.bias.{s}" for s in ("x", "B", "C")}

    def test_optimizer_state_split(self):
        key = "optimizer.state.exp_avg.decoder.layers.0.mixer.in_proj.weight"
        out, n = _split_mamba_projections({key: torch.randn(132, 24)}, self.ARGS)
        assert n == 1
        assert f"{key}.dt" in out and out[f"{key}.dt"].shape[0] == 4

    def test_stacked_block_splits_second_dim(self):
        # Homogeneous all-Mamba block: leading num-layers axis, split dim 1.
        out, n = _split_mamba_projections(
            {"decoder.layers.mixer.in_proj.weight": torch.randn(3, 132, 24)}, self.ARGS
        )
        assert n == 1
        assert out["decoder.layers.mixer.in_proj.weight.z"].shape == (3, 32, 24)

    def test_passthrough_and_noop_keys_untouched(self):
        src = {
            "decoder.layers.0.mixer.A_log": torch.randn(4),
            "decoder.layers.0.mixer.out_proj.weight": torch.randn(24, 32),
            "decoder.layers.0.mlp.linear_fc1.weight": torch.randn(8, 8),
        }
        out, n = _split_mamba_projections(dict(src), self.ARGS)
        assert n == 0 and set(out) == set(src)

    def test_missing_args_raises(self):
        with pytest.raises(NotImplementedError, match="Mamba"):
            _split_mamba_projections(
                {"decoder.layers.0.mixer.in_proj.weight": torch.randn(132, 8)}, None
            )


class TestOutputGroupId:
    """Sharded-convert grouping: keys of one transform group share a group id."""

    def test_swiglu_halves_share_group(self):
        w = "decoder.layers.0.mlp.linear_fc1.weight_w"
        v = "decoder.layers.0.mlp.linear_fc1.weight_v"
        assert _output_group_id(w, False) == _output_group_id(v, False)

    def test_grouped_experts_share_group(self):
        g0 = "decoder.layers.0.mlp.experts.linear_fc1.weight0"
        g3 = "decoder.layers.0.mlp.experts.linear_fc1.weight3"
        assert _output_group_id(g0, False) == _output_group_id(g3, False)

    def test_sequential_experts_share_group(self):
        e0 = "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight"
        e5 = "decoder.layers.0.mlp.experts.local_experts.5.linear_fc1.weight"
        assert _output_group_id(e0, False) == _output_group_id(e5, False)

    def test_layer_index_grouped_only_when_stacking(self):
        l0 = "decoder.layers.0.mlp.linear_fc1.weight"
        l1 = "decoder.layers.1.mlp.linear_fc1.weight"
        # homogeneous/stacked -> all layers of a param share a group (co-resident to stack)
        assert _output_group_id(l0, True) == _output_group_id(l1, True)
        # per-layer -> each layer is its own group (finer sharding)
        assert _output_group_id(l0, False) != _output_group_id(l1, False)

    def test_distinct_params_distinct_groups(self):
        a = "decoder.layers.0.self_attention.linear_qkv.weight"
        b = "decoder.layers.0.mlp.linear_fc2.weight"
        assert _output_group_id(a, True) != _output_group_id(b, True)


def _gdp_args(num_householder=3, heads=24, head_dim=64, groups=24, state=128):
    """Checkpoint args for a Gated DeltaProduct mixer (nm4-style defaults)."""
    return SimpleNamespace(
        gdp_num_householder=num_householder,
        mamba_num_heads=heads,
        mamba_head_dim=head_dim,
        mamba_num_groups=groups,
        mamba_state_dim=state,
    )


def _gdp_widths(args):
    d_inner = args.mamba_num_heads * args.mamba_head_dim
    gds = args.mamba_num_groups * args.mamba_state_dim
    m = args.gdp_num_householder
    in_proj = d_inner * (1 + m) + gds * (m + 1) + args.mamba_num_heads * (m + 1)
    conv = d_inner * m + gds * (m + 1)
    return in_proj, conv


class TestGdpSplitNames:
    """The concatenation order is the contract between the two directions."""

    def test_in_proj_order(self):
        assert _gdp_split_names(3, is_conv=False) == [
            "z", "V0", "V1", "V2", "K0", "K1", "K2", "Q", "b0", "b1", "b2", "a"
        ]

    def test_conv_order(self):
        assert _gdp_split_names(3, is_conv=True) == [
            "V0", "V1", "V2", "K0", "K1", "K2", "Q"
        ]

    def test_single_householder(self):
        assert _gdp_split_names(1, is_conv=False) == ["z", "V0", "K0", "Q", "b0", "a"]



class TestSplitDeltaProductProjections:
    def test_sections_match_gated_delta_product_layout(self):
        args = _gdp_args()
        in_proj_w, conv_w = _gdp_widths(args)
        prefix = "decoder.layers.0.mixer."
        tensors = {
            f"{prefix}in_proj.weight": torch.arange(in_proj_w * 4, dtype=torch.float32).reshape(
                in_proj_w, 4
            ),
            f"{prefix}conv1d.weight": torch.arange(conv_w * 4, dtype=torch.float32).reshape(
                conv_w, 1, 4
            ),
            f"{prefix}out_proj.weight": torch.zeros(4, 4),
        }
        out, n = _split_deltaproduct_projections(tensors, args)
        assert n == 2
        # Fused keys are replaced by their sections; unrelated keys pass through.
        assert f"{prefix}in_proj.weight" not in out
        assert f"{prefix}out_proj.weight" in out
        d_inner = args.mamba_num_heads * args.mamba_head_dim
        gds = args.mamba_num_groups * args.mamba_state_dim
        assert out[f"{prefix}in_proj.weight.z"].shape[0] == d_inner
        assert out[f"{prefix}in_proj.weight.V0"].shape[0] == d_inner
        assert out[f"{prefix}in_proj.weight.K0"].shape[0] == gds
        assert out[f"{prefix}in_proj.weight.Q"].shape[0] == gds
        assert out[f"{prefix}in_proj.weight.b0"].shape[0] == args.mamba_num_heads
        assert out[f"{prefix}in_proj.weight.a"].shape[0] == args.mamba_num_heads
        assert out[f"{prefix}conv1d.weight.Q"].shape == (gds, 1, 4)

    def test_noop_without_mamba_dims_in_args(self):
        tensors = {"decoder.layers.0.mixer.in_proj.weight": torch.zeros(8, 4)}
        out, n = _split_deltaproduct_projections(tensors, SimpleNamespace())
        assert n == 0 and out == tensors

    def test_width_mismatch_passes_through_for_mamba(self):
        """A non-DeltaProduct width is left alone, not force-split."""
        args = _gdp_args()
        tensors = {"decoder.layers.0.mixer.in_proj.weight": torch.zeros(7, 4)}
        out, n = _split_deltaproduct_projections(tensors, args)
        assert n == 0 and out == tensors

    def test_gdn_self_attention_keys_are_out_of_scope(self):
        args = _gdp_args()
        in_proj_w, _ = _gdp_widths(args)
        key = "decoder.layers.0.self_attention.in_proj.weight"
        tensors = {key: torch.zeros(in_proj_w, 4)}
        out, n = _split_deltaproduct_projections(tensors, args)
        assert n == 0 and out == tensors



class TestDeltaProductVsMambaDisambiguation:
    """Both mixers expose ``mixer.in_proj.weight``; the fused width decides which."""

    def test_mamba_checkpoint_still_splits_despite_gdp_default_in_args(self):
        """gdp_num_householder defaults to 3 and is back-filled onto old checkpoints,
        so it must not be used as a presence test for DeltaProduct."""
        args = _gdp_args()  # carries gdp_num_householder=3 like every fork checkpoint
        d_inner = args.mamba_num_heads * args.mamba_head_dim
        gds = args.mamba_num_groups * args.mamba_state_dim
        mamba_w = 2 * d_inner + 2 * gds + args.mamba_num_heads
        key = "decoder.layers.0.mixer.in_proj.weight"
        tensors = {key: torch.randn(mamba_w, 4)}

        # The DeltaProduct split must decline it...
        out, n_gdp = _split_deltaproduct_projections(dict(tensors), args)
        assert n_gdp == 0 and set(out) == {key}
        # ...and the Mamba split must still handle it.
        out, n_mamba = _split_mamba_projections(out, args)
        assert n_mamba == 1
        assert f"{key}.z" in out and f"{key}.dt" in out

    def test_deltaproduct_width_is_claimed_by_the_gdp_split(self):
        args = _gdp_args()
        in_proj_w, _ = _gdp_widths(args)
        key = "decoder.layers.0.mixer.in_proj.weight"
        out, n = _split_deltaproduct_projections({key: torch.randn(in_proj_w, 4)}, args)
        assert n == 1 and f"{key}.V2" in out

    def test_widths_never_collide(self):
        """The discriminator is only sound if the two layouts cannot share a width."""
        for m in (1, 2, 3, 4):
            args = _gdp_args(num_householder=m)
            gdp_w, _ = _gdp_widths(args)
            d_inner = args.mamba_num_heads * args.mamba_head_dim
            gds = args.mamba_num_groups * args.mamba_state_dim
            mamba_w = 2 * d_inner + 2 * gds + args.mamba_num_heads
            assert gdp_w != mamba_w, m



class TestMambaNotClaimedWhenHeadsAbsent:
    """Regression: mamba_num_heads is Optional and defaults to None. Deriving the head
    count from the in_proj width made the DeltaProduct/Mamba-2 width discriminator
    self-fulfilling -- the layout summed to the observed width by construction, so a pure
    Mamba-2 in_proj was claimed and silently split into [z,V*,K*,Q,b*,a]."""

    @staticmethod
    def _args(head_dim=64, groups=8, state=128):
        return SimpleNamespace(
            mamba_head_dim=head_dim, mamba_num_groups=groups, mamba_state_dim=state,
            mamba_num_heads=None,   # the default
            gdp_num_householder=3,  # back-filled onto every checkpoint
        )

    def test_pure_mamba_checkpoint_is_left_for_the_mamba_split(self):
        args = self._args()
        nheads, d_inner, gds = 32, 32 * 64, 8 * 128
        width = 2 * d_inner + 2 * gds + nheads
        key = "decoder.layers.0.mixer.in_proj.weight"
        tensors = {key: torch.randn(width, 8)}

        out, n_gdp = _split_deltaproduct_projections(dict(tensors), args)
        assert n_gdp == 0 and set(out) == {key}

        out, n_mamba = _split_mamba_projections(out, args)
        assert n_mamba == 1
        assert sorted(k.rsplit(".", 1)[-1] for k in out) == ["B", "C", "dt", "x", "z"]

    @pytest.mark.parametrize(
        "head_dim,groups,state,nheads",
        [(64, 1, 128, 4), (64, 8, 128, 32), (64, 8, 256, 64), (128, 8, 256, 32)],
    )
    def test_known_false_positive_widths_are_not_claimed(self, head_dim, groups, state, nheads):
        """Configurations whose Mamba-2 in_proj width factored into a bogus layout."""
        args = self._args(head_dim, groups, state)
        d_inner, gds = nheads * head_dim, groups * state
        width = 2 * d_inner + 2 * gds + nheads
        tensors = {"decoder.layers.0.mixer.in_proj.weight": torch.zeros(width, 4)}
        _, n = _split_deltaproduct_projections(tensors, args)
        assert n == 0

    def test_deltaproduct_still_works_without_mamba_num_heads(self):
        """The dotted conv1d is DeltaProduct-only, so it can still supply the head count."""
        args = self._args(head_dim=64, groups=24, state=128)
        m, head_dim, gds = 3, 64, 24 * 128
        nheads = 24
        d_inner = nheads * head_dim
        in_w = d_inner * (1 + m) + gds * (m + 1) + nheads * (m + 1)
        conv_w = d_inner * m + gds * (m + 1)
        pre = "decoder.layers.0.mixer."
        tensors = {f"{pre}in_proj.weight": torch.randn(in_w, 4),
                   f"{pre}conv1d.weight": torch.randn(conv_w, 1, 4)}
        out, n = _split_deltaproduct_projections(tensors, args)
        assert n == 2, "conv1d width should have supplied nheads"
        assert f"{pre}in_proj.weight.V2" in out and f"{pre}conv1d.weight.Q" in out

    def test_conv_width_factorisation_rejects_non_deltaproduct(self):
        assert _gdp_heads_from_conv_width(24 * 64 * 3 + 3072 * 4, 3, 64, 3072) == 24
        assert _gdp_heads_from_conv_width(7, 3, 64, 3072) is None
        assert _gdp_heads_from_conv_width(None, 3, 64, 3072) is None



class TestHeadsRecoveredFromWidth:
    """mamba_num_heads is Optional and defaults to None."""

    def test_in_proj_alone_does_not_supply_the_head_count(self):
        """An in_proj width must never be used to derive nheads: the layout is built from
        that very number, so any width factors and the discriminator self-fulfils."""
        args = _gdp_args()
        in_proj_w, _ = _gdp_widths(args)
        args_no_heads = SimpleNamespace(
            gdp_num_householder=args.gdp_num_householder,
            mamba_head_dim=args.mamba_head_dim,
            mamba_num_groups=args.mamba_num_groups,
            mamba_state_dim=args.mamba_state_dim,
        )
        key = "decoder.layers.0.mixer.in_proj.weight"
        out, n = _split_deltaproduct_projections({key: torch.randn(in_proj_w, 3)}, args_no_heads)
        assert n == 0 and set(out) == {key}

    def test_dotted_conv1d_supplies_the_head_count(self):
        args = _gdp_args()
        in_proj_w, conv_w = _gdp_widths(args)
        args_no_heads = SimpleNamespace(
            gdp_num_householder=args.gdp_num_householder,
            mamba_head_dim=args.mamba_head_dim,
            mamba_num_groups=args.mamba_num_groups,
            mamba_state_dim=args.mamba_state_dim,
        )
        pre = "decoder.layers.0.mixer."
        tensors = {f"{pre}in_proj.weight": torch.randn(in_proj_w, 3),
                   f"{pre}conv1d.weight": torch.randn(conv_w, 1, 4)}
        out, n = _split_deltaproduct_projections(tensors, args_no_heads)
        assert n == 2
        assert out[f"{pre}in_proj.weight.a"].shape[0] == args.mamba_num_heads


class TestMixerProjectionsShareARank:
    """A DeltaProduct in_proj split needs the sibling conv1d width to recover the head
    count, so a multi-rank convert must not hand them to different ranks."""

    PRE = "decoder.layers.0.mixer."

    def test_in_proj_and_conv1d_reduce_to_one_group(self):
        ids = {_output_group_id(self.PRE + n, do_stack=False)
               for n in ("in_proj.weight", "conv1d.weight", "conv1d.bias", "in_proj.bias")}
        assert len(ids) == 1, ids

    def test_mamba_fused_conv_joins_the_same_group(self):
        ids = {_output_group_id(self.PRE + n, do_stack=False)
               for n in ("in_proj.weight", "conv1d_weight", "conv1d_bias")}
        assert len(ids) == 1, ids

    def test_other_mixer_params_stay_separate(self):
        base = _output_group_id(self.PRE + "in_proj.weight", do_stack=False)
        for other in ("out_proj.weight", "norm.weight", "A_log", "dt_bias"):
            assert _output_group_id(self.PRE + other, do_stack=False) != base

    def test_distinct_layers_stay_separate(self):
        a = _output_group_id("decoder.layers.0.mixer.in_proj.weight", do_stack=False)
        b = _output_group_id("decoder.layers.1.mixer.in_proj.weight", do_stack=False)
        assert a != b

    def test_optimizer_subkey_follows_its_param(self):
        a = _output_group_id(self.PRE + "in_proj.weight", do_stack=False)
        b = _output_group_id(self.PRE + "conv1d.weight", do_stack=False)
        assert a == b


class TestDeltaProductWidthMismatchIsLoud:
    """A mixer with a dotted conv1d IS DeltaProduct. If its in_proj width disagrees with
    the args-derived layout, passing it on would hand it to the Mamba-2 transform, whose
    own width assert can coincidentally hold -- emitting z/x/B/C/dt with wrong section
    sizes and no error."""

    @staticmethod
    def _args(m):
        return SimpleNamespace(mamba_head_dim=64, mamba_num_groups=24, mamba_state_dim=64,
                               mamba_num_heads=8, gdp_num_householder=m)

    def test_args_disagreeing_with_tensors_raises(self):
        true_m = 2
        a = self._args(3)                      # back-filled default, disagrees with tensors
        d, g, n = 8 * 64, 24 * 64, 8
        in_w = d * (1 + true_m) + g * (true_m + 1) + n * (true_m + 1)
        cv_w = d * true_m + g * (true_m + 1)
        pre = "decoder.layers.0.mixer."
        tensors = {f"{pre}in_proj.weight": torch.randn(in_w, 4),
                   f"{pre}conv1d.weight": torch.randn(cv_w, 1, 4)}
        with pytest.raises(NotImplementedError, match="does not match the layout"):
            _split_deltaproduct_projections(tensors, a)

    def test_mamba_only_mixer_still_passes_through_silently(self):
        """No dotted conv1d -> not DeltaProduct -> must NOT raise, just decline."""
        a = self._args(3)
        d, g, n = 8 * 64, 24 * 64, 8
        pre = "decoder.layers.0.mixer."
        tensors = {f"{pre}in_proj.weight": torch.randn(2 * d + 2 * g + n, 4),
                   f"{pre}conv1d_weight": torch.randn(d + 2 * g, 4)}
        out, k = _split_deltaproduct_projections(dict(tensors), a)
        assert k == 0 and set(out) == set(tensors)

    def test_hybrid_splits_each_mixer_by_its_own_kind(self):
        a = self._args(3)
        d, g, n, m = 8 * 64, 24 * 64, 8, 3
        gdp_in = d * (1 + m) + g * (m + 1) + n * (m + 1)
        gdp_cv = d * m + g * (m + 1)
        tensors = {
            "decoder.layers.0.mixer.in_proj.weight": torch.randn(2 * d + 2 * g + n, 4),
            "decoder.layers.0.mixer.conv1d_weight": torch.randn(d + 2 * g, 4),
            "decoder.layers.1.mixer.in_proj.weight": torch.randn(gdp_in, 4),
            "decoder.layers.1.mixer.conv1d.weight": torch.randn(gdp_cv, 1, 4),
        }
        out, k = _split_deltaproduct_projections(dict(tensors), a)
        assert k == 2                                    # only the DeltaProduct mixer
        assert "decoder.layers.0.mixer.in_proj.weight" in out     # Mamba-2 untouched
        assert "decoder.layers.1.mixer.in_proj.weight.V2" in out
