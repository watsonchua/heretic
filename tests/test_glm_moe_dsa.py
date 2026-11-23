"""Offline GLM-5.3 packed-expert and abliteration regressions."""

import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

from heretic.config import QuantizationMethod, Settings
from heretic.glm import GlmExpertViews
from heretic.model import Model
from heretic.modifiers.abliteration import (
    Abliteration,
    Parameters,
    RowNormalization,
    WeightDistribution,
)
from heretic.modifiers.abliteration import Settings as AbliterationSettings
from heretic.plugin import Context


def tiny_model():
    config = GlmMoeDsaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=4,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=2,
        mlp_layer_types=["dense", "sparse", "sparse"],
        indexer_types=["full", "full", "shared"],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    model = Model.__new__(Model)
    model.model = GlmMoeDsaForCausalLM(config).eval()
    model.model.config.name_or_path = "tiny-glm-dsa"
    model.settings = Settings.model_construct(
        model="tiny-glm-dsa",
        quantization=QuantizationMethod.NONE,
        seed=42,
        batch_size=1,
        offload_outputs_to_cpu=True,
    )
    model.needs_reload = False
    return model


class GlmMoeDsaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = tiny_model()
        self.tokens = torch.tensor([[3, 4, 5]])

    def test_routing_storage_and_projection_discovery(self):
        packed = self.model.get_layers()[1].mlp.experts
        hidden = torch.randn(3, 16)
        indices = torch.tensor([[0, 1], [0, 2], [2, 1]])
        weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])
        with torch.no_grad():
            expected = packed(hidden, indices, weights)
        self.assertEqual(len(self.model.get_layer_modules(0)["mlp.down_proj"]), 1)
        modules = self.model.get_layer_modules(1)["mlp.down_proj"]
        self.assertEqual(len(modules), 5)
        self.assertEqual(len({id(m) for m in modules}), 5)
        views = self.model.get_layers()[1].mlp.experts
        self.assertIsInstance(views, GlmExpertViews)
        with torch.no_grad():
            torch.testing.assert_close(views(hidden, indices, weights), expected)
        for index, expert in enumerate(views.experts):
            self.assertEqual(
                expert.down_proj.weight.data_ptr(), packed.down_proj[index].data_ptr()
            )
        self.assertEqual(
            self.model.get_modifiable_components(), ["attn.o_proj", "mlp.down_proj"]
        )

    def test_residuals_abliteration_reset_and_native_export(self):
        # Use the real Transformers generation path without downloading a tokenizer.
        def generate(prompts, **kwargs):
            tokens = torch.tensor([prompts[0]])
            with torch.no_grad():
                return {"input_ids": tokens}, self.model.model.generate(
                    tokens, **kwargs
                )

        self.model.generate = generate
        good = [[3, 4, 5]]
        bad = [[6, 7, 8]]
        residuals = self.model.get_residuals(good)
        self.assertEqual(tuple(residuals.shape), (1, 4, 16))
        self.assertTrue(torch.isfinite(residuals).all())
        existing_hooks = {
            id(m): set(m._forward_hooks) for m in self.model.model.modules()
        }
        io = self.model.get_module_io(good)
        self.assertEqual(len(io), 3)
        self.assertTrue(any(index > 0 for index in io[1]["mlp.down_proj"]))
        for module in self.model.model.modules():
            self.assertEqual(
                set(module._forward_hooks), existing_hooks.get(id(module), set())
            )
        modifier = Abliteration(
            self.model.settings,
            AbliterationSettings(
                row_normalization=RowNormalization.NONE, orthogonalize_direction=False
            ),
        )
        ctx = Context(settings=self.model.settings, model=self.model)
        with patch.object(Context, "load_prompts", side_effect=[good, bad]):
            modifier.init(ctx)
        parameters = Parameters(
            direction_index=None,
            weight_distributions={
                component: WeightDistribution(
                    max_weight=0.5,
                    max_weight_position=1.0,
                    min_weight=0.5,
                    min_weight_distance=3.0,
                )
                for component in self.model.get_modifiable_components()
            },
        )
        with torch.no_grad():
            baseline = self.model.model(self.tokens, use_cache=False).logits
        modifier.modify_model(ctx, parameters)
        adapters = [p for n, p in self.model.model.named_parameters() if "lora_" in n]
        self.assertTrue(adapters)
        self.assertTrue(all(torch.isfinite(p).all() for p in adapters))
        for layer_index in range(3):
            for modules in self.model.get_layer_modules(layer_index).values():
                for module in modules:
                    self.assertGreater(
                        module.lora_B["default"].weight.count_nonzero(), 0
                    )
        with torch.no_grad():
            changed = self.model.model(self.tokens, use_cache=False).logits
        self.assertFalse(torch.allclose(changed, baseline))
        self.assertTrue(self.model.reset_model())
        self.assertTrue(all(p.count_nonzero() == 0 for p in adapters))
        with torch.no_grad():
            torch.testing.assert_close(
                self.model.model(self.tokens, use_cache=False).logits, baseline
            )
        modifier.modify_model(ctx, parameters)
        merged = self.model.get_merged_model()
        self.assertTrue(self.model.needs_reload)
        self.assertFalse(any(isinstance(m, GlmExpertViews) for m in merged.modules()))
        self.assertFalse(
            any("lora_" in n or "heretic_projection" in n for n in merged.state_dict())
        )
        with tempfile.TemporaryDirectory() as directory:
            merged.save_pretrained(directory)
            restored = GlmMoeDsaForCausalLM.from_pretrained(
                directory, attn_implementation="eager"
            ).eval()
            with torch.no_grad():
                torch.testing.assert_close(
                    restored(self.tokens, use_cache=False).logits,
                    changed,
                    rtol=1e-4,
                    atol=1e-5,
                )
                self.assertTrue(
                    torch.isfinite(
                        restored.generate(self.tokens, max_new_tokens=2)
                    ).all()
                )

    def test_quantization_rejected_before_loading(self):
        with (
            patch(
                "heretic.model.PretrainedConfig.get_config_dict",
                return_value=(
                    {
                        "model_type": "glm_moe_dsa",
                        "quantization_config": {"quant_method": "fp8"},
                    },
                    {},
                ),
            ),
            patch("heretic.model.AutoTokenizer.from_pretrained") as tokenizer,
            patch("heretic.model.AutoModelForCausalLM.from_pretrained") as loader,
        ):
            with self.assertRaisesRegex(ValueError, "unquantized weights"):
                Model(self.model.settings)
            tokenizer.assert_not_called()
            loader.assert_not_called()
        self.model.settings.quantization = QuantizationMethod.BNB_4BIT
        for method in ("get_layer_modules", "apply_lora"):
            with self.assertRaisesRegex(ValueError, "unquantized weights"):
                getattr(self.model, method)(1)


if __name__ == "__main__":
    unittest.main()
