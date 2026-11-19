"""Offline GLM-4.7 regressions for Heretic 2.1."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM

from heretic.config import ExportStrategy, QuantizationMethod, Settings
from heretic.glm import GlmExpertViews
from heretic.model import Model, check_model_resident
from heretic.modifiers.ara import ARA, Parameters
from heretic.modifiers.ara import Settings as ARASettings
from heretic.plugin import Context


def tiny_model():
    config = Glm4MoeLiteConfig(
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
        mlp_layer_types=["dense", "sparse", "sparse"],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    model = Model.__new__(Model)
    model.model = Glm4MoeLiteForCausalLM(config).eval()
    model.model.config.name_or_path = "tiny-glm47"
    model.settings = Settings.model_construct(
        model="tiny-glm47",
        quantization=QuantizationMethod.NONE,
        seed=42,
        batch_size=1,
    )
    model.needs_reload = False
    return model


class Glm47V2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = tiny_model()
        self.tokens = torch.tensor([[3, 4, 5]])

    def test_reference_routing_storage_and_unique_discovery(self):
        packed = self.model.model.model.layers[1].mlp.experts
        hidden = torch.randn(3, 16)
        indices = torch.tensor([[0, 1], [0, 2], [2, 1]])
        weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])
        with torch.no_grad():
            expected = packed(hidden, indices, weights)
        modules = self.model.get_layer_modules(1)["mlp.down_proj"]
        views = self.model.model.model.layers[1].mlp.experts
        self.assertEqual(len(modules), 5)
        self.assertEqual(len({id(m) for m in modules}), 5)
        self.assertIs(
            modules[0], self.model.model.model.layers[1].mlp.shared_experts.down_proj
        )
        self.assertEqual(
            self.model.get_modifiable_components(), ["attn.o_proj", "mlp.down_proj"]
        )
        with torch.no_grad():
            torch.testing.assert_close(views(hidden, indices, weights), expected)
        for index, expert in enumerate(views.experts):
            self.assertEqual(
                expert.down_proj.weight.data_ptr(), packed.down_proj[index].data_ptr()
            )

    def test_hook_cleanup_and_cache_disabled(self):
        def fail(prompts, **kwargs):
            self.assertFalse(kwargs["use_cache"])
            raise RuntimeError("capture failed")

        self.model.generate = fail
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            self.model.get_module_io([])
        for module in self.model.model.modules():
            self.assertEqual(len(module._forward_hooks), 0)

    def test_real_ara_step_reset_and_native_export(self):
        # Exercise actual 2-D routed hooks and optimizer, not just the loss helper.
        def generate(prompts, **kwargs):
            with torch.no_grad():
                self.model.model(self.tokens, use_cache=kwargs.get("use_cache", False))

        self.model.generate = generate
        io = self.model.get_module_io([object()])
        modifier = ARA(
            self.model.settings,
            ARASettings(
                lora_rank=1,
                n_optimization_steps=1,
                max_iter=1,
                history_size=1,
                preserve_row_magnitudes=False,
            ),
        )
        modifier.good_module_io = io
        modifier.bad_module_io = io
        self.model.apply_lora(1)
        with torch.no_grad():
            baseline = self.model.model(self.tokens).logits
        ctx = Context(settings=self.model.settings, model=self.model)
        modifier.modify_model(ctx, Parameters(0, 3, 1.0, 0.1, 0.0, 15))
        adapters = [
            (n, p) for n, p in self.model.model.named_parameters() if "lora_" in n
        ]
        self.assertTrue(adapters)
        self.assertTrue(all(torch.isfinite(p).all() for _, p in adapters))
        self.assertTrue(all(p.grad is None for _, p in adapters))
        self.assertTrue(self.model.reset_model())
        self.assertTrue(all(p.count_nonzero() == 0 for _, p in adapters))
        with torch.no_grad():
            torch.testing.assert_close(self.model.model(self.tokens).logits, baseline)
        merged = self.model.get_merged_model()
        self.assertTrue(self.model.needs_reload)
        self.assertFalse(any(isinstance(m, GlmExpertViews) for m in merged.modules()))
        self.assertFalse(
            any("heretic_projection" in n or "lora_" in n for n in merged.state_dict())
        )
        with tempfile.TemporaryDirectory() as directory:
            merged.save_pretrained(directory)
            restored = Glm4MoeLiteForCausalLM.from_pretrained(directory).eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(self.tokens).logits, baseline)

    def test_quantization_rejected_early_and_in_discovery(self):
        with (
            patch(
                "heretic.model.PretrainedConfig.get_config_dict",
                return_value=(
                    {
                        "model_type": "glm4_moe_lite",
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

    def test_export_policy_and_residency(self):
        from heretic.main import obtain_export_strategy

        self.model.settings.export_strategy = ExportStrategy.ADAPTER
        with self.assertRaisesRegex(ValueError, "standalone PEFT adapters"):
            obtain_export_strategy(self.model.settings, self.model)
        for strategy in (None, ExportStrategy.MERGE):
            self.model.settings.export_strategy = strategy
            self.assertEqual(
                obtain_export_strategy(self.model.settings, self.model),
                ExportStrategy.MERGE,
            )
        check_model_resident(self.model.model)
        with self.assertRaisesRegex(Exception, "does not fit"):
            check_model_resident(torch.nn.Linear(2, 2, device="meta"))

    def test_dedicated_config_uses_classic_abliteration(self):
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        from heretic.config import InitOnlySettings

        path = Path(__file__).resolve().parents[1] / "config.glm-4.7.toml"
        settings = InitOnlySettings.model_validate(tomllib.loads(path.read_text()))
        self.assertEqual(
            settings.modifiers[0].plugin, "heretic.modifiers.abliteration.Abliteration"
        )
        self.assertEqual(settings.export_strategy, ExportStrategy.MERGE)
        self.assertEqual(settings.quantization, QuantizationMethod.NONE)
        self.assertEqual(
            settings.model_extra["modifier"]["Abliteration"]["row_normalization"],
            "none",
        )


if __name__ == "__main__":
    unittest.main()
