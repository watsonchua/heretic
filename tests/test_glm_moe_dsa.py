import tempfile
import unittest
from unittest.mock import patch

import torch
from glm_dsa_preflight import tiny_model
from transformers import AutoModelForCausalLM

from heretic.config import ExportStrategy, QuantizationMethod, Settings
from heretic.glm import GlmExpertViews, validate_glm_quantization
from heretic.model import Model


class GlmDsaIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = Model.__new__(Model)
        self.model.model = tiny_model()
        self.model.model.config.name_or_path = "tiny-glm-dsa"
        self.model.settings = Settings.model_construct(
            model="tiny-glm-dsa", quantization=QuantizationMethod.NONE
        )
        self.model.needs_reload = False
        self.tokens = torch.tensor([[3, 4, 5]])

    def test_adapters_reset_merge_and_native_export(self):
        with torch.no_grad():
            baseline = self.model.model(self.tokens).logits
        modules = [self.model.get_layer_modules(i) for i in range(3)]
        self.assertEqual(sum(len(m["attn.o_proj"]) for m in modules), 3)
        self.assertEqual(sum(len(m["mlp.down_proj"]) for m in modules), 11)
        self.model.apply_lora(1)
        with torch.no_grad():
            torch.testing.assert_close(self.model.model(self.tokens).logits, baseline)
        expert = self.model.get_layer_modules(1)["mlp.down_proj"][1]
        original = expert.base_layer.weight.detach().clone()
        direction = torch.nn.functional.normalize(torch.arange(16).float(), dim=0)
        expected = original - direction[:, None] * (direction @ original)[None, :]

        def update():
            expert.lora_A["default"].weight.data.copy_(
                (direction @ original).unsqueeze(0)
            )
            expert.lora_B["default"].weight.data.copy_(-direction.unsqueeze(1))

        update()
        torch.testing.assert_close(
            expert.get_delta_weight("default"), expected - original
        )
        self.assertTrue(self.model.reset_model())
        self.assertEqual(expert.lora_B["default"].weight.count_nonzero().item(), 0)
        with torch.no_grad():
            torch.testing.assert_close(self.model.model(self.tokens).logits, baseline)
        update()
        calls = []
        handle = expert.register_forward_hook(
            lambda module, inputs, output: calls.append((inputs[0].shape, output.shape))
        )
        try:
            with torch.no_grad():
                modified = self.model.model(self.tokens).logits
        finally:
            handle.remove()
        self.assertTrue(all(len(i) == len(o) == 2 for i, o in calls))
        merged = self.model.get_merged_model()
        self.assertTrue(self.model.needs_reload)
        torch.testing.assert_close(
            merged.model.layers[1].mlp.experts.down_proj[0], expected
        )
        self.assertFalse(any(isinstance(m, GlmExpertViews) for m in merged.modules()))
        self.assertFalse(
            any("heretic_projection" in k or "lora_" in k for k in merged.state_dict())
        )
        with tempfile.TemporaryDirectory() as directory:
            merged.save_pretrained(directory)
            restored = AutoModelForCausalLM.from_pretrained(directory).eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(self.tokens).logits, modified)
                self.assertGreater(
                    restored.generate(self.tokens, max_new_tokens=2).shape[1], 3
                )

    def test_quantization_rejected_by_discovery_and_adapter_preparation(self):
        for method in ("get_layer_modules", "apply_lora"):
            self.model.model.config.quantization_config = {"quant_method": "fp8"}
            with self.assertRaisesRegex(ValueError, "GLM-5.3-BF16"):
                getattr(self.model, method)(1)
            del self.model.model.config.quantization_config
            self.model.settings.quantization = QuantizationMethod.BNB_4BIT
            with self.assertRaisesRegex(ValueError, "unquantized weights"):
                getattr(self.model, method)(1)
            self.model.settings.quantization = QuantizationMethod.NONE

    def test_fp8_rejected_before_tokenizer_or_weight_loading(self):
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
            with self.assertRaisesRegex(ValueError, "GLM-5.3-BF16"):
                Model(self.model.settings)
            tokenizer.assert_not_called()
            loader.assert_not_called()
        validate_glm_quantization(
            "other-model", QuantizationMethod.BNB_4BIT, {"quant_method": "fp8"}
        )

    def test_merged_export_only(self):
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


if __name__ == "__main__":
    unittest.main()
