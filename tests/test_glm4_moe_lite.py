import tempfile
import unittest

import torch
from transformers import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM

from heretic.config import QuantizationMethod, Settings
from heretic.glm import GlmExpertViews
from heretic.model import Model


class GlmTests(unittest.TestCase):
    def test_experts_adapters_reset_and_native_export(self):
        torch.manual_seed(42)
        config = Glm4MoeLiteConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            moe_intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            q_lora_rank=8,
            kv_lora_rank=4,
            qk_rope_head_dim=4,
            qk_nope_head_dim=4,
            v_head_dim=4,
        )
        native = Glm4MoeLiteForCausalLM(config).eval()
        native.config.name_or_path = "tiny-glm"
        model = Model.__new__(Model)
        model.model = native
        model.settings = Settings.model_construct(
            model="tiny-glm",
            quantization=QuantizationMethod.NONE,
        )
        model.needs_reload = False
        tokens = torch.tensor([[3, 4, 5]])
        with torch.no_grad():
            baseline = native(tokens).logits
        self.assertEqual(len(model.get_layers()), 2)
        self.assertEqual(len(model.get_layer_modules(0)["mlp.down_proj"]), 1)
        self.assertEqual(len(model.get_layer_modules(1)["mlp.down_proj"]), 5)
        views = native.model.layers[1].mlp.experts
        self.assertIsInstance(views, GlmExpertViews)
        with torch.no_grad():
            torch.testing.assert_close(native(tokens).logits, baseline)
        # All active routed projections are invoked once and have 2-D I/O.
        calls = []
        handles = [
            e.down_proj.register_forward_hook(
                lambda module, inputs, outputs: calls.append(
                    (inputs[0].shape, outputs.shape)
                )
            )
            for e in views.experts
        ]
        with torch.no_grad():
            native(tokens)
        for handle in handles:
            handle.remove()
        self.assertGreater(len(calls), 0)
        self.assertLessEqual(len(calls), 4)
        self.assertTrue(all(len(i) == len(o) == 2 for i, o in calls))
        model.apply_lora(1)
        expert = model.get_layer_modules(1)["mlp.down_proj"][1]
        direction = torch.nn.functional.normalize(torch.arange(16).float(), dim=0)
        original = expert.base_layer.weight.detach().clone()
        expert.lora_A["default"].weight.data.copy_((direction @ original).unsqueeze(0))
        expert.lora_B["default"].weight.data.copy_(-direction.unsqueeze(1))
        expected = original - direction[:, None] * (direction @ original)[None, :]
        torch.testing.assert_close(
            expert.get_delta_weight("default"), expected - original
        )
        self.assertTrue(model.reset_model())
        self.assertEqual(expert.lora_B["default"].weight.count_nonzero().item(), 0)
        expert.lora_A["default"].weight.data.copy_((direction @ original).unsqueeze(0))
        expert.lora_B["default"].weight.data.copy_(-direction.unsqueeze(1))
        with torch.no_grad():
            modified = model.model(tokens).logits
        merged = model.get_merged_model()
        torch.testing.assert_close(
            merged.model.layers[1].mlp.experts.down_proj[0], expected
        )
        with tempfile.TemporaryDirectory() as directory:
            merged.save_pretrained(directory)
            restored = Glm4MoeLiteForCausalLM.from_pretrained(directory).eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(tokens).logits, modified)
            self.assertGreater(restored.generate(tokens, max_new_tokens=2).shape[1], 3)


if __name__ == "__main__":
    unittest.main()
