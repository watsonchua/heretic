"""Config-only and tiny-model GLM-5.3 checks; never load checkpoint weights."""

import json
import tempfile
import unittest
import urllib.request

import peft
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, GlmMoeDsaConfig

from heretic.config import QuantizationMethod, Settings
from heretic.glm import GlmExpertViews
from heretic.model import Model


def inspect_config(model_id):
    with urllib.request.urlopen(
        f"https://huggingface.co/api/models/{model_id}"
    ) as response:
        metadata = json.load(response)
    revision = metadata["sha"]
    with urllib.request.urlopen(
        f"https://huggingface.co/{model_id}/raw/{revision}/config.json"
    ) as response:
        raw = json.load(response)
    config = AutoConfig.for_model(
        raw["model_type"], **{k: v for k, v in raw.items() if k != "model_type"}
    )
    sparse = config.mlp_layer_types.count("sparse")
    dense = config.mlp_layer_types.count("dense")
    # Resolve the class only; do not instantiate a full-sized model.
    model_class = AutoModelForCausalLM._model_mapping[type(config)]
    return {
        "model_id": model_id,
        "revision": revision,
        "model_class": model_class.__name__,
        "raw_config": raw,
        "resolved_mlp_layer_types": config.mlp_layer_types,
        "resolved_indexer_types": config.indexer_types,
        "attention_projections": config.num_hidden_layers,
        "dense_projections": dense,
        "shared_projections": sparse,
        "routed_projections": sparse * config.n_routed_experts,
        "safetensors": metadata.get("safetensors"),
    }


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
        n_group=1,
        topk_group=1,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        first_k_dense_replace=1,
        mlp_layer_types=["dense", "sparse", "sparse"],
        indexer_types=["full", "shared", "full"],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    return AutoModelForCausalLM.from_config(config).eval()


class GlmDsaPreflightTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.native = tiny_model()
        self.tokens = torch.tensor([[3, 4, 5]])

    def test_native_forward_and_cached_generation(self):
        with torch.no_grad():
            output = self.native(self.tokens, output_hidden_states=True)
            generated = self.native.generate(self.tokens, max_new_tokens=2)
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertEqual(len(output.hidden_states), 4)
        self.assertGreater(generated.shape[1], self.tokens.shape[1])

    def test_expert_reference_hooks_and_storage(self):
        packed = self.native.model.layers[1].mlp.experts
        views = GlmExpertViews(packed)
        hidden = torch.randn(3, 16)
        # Expert 3 is inactive; 4 is the native sentinel expert index.
        indices = torch.tensor([[0, 1], [0, 2], [4, 0]])
        weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]])
        calls = []
        handles = [
            e.down_proj.register_forward_hook(
                lambda module, inputs, output: calls.append(
                    (inputs[0].shape, output.shape)
                )
            )
            for e in views.experts
        ]
        try:
            with torch.no_grad():
                torch.testing.assert_close(
                    views(hidden, indices, weights), packed(hidden, indices, weights)
                )
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(len(i) == len(o) == 2 for i, o in calls))
        for index, expert in enumerate(views.experts):
            self.assertEqual(
                expert.down_proj.weight.data_ptr(), packed.down_proj[index].data_ptr()
            )

    def test_wrapped_logits_and_native_reload(self):
        with torch.no_grad():
            baseline = self.native(self.tokens).logits
        for layer in self.native.model.layers[1:]:
            layer.mlp.experts = GlmExpertViews(layer.mlp.experts)
        with torch.no_grad():
            torch.testing.assert_close(self.native(self.tokens).logits, baseline)
            self.native.generate(self.tokens, max_new_tokens=2)
        for layer in self.native.model.layers[1:]:
            layer.mlp.experts = layer.mlp.experts.repack()
        with tempfile.TemporaryDirectory() as directory:
            self.native.save_pretrained(directory)
            restored = AutoModelForCausalLM.from_pretrained(directory).eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(self.tokens).logits, baseline)

    def test_heretic_discovers_routed_experts(self):
        model = Model.__new__(Model)
        model.model = self.native
        model.settings = Settings.model_construct(
            model="tiny-glm-dsa", quantization=QuantizationMethod.NONE
        )
        self.assertEqual(len(model.get_layers()), 3)
        self.assertEqual(len(model.get_layer_modules(0)["mlp.down_proj"]), 1)
        modules = model.get_layer_modules(1)["mlp.down_proj"]
        self.assertEqual(len(modules), 5)
        self.assertEqual(len({id(module) for module in modules}), 5)
        self.assertIs(
            modules[0], self.native.model.layers[1].mlp.shared_experts.down_proj
        )
        self.assertIsInstance(self.native.model.layers[1].mlp.experts, GlmExpertViews)


if __name__ == "__main__":
    configs = [
        inspect_config("zai-org/GLM-5.3"),
        inspect_config("zai-org/GLM-5.3-BF16"),
    ]
    summaries = []
    for config in configs:
        summary = {
            key: value
            for key, value in config.items()
            if key
            not in (
                "raw_config",
                "resolved_mlp_layer_types",
                "resolved_indexer_types",
            )
        }
        summary["quantization_method"] = (
            config["raw_config"].get("quantization_config", {}).get("quant_method")
        )
        summary["full_indexer_layers"] = config["resolved_indexer_types"].count("full")
        summary["shared_indexer_layers"] = config["resolved_indexer_types"].count(
            "shared"
        )
        summaries.append(summary)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
                "cuda_available": torch.cuda.is_available(),
                "configs": summaries,
            },
            indent=2,
        ),
        flush=True,
    )
    unittest.main(verbosity=2)
