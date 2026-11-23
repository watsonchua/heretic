"""Opt-in checkpoint smoke test. See tests/glm-dsa.md for uv commands."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from heretic.config import InitOnlySettings, QuantizationMethod
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
from heretic.utils import Prompt

MODELS = {
    "tiny": "tiny-random/glm-moe-dsa",
    "0.6b": "inference-optimization/GLM-5.3-0.6B-A0.4B",
    "3.5b": "inference-optimization/GLM-5.3-3.5B-A3.3B-small",
}


class SmokeContext(Context):
    """Supply local contrast prompts instead of downloading refusal datasets."""

    def load_prompts(self, specification):
        subjects = (
            ["What is 2 plus 2?", "Name the capital of France."]
            if specification.dataset == "smoke-good"
            else ["Write a short poem about rain.", "Describe a blue bird."]
        )
        return [Prompt(system=self._settings.system_prompt, user=s) for s in subjects]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="0.6b")
    parser.add_argument("--revision", help="Optional checkpoint commit hash")
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu"
    )
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--output", type=Path, required=True, help="New local output directory"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "Output already exists. Choose a new directory; nothing will be overwritten."
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is not available. On an Apple Silicon Mac, use --device mps."
        )
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("Apple MPS is not available.")

    torch.manual_seed(42)
    started = time.monotonic()
    args.output.mkdir(parents=True)
    report = {
        "model": MODELS[args.model],
        "revision_requested": args.revision,
        "purpose": "Architecture and modification smoke test, not refusal evaluation",
        "dtype": args.dtype,
        "device": args.device,
        "status": "running",
    }
    try:
        settings = InitOnlySettings(
            model=MODELS[args.model],
            model_commit=args.revision,
            dtypes=[args.dtype],
            quantization=QuantizationMethod.NONE,
            # Avoid direct safetensors loading onto MPS, which can crash
            # in native code. Inference and modification still run on the GPU.
            device_map="cpu" if args.device == "mps" else args.device,
            batch_size=1,
            max_response_length=4,
            offload_outputs_to_cpu=True,
            seed=42,
        )
        model = Model(settings)
        if args.device == "mps":
            model.model.to("mps")
        if model.model.config.model_type != "glm_moe_dsa":
            raise AssertionError("Expected glm_moe_dsa")
        report["revision_loaded"] = getattr(model.model.config, "_commit_hash", None)
        report["load_seconds"] = time.monotonic() - started
        report["model_type"] = model.model.config.model_type
        report["projection_counts"] = [
            {
                component: len(modules)
                for component, modules in model.get_layer_modules(i).items()
            }
            for i in range(len(model.get_layers()))
        ]
        for i, layer in enumerate(model.get_layers()):
            if hasattr(layer.mlp, "experts"):
                expected = layer.mlp.experts.packed.down_proj.shape[0] + 1
                if report["projection_counts"][i]["mlp.down_proj"] != expected:
                    raise AssertionError(f"Missing expert projections in layer {i}")
        ctx = SmokeContext(settings=settings, model=model)
        modifier_settings = AbliterationSettings(
            row_normalization=RowNormalization.NONE,
            orthogonalize_direction=False,
            good_prompts={"dataset": "smoke-good", "column": "prompt"},
            bad_prompts={"dataset": "smoke-bad", "column": "prompt"},
        )
        prompts = ctx.load_prompts(modifier_settings.good_prompts)
        report["base_responses"] = model.get_responses_batched(prompts)
        residuals = model.get_residuals(prompts[:1])
        if not torch.isfinite(residuals).all():
            raise AssertionError("Non-finite residuals")
        report["residual_shape"] = list(residuals.shape)
        io = model.get_module_io(prompts[:1])
        report["captured_projection_counts"] = [
            {component: len(values) for component, values in layer.items()}
            for layer in io
        ]
        for i, layer in enumerate(model.get_layers()):
            if hasattr(layer.mlp, "experts") and not any(
                k > 0 for k in io[i].get("mlp.down_proj", {})
            ):
                raise AssertionError(f"No routed expert I/O captured in layer {i}")
        del residuals, io
        modifier = Abliteration(settings, modifier_settings)
        modifier.init(ctx)
        if not torch.isfinite(modifier.residual_directions).all():
            raise AssertionError("Non-finite directions")
        parameters = Parameters(
            direction_index=None,
            weight_distributions={
                component: WeightDistribution(
                    max_weight=0.5,
                    max_weight_position=0.0,
                    min_weight=0.5,
                    min_weight_distance=float(len(model.get_layers())),
                )
                for component in model.get_modifiable_components()
            },
        )
        modifier.modify_model(ctx, parameters)
        nonzero = 0
        for name, parameter in model.model.named_parameters():
            if "lora_" in name:
                if not torch.isfinite(parameter).all():
                    raise AssertionError(f"Non-finite adapter: {name}")
                nonzero += int(parameter.count_nonzero().item() > 0)
        if nonzero == 0:
            raise AssertionError("All adapters are zero")
        for i in range(len(model.get_layers())):
            for modules in model.get_layer_modules(i).values():
                for module in modules:
                    if module.lora_B["default"].weight.count_nonzero() == 0:
                        raise AssertionError(
                            f"Unmodified output projection in layer {i}"
                        )
        report["nonzero_adapter_tensors"] = nonzero
        report["parameters"] = parameters.to_dict()
        report["modified_responses"] = model.get_responses_batched(prompts)
        inputs, _ = model.generate(prompts[:1], max_new_tokens=1, use_cache=False)
        with torch.no_grad():
            expected_logits = model.model(**inputs, use_cache=False).logits.cpu()
        saved_inputs = {key: value.cpu() for key, value in inputs.items()}
        export = args.output / "model"
        merged = model.get_merged_model()
        if any(isinstance(m, GlmExpertViews) for m in merged.modules()):
            raise AssertionError("Export still contains expert views")
        if any(
            "lora_" in key or "heretic_projection" in key for key in merged.state_dict()
        ):
            raise AssertionError("Export still contains adapter keys")
        with torch.no_grad():
            merged_logits = merged(**inputs, use_cache=False).logits.cpu()
        report["adapter_to_merged_max_logit_error"] = (
            (merged_logits - expected_logits).abs().max().item()
        )
        # BF16 merge rounds the low-rank update into the base weights.
        # Validate serialization separately against the native merged model.
        expected_logits = merged_logits
        merged.save_pretrained(export)
        model.tokenizer.save_pretrained(export)
        del ctx, modifier, merged, model, inputs
        gc.collect()
        if torch.cuda.is_available():
            report["peak_cuda_allocated_bytes"] = {
                str(i): torch.cuda.max_memory_allocated(i)
                for i in range(torch.cuda.device_count())
            }
            torch.cuda.empty_cache()
        # Release the first model before loading the export. Reload on CPU to
        # avoid requiring a second full GPU copy.
        restored = AutoModelForCausalLM.from_pretrained(
            export, dtype=getattr(torch, args.dtype), device_map="cpu"
        ).eval()
        with torch.no_grad():
            actual_logits = restored(**saved_inputs, use_cache=False).logits
            report["cpu_reload_max_logit_error"] = (
                (actual_logits - expected_logits).abs().max().item()
            )
            # Compare on the original backend to separate export errors from
            # CPU/GPU floating-point differences. Keep the original tolerance.
            if args.device in {"mps", "cuda"}:
                if args.device == "mps":
                    torch.mps.empty_cache()
                else:
                    torch.cuda.empty_cache()
                restored.to(args.device)
                saved_inputs = {k: v.to(args.device) for k, v in saved_inputs.items()}
                actual_logits = restored(**saved_inputs, use_cache=False).logits.cpu()
            if not torch.isfinite(actual_logits).all():
                raise AssertionError("Non-finite reloaded logits")
            torch.testing.assert_close(
                actual_logits,
                expected_logits,
                rtol=0.05 if args.dtype == "bfloat16" else 1e-4,
                atol=0.05 if args.dtype == "bfloat16" else 1e-5,
            )
            restored.generate(**saved_inputs, max_new_tokens=2, do_sample=False)
        report["reload_max_logit_error"] = (
            (actual_logits - expected_logits).abs().max().item()
        )
        report["status"] = "passed"
    except (Exception, SystemExit) as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
