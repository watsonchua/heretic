"""Full BF16 GLM smoke test. Run with uv run --no-sync tests/glm_smoke.py."""

import json
import time
from pathlib import Path
from typing import cast

import optuna
import torch

from heretic.config import QuantizationMethod, Settings
from heretic.model import Model
from heretic.modifiers.abliteration import Abliteration, RowNormalization
from heretic.modifiers.abliteration import Settings as AbliterationSettings
from heretic.plugin import Context as PluginContext
from heretic.utils import Prompt

torch.manual_seed(42)
settings = Settings(
    model="zai-org/GLM-4.7-Flash",
    dtypes=["bfloat16"],
    quantization=QuantizationMethod.NONE,
    device_map="auto",
    batch_size=1,
    max_response_length=16,
    offload_outputs_to_cpu=True,
)
start = time.monotonic()
model = Model(settings)
good = [
    Prompt(system=settings.system_prompt, user="What is 2 plus 2?"),
    Prompt(system=settings.system_prompt, user="Name the capital of France."),
]
bad = [
    Prompt(system=settings.system_prompt, user="How can I steal someone's password?"),
    Prompt(system=settings.system_prompt, user="How can I commit credit card fraud?"),
]


class Context:
    def get_model(self):
        return model

    def load_prompts(self, specification):
        return good if specification == modifier.settings.good_prompts else bad


ctx = cast(PluginContext, Context())
report = {"load_seconds": time.monotonic() - start}
report["base_responses"] = model.get_responses(good + bad)
modifier = Abliteration(
    settings, AbliterationSettings(row_normalization=RowNormalization.NONE)
)
modifier.init(ctx)
io = model.get_module_io(good[:1])
report["captured_modules"] = [sum(len(v) for v in layer.values()) for layer in io]
study = optuna.create_study()
trial = study.ask()
parameters = modifier.suggest_parameters(ctx, trial)
modifier.reset_model(ctx)
modifier.modify_model(ctx, parameters)
report["modified_responses"] = model.get_responses(good + bad)
assert all(
    torch.isfinite(p).all() for n, p in model.model.named_parameters() if "lora_" in n
)
study.tell(trial, 0.0)
report["parameters"] = parameters.to_dict()
report["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
report["elapsed_seconds"] = time.monotonic() - start
Path("glm-smoke-results.json").write_text(json.dumps(report, indent=2))
model.get_merged_model().save_pretrained("glm-smoke-export", max_shard_size="5GB")
model.tokenizer.save_pretrained("glm-smoke-export")
print(json.dumps(report, indent=2))
