"""Fresh-process reload and small PIQA regression comparison."""

import json
import sys
from pathlib import Path
from typing import Any, cast

import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype="bfloat16", device_map="auto"
)
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 2 plus 2?"}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = cast(Any, model).generate(**inputs, max_new_tokens=32, do_sample=False)
response = tokenizer.decode(
    output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
)
assert response.strip()
results = lm_eval.simple_evaluate(
    model=HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1),
    tasks=["piqa"],
    num_fewshot=0,
    limit=8,
    random_seed=42,
    numpy_random_seed=42,
    torch_random_seed=42,
)
Path(sys.argv[2]).write_text(
    json.dumps(
        {
            "model": model_id,
            "response": response,
            "benchmark": results["results"],
            "limit": 8,
            "num_fewshot": 0,
        },
        indent=2,
    )
)
