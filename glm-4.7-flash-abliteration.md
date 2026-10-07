# GLM-4.7-Flash support and evaluation plan

## Goal

Add support for running Heretic's complete abliteration workflow on `zai-org/GLM-4.7-Flash`, including model load, residual and module-I/O collection, optimization, adapter merge/export, and post-change evaluation. Make all work on a dedicated branch and keep the existing default behavior for other architectures unchanged.

## Model facts and current compatibility

- The model is a text-only `Glm4MoeLiteForCausalLM` with 47 decoder layers. Its layer stack is `model.model.layers`, so the existing `get_model_class()` and `get_layers()` paths in `src/heretic/model.py` are expected to apply.
- The model card requires a current Transformers build for native loading. Check that the resolved/pinned Transformers release in this checkout exposes `glm4_moe_lite`; if not, update `pyproject.toml` and `uv.lock` to a release that does. Avoid a floating dependency in the final branch.
- Its attention output projection is an ordinary module (`self_attn.o_proj`). Dense and shared MLP projections are ordinary modules. In sparse layers, routed expert weights are packed parameters on `layer.mlp.experts` (including a 3-D `down_proj` parameter), not a list of per-expert `nn.Linear` modules. The current generic expert loop in `get_layer_modules()` does not handle this representation.
- Heretic's optimizer and forward-hook collection currently operate on `nn.Module` objects; `apply_lora()` also uses PEFT module targets. Do not treat `Parameter` tensors as if they were individual modules. Decide and document whether initial support will cover attention plus shared/dense MLP only, or implement routed packed-expert support end-to-end. Full support should include that end-to-end path; silently reporting success while omitting routed experts is not full component coverage.
- The repository snapshot currently declares `transformers[kernels]~=5.18`. Verify the actual installed/resolved version rather than assuming the version number alone proves architecture support.

## Concrete files to change

1. `src/heretic/model.py`
   - Keep the existing causal-LM detection and validate model loading, chat template rendering, greedy generation, and hidden-state capture.
   - Add GLM-4.7-Flash component discovery. Confirm `self_attn.o_proj`, dense `mlp.down_proj`, and `mlp.shared_experts.down_proj` paths against the installed model class.
   - Design routed expert support around packed `experts.down_proj` parameters. A robust implementation must define how to gather per-expert input/output activations (or otherwise derive the optimization data), apply the transformation across the packed expert matrices, and preserve/reset/export the result across trials. PEFT cannot target the packed tensor as though it were a module. If a narrowly scoped first patch omits routed experts, explicitly expose/report that limitation and make it a separate milestone.
   - Ensure module discovery still finds at least one usable component in every layer, including sparse and dense layer variants.
   - Check save/merge/reload behavior after any custom parameter or adapter path is added.

2. `tests/` (add focused tests; suggested new file `tests/test_glm4_moe_lite.py`)
   - Use a tiny/random GLM-4.7-Flash config/model where practical; tests should not download the full checkpoint.
   - Assert architecture detection, 47-layer-compatible stack access, per-layer component discovery, sparse/dense coverage, and no duplicate/missing module-I/O capture.
   - Test packed expert transformation against an explicit small tensor reference, including identity initialization, reset between trials, and merge/export round-trip if custom adapter logic is used.
   - Add a short generation regression check using the chat template and EOS handling.

3. `pyproject.toml` and `uv.lock`
   - Change only if the supported Transformers release does not include GLM-4.7-Flash or if another required dependency is missing. Pin a stable compatible version where possible and regenerate the lockfile using the repository's `uv` workflow.

4. `config.default.toml` or `README.md` (only if needed)
   - Document that full BF16 testing needs a large-memory GPU and show the model ID/config settings. Do not change global defaults just for this model; use a dedicated test config or CLI overrides.

## Branch and change management

- Start from the intended base branch and create a topic branch such as `support/glm-4.7-flash`.
- Keep GLM-specific logic and tests in focused commits. Do not include unrelated working-tree changes.
- Before committing, inspect `git diff` and `git status`; commit only the intended files. Report the branch name and commit hash in the handoff.

## Hardware and test configuration

The Hugging Face checkpoint is listed at roughly 62.5 GB, with BF16 weights alone requiring about 58 GiB before temporary loading, activations, CUDA context, and Heretic analysis. Recommended initial target: one NVIDIA 80 GB GPU (A100 80 GB or H100 80 GB) with ample host RAM and disk for the checkpoint/cache and exported model. Multi-GPU with `device_map="auto"` is also reasonable. A 48 GB GPU is not a reliable BF16 target. Do not assume 4-bit loading solves the fit issue until the packed expert parameters are confirmed to be quantized by the selected bitsandbytes path.

For the first full-checkpoint smoke test, use a dedicated TOML config with:

```toml
model = "zai-org/GLM-4.7-Flash"
quantization = "none"
device_map = "auto"
offload_outputs_to_cpu = true
batch_size = 1
max_response_length = 16
```

Use the model's native BF16 (`dtypes = ["auto"]` or `dtypes = ["bfloat16"]`) for a predictable first run. Keep prompts/datasets tiny in the smoke-test config. Avoid a production abliteration run until module coverage and optimizer behavior have been checked.

## Evaluation sequence and acceptance criteria

### A. Dependency and architecture preflight

- On the target server, record GPU model/VRAM (`nvidia-smi`), host RAM, Python, PyTorch/CUDA, Transformers, Accelerate, and PEFT versions.
- Confirm `AutoConfig.from_pretrained("zai-org/GLM-4.7-Flash").model_type == "glm4_moe_lite"` and that `AutoModelForCausalLM` resolves `Glm4MoeLiteForCausalLM`.
- If the pinned Transformers install cannot resolve it, fix/pin the dependency before proceeding.

### B. Fast unit tests (no full model download)

- Run the focused GLM architecture tests and existing model/config tests that cover shared loader and reproducibility behavior.
- Check dense and sparse layers independently. Confirm Heretic's reported component counts match discovered modules/packed tensors and that every intended component is transformed.
- For a packed expert transform, compare against a small deterministic reference tensor calculation; assert finite values, expected shapes, identity at initialization/reset, and expected changed weights after applying a nonzero transform.

### C. Full-checkpoint load/generation smoke test

- Load the full model with the settings above. Confirm no architecture, dtype, tokenizer, chat-template, generation, or out-of-memory error.
- Confirm Heretic reports 47 transformer layers and the expected components. Generate a short response from a simple prompt and confirm nonempty decoded output.
- Capture peak VRAM and elapsed load/generation time. Record the exact hardware and package versions.

### D. Heretic pipeline smoke test

- Run one short optimization trial with tiny harmful/harmless prompt sets, batch size 1, and low response length. Confirm residual collection, component I/O collection, scoring, and adapter updates complete without missing-hook assertions, NaNs, or OOM.
- Confirm the intended routed expert path participates. If only attention/shared modules are supported in the first patch, the run must display that scope clearly and must not be presented as complete GLM-4.7-Flash coverage.
- Save/export the result, reload it in a fresh process, and generate again. Compare adapter and merged export paths if both are supported.

### E. Quality and regression evaluation

- Use fixed seeds and a small, versioned prompt set to compare base and abliterated outputs on refusal-sensitive prompts plus harmless control prompts. Record refusal rate, control-task quality, and any obvious quality regressions; keep prompt material in a suitable test fixture without embedding sensitive secrets.
- Run at least one configured lm-eval benchmark on both base and modified models, choosing a small practical task first. Record task/version, few-shot settings, hardware, and base/modified scores. A benchmark is a regression signal, not proof of safety or general quality.
- Run existing repository tests and lint/type checks relevant to changed code. Do not update reproducibility hashes unless output changes are intentional and independently reviewed.

## Completion checklist

- [ ] GLM-4.7-Flash loads under a pinned supported dependency set.
- [ ] Intended components, including routed experts if claiming full support, are discovered, measured, transformed, reset, and exported correctly.
- [ ] Unit/reference tests pass without requiring the full checkpoint.
- [ ] Full-model generation and one end-to-end short Heretic run complete on documented hardware.
- [ ] Exported model reloads and generates in a clean process.
- [ ] Base-versus-modified refusal/control and benchmark results are recorded.
- [ ] Changes are committed on a dedicated branch with a concise summary of limitations and hardware/version details.

## Upstream references

- Model card and load instructions: <https://huggingface.co/zai-org/GLM-4.7-Flash>
- Transformers model documentation: <https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/glm4_moe_lite.md>
- Transformers implementation (packed routed expert tensors): <https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py>
