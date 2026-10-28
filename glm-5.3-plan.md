# GLM-5.3 compatibility plan

## Goal and scope

Adapt the existing GLM-4.7-Flash integration to GLM-5.3 in two phases:

1. Now: validate config/source compatibility and tiny random models without downloading or loading full weights. An A100 80GB test environment is sufficient for tiny CUDA/BF16 tests if needed.
2. On an 8×B300 target system: validate the official BF16 checkpoint, multi-GPU execution, a short optimization run, merged export/reload, and base-versus-modified evaluation.

Tiny-model success is not proof of full-checkpoint compatibility or quality. Do not download the full checkpoint or run jobs on the future system until the user authorizes that phase.

Branch: `explore/glm-5.3-config`.
Base: `ccc8d46646b1a9dcc1d4fb16d5ba30ce2915e487`, including GLM-4.7 implementation `c30d653` and the sparse ARA fix. The original plan-only branch did not contain these changes.

## Findings so far

- Published config: `https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json`.
- Architecture: `GlmMoeDsaForCausalLM`; model type: `glm_moe_dsa`.
- Config reports 78 layers, hidden size 6,144, 256 routed experts, eight selected experts per token, one shared expert, and three initial dense MLP layers.
- Installed environment: Transformers 5.18.0, PyTorch 2.14.1, PEFT 0.21.2. Transformers includes the native architecture. No dependency synchronization was performed.
- Installed `GlmMoeDsaExperts` has the same basic interface as the existing wrapper: `forward(hidden_states, top_k_index, top_k_weights)`, packed `gate_up_proj` of shape `[experts, 2 * intermediate, hidden]`, packed `down_proj` of shape `[experts, hidden, intermediate]`, and an activation function.
- Dense/shared output projections are ordinary `nn.Linear` modules. Attention exposes `self_attn.o_proj`.
- DSA means sparse attention with a token-selection indexer. Some layers can reuse an earlier layer's selection. Tiny tests must cover both full and shared indexers, including cached generation.
- Existing GLM wrapping, adapter preparation, and export restrictions are gated specifically on `glm4_moe_lite`. GLM-5.3 does not yet receive those paths.
- `zai-org/GLM-5.3` declares FP8 checkpoint quantization. Heretic's `quantization='none'` setting does not by itself remove a checkpoint's quantization metadata. Unquantized tiny-model success must not be described as support for this published FP8 checkpoint.
- An official unquantized checkpoint exists at `https://huggingface.co/zai-org/GLM-5.3-BF16`, verified at revision `abf42e6c0e53bc29a875511301e7b7fcd220315b`. Its config has `dtype='bfloat16'`, `model_type='glm_moe_dsa'`, and no quantization config. Hub metadata lists 753,329,921,024 BF16 parameters and 19,456 FP32 parameters. Use this variant for the unquantized compatibility target; no weights were downloaded.

These are config/source findings, not runtime validation. Record the exact upstream revision and raw config in the first preflight step.

## Steps

### 1. Config-only preflight

Fetch only `config.json` and repository revision metadata. Do not fetch checkpoint shards. Resolve `AutoConfig` and the causal-LM mapping without calling the full checkpoint loader. Record raw and resolved layer patterns, attention/indexer settings, dimensions, and quantization metadata.

Derive expected component counts from resolved MLP layer types. If the config resolves to three dense plus 75 sparse layers, expect 78 attention projections and 19,278 MLP projections: three dense, 75 shared, and 19,200 routed.

Optionally inspect the weight-index JSON for tensor names and checkpoint size. Do not fetch any tensor data. A config cannot prove tensor layout; use installed source and, if needed, index metadata to check it.

### 2. Tiny native-model reference checks

Create a CPU model with reduced vocabulary, hidden dimensions, heads, expert count, and layer count. Explicitly set coherent MLP and indexer patterns. No real model weights are involved.

Before changing production code, compare native packed-expert outputs against `GlmExpertViews` using deterministic inputs and routing. Cover repeated token selections, inactive experts, routing weights, and native sentinel expert behavior. Check storage sharing and numerical agreement.

Check native forward, hidden-state collection, full/shared DSA indexers, and short cached generation on the tiny model.

### 3. Extend the existing integration

Change `src/heretic/model.py` and `src/heretic/main.py` to apply the proven packed-expert preparation and merged-export policy to `glm_moe_dsa` as well as `glm4_moe_lite`. Keep the accepted architectures explicit rather than assuming all packed-expert models are identical.

Reuse `src/heretic/glm.py` only after reference tests pass. Preserve dense/shared/routed component discovery, PEFT target-name handling, and restoration of the native checkpoint layout.

Add an early, actionable rejection for unsupported checkpoint quantization. Separately investigate whether Transformers supports explicit FP8-to-BF16 dequantization for this architecture; do not silently strip metadata or claim that path works without testing it.

### 4. Tiny end-to-end regression tests

Add `tests/test_glm_moe_dsa.py`. Cover:

- Layer access and complete attention/dense/shared/routed module discovery.
- Identical logits before and after installing expert views and zero-initialized adapters.
- Routed hooks with 2-D selected-token activations and inactive expert handling.
- Deterministic rank-one update, reset between trials, and native packed-weight merge.
- Save/reload of a tiny merged model with equivalent logits and generation.
- No private adapter paths remaining in the native export.
- Quantization and standalone-adapter rejection.
- Existing sparse ARA regression tests and GLM-4.7 tests still passing.

Run Python with `uv`. Use the existing environment without syncing where appropriate, and report the exact package versions. Run changed-file lint/format checks and focused unit tests. Do not update reproducibility hashes.

### 5. Report scope and remaining blockers

Report config/source compatibility separately from tiny-model runtime compatibility. Document the published FP8 checkpoint limitation, CPU-only test coverage, and any required full-checkpoint follow-up.

The 753B model's rough BF16 weight requirement is 1.506 TB decimal (about 1.37 TiB), before activations, adapters, loading overhead, and analysis. Active expert count does not reduce total resident weight storage. Do not assume the existing Python expert loop or ARA activation collection will scale acceptably to 19,200 routed expert projections.

## Phase 2: full BF16 validation on 8×B300

### 6. Hardware and storage preflight

Use the official `zai-org/GLM-5.3-BF16` checkpoint, not the FP8 repository with a dtype override. Pin the model/tokenizer revision and record the code commit, config, datasets, seeds, and package versions for every run.

NVIDIA lists 2.1 TB total GPU memory for DGX B300's eight GPUs. Other systems may have a different vendor configuration. Measure actual usable per-GPU capacity with `nvidia-smi`; do not derive an exact per-GPU capacity from a rounded marketing total. Confirm all eight GPUs, NVLink topology, driver/CUDA compatibility, BF16 operation, and supported PyTorch wheels. Run `nvidia-smi topo -m` and a small cross-GPU transfer test.

Capacity estimates:

- BF16 checkpoint weights: about 1.507 TB decimal, averaging 188.3 GB per GPU if perfectly balanced.
- If usable aggregate HBM is near 2.1 TB, gross remaining capacity is about 0.59 TB. This is not guaranteed free working memory: placement imbalance, CUDA, activations, adapters, temporary tensors, and merge buffers consume it.
- Aim for at least 20–30 GiB spare on each GPU after model loading as an initial smoke-test floor. Measure later phases before increasing prompts or context length.
- Plan at least 4 TB free local NVMe for source weights, one merged export, logs, and working files. Prefer 5 TB or more. Check the pinned weight index's exact size first. A second preserved base copy, extra exports, or temporary duplicate saves need additional disk.
- Record host RAM. Prefer roughly 2 TB or more; less may work only with verified streaming/sharded loading and saving. Do not assume the loader or exporter will avoid assembling large host tensors.

Stop before download if disk, host memory, dependency support, or GPU headroom planning is inadequate. Download only the pinned BF16 checkpoint after this gate passes and the user authorizes the transfer.

### 7. Multi-GPU placement and wrapper gate

Start with one process and Accelerate layer-wise placement. Eight GPUs should hold different decoder layers; do not launch eight independent model copies with `torchrun`. `device_map='auto'` is the starting point, not proof of correct or balanced placement. Tensor/expert parallel execution is a separate implementation path and is not covered by the existing wrapper.

Before full loading:

- Test a tiny GLM split across at least two GPUs. Exercise wrapped forward, cached DSA generation, residual and module-I/O hooks, adapter preparation/reset, and merged export/reload.
- Audit `GlmExpertViews` construction under Accelerate. Replacing a dispatched expert module can lose device hooks; allocating temporary `nn.Linear` storage per expert or creating adapters on the wrong GPU can raise peak memory. Require correct placement, unchanged packed-weight storage sharing, and no unintended full packed-tensor copies.
- Keep each decoder layer intact. Inspect the resulting device map and assert no unexpected CPU/disk offload. If automatic placement is too tight or unbalanced, add a tested loader option for per-device `max_memory` budgets or an explicit layer map. The current plan does not assume Heretic already exposes `max_memory`.
- Reserve extra space on the input/output GPU for logits and scoring. Log memory both before and after wrapper and LoRA installation.

The current per-expert Python loop may be slow. Measure it before a long run; preserve hooks and numerical equivalence if optimizing it. NVLink does not automatically make this single-process layer-wise path tensor parallel.

### 8. Full-checkpoint staged smoke tests

Use a dedicated config rather than changing defaults. Create it after Phase 1 support and multi-GPU tests pass:

```toml
model = "zai-org/GLM-5.3-BF16"
dtypes = ["bfloat16"]
quantization = "none"
device_map = "auto"
offload_outputs_to_cpu = true
batch_size = 1
max_response_length = 16
n_trials = 1
modifiers = [{ plugin = "heretic.modifiers.abliteration.Abliteration" }]

[modifier.Abliteration]
row_normalization = "none"

[modifier.Abliteration.good_prompts]
dataset = "mlabonne/harmless_alpaca"
split = "train[:4]"
column = "text"

[modifier.Abliteration.bad_prompts]
dataset = "mlabonne/harmful_behaviors"
split = "train[:4]"
column = "text"
```

This is a proposed starting config, not a validated runnable recipe. Also limit each chosen scorer's dataset to a few examples, set an explicit input-token limit in the smoke harness, and verify actual effective settings. Select standard Abliteration explicitly because the repository defaults to ARA. `row_normalization='none'` avoids the full-normalization SVD path for the first run.

Run each gate separately and preserve logs:

1. Native BF16 load and short greedy generation before installing Heretic wrappers. Validate native chat template, reasoning controls, EOS, finite outputs, and the resolved model/config. Inspect loading warnings, especially unexpected extra prediction-layer tensors, rather than hiding them.
2. Install expert views and zero adapters. Compare native/wrapped logits on fixed prompts within a documented BF16 tolerance. Confirm expected component counts from the resolved config, weight placement, and memory headroom.
3. Collect a few good/bad residuals. Check shapes, finite values, cross-device transfers, and CPU output offload.
4. Run one actual standard Abliteration trial with real scoring, rather than a constant placeholder objective. Confirm finite adapters and reset behavior. A 16-token reasoning-only output proves execution, not refusal reduction.
5. Separately collect routed module I/O on a few prompts. Measure host RAM, activation row counts, and the number of experts without data. This gates ARA; it is not required to call standard Abliteration validated.

At every gate record elapsed time and per-GPU allocated/reserved/peak memory, free GPU memory, host RAM, and disk use. Stop on OOM, unexpected offload, missing coverage, NaNs, or sustained unusable latency. Do not start a 100/200-trial run yet.

### 9. Merge, export, and fresh-process reload

Only merged native export is supported by the existing expert-view approach. Standalone adapters contain private module paths absent from the original checkpoint.

Inspect and test the merge/save code for multi-GPU memory peaks. Merge sequentially and save in shards without cloning the whole model or collecting the full state dictionary on one GPU/CPU. If the existing path does not meet this requirement, implement and validate a bounded-memory export path before using the full checkpoint. Confirm each expert is repacked correctly, native names are restored, and private wrapper/adapter tensors are absent.

Export to a new directory; never overwrite the base checkpoint. Free the modified process before fresh-process reload so two full models are not resident together. Reload through the native Transformers class, compare logits/generation against the pre-export modified model, and record shard size and exact output bytes.

### 10. Evaluation and controlled scale-up

Use fixed, versioned evaluation prompts separate from optimization prompts. Compare base and modified models sequentially on the same hardware and settings. Test longer responses with verified reasoning controls so final answers, not just reasoning prefixes, are evaluated.

Record refusal rate, harmless-control correctness, response truncation, and a practical lm-eval task such as PIQA. Start small to verify the harness, then use a sample large enough to report uncertainty meaningfully. An eight-example benchmark or a few controls do not establish general quality preservation. Record actual optimization objective values and holdout results.

Scale batch size, prompt counts, response length, and trial count one at a time. Recheck peak memory and latency after each increase. Profile the standard Abliteration FP32 per-matrix workspace before enabling row normalization; test the full-normalization SVD separately.

Treat ARA as a separate milestone. Its routed activation collection, FP32 matrix work, effective-weight gradients, and nearest-neighbor work may have substantial memory/time costs. Require the sparse-expert regression tests, a full-model tiny ARA trial with meaningful scoring, and explicit skipped-expert counts before scaling. If too large, design layer-wise/streamed activation collection as a separate change. Do not silently skip routed components and claim full coverage.

## Acceptance criteria

### Phase 1: config and tiny-model testing

- No full checkpoint shard is downloaded or loaded.
- Exact config revision and installed dependency versions are recorded.
- Tiny native/wrapped forward equivalence, adapter/reset/merge, export/reload, and DSA generation tests pass.
- Existing GLM-4.7 and sparse ARA tests pass.
- Unsupported FP8 loading and standalone export paths fail clearly.
- Final report does not claim real-checkpoint compatibility, performance, refusal reduction, or quality preservation from config or tiny-model tests.

### Phase 2: on 8×B300

- Actual GPU capacity/topology, host RAM, disk capacity, and exact software/model revisions are recorded.
- Tiny multi-GPU tests pass before full-checkpoint loading.
- Native and wrapped BF16 generation pass with measured per-GPU headroom and complete intended component coverage.
- One meaningful standard Abliteration trial passes with finite adapters, successful reset, and recorded scoring.
- Native merged export/reload passes without retaining two resident full models or making unbounded full-model copies.
- Base-versus-modified holdout refusal/control and benchmark results are recorded with clear sample-size limits.
- ARA and full-normalization support are reported separately; neither is implied by the standard smoke run.
- Longer optimization starts only after memory, latency, and evaluation gates pass.

Do not commit or push unless requested.

## Hardware reference

- NVIDIA DGX B300 specifications (8 GPUs, advertised 2.1 TB total GPU memory): <https://www.nvidia.com/en-us/data-center/dgx-b300/>. Measure the actual target system rather than assuming it matches DGX specifications.
