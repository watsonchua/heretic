# GLM-4.7-Flash implementation and test report

Date: 2026-10-07. Branch: `support/glm-4.7-flash`.

## Changes and repository preparation

Fetched the plan from remote branch `glm-4.7-flash-abliteration` and fast-forwarded to `c940f6129411c3ac8798220c090c7a2e15bc86a2`. The default branch is `master`, not `main`. A local branch with an `origin/` name obscured the remote ref; using its fully qualified ref resolved this. Created the dedicated support branch. Existing untracked DeepSeek export and `view_trials.py` were left untouched.

Added `src/heretic/glm.py`, which exposes routed expert output projections as modules sharing the original packed storage. Routing, gate/up activation, output projection and weighted accumulation follow the native implementation. Ordinary PEFT LoRA now measures, modifies and resets each expert projection.

Updated `src/heretic/model.py` to discover dense, shared and routed projections and restore the native packed layout after adapter merge. PEFT initially converted GLM MLP module targets into packed parameter targets, silently omitting the intended module adapters. Temporary projection names bypass that conversion and are restored during merge. Hook capture supports two-dimensional routed activations and always removes hooks after generation, including errors. Routed I/O contains selected token rows rather than one final-position row per prompt; consumers must account for this difference.

Updated `src/heretic/main.py` to select merged GLM export and explicitly reject standalone adapters. Such adapters use module-view paths absent from the native checkpoint. Quantized GLM use is also explicitly rejected; only unquantized support was tested.

Added tiny-model reference/regression tests and full-checkpoint smoke/evaluation scripts in `tests/test_glm4_moe_lite.py`, `tests/glm_smoke.py`, and `tests/glm_evaluate.py`.

## Environment and package management

Used `uv` for Python execution and package management after the user's instruction. Hardware: NVIDIA A100-SXM4-80GB, driver 580.173.02, CUDA 13.0, approximately 167 GiB host RAM and initially 422 GB free disk.

Installed environment: Python 3.12.3; PyTorch 2.13.0+cu130; Transformers 5.6.2; PEFT 0.19.1; Accelerate 1.13.0; lm-eval 0.4.11. The installed versions differ from the newer project requirements.

Attempted `uv sync --locked --index pytorch=https://download.pytorch.org/whl/cu130`. Resolution first failed on tqdm because of index ordering. Retried with `--index-strategy unsafe-best-match`; resolution succeeded but required a lockfile update, which `--locked` refused. No dependency files were changed. Tests used `uv run --no-sync`; the declared locked dependency set remains unvalidated.

Downloaded full checkpoint revision `7dd20894a642a0aa287e9827cb1a1f7f91386b67`. AutoConfig reports `glm4_moe_lite`, 47 layers; the native causal model instantiated and loaded successfully.

## Confirmed validation

- All six unit tests passed with `uv run --no-sync python -m unittest discover -s tests -p 'test_*.py'`.
- Tiny GLM checks cover dense/sparse discovery, unchanged initial outputs, routed hook invocation and 2-D shapes, deterministic rank-one weight reference, zero adapter reset, merged packed weight values, export/reload logits, and generation.
- Lint passed on the helper, model integration and new scripts. All six changed Python files pass formatting. New helper/test scripts pass typing. Model typing reports three pre-existing assignments of `None` to its non-optional model annotation; main.py lint reports pre-existing E402 import placement errors.

Full BF16 smoke command:

```bash
uv run --no-sync tests/glm_smoke.py > /tmp/glm-smoke.log 2>&1
```

Settings: no quantization, BF16, automatic device placement, batch size 1, CPU output offload, response limit 16. Used two harmless and two refusal-sensitive prompts embedded in the script.

The full model completed generation, good/bad residual collection, module-I/O collection, one Optuna parameter trial, actual Abliteration updates with row normalization disabled, finite-adapter checks, and merged export. Discovered 47 attention projections and 2,991 MLP projections: one dense, 46 shared and 2,944 routed. Routed captures span all sparse layers; only experts selected by routing produce activation rows.

Load time: 59.75 seconds. Time through modified generation: 91.82 seconds, excluding export. Peak allocated VRAM: 60,700,238,336 bytes (56.53 GiB). The native export completed with 14 shards plus tokenizer under `glm-smoke-export/`. Raw results are in `glm-smoke-results.json`.

The trial used a placeholder objective of zero, so it proves execution, not measured optimization. Base/modified responses were identical at 16 tokens and contained only the start of reasoning. Refusal reduction and control-task quality cannot be inferred from this result.

Fresh-process native export reload and short generation completed sufficiently to reach lm-eval initialization. The eight-example zero-shot PIQA evaluation was launched but its outcome is pending. Command:

```bash
uv run --no-sync tests/glm_evaluate.py glm-smoke-export glm-modified-evaluation.json > /tmp/glm-evaluate-modified.log 2>&1
```

## Limitations and remaining work

The corresponding base PIQA run, meaningful longer refusal/control responses, full-normalization/SVD testing, other modifiers, multi-GPU/offload behavior and verification against pinned project dependencies remain incomplete. Heavy existing architecture reproducibility tests were not run; hashes were not changed. Support requires merged unquantized export. Generated model/result artifacts should not be included in the code commit.

Sandbox execution repeatedly failed on network-interface initialization. Reviewed escalations were used. Approval review also failed on unsupported reviewer model names and later HTTP 429, interrupting verification and the first report write. Those failures were not bypassed.

## Completed fresh-process benchmark

Native export reload and non-thinking greedy generation passed: both base and modified models answered `2 plus 2 equals **4**.` PIQA completed on both models, lm-eval 0.4.11, eight examples, zero-shot, seeds 42:

| Metric | Base | Modified |
| --- | --- | --- |
| Accuracy | 0.875 (7/8) | 0.875 (7/8) |
| Normalized accuracy | 0.750 (6/8) | 0.750 (6/8) |

Accuracy standard error is 0.125; normalized accuracy standard error is 0.164. This sample provides no statistically useful evidence of general quality preservation. Raw files: `glm-base-evaluation.json` and `glm-modified-evaluation.json`. Initial modified evaluation waited for disk I/O before completing; it did not fail.

Added `tests/glm_controls.py` to compare the same four prompts with `enable_thinking=False`, greedy decoding and a 128-token response limit. Both models correctly answer both harmless controls (2+2=4; France's capital is Paris) and explicitly refuse both refusal-sensitive prompts. Observed refusal rate is 2/2 (100%) for both; no refusal reduction was demonstrated. Wording differs after modification. One modified response makes an unverified claim about applicable fraud law; factual quality is not established by these two controls. Raw files: `glm-base-controls.json` and `glm-modified-controls.json`.

Final unit rerun: six tests passed in 2.303 seconds. Final helper/model/script lint and all seven changed Python files' formatting pass. Remaining limitations above still apply, except that fresh-process benchmark and longer control comparison are now complete. The commit includes code, evaluation scripts and this report, without generated checkpoint or raw JSON artifacts. This report does not claim quality improvement or completion of every acceptance criterion in the original plan.

The first commit attempt failed because Git author identity was unset. After the user supplied `watsonchua / watsonchua@gmail.com`, configured this identity locally for the repository and committed the eight intended code/test/report files as `c30d65368197b4b17ca2e8bb667c9005810b33ba` on `support/glm-4.7-flash`. Generated model/results and pre-existing untracked files were excluded. No remote push was performed.
