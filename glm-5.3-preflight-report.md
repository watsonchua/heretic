# GLM-5.3 preflight results

## Scope

Config metadata and tiny random CPU models only. No full checkpoint weights were downloaded. CUDA was unavailable; no GPU or multi-GPU tests ran. This report does not claim full Heretic support, refusal reduction, or quality preservation.

Branch: `explore/glm-5.3-config`. Original upstream master `d34dd9192da06d203eb7e80d20229117bb4e981c` was merged as `54779ab`. The merge includes 14 upstream commits. No dependency sync was performed; tests used the existing environment with local checkout source explicitly first on the import path.

## Config findings

| Checkpoint | Pinned config revision | Quantization |
| --- | --- | --- |
| `zai-org/GLM-5.3` | `ba32f8c19c8738e2c57c71798c601d477a7bf844` | FP8 |
| `zai-org/GLM-5.3-BF16` | `abf42e6c0e53bc29a875511301e7b7fcd220315b` | None |

Both resolve to `GlmMoeDsaForCausalLM`. Resolved config patterns give 78 attention projections, three dense MLP projections, 75 shared MLP projections, and 19,200 routed MLP projections. There are 21 full attention indexers and 57 shared-indexer layers.

Hub metadata lists 753,329,921,024 BF16 parameters and 19,456 FP32 parameters for the BF16 variant. The FP8 variant has 751,226,191,872 FP8 parameters, 2,103,729,152 BF16 parameters, and 19,456 FP32 parameters. Dtype badges alone must not be used to infer an unquantized checkpoint.

## Tests

Environment: PyTorch 2.14.1, Transformers 5.18.0, PEFT 0.21.2. CPU-only execution.

The user executed the final test command in this session after Auto mode blocked assistant execution of merged upstream code:

```bash
uv run --no-sync python -c "import sys,unittest; sys.path[:0]=['src','tests']; import glm_dsa_preflight; suite=unittest.defaultTestLoader.discover('tests',pattern='test_*.py'); suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(glm_dsa_preflight)); result=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not result.wasSuccessful())"
```

Result: all 31 tests passed in 0.199 seconds. This includes 27 existing unit tests and four GLM-5.3 preflight tests:

- Native forward with finite logits, hidden-state capture, and cached generation using tiny full/shared DSA layers.
- Expert-wrapper equivalence to native packed-expert outputs, including repeated routing, an inactive expert, and a sentinel expert index. Hooks receive 2-D selected-token activations; expert output weights share packed storage.
- Unchanged wrapped logits, short wrapped generation, repack, tiny native save/reload, and restored logits.
- Explicit confirmation that current Heretic discovery does not automatically wrap GLM-5.3 routed experts. It finds only the shared projection in a sparse layer.

The upstream merge introduced duplicate shared-expert discovery. Removing the duplicate initially changed GLM-4.7 module order and failed its expert update/export test. The final fix keeps one shared-expert discovery block before routed-expert discovery. The GLM-4.7 adapter/reset/merge/export regression now passes.

## Integration follow-up

The initial results above describe the pre-integration baseline. The implementation now extends the explicit packed-GLM architecture guards to `glm_moe_dsa` in discovery, adapter preparation, and merged-only export. The expert wrapper and native repack calculation are reused unchanged.

A shared validation helper rejects requested quantization and checkpoint quantization metadata. `Model.__init__` checks checkpoint config before tokenizer or weight loading, so the FP8 repository fails early with guidance to use `zai-org/GLM-5.3-BF16`.

Added `tests/test_glm_moe_dsa.py` and updated the preflight discovery assertion to require routed-expert coverage. The user ran the same test-suite command after these changes: **all 35 tests passed in 0.338 seconds** on CPU, with the same package versions. This includes the existing GLM-4.7 regression, four updated preflight tests, and four new GLM-5.3 integration tests.

Confirmed integration coverage:

- Tiny-model discovery finds three attention projections and 11 MLP projections: one dense, two shared, and eight routed.
- Zero-initialized adapters preserve native logits.
- A routed expert's rank-one adapter matches a deterministic weight-update reference.
- Reset clears the adapter's output factor and restores baseline logits; the update can then be reapplied.
- Merge produces the expected native packed expert weight, removes expert views and private adapter/projection paths, and marks the model for reload.
- Tiny native save/reload preserves modified logits and supports short generation.
- FP8 metadata is rejected before tokenizer or checkpoint weight loading; requested quantization is rejected by both discovery and adapter preparation.
- Standalone adapters are rejected; default and explicit merged export select native merged output.

Lint and formatting passed for the helper and new/updated test files. These tests do not run a scored optimizer trial or prove full-model CUDA, BF16, offload, or distributed behavior.

## Remaining work

Validate CUDA/BF16 and small multi-GPU placement. Full BF16 loading and optimization remain gated by the 8×B300 hardware plan in `glm-5.3-plan.md`. Full-normalization and ARA must be validated separately. No full checkpoint weights have been downloaded.
