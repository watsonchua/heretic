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

## Remaining work

Extend the explicit architecture guards and adapter/export policy to `glm_moe_dsa`, with tiny end-to-end adapter/reset/merge/export tests and quantization rejection. The tests above manually install expert views; they do not exercise GLM-5.3 adapters through Heretic.

Then validate CUDA/BF16 and small multi-GPU placement. Full BF16 loading and optimization remain gated by the 8×B300 hardware plan in `glm-5.3-plan.md`. Full-normalization and ARA must be validated separately.
