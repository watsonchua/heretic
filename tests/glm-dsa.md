# GLM-5.3 small-model tests

These tests check the `glm_moe_dsa` modification path. They do not measure refusal removal. The small checkpoints have random or toy-trained weights, not representative instruction-following behavior.

## Offline regression tests

```bash
uv run --no-sync tests/test_glm_moe_dsa.py
```

No weights are downloaded. The test builds a three-layer model locally, including dense and sparse MLPs and full/shared attention indexers. It checks expert routing, shared weight storage, projection discovery, residual capture, classic abliteration, reset, native export, and reload equivalence. Quantized checkpoints are rejected before loading.

## Checkpoint smoke tests

Run from the repository root. Each command downloads the named public checkpoint and writes only local files. Choose a new output directory for every run. Existing output directories are never overwritten. Settings are isolated from Heretic CLI arguments, environment settings, and config files.

Start with the tiny random checkpoint on CPU:

```bash
uv run --no-sync tests/glm_dsa_smoke.py --model tiny --device cpu --dtype float32 --output /tmp/glm-dsa-tiny
```

Then run the 0.6B checkpoint. This is also the default `--model`:

```bash
uv run --no-sync tests/glm_dsa_smoke.py --model 0.6b --device cuda --dtype bfloat16 --output /tmp/glm-dsa-06b
```

Finally test the 3.5B checkpoint with more representative tensor dimensions:

```bash
uv run --no-sync tests/glm_dsa_smoke.py --model 3.5b --device cuda --dtype bfloat16 --output /tmp/glm-dsa-35b
```

On an Apple Silicon Mac, use the Apple GPU through MPS:

```bash
uv run --no-sync tests/glm_dsa_smoke.py --model 0.6b --device mps --dtype float32 --output /tmp/glm-dsa-06b-mps
```

For MPS, the script loads weights on CPU first, then moves the model to the GPU. Direct MPS weight loading crashed in native code on the local setup.

Use `--device cpu --dtype float32` without a supported GPU. Float32 doubles weight memory relative to BF16. The 3.5B checkpoint needs about 7 GB for BF16 weights alone. Allow extra memory for activations, expert projections, adapters, and export. A 24 GB GPU is a starting estimate, not a verified capacity requirement. Export reload runs on CPU, so also provide enough system RAM.

Use `--revision COMMIT_HASH` to pin a checkpoint. `--device auto` allows Transformers to choose device placement, but Heretic rejects weights offloaded as meta tensors. For this path, use unquantized weights: setting a BF16 dtype does not dequantize an FP8 checkpoint.

## Checkpoints

| Alias | Repository | Purpose |
| --- | --- | --- |
| `tiny` | `tiny-random/glm-moe-dsa` | Fast pipeline checks with random weights |
| `0.6b` | `inference-optimization/GLM-5.3-0.6B-A0.4B` | Reduced dimensions and toy training |
| `3.5b` | `inference-optimization/GLM-5.3-3.5B-A3.3B-small` | Original hidden/attention dimensions, three decoder layers and ten routed experts per sparse layer |

## Outputs

- `report.json`: status, checkpoint ID, timing, projection counts, captured expert counts, residual shape, nonzero adapter count, generated text, and reload logit error.
- `model/`: merged checkpoint and tokenizer, with native packed experts and no Heretic adapter names.

The script uses two groups of benign contrast prompts. It applies classic directional ablation with fixed nonzero weights to both attention and MLP output projections. It checks that routed expert hooks activate, adapters are finite and nonzero, and saved/reloaded logits agree. Generated text is diagnostic only.

## Verified locally

The CPU float32 tiny checkpoint run passed. It discovered 260 output projections, including all 256 routed experts. The saved/reloaded maximum logit difference was approximately `1.01e-7`.

Transformers reported unexpected keys under `model.layers.2` when loading this checkpoint. Those keys belong to its extra prediction layer, which the installed causal-LM implementation does not load. The two decoder layers still completed the smoke test. This result is not validation of that extra prediction layer.

The 0.6B checkpoint passed on the Apple GPU (MPS) in float32 in approximately 21 seconds. It discovered and modified all 172 output projections across six layers, including 160 routed experts. All 344 adapter tensors were nonzero and finite. Export/reload on CPU matched the GPU logits with a maximum absolute difference of `8.94e-6`.

The 3.5B checkpoint completed CUDA/BF16 modification and serialization checks on one NVIDIA L4 (24 GB) with PyTorch 2.13.0+cu129 and Transformers 5.18.0. The final cached run took 44 seconds and peaked at 7,963,730,432 bytes (7.42 GiB) of CUDA allocated memory. All 26 output projections, including 20 routed experts, were modified; all 52 adapter tensors were finite and nonzero. The merged model and its saved/reloaded copy produced identical CUDA logits (maximum error 0).

The original adapter-to-export equivalence check failed. The final test separates stages rather than loosening tolerance: adapter-to-native-merged maximum logit drift was 0.09375 in BF16; native-merged-to-CPU-reload drift was 0.3515625; native-merged-to-CUDA-reload drift was 0. This validates serialization, not exact adapter/merged equivalence or the acceptability of merge drift for full GLM-5.3. BF16 rounding and different evaluation paths need to be considered when evaluating merged-model quality.

The temporary cloud VM and its disk were deleted after collecting the report. The local model outputs and the three named checkpoint caches were deleted as requested. The final lightweight report is at `/tmp/glm-dsa-35b-cuda-report.json`.

The offline three-layer regression covers the full/shared indexer path missing from the two-layer remote tiny fixture.
