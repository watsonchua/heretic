# Combined cybersecurity evaluation prompts

The combined corpus has **1,085 harmful** and **947 harmless** source prompts. JSONL and Parquet exports contain exactly one string column, `text`, matching the schema of `mlabonne/harmless_alpaca` and `mlabonne/harmful_behaviors`.

| Source | Harmful | Harmless |
|---|---:|---:|
| StrongREJECT | 25 | 0 |
| HarmBench | 67 | 0 |
| ClearHarm | 12 | 0 |
| CyberSecEval MITRE | 981 | 0 |
| harmless_alpaca | 0 | 197 |
| CyberSecEval false-refusal | 0 | 750 |

Use `data/harmful.jsonl` and `data/harmless.jsonl` (or their `.parquet` equivalents) for the complete sets. Each label directory also provides train/test exports: harmful 856/229, harmless 757/190. Companion `.metadata.jsonl` files and `evaluation.jsonl` preserve provenance, original row, source field, source revision, label and expected behavior. `manifest.json` records source counts and checksums.

```python
from datasets import load_dataset
harmful = load_dataset('parquet', data_files='dataset-tools/cyber-eval/data/harmful.parquet', split='train')
harmless = load_dataset('parquet', data_files='dataset-tools/cyber-eval/data/harmless.parquet', split='train')
print(harmless.column_names)  # ['text']
```

No local paraphrases, formatting variants or repetitions are included. CyberSecEval MITRE uses original `base_prompt`, not `mutated_prompt`. FRR uses the benchmark user prompt labeled `is_malicious=False`. HarmBench contextual tasks retain supplied context followed by the original behavior. Original source data can itself be model-generated; these are distinct source requests, not necessarily human-authored requests.

Nineteen normalized exact duplicates were removed. Identical normalized behavior groups, including contextual HarmBench requests, stay in one split. Semantic near-duplicates across sources may remain: exact uniqueness does not prove independent behaviors. Labels follow benchmark conventions; some offensive benchmark requests are dual-use and require contextual review for the policy being evaluated. No decision model or CLM key was used. These exports are ready to load, but not certified as 1,000 semantically independent examples per class.

Reproduce offline from the saved sources:

```bash
.venv/bin/python dataset-tools/cyber-eval/build.py
.venv/bin/python dataset-tools/cyber-eval/verify.py
```

`build.py` is the sole builder in this standalone folder. It includes no augmentation workflow.

Evaluate harmful compliance and harmless over-refusal separately, with a response judge that checks substance rather than refusal keywords. Do not evaluate on prompts used for tuning. No attack answers were generated and no datasets were published.

StrongREJECT and HarmBench upstream MIT licenses are saved under `sources/`. Cite StrongREJECT (https://arxiv.org/abs/2402.10260), HarmBench (https://arxiv.org/abs/2402.04249), and CyberSecEval (https://arxiv.org/abs/2404.13161). Harmless Alpaca, ClearHarm and the CyberSecEval mirrors' cards do not declare source licensing sufficient for this package to grant redistribution rights; verify upstream terms before redistribution. Additional source search details are in `DATASET_SEARCH.md`.
