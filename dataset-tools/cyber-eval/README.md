# Cyber evaluation dataset workflow

Standalone code and methodology for combining source cyber prompts into the single-column `text` schema used by harmless_alpaca. The verified corpus contains 1,085 harmful and 947 harmless prompts. Source files and generated datasets are excluded from version control.

From the repository root, using the existing environment:

```bash
.venv/bin/python dataset-tools/cyber-eval/download_sources.py
.venv/bin/python dataset-tools/cyber-eval/build.py
.venv/bin/python dataset-tools/cyber-eval/verify.py
```

For a separate environment, create a virtual environment and install `requirements.txt` first. Downloads require internet access; building and verification run offline. The downloader uses the `hf` CLI installed alongside Python and pinned GitHub URLs. It does not read `.env` or use CLM.

Outputs are created in this folder's `data/` directory with a checksum manifest. Complete sets, train/test splits, JSONL, Parquet, and companion provenance files are produced. The verifier checks source fidelity, exact duplicate removal, grouped split isolation, checksums and text-only export schemas.

See [METHODOLOGY.md](METHODOLOGY.md) for selection rules, label limitations and evaluation guidance, and [DATASET_SEARCH.md](DATASET_SEARCH.md) for the source search. No local augmentation is used. Semantic independence and prompt-level labels have not been fully audited; the workflow does not claim 1,000 independently validated behaviors per label.
