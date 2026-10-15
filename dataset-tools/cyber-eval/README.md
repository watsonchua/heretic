# Cyber evaluation dataset workflow

Standalone code and methodology for combining source cyber prompts into the single-column `text` schema used by harmless_alpaca. The current corpus is **v0**: 1,085 harmful and 947 harmless prompts. Source files and generated datasets are excluded from version control.

> **v0 is not fit for abliteration or for refusal evaluation as it stands.** Measurement showed
> the label is predictable from the first three words of the prompt at 91% accuracy, because the
> harmful and harmless sides were drawn from differently-templated fields of CyberSecEval.
> A rebuild is specified in [PLAN.md](PLAN.md). Read that before using these exports.

From the repository root, using the existing environment:

```bash
.venv/bin/python dataset-tools/cyber-eval/download_sources.py
.venv/bin/python dataset-tools/cyber-eval/build.py
.venv/bin/python dataset-tools/cyber-eval/verify.py
```

Corpus quality is measured separately, and `diagnostics.py` is the only check in the workflow
that uses no model at all:

```bash
# model-free metrics; writes the tracked diagnostics.<version>.json baseline
python diagnostics.py --data ../../cyber-eval/data --version v0

# strip MITRE roleplay pretexts; model-free, writes templates.json + cores.jsonl
python detemplate.py --sources ../../cyber-eval/sources --out .

# collapse to one prompt per behaviour, then cap MITRE's share
python select_cores.py --target 220

# calibrate System One before any gate depends on it
python calibrate.py sample --data ../../cyber-eval/data   # stratified 100
python triage.py --sample-agreements 20                   # 2nd opinion; shrinks the hand-label set
python calibrate.py score                                 # agreement and thresholds

# score one model's actual behaviour; results are per (prompt, model), not dataset properties
python evaluate.py --prompts selected.mitre.jsonl --field core --model <id> --label harmful
```

Models: `clef-flash` (System One) labels and gates, `kimi-k3` writes the harmless twins,
`glm-5.2` judges responses and gives the second labelling opinion, and
`azure/text-embedding-3-large` clusters. No model ever grades its own output.

For a separate environment, create a virtual environment and install `requirements.txt` first. Downloads require internet access; building and verification run offline. The downloader uses the `hf` CLI installed alongside Python and pinned GitHub URLs. It does not read `.env`. The v0 build itself makes no model calls; the planned v1 rebuild uses the CLM System One classifier for labelling and gating, as described in [PLAN.md](PLAN.md).

`systemone.py` is the shared classifier client and holds the question bank; rewording a question there invalidates its calibration. Credentials load from the repository-root `.env` with `python-dotenv` (two lines in that file put spaces around `=`, so shell `source` fails on it).

Outputs are created in this folder's `data/` directory with a checksum manifest. Complete sets, train/test splits, JSONL, Parquet, and companion provenance files are produced. The verifier checks source fidelity, exact duplicate removal, grouped split isolation, checksums and text-only export schemas.

See [PLAN.md](PLAN.md) for the v1 rebuild specification, [METHODOLOGY.md](METHODOLOGY.md) for v0 selection rules, label limitations and evaluation guidance, and [DATASET_SEARCH.md](DATASET_SEARCH.md) for the source search. v0 uses no local augmentation. Semantic independence and prompt-level labels have not been audited; the workflow does not claim 1,000 independently validated behaviors per label.
