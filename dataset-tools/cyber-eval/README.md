# Cyber refusal evaluation corpus

A paired corpus for measuring refusal behaviour on cybersecurity prompts, and the tooling that
builds and scores it. Each record is a **pair**: a harmful prompt and a harmless twin matched on
topic, target, named technologies, task verb, output format and length, differing only in intent.

Source files and generated data are excluded from version control. See
[METHODOLOGY.md](METHODOLOGY.md) for the approach and the measurements behind it,
[DATASET_SEARCH.md](DATASET_SEARCH.md) for the source inventory, and [PLAN.md](PLAN.md) for
remaining work.

## Build

```bash
pip install -r requirements.txt
python download_sources.py          # pinned revisions into sources/

python detemplate.py                # strip roleplay wrappers -> cores.jsonl, templates.json
python select_cores.py --target 220 # one prompt per behaviour, capped per source
python screen.py --input sources/<file> --field <col> --name <id> --label harmful
python twins.py                     # generate harmless twins -> pairs.jsonl
```

## Check

```bash
python diagnostics.py               # model-free corpus metrics
python gates.py                     # all seven gates -> gates.json
```

Calibrate the classifier before trusting any gate that depends on it:

```bash
python calibrate.py sample          # stratified 100 from pairs.jsonl
python triage.py                    # second opinion; aims the hand-label effort
# hand-label calibration/to_label.jsonl
python calibrate.py score           # per-question agreement and thresholds
```

## Score a model

Refusal is a property of a (prompt, model) pair, not of the dataset, so results live separately
and accumulate one file per model.

```bash
python evaluate.py --prompts pairs.jsonl --field harmful --model <id> --label harmful
```

## Models

| Model | Role |
|---|---|
| `clef-flash` (System One) | prompt labels, response verdicts, gates |
| `kimi-k3` | writes the harmless twins |
| `glm-5.2` | judges responses; second opinion during calibration |
| `azure/text-embedding-3-large` | behaviour clustering, pairing similarity |

No model grades its own output. `diagnostics.py` imports none of them and is the only check in
the workflow that is not a model's opinion.

Credentials load from the repository-root `.env` with `python-dotenv`. Two lines in that file
put spaces around `=`, so shell `source` fails on it. `.env` is gitignored.
