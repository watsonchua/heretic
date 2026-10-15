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
python download_sources.py      # pinned revisions into sources/

python detemplate.py            # strip roleplay wrappers -> cores.jsonl, templates.json
python select_cores.py          # one prompt per behaviour -> selected.mitre.jsonl
python build_pool.py            # screen every source -> harmful_pool.jsonl, control_pool.jsonl
python twins.py --prompts harmful_pool.jsonl --field text   # -> pairs.jsonl
python export.py                # matched-order exports, splits, manifest -> export/
```

`detemplate.py` and `select_cores.py` prepare CyberSecEval MITRE, which needs de-templating and
behaviour-level clustering before it is usable. Every other source goes straight into
`build_pool.py`.

## Check

```bash
python diagnostics.py           # model-free corpus metrics
python gates.py                 # all seven gates -> gates.json
```

## Calibrate

Thresholds come from data, not from guesses. The harm question calibrates against upstream human
labels; the rest need hand labels.

```bash
python calibrate_human.py       # harm question vs WildGuardMix human labels (n=632)

python calibrate.py sample      # stratified 100 from pairs.jsonl
python triage.py                # second opinion; aims the hand-label effort
# hand-label calibration/to_label.jsonl
python calibrate.py score       # per-question agreement and thresholds
```

## Score a model

Refusal is a property of a (prompt, model) pair, not of the dataset, so results live separately
and accumulate one file per model.

```bash
python evaluate.py --prompts pairs.jsonl --field harmful --model <id> --label harmful
```

## Layout

| Module | Role |
|---|---|
| `common.py` | env, JSONL/table IO, disk cache, shared rate gate |
| `llm.py` | LiteLLM gateway: embeddings, twin generator, response judge |
| `systemone.py` | System One classifier and the question bank |

| Model | Role |
|---|---|
| `clef-flash` (System One) | prompt labels, response verdicts, gates |
| `gemini-3.8-flash-global` | writes the harmless twins |
| `glm-5.2` | judges responses; second opinion during calibration |
| `azure/text-embedding-3-large` | behaviour clustering, pairing similarity |

No model grades its own output. `diagnostics.py` imports none of them and is the only check in
the workflow that is not a model's opinion.

Credentials load from the repository-root `.env` with `python-dotenv`. Two lines in that file
put spaces around `=`, so shell `source` fails on it. `.env` is gitignored.
