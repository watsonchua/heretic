"""Strip CyberSecEval MITRE roleplay pretexts, recovering the plain request underneath.

Decision 3 in PLAN.md: the corpus measures refusal on plain requests, so the ~190 pretext
wrappers come off. Model-free and deterministic, so it needs no calibration and its output is
auditable.

Boilerplate is identified by distribution, not by a hand-written regex: text that recurs across
unrelated MITRE categories is a wrapper, text that tracks the technique is the request.

  pass 1  punctuation-delimited clauses recurring >=3 times across >=2 categories
  pass 2  leading phrases recurring >=5 times across >=3 categories that name a role or open a
          scenario. Polite framings ("could you", "how would you") are request register, not
          pretext, and are kept.
  pass 3  structural: a leading clause carrying no request verb, when a later clause does carry
          one, is framing. Catches one-off wrappers the frequency passes cannot see.

Writes templates.json (the reviewable catalogue) and cores.jsonl (text + provenance).

  python detemplate.py
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

CLAUSE = re.compile(r"(?<=[,;.?!])\s+")
# A pass-2 prefix is a pretext only if it casts a role or opens a scenario. Without this guard
# the pass also eats "could you" and "how would you", which are register, not framing.
PRETEXT = re.compile(
    r"\b(researcher|analyst|tester|engineer|professor|student|instructor|expert|"
    r"specialist|consultant|developer|hacker|team|group|class|course|lecture|seminar|workshop|"
    r"exercise|project|module|paper|study|segment|challenge|assignment|lab|training|curriculum|"
    r"while|when|since|given|whilst|during|amid|imagine|suppose|assume|pretend|acting|roleplay|"
    r"simulate|simulating|aiming|thinking|considering|exploring|envisioning|"
    r"let'?s|as\s+you|in\s+light\s+of|in\s+the\s+pursuit|for\s+our|for\s+this|for\s+today)\b",
    re.I,
)
REQUEST = re.compile(
    r"\b(write|create|craft|design|develop|generate|implement|produce|build|draft|"
    r"provide|show|outline|illustrate|demonstrate|detail|construct|showcase|engineer|compose|devise|"
    r"present|prepare|make|give|explain|script|code|program)\b",
    re.I,
)
ERROR_ROW = "#ERROR!"


def clauses(text):
    return [c.strip() for c in CLAUSE.split(" ".join(text.split())) if c.strip()]


def pass1(rows):
    """Clause-level boilerplate: recurs >=3 times and spans >=2 MITRE categories."""
    freq = Counter()
    cats = defaultdict(set)
    for r in rows:
        for c in clauses(r["base_prompt"]):
            k = c.casefold()
            freq[k] += 1
            cats[k].add(r["mitre_category"])
    return {
        k: {"count": freq[k], "categories": sorted(cats[k])}
        for k, n in freq.items()
        if n >= 3 and len(cats[k]) >= 2
    }


def pass2(cores, rows):
    """Leading-phrase pretexts, for wrappers the source left unpunctuated."""
    freq = Counter()
    cats = defaultdict(set)
    for core, r in zip(cores, rows, strict=False):
        w = core.split()
        for n in range(1, min(13, len(w) + 1)):
            k = " ".join(w[:n]).casefold()
            freq[k] += 1
            cats[k].add(r["mitre_category"])
    return {
        k: {"count": freq[k], "categories": sorted(cats[k])}
        for k, n in freq.items()
        if n >= 5 and len(cats[k]) >= 3 and PRETEXT.search(k) and not REQUEST.search(k)
    }


def strip_framing(text):
    """Pass 3. Drop leading clauses with no request verb while a later clause still has one."""
    cs = clauses(text)
    removed = []
    while len(cs) > 1 and not REQUEST.search(cs[0]) and any(REQUEST.search(c) for c in cs[1:]):
        removed.append(cs.pop(0))
    return " ".join(cs), " ".join(removed)


def strip_prefix(text, catalogue):
    """Remove the longest catalogued pretext prefix. Returns (core, removed)."""
    w = text.split()
    best = 0
    for n in range(1, min(13, len(w) + 1)):
        if " ".join(w[:n]).casefold() in catalogue:
            best = n
    return " ".join(w[best:]), " ".join(w[:best])


def run(sources, out):
    rows = pq.read_table(
        sources / "cyberseceval-mitre/data/test-00000-of-00001.parquet"
    ).to_pylist()
    # The upstream parquet carries 7 literal "#ERROR!" rows, a spreadsheet artefact. Drop them.
    dropped = [r for r in rows if r["base_prompt"].strip() == ERROR_ROW]
    rows = [r for r in rows if r["base_prompt"].strip() != ERROR_ROW]

    boiler = pass1(rows)
    stage1 = []
    removed1 = []
    for r in rows:
        cs = clauses(r["base_prompt"])
        stage1.append(" ".join(c for c in cs if c.casefold() not in boiler))
        removed1.append([c for c in cs if c.casefold() in boiler])
    prefixes = pass2(stage1, rows)
    records = []
    for r, s1, rm1 in zip(rows, stage1, removed1, strict=False):
        s2, rm2 = strip_prefix(s1, prefixes)
        core, rm3 = strip_framing(s2)
        records.append(
            {
                "task_id": r["task_id"],
                "mitre_category": r["mitre_category"],
                "ttp_id": r["ttp_id_name_mapping"]["TTP_ID"],
                "ttp_name": r["ttp_id_name_mapping"]["TTP_Name"],
                "original": " ".join(r["base_prompt"].split()),
                "core": core,
                "pretext": " ".join(x for x in ([rm2, rm3] + rm1) if x).strip(),
                # A core left as a bare technique phrase needs a canonical imperative in Phase 3. Flagged,
                # never invented here: this script only removes text.
                "needs_verb": not bool(REQUEST.search(core)),
            }
        )

    # The core must be a subsequence of the original; nothing may be invented.
    for x in records:
        o = x["original"].casefold()
        assert all(w in o for w in x["core"].casefold().split()[:40]), x["task_id"]

    usable = [x for x in records if x["core"].strip() and not x["needs_verb"]]
    catalogue = {
        "source_rows": len(rows) + len(dropped),
        "dropped_error_rows": len(dropped),
        "clause_boilerplate": dict(sorted(boiler.items(), key=lambda kv: -kv[1]["count"])),
        "prefix_pretexts": dict(sorted(prefixes.items(), key=lambda kv: -kv[1]["count"])),
        "cores_recovered": len(records),
        "cores_with_request_verb": len(usable),
        "cores_needing_canonical_verb": sum(1 for x in records if x["needs_verb"]),
        "distinct_cores": len(
            {" ".join(x["core"].split()).casefold().rstrip(".?! ") for x in usable}
        ),
        "distinct_ttp": len({x["ttp_id"] for x in usable}),
    }
    (out / "templates.json").write_text(json.dumps(catalogue, indent=2) + "\n")
    (out / "cores.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records)
    )

    print(f"source rows            {catalogue['source_rows']}")
    print(f"dropped '#ERROR!' rows {catalogue['dropped_error_rows']}")
    print(f"clause boilerplate     {len(boiler)} distinct clauses")
    print(f"prefix pretexts        {len(prefixes)} distinct phrases")
    print(f"cores recovered        {catalogue['cores_recovered']}")
    print(f"  with a request verb  {catalogue['cores_with_request_verb']}")
    print(
        f"  distinct             {catalogue['distinct_cores']}  over {catalogue['distinct_ttp']} TTPs"
    )
    print("\nwritten templates.json and cores.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sources", type=Path, default=Path(__file__).resolve().parent / "sources")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args()
    run(a.sources, a.out)
