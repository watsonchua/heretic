"""Run the seven gates over a paired corpus and report pass/fail.

Nothing ships unless all seven pass. Thresholds are provisional until calibrate.py fixes them
against hand labels; the harm questions in particular had System One and glm-5.2 disagreeing on
28% of the calibration set, all of it in the dual-use band.

G1 is the only gate that uses no model, which makes it the only check that survives the
generator and the judge sharing a blind spot. Treat a corpus that passes G3-G6 but fails G1 as
failing.

  python gates.py --pairs pairs.jsonl
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import diagnostics as dg
import llm
import numpy as np
import systemone
from common import read_jsonl

R = Path(__file__).resolve().parent
# Three System One calls per pair, so a few hundred pairs is a long run. Cache every answer on
# disk keyed by (question set, text) so a failure or an interrupt costs only the current pair.
CACHE = R / "gates.cache.json"
T = {
    "g1_lead_in": 0.60,
    "g1_full_text": 0.75,
    "g1_length_ratio": 0.10,
    "g2_cosine_median": 0.45,
    "g2_top1": 0.80,
    "g3_twin_clear": 0.95,
    "g4_answerable": 0.90,
    "g5_harmful_harm": 0.90,
    "g6_match": 0.90,
    "g7_max_axis": 0.40,
}
# G5 asks whether the harmful side is really harmful. Most of the pool now carries a human
# judgement, so asking a classifier to re-adjudicate it is either second-guessing the humans or,
# for the prompts the classifier itself selected, circular. The gate therefore applies only to
# the `classifier` tier, and the two human tiers are reported as an independent cross-check:
# if benchmark_membership prompts score like upstream_human ones, treating membership as a
# label is sound; if they score far lower, that assumption needs revisiting.
G5_TIER = "classifier"
HARM_CLEAR, HARM_HARMFUL, REFUSE_CLEAR, MATCH_CLEAR = 0.7, 1.3, 0.3, 0.7
# Realised-axis keys, in the order the assigned axis numbers 1-4 map onto.
AXIS_KEYS = {"defensive": 1, "benign_action": 2, "benign_artefact": 3, "forensic": 4, "none": 0}


def run(a):
    pairs = read_jsonl(R / a.pairs)
    if a.limit:
        pairs = pairs[: a.limit]
    harmful = [p["harmful"] for p in pairs]
    harmless = [p["harmless"] for p in pairs]
    print(f"{len(pairs)} pairs\n")
    res = {}

    # --- G1, model-free -------------------------------------------------------
    texts = harmful + harmless
    labels = ["harmful"] * len(harmful) + ["harmless"] * len(harmless)
    res["g1_full_text"] = round(dg.cv_accuracy(texts, labels, ngram_range=(1, 2), min_df=2), 4)
    lead = [" ".join(t.split()[: dg.LEAD_WORDS]) for t in texts]
    res["g1_lead_in"] = round(
        dg.cv_accuracy(lead, labels, analyzer="word", ngram_range=(1, dg.LEAD_WORDS)), 4
    )
    import statistics

    mh = statistics.median([len(t.split()) for t in harmful])
    ml = statistics.median([len(t.split()) for t in harmless])
    res["g1_length_ratio"] = round(abs(mh - ml) / max(mh, ml), 4)
    print(f"G1 length medians: harmful {mh}, harmless {ml}")

    # --- G2, embeddings -------------------------------------------------------
    vecs = np.array(llm.embed(harmful + harmless, progress=True), dtype=np.float32)
    A = vecs[: len(harmful)]
    B = vecs[len(harmful) :]
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    S = A @ B.T
    res["g2_cosine_median"] = round(float(np.median(np.diag(S))), 4)
    res["g2_top1"] = round(float(np.mean(S.argmax(1) == np.arange(len(pairs)))), 4)

    # --- G3-G7, System One ----------------------------------------------------
    url, key = systemone.credentials()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    def ask(state, names):
        k = hashlib.sha256(("|".join(names) + "\x00" + state).encode()).hexdigest()
        if k not in cache:
            cache[k] = systemone.ask(state, names, url, key)
        return cache[k]

    twin_harm = []
    harm_harm = []
    twin_refuse = []
    match_scores = []
    axes = []
    for i, p in enumerate(pairs, 1):
        t = ask(p["harmless"], ["harm", "refuse"])
        h = ask(p["harmful"], ["harm"])
        m = ask(systemone.pair_state(p["harmful"], p["harmless"]), ["match", "realised_axis"])
        twin_harm.append(t["harm"]["score"])
        twin_refuse.append(t["refuse"]["noul"])
        harm_harm.append(h["harm"]["score"])
        match_scores.append(m["match"]["noul"])
        axes.append(m["realised_axis"]["choice"])
        if i % 25 == 0 or i == len(pairs):
            CACHE.write_text(json.dumps(cache))
            print(f"  gated {i}/{len(pairs)}", flush=True)
    CACHE.write_text(json.dumps(cache))
    res["g3_twin_clear"] = round(sum(1 for x in twin_harm if x < HARM_CLEAR) / len(pairs), 4)
    res["g4_answerable"] = round(sum(1 for x in twin_refuse if x < REFUSE_CLEAR) / len(pairs), 4)
    tiers = {}
    for p, h in zip(pairs, harm_harm, strict=False):
        tiers.setdefault(p.get("harm_label_source", "unknown"), []).append(h)
    judged = tiers.get(G5_TIER)
    # Fall back to the whole corpus when no tier information is present (older pair files).
    basis = judged if judged else harm_harm
    res["g5_harmful_harm"] = round(sum(1 for x in basis if x > HARM_HARMFUL) / len(basis), 4)
    res["g5_basis"] = f"{G5_TIER} tier, n={len(basis)}" if judged else f"all pairs, n={len(basis)}"
    if len(tiers) > 1:
        import statistics as _s

        print("\nharm score by label source (G5 scores only the classifier tier;")
        print("the human tiers are a cross-check on treating membership as a label)")
        for k in sorted(tiers):
            v = tiers[k]
            print(
                f"  {k:<22} n={len(v):<5} median {_s.median(v):.2f}  "
                f">1.3 {sum(1 for x in v if x > HARM_HARMFUL) / len(v):>6.1%}"
            )
    res["g6_match"] = round(sum(1 for x in match_scores if x >= MATCH_CLEAR) / len(pairs), 4)
    realised = Counter(axes)
    res["g7_max_axis"] = round(max(realised.values()) / len(pairs), 4)

    assigned = Counter(p["axis"] for p in pairs)
    drift = sum(1 for p, ax in zip(pairs, axes, strict=False) if AXIS_KEYS.get(ax) != p["axis"])
    print("\nassigned axes : " + "  ".join(f"{k}:{v}" for k, v in sorted(assigned.items())))
    print("realised axes : " + "  ".join(f"{k}:{v}" for k, v in realised.most_common()))
    print(f"axis drift    : {drift}/{len(pairs)} ({drift / len(pairs):.0%}) realised != assigned")
    if realised.get("none"):
        print(f"  WARNING: {realised['none']} pairs judged to not change the intent at all")

    lower = {"g1_lead_in", "g1_full_text", "g1_length_ratio", "g7_max_axis"}
    print("\ngates")
    failed = []
    for k, target in T.items():
        v = res[k]
        ok = v <= target if k in lower else v >= target
        if not ok:
            failed.append(k)
        print(
            f"  {k:<20} {v:>8.4f}   {'PASS' if ok else 'FAIL'} (target {'<=' if k in lower else '>='} {target})"
        )
    out = {
        "pairs": len(pairs),
        "results": res,
        "failed": failed,
        "assigned_axes": dict(assigned),
        "realised_axes": dict(realised),
        "axis_drift": drift,
        "thresholds": T,
        "provisional": True,
        "harm_by_label_source": {
            k: {
                "n": len(v),
                "median": round(sorted(v)[len(v) // 2], 3),
                "above_1.3": round(sum(1 for x in v if x > HARM_HARMFUL) / len(v), 4),
            }
            for k, v in tiers.items()
        },
    }
    (R / "gates.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"\n{len(failed)} of {len(T)} gates failing" + (f": {', '.join(failed)}" if failed else "")
    )
    print("thresholds are provisional until calibrate.py runs against hand labels")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pairs", default="pairs.jsonl")
    ap.add_argument("--limit", type=int)
    run(ap.parse_args())
