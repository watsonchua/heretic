"""Write the final exports: matched-order pairs, control pool, splits and a manifest.

Two properties the consumer depends on:

  Matched order. harmful.jsonl row i and harmless.jsonl row i are the two halves of one pair.
  Anything computing a per-pair difference relies on this, and nothing in the file format
  enforces it, so it is asserted here.

  Grouped splits. A twin must never sit opposite its partner across the train/test boundary,
  and neither must a near-duplicate. Pairs are clustered on the harmful side and whole clusters
  are assigned to one split, so leakage cannot occur through either route.

Twins a careful model would refuse are routed to harmless_borderline rather than dropped: they
are poor material for a refusal direction but good over-refusal probes. Pairs whose harmful
side lands in the dual-use band go to dual_use for the same reason, kept and labelled rather
than forced into a class.

  python export.py --test-fraction 0.2
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import llm
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import systemone
from common import read_jsonl
from sklearn.cluster import AgglomerativeClustering

R = Path(__file__).resolve().parent
OUT = R / "export"
DUAL_LO, DUAL_HI = 0.7, 1.3  # calibrated: see calibration/human_harm.json
REFUSE_MAX = 0.3


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    pq.write_table(pa.Table.from_pylist(rows), path.with_suffix(".parquet"))


def cluster_split(texts, test_fraction, threshold=0.85, seed=0):
    """Assign whole near-duplicate clusters to train or test, never splitting one."""
    X = np.array(llm.embed(texts, progress=True), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1 - threshold, metric="cosine", linkage="average"
    ).fit_predict(X)
    # Deterministic assignment: hash the cluster id so a rebuild reproduces the same split.
    test = {
        c
        for c in set(labels)
        if int(hashlib.sha256(f"{seed}:{c}".encode()).hexdigest()[:8], 16) % 1000
        < test_fraction * 1000
    }
    return ["test" if c in test else "train" for c in labels], len(set(labels))


def run(a):
    pairs = read_jsonl(R / "pairs.jsonl")
    control = read_jsonl(R / "control_pool.jsonl")
    pool = {x["id"]: x for x in read_jsonl(R / "harmful_pool.jsonl")}
    print(f"{len(pairs)} pairs, {len(control)} control prompts")

    url, key = systemone.credentials()
    cpath = R / "export.cache.json"
    cache = json.loads(cpath.read_text()) if cpath.exists() else {}
    core, borderline, dual = [], [], []
    for i, p in enumerate(pairs, 1):
        src = pool.get(p["pair_id"], {})
        h = src.get("harm_score")
        if p["harmless"] not in cache:
            cache[p["harmless"]] = systemone.ask(p["harmless"], ["refuse"], url, key)["refuse"][
                "noul"
            ]
        r = cache[p["harmless"]]
        rec = {
            **p,
            "harm_label_source": src.get("harm_label_source"),
            "harmful_harm_score": h,
            "twin_p_refuse": round(r, 4),
            "source": src.get("source"),
        }
        if h is not None and DUAL_LO <= h <= DUAL_HI:
            dual.append(rec)
        elif r >= REFUSE_MAX:
            borderline.append(rec)
        else:
            core.append(rec)
        if i % 100 == 0 or i == len(pairs):
            cpath.write_text(json.dumps(cache))
            print(f"  routed {i}/{len(pairs)}")

    print(f"\ncore {len(core)}  borderline {len(borderline)}  dual_use {len(dual)}")
    splits, nclusters = cluster_split([x["harmful"] for x in core], a.test_fraction)
    for x, s in zip(core, splits, strict=False):
        x["split"] = s
    print(
        f"{nclusters} clusters -> "
        + "  ".join(f"{k} {v}" for k, v in Counter(splits).most_common())
    )

    OUT.mkdir(exist_ok=True)
    write(OUT / "harmful.jsonl", [{"text": x["harmful"]} for x in core])
    write(OUT / "harmless.jsonl", [{"text": x["harmless"]} for x in core])
    for s in ("train", "test"):
        sel = [x for x in core if x["split"] == s]
        write(OUT / f"harmful/{s}.jsonl", [{"text": x["harmful"]} for x in sel])
        write(OUT / f"harmless/{s}.jsonl", [{"text": x["harmless"]} for x in sel])
    write(OUT / "harmless_control.jsonl", [{"text": x["text"]} for x in control])
    write(OUT / "harmless_borderline.jsonl", [{"text": x["harmless"]} for x in borderline])
    write(OUT / "dual_use.jsonl", [{"text": x["harmful"]} for x in dual])
    (OUT / "pairs.metadata.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in core + borderline + dual)
    )

    # Matched order is the one property the file format cannot express. Assert it.
    hf = [r["text"] for r in read_jsonl(OUT / "harmful.jsonl")]
    hl = [r["text"] for r in read_jsonl(OUT / "harmless.jsonl")]
    assert len(hf) == len(hl) == len(core)
    for i, x in enumerate(core):
        assert hf[i] == x["harmful"] and hl[i] == x["harmless"], f"order broken at {i}"
    groups = {}
    for x in core:
        groups.setdefault(x["pair_id"], x["split"])
    assert all(groups[x["pair_id"]] == x["split"] for x in core), "a pair straddles the split"

    man = {
        "pairs_total": len(pairs),
        "core": len(core),
        "borderline": len(borderline),
        "dual_use": len(dual),
        "control": len(control),
        "splits": dict(Counter(x["split"] for x in core)),
        "clusters": nclusters,
        "generation_model": llm.CHAT_MODEL,
        "decision_model": systemone.MODEL,
        "judge_model": llm.JUDGE_MODEL,
        "embedding_model": llm.EMBED_MODEL,
        "harm_label_source": dict(Counter(x.get("harm_label_source") for x in core)),
        "source_counts": dict(Counter(x.get("source") for x in core)),
        "axis": dict(Counter(x["axis"] for x in core)),
        "thresholds": {"dual_use_band": [DUAL_LO, DUAL_HI], "twin_refuse_max": REFUSE_MAX},
        "augmentation": True,
        "harmful_side_generated": False,
    }
    man["sha256"] = {
        str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(OUT.rglob("*"))
        if p.is_file()
    }
    (R / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print("\nwritten export/ and manifest.json")
    for k in ("core", "borderline", "dual_use", "control", "splits"):
        print(f"  {k}: {man[k]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--test-fraction", type=float, default=0.2)
    run(ap.parse_args())
