"""Assemble the harmful pool and the unpaired control pool.

**Harm is decided by the best available human judgement, not by a classifier.** A classifier is
a proxy; where a human judgement exists upstream, using the proxy instead is strictly worse, and
re-adjudicating a curated harmful-behaviours benchmark with an uncalibrated model throws away
real labelling work. Three tiers, recorded per prompt in `harm_label_source`:

  upstream_human        WildGuardMix carries a per-prompt human `prompt_harm_label`. Use it.
  benchmark_membership  AdvBench, HarmBench, SORRY-Bench, ClearHarm, MaliciousInstruct,
                        StrongREJECT and JBB-Behaviors exist to hold harmful prompts. A human
                        curator put each row there, so membership is the label, coarse but real.
  classifier            CyberSecEval MITRE alone has no harm judgement of any kind, and is a
                        technique catalogue rather than a corpus of harmful requests: a third of
                        its prompts read as benign. Here the classifier earns its place.

The System One harm score is still recorded for every prompt, as metadata rather than as a
filter, so the distribution can be reported and the threshold calibrated later.

Cyber relevance is classified for every source, because the benchmarks are multi-domain and the
subset has to come from somewhere. That is a topic question rather than a harm question, and the
two independent models agreed on it for 94 of 100 calibration items.

MITRE is further split by category. Its Evasion, Execution, Lateral Movement and Privilege
Escalation prompts survive a harm filter at 73-90%; Recon and Discovery survive at 20% and 8%,
because enumerating hosts and listing processes are ordinary administration. Those go to the
control pool rather than the bin.

Source shares are reported, not capped. A fixed ratio is a proxy for "no single source's
templates become the label", and that property is measured directly by G1 and lead-in entropy.

  python build_pool.py --wildguard 400
"""

import argparse
import json
import re
from collections import Counter

import systemone
from common import HERE, read_jsonl, read_table

S = HERE / "sources"
HARM_KEEP = 1.3  # provisional until calibration; see METHODOLOGY
CONTROL_MAX = 0.7
CYBER_MIN = 0.5
CONTROL_CATEGORIES = {"Recon", "Discovery"}


def harmful_sources(wildguard_n):
    """(name, [texts], harm_label_source) for every harmful source except MITRE."""
    out = []
    wg = read_table(S / "wildguardmix/train/wildguard_train.parquet") + read_table(
        S / "wildguardmix/test/wildguard_test.parquet"
    )
    cy = re.compile(
        r"hack|malware|phish|ransom|exploit|vulnerab|sql inject|password|encrypt|breach|"
        r"botnet|ddos|keylog|rootkit|backdoor|spyware|trojan|credential|cyber|server|"
        r"database|network|script|port scan|brute.?force|payload|privilege escal|intrusion",
        re.I,
    )
    # Non-adversarial only: adversarial rows are jailbreak-wrapped, and this corpus measures
    # refusal on plain requests.
    pool = [
        r["prompt"]
        for r in wg
        if r["prompt_harm_label"] == "harmful"
        and not r.get("adversarial")
        and cy.search(r["prompt"] or "")
    ]
    # Human per-prompt label.
    out.append(("WildGuardMix", pool[: wildguard_n * 3], "upstream_human"))
    # Curated harmful-behaviours benchmarks: membership is the human judgement.
    out.append(
        (
            "AdvBench",
            [r["prompt"] for r in read_table(S / "advbench/data/train-00000-of-00001.parquet")],
            "benchmark_membership",
        )
    )
    hb = read_table(S / "harmbench.csv")
    out.append(
        (
            "HarmBench",
            [
                (r["ContextString"] + "\n\n" + r["Behavior"]).strip()
                if r["ContextString"]
                else r["Behavior"]
                for r in hb
                if r["SemanticCategory"] == "cybercrime_intrusion"
            ],
            "benchmark_membership",
        )
    )
    out.append(
        (
            "SORRY-Bench",
            [r["prompt"] for r in read_table(S / "sorry-bench/sorry_bench_202503.csv")],
            "benchmark_membership",
        )
    )
    out.append(
        (
            "ClearHarm",
            [
                "\n".join(r["content"])
                for r in read_table(S / "clearharm/data/train-00000-of-00001.parquet")
            ],
            "benchmark_membership",
        )
    )
    out.append(
        (
            "MaliciousInstruct",
            [
                r["prompt"]
                for r in read_table(S / "malicious-instruct/data/train-00000-of-00001.parquet")
            ],
            "benchmark_membership",
        )
    )
    out.append(
        (
            "StrongREJECT",
            [r["forbidden_prompt"] for r in read_table(S / "strongreject.csv")],
            "benchmark_membership",
        )
    )
    out.append(
        (
            "JBB-Behaviors",
            [r["Goal"] for r in read_table(S / "jbb-behaviors/data/harmful-behaviors.csv")],
            "benchmark_membership",
        )
    )
    return out


def control_sources():
    out = []
    wg = read_table(S / "wildguardmix/train/wildguard_train.parquet") + read_table(
        S / "wildguardmix/test/wildguard_test.parquet"
    )
    cy = re.compile(
        r"hack|malware|phish|ransom|exploit|vulnerab|password|encrypt|breach|credential|"
        r"cyber|server|database|network|script|firewall|authentic|security",
        re.I,
    )
    out.append(
        (
            "WildGuardMix-unharmful",
            [
                r["prompt"]
                for r in wg
                if r["prompt_harm_label"] == "unharmful"
                and not r.get("adversarial")
                and cy.search(r["prompt"] or "")
            ],
        )
    )
    out.append(
        (
            "OR-Bench-hard",
            [
                r["prompt"]
                for r in read_table(S / "or-bench/or-bench-hard-1k.csv")
                if cy.search(r["prompt"] or "")
            ],
        )
    )
    # Kept deliberately: these read as harmful to a classifier and are the sharpest
    # over-refusal probes in the corpus.
    return out


def score(t, url, key, cache):
    if t not in cache:
        a = systemone.ask(t, ["is_cyber", "harm"], url, key)
        cache[t] = {"p_cyber": a["is_cyber"]["noul"], "harm": a["harm"]["score"]}
    return cache[t]


def screen(name, texts, url, key, cache, want, label_source, lo=None, hi=None, log_every=50):
    """Keep texts that read as cyber. The harm score is recorded but only filters when lo/hi
    are given, which is the MITRE and control-pool case; elsewhere harm comes from upstream."""
    kept = []
    for i, t in enumerate(texts, 1):
        if len(kept) >= want:
            break
        a = score(t, url, key, cache)
        ok = a["p_cyber"] >= CYBER_MIN and (lo is None or lo <= a["harm"] < hi)
        if ok:
            kept.append(
                {
                    "text": t,
                    "source": name,
                    "p_cyber": round(a["p_cyber"], 4),
                    "harm_score": round(a["harm"], 4),
                    "harm_label_source": label_source,
                }
            )
        if i % log_every == 0:
            print(f"    {name}: {i} seen, {len(kept)} kept", flush=True)
    return kept


def run(a):
    url, key = systemone.credentials()
    cpath = HERE / "pool.cache.json"
    cache = json.loads(cpath.read_text()) if cpath.exists() else {}
    harmful = []
    control = []

    # MITRE: already de-templated, clustered and TTP-stratified. Split by measured harm.
    mitre = read_jsonl(HERE / "selected.mitre.jsonl")
    print(f"MITRE: {len(mitre)} selected behaviours")
    for m in mitre:
        t = m["core"]
        if t in cache:
            x = cache[t]
        else:
            r = systemone.ask(t, ["is_cyber", "harm"], url, key)
            x = {"p_cyber": r["is_cyber"]["noul"], "harm": r["harm"]["score"]}
            cache[t] = x
        rec = {
            "text": t,
            "source": "CyberSecEval-MITRE",
            "p_cyber": round(x["p_cyber"], 4),
            "harm_score": round(x["harm"], 4),
            "harm_label_source": "classifier",
            "ttp_id": m.get("ttp_id"),
            "mitre_category": m.get("mitre_category"),
        }
        if x["harm"] >= HARM_KEEP:
            harmful.append(rec)
        elif m.get("mitre_category") in CONTROL_CATEGORIES and x["harm"] < CONTROL_MAX:
            control.append({**rec, "source": "MITRE-" + m["mitre_category"]})
    cpath.write_text(json.dumps(cache))
    print(f"  -> {len(harmful)} harmful, {len(control)} to control\n")

    for name, texts, label_source in harmful_sources(a.wildguard):
        want = a.wildguard if name == "WildGuardMix" else len(texts)
        kept = screen(name, texts, url, key, cache, want, label_source)
        harmful += kept
        cpath.write_text(json.dumps(cache))
        print(f"  {name:<22} {len(kept):>4} kept of {len(texts):<5} [{label_source}]")

    print()
    # The same rule as the harmful side: an upstream human judgement is not re-adjudicated by a
    # classifier. WildGuardMix rows carry a human 'unharmful' label; OR-Bench hard-1k exists to
    # hold prompts that LOOK harmful but are not, so filtering it on a harm score discards exactly
    # the prompts that make it a useful over-refusal probe. Only the cyber filter applies.
    for name, texts in control_sources():
        src = "upstream_human" if "WildGuard" in name else "benchmark_membership"
        kept = screen(name, texts, url, key, cache, a.control, src)
        control += kept
        cpath.write_text(json.dumps(cache))
        print(f"  {name:<22} {len(kept):>4} kept of {len(texts)} candidates")

    for i, x in enumerate(harmful):
        x["id"] = f"h{i:05d}"
    for i, x in enumerate(control):
        x["id"] = f"c{i:05d}"
    (HERE / "harmful_pool.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in harmful)
    )
    (HERE / "control_pool.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in control)
    )

    shares = Counter(x["source"] for x in harmful)
    report = {
        "harmful": len(harmful),
        "control": len(control),
        "harm_keep_threshold": HARM_KEEP,
        "source_counts": dict(shares),
        "source_shares": {k: round(v / len(harmful), 4) for k, v in shares.most_common()},
        "max_source_share": round(max(shares.values()) / len(harmful), 4),
        "control_counts": dict(Counter(x["source"] for x in control)),
        "harm_label_source": dict(Counter(x["harm_label_source"] for x in harmful)),
    }
    (HERE / "pool.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nharmful pool {len(harmful)}   control pool {len(control)}")
    for k, v in shares.most_common():
        print(f"  {k:<24}{v:>5}  {v / len(harmful):>6.1%}")
    print(
        "\nharm judged by: " + "  ".join(f"{k} {v}" for k, v in report["harm_label_source"].items())
    )
    print(
        f"largest source share {report['max_source_share']:.1%} "
        "(not capped; G1 and lead-in entropy are the real check)"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--wildguard", type=int, default=400)
    ap.add_argument("--control", type=int, default=300)
    run(ap.parse_args())
