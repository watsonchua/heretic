# Source inventory

Revisions pinned in `download_sources.py`. Counts are measured, not taken from dataset cards.
Selection rules and the screening method are in [METHODOLOGY.md](METHODOLOGY.md).

## In the harmful pool (976)

`drawn` is what `build_pool.py` selected; `of` is the candidate rows it screened. Harm label
tiers are explained in [METHODOLOGY.md](METHODOLOGY.md) section 3.

| Source | Revision | Drawn | of | Share | Harm label | Licence |
|---|---|---:|---:|---:|---|---|
| [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) | `d29c47f41c8b` | 400 | 1,200 | 41.0% | upstream_human | ODC-BY, gated |
| [AdvBench](https://huggingface.co/datasets/walledai/AdvBench) | `9d4730540082` | 236 | 520 | 24.2% | benchmark_membership | MIT, gated |
| [CyberSecEval MITRE](https://huggingface.co/datasets/purpcode/CyberSecEval-MITRE) | `70213a497f4e` | 111 | 220 | 11.4% | classifier | — |
| [HarmBench](https://github.com/centerforaisafety/HarmBench) | `8e1604d1171f` | 59 | 67 | 6.0% | benchmark_membership | MIT |
| [StrongREJECT](https://github.com/alexandrasouly/strongreject) | `f7cad6c17e62` | 56 | 313 | 5.7% | benchmark_membership | MIT |
| [SORRY-Bench](https://huggingface.co/datasets/sorry-bench/sorry-bench-202503) | `612a4e1f45db` | 33 | 440 | 3.4% | benchmark_membership | other, gated |
| [ClearHarm](https://huggingface.co/datasets/AlignmentResearch/ClearHarm) | `79464fb6b3c2` | 30 | 179 | 3.1% | benchmark_membership | — |
| [MaliciousInstruct](https://huggingface.co/datasets/walledai/MaliciousInstruct) | `4669349f48fe` | 28 | 100 | 2.9% | benchmark_membership | CC-BY-SA-4.0 |
| [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | `886acc352a31` | 23 | 100 | 2.4% | benchmark_membership | MIT |

JBB also supplies the **10 human-written Malware/Hacking pairs** used as few-shot exemplars for
twin generation and as a validation set for gate G6.

MITRE's 1,000 rows reduce to 220 distinct behaviours over 154 TTPs after de-templating and
clustering, then to 111 after a harm filter. Its Recon and Discovery prompts go to the control
pool rather than the harmful pool; see METHODOLOGY.md section 3.

## In the control pool (288)

| Source | Drawn | of | Why |
|---|---:|---:|---|
| [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) unharmful | 137 | 607 | human `unharmful` label, cyber, non-adversarial |
| [OR-Bench hard-1k](https://huggingface.co/datasets/bench-llm/or-bench) | 130 | 196 | built from prompts that look harmful but are not |
| CyberSecEval MITRE Recon/Discovery | 21 | — | enumeration and host discovery are ordinary administration |

OR-Bench is kept **without** a harm filter. Its prompts are designed to read as harmful, which
is what makes them over-refusal probes; screening them on a harm score discarded 87% of them
(130 down to 16) before that was corrected.

ClearHarm's `rep40` configuration repeats its 179 prompts 40 times and cannot supply independent
examples; only the base configuration is used. HarmBench includes contextual prompts needing
review.

## Excluded

| Source | Reason |
|---|---|
| [XSTest](https://huggingface.co/datasets/walledai/XSTest) | 450 rows, 2 cyber, 1 of them safe. Built for safe-looking-unsafe prompts generally, but has almost nothing in this domain. The public `natolambert/xstest-v2-copy` mirror also has an entirely null `label` column. |
| [CyberNative Code_Vulnerability_Security_DPO](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO) | Its own correction states the chosen/rejected preference labels are not verified secure/insecure classifications. |
| [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | Labels apply to QA pairs rather than prompts, so they cannot determine prompt harmfulness directly. CC-BY-NC-4.0. |
| [CyberSecEval false-refusal](https://huggingface.co/datasets/purpcode/CyberSecEval-FRR) | 750 prompts across four templates covering 740 of them. Too templated to serve as a harmless side; see METHODOLOGY.md section 1. |
| CySecBench | Does not resolve on the Hub. |

## Not yet audited

| Candidate | Why it might help |
|---|---|
| [Fenrir v2.1](https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1) | Card describes 99,870 defensive instruction triples, Apache-2.0. Rows not downloaded; publisher claims unverified. |
| SecQA | Ordinary cyber knowledge questions for the control pool. CC-BY-NC-SA-4.0, so non-commercial only. |

## Gated access

WildGuardMix, AdvBench, SORRY-Bench and XSTest require accepting their terms on the Hub before
`download_sources.py` can fetch them. Without that the downloader reports
`Access denied. This repository requires approval.`

`source_search.json` holds the measured CyberSecEval counts.
