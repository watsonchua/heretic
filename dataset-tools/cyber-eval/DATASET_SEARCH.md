# Source inventory

Revisions pinned in `download_sources.py`. Counts are measured, not taken from dataset cards.
Selection rules and the screening method are in [METHODOLOGY.md](METHODOLOGY.md).

## In use

| Source | Revision | Measured | Licence | Role |
|---|---|---|---|---|
| [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) | `d29c47f41c8b` | 88,458 rows; 4,696 harmful and 587 unharmful cyber, non-adversarial | ODC-BY, gated | harmful + control |
| [CyberSecEval MITRE](https://huggingface.co/datasets/purpcode/CyberSecEval-MITRE) | `70213a497f4e` | 1,000 rows, 981 distinct prompts, 185 TTPs, 10 categories; 220 behaviours after collapsing | — | harmful |
| [AdvBench](https://huggingface.co/datasets/walledai/AdvBench) | `9d4730540082` | 520 rows, 165 cyber | MIT, gated | harmful |
| [HarmBench](https://github.com/centerforaisafety/HarmBench) | `8e1604d1171f` | 67 cybercrime_intrusion behaviours | MIT | harmful |
| [SORRY-Bench](https://huggingface.co/datasets/sorry-bench/sorry-bench-202503) | `612a4e1f45db` | 440 rows (44 categories x 10), 47 cyber | other, gated | harmful |
| [ClearHarm](https://huggingface.co/datasets/AlignmentResearch/ClearHarm) | `79464fb6b3c2` | 179 harmful rows, all distinct; 30 cyber | — | harmful |
| [MaliciousInstruct](https://huggingface.co/datasets/walledai/MaliciousInstruct) | `4669349f48fe` | 100 rows, 28 cyber | CC-BY-SA-4.0 | harmful |
| [StrongREJECT](https://github.com/alexandrasouly/strongreject) | `f7cad6c17e62` | 25 cyber seeds | MIT | harmful |
| [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | `886acc352a31` | 100 harmful + 100 index-aligned benign; 23 cyber, 10 Malware/Hacking pairs | MIT | harmful + few-shot exemplars |
| [OR-Bench hard-1k](https://huggingface.co/datasets/bench-llm/or-bench) | `e36d8b80e818` | 1,319 rows, 243 cyber | CC-BY-4.0 | control pool |

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
