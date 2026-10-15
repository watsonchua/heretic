# Additional source search

Only original source prompts are eligible for the **harmful** side, and no locally generated
formatting variants are included. The v0 sources below are combined by `build.py`.

The search goal has changed. v0 drew 90% of its harmful prompts from CyberSecEval MITRE and the
two classes separate on template style at 91% accuracy; see [METHODOLOGY.md](METHODOLOGY.md).
[PLAN.md](PLAN.md) Phase 2 therefore sets a hard cap of 25% per source per class and needs
roughly 700-800 further harmful cyber prompts across at least four sources.

| Source | Verified evidence | Suitability |
|---|---|---|
| [CyberSecEval MITRE](https://huggingface.co/datasets/purpcode/CyberSecEval-MITRE) | Downloaded pinned revision `70213a497f4e7d5ee5b0089eac5b8d3244eba2af`: 1,000 rows, 981 distinct normalized `base_prompt` values, 269 technique mappings, 10 categories | Strong candidate for harmful set. Use one original base prompt per duplicate group; do not count mutated versions as independent. Needs intent and semantic overlap review, including dual-use requests. |
| [CyberSecEval false-refusal](https://huggingface.co/datasets/purpcode/CyberSecEval-FRR) | Downloaded pinned revision `73294ca390f12ebb075d763550b942875dee5fc2`: 750 distinct normalized prompts, all `is_malicious=False` | Strong candidate for harmless set. Source-generated benchmark prompts; uniqueness is not proof of independent behaviors. Needs semantic and intent review. |
| [Fenrir v2.1](https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1) | Metadata/card describes 99,870 defensive instruction triples, Apache-2.0, OWASP/MITRE/authentication/cloud/cryptography coverage. Revision `fd7967ddda760281a2f01f4367f7b78bd128f3ec` | Candidate for remaining harmless prompts. Rows have not yet been downloaded or audited; publisher quality claims are unverified. |
| [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) | Human-labeled prompt harmfulness; 10K–100K size category; ODC-BY; gated access | Candidate for harmful and harmless prompts after cyber filtering. No usable count verified. |
| [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | 330K QA pairs, human safety labels, CC-BY-NC-4.0 | Supplemental candidate, but labels apply to QA pairs/responses and cannot directly determine whether a prompt is harmful. Requires prompt-level classification. |

The initial requested sources remain included: StrongREJECT (25 selected cyber seeds), HarmBench (67 cyber-category candidates), ClearHarm (12 selected cyber candidates). ClearHarm's rep40 configuration repeats 179 prompts 40 times; it cannot supply independent examples. HarmBench includes contextual prompts and dual-use cases needing review.

Excluded as ground truth: CyberNative/Code_Vulnerability_Security_DPO. Its October 2026 correction explicitly states that chosen/rejected preference labels are not verified secure/insecure classifications.

## Phase 2 candidates (all unverified)

None of these has a verified row count or revision hash yet. Do not take dataset card claims on
trust; record a measured count and a pinned revision before use, or discard the source.

| Candidate | Intended side | Why |
|---|---|---|
| CySecBench | harmful | cyber-specific jailbreak prompts, categorised |
| JailbreakBench JBB-Behaviors | harmful | has a Malware/Hacking category |
| AdvBench `harmful_behaviors` | harmful | cyber subset |
| SORRY-Bench | harmful | fine-grained categories including malware |
| MaliciousInstruct | harmful | small, hacking category |
| XSTest | harmless control | built for safe prompts that superficially look unsafe |
| OR-Bench | harmless control | over-refusal prompts, categorised |
| SecQA / Fenrir-style corpora | harmless control | ordinary cyber knowledge questions |

`source_search.json` contains measured CyberSecEval counts for v0, whose exports hold 1,085
harmful and 947 harmless source prompts.

The CLM System One classifier replaces the regex cyber filter in `build.py` for v1 source
screening, and supplies the harm and dual-use labels. It must be calibrated against 100
hand-labelled items before any gate depends on it; see [PLAN.md](PLAN.md) section 3. No System
One call has been made yet, and neither the base URL nor `CLM_API_KEY` is recorded in this
workspace.

Semantic review remains necessary before claiming independently validated behaviors.
