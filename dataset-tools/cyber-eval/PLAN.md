# Remaining work

The approach and its measurements are in [METHODOLOGY.md](METHODOLOGY.md). This file tracks
what is not yet done.

## Blocking

### Calibration — needs a human
`calibration/to_label.jsonl` holds 59 items awaiting blind hand-labelling: 39 where System One
and glm-5.2 disagree, plus a 20-item random audit of the cases where they agree.

Label all four fields on every item. Values: `hand_is_cyber`, `hand_refuse`, `hand_pretext`
true/false; `hand_harm` 0 (clearly benign), 1 (dual-use), 2 (clearly harmful).

Priority if effort is limited: `hand_harm` first — it is the sole input to gates G3 and G5 and
the question the two models split on 28% of the time. Then `hand_refuse`, then `hand_is_cyber`.
`hand_pretext` last; nothing currently consumes it.

The 20 audit items are not optional. They are the only unbiased estimate of accuracy; the
disagreement rows are enriched for hard cases and cannot stand in for them.

Then `python calibrate.py score`, and record the resulting thresholds in `manifest.json`.
Until this lands, every System One threshold in the gates is a guess.

## Open problems

### Realised axis drifts from assigned
**47%** over 204 pairs. Assignment made the *assigned* distribution perfect and changed nothing
else. G7 passes at 0.3824 only because the drift spreads across categories rather than
collapsing into one, and it sits two points from its 0.40 limit.

Fix: enforce the realised axis during candidate selection, the same way G3 was moved into the
generation loop. Generate candidates, judge the realised axis, reject the ones that drifted.
This turns axis assignment from a hint into a constraint.

### G6 is mis-specified
Re-specify before using. Human-written JBB pairs score median 0.726 against our 0.701, and only
6 of 10 gold pairs clear the 0.7 threshold, so "90% above 0.7" is unreachable by hand-made
pairing. Options: reword the match question, set the threshold from the gold distribution, or
drop G6 in favour of G2's cosine, which already separates cleanly at 0.86. Do not simply lower
the number until the corpus passes.

Axis drift is not the cause: kept-axis pairs pass at 55%, drifted at 44%, an 11-point gap
against a 40-point shortfall.

### G5 harmful-side harm at 0.51
A real shortfall, not a threshold artefact. Two compounding causes: MITRE is dual-use at median
harm 1.00 (against 1.89-1.90 for ClearHarm, JBB-Behaviors and MaliciousInstruct), and generation
dropped its harshest 7%. Remedy is harsher sources — WildGuardMix and AdvBench — rather than
adjusting the cut, though 1.3 is itself uncalibrated until the hand labels land.

### g1_full_text at 0.777 against a 0.75 target
Narrowly failing, and the target was set by guess. This measures semantic separability, which
should not reach chance: words like *exfiltrate* carry real signal. Decide whether 0.75 is
defensible or whether the right target is "materially below the unpaired baseline of 0.9027",
which 0.777 satisfies. Resolve by argument, not by moving it after seeing the result.

## Not yet built

- **Harvest the screened sources into the harmful pool.** WildGuardMix (4,696 cyber harmful,
  non-adversarial), AdvBench (165), SORRY-Bench (47) are downloaded and sized but not yet drawn
  into a build, and the 25% per-source cap is not yet asserted anywhere in code.
- **The ~300 unpaired control pool**, from WildGuardMix's 587 unharmful cyber prompts and
  OR-Bench's 243.
- **Twins for the non-MITRE harmful sources.** `twins.py` currently runs over the MITRE
  selection only.
- **Final exports**: `harmful.jsonl` / `harmless.jsonl` in matched order so row *i* pairs with
  row *i*, plus `harmless_control.jsonl`, `harmless_borderline.jsonl`, `dual_use.jsonl`.
- **Splits** grouped by `pair_id` and topic cluster, so a twin or near-duplicate never straddles
  train and test.
- **`manifest.json`**: counts, per-axis distribution, calibration agreement rates and chosen
  thresholds, all seven gate results, and the models used.

## Decided, for the record

- **Target size**: 800-1,000 pairs plus ~300 unpaired control prompts.
- **Pretexts are stripped**, not reapplied to either class. The corpus measures refusal on plain
  requests; jailbreak robustness would need its own corpus.
- **Generator is `kimi-k3`**, retained after re-testing. It is rate-limited to roughly 2 pairs
  per minute but refuses nothing and drops nothing. `gemini-3.5-flash-global` is the fallback
  the data supports.
- **Only the harmless side is generated.** The harmful side stays source-derived.
