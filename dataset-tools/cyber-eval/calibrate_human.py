"""Calibrate the System One harm question against upstream human labels.

`calibrate.py` measures agreement on a hand-labelled sample of about 60 items. This script does
the same job with roughly ten times the data and no human effort, by exploiting a source that
already carries per-prompt human judgements: WildGuardMix's `prompt_harm_label`.

Every prompt screened during `build_pool.py` is cached with its System One harm score. Where
that prompt also appears in WildGuardMix, its human label is known, so the two can be compared
directly and a threshold chosen from data rather than guessed.

Scope, and what this does not establish:
  - It calibrates the harmful/unharmful boundary only. The three-level scale's middle band
    (dual-use) has no human counterpart here and stays uncalibrated.
  - It covers the `harm` question alone. `refuse`, `is_cyber` and `match` still need
    hand labels via calibrate.py.
  - WildGuardMix's labellers applied their own standard to a general-purpose corpus, not to
    cyber dual-use specifically.
  - The sample is the cyber subset that passed screening, so a selection effect remains.

  python calibrate_human.py
"""
import json,statistics
from pathlib import Path
import pyarrow.parquet as pq

R=Path(__file__).resolve().parent
CACHE=R/'pool.cache.json'
OUT=R/'calibration/human_harm.json'
MIN_AGREEMENT=0.85

def human_labels():
 rows=[]
 for split in ('train','test'):
  p=R/f'sources/wildguardmix/{split}/wildguard_{split}.parquet'
  if p.exists():rows+=pq.read_table(p).to_pylist()
 return {r['prompt']:r['prompt_harm_label'] for r in rows
         if r.get('prompt') and r.get('prompt_harm_label') in ('harmful','unharmful')}

def run():
 if not CACHE.exists():raise SystemExit('no pool.cache.json; run build_pool.py first')
 cache=json.loads(CACHE.read_text());hum=human_labels()
 pairs=[(cache[t]['harm'],hum[t]) for t in cache if t in hum]
 if not pairs:raise SystemExit('no overlap between the screening cache and WildGuardMix')
 H=[s for s,l in pairs if l=='harmful'];U=[s for s,l in pairs if l=='unharmful']
 if not U or not H:
  raise SystemExit(f'need both classes; have {len(H)} harmful and {len(U)} unharmful. '
                   'A one-class sample makes the threshold scan degenerate: with no negatives '
                   'a lower cut always scores better, all the way to accepting everything.')
 def metrics(thr):
  rec=sum(1 for s in H if s>=thr)/len(H);spec=sum(1 for s in U if s<thr)/len(U)
  return (sum(1 for s in H if s>=thr)+sum(1 for s in U if s<thr))/len(pairs),rec,spec
 scan=[(round(i/20,2),)+metrics(i/20) for i in range(1,41)]
 best=max(scan,key=lambda r:r[1])
 report={'n':len(pairs),'n_harmful':len(H),'n_unharmful':len(U),
  'base_rate_harmful':round(len(H)/len(pairs),4),
  'median_harm_human_harmful':round(statistics.median(H),3),
  'median_harm_human_unharmful':round(statistics.median(U),3),
  'best_threshold':best[0],'accuracy_at_best':round(best[1],4),
  'scan':{str(t):{'accuracy':round(a,4),'recall':round(r,4),'specificity':round(s,4)}
          for t,a,r,s in scan if t in (0.7,1.0,1.3,1.4,1.5,best[0])},
  'passes':best[1]>=MIN_AGREEMENT,'min_agreement':MIN_AGREEMENT,
  'scope':'harmful/unharmful boundary only; dual-use band uncalibrated; harm question only'}
 OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(report,indent=2)+'\n')
 print(f"n={len(pairs)}  ({len(H)} human-harmful, {len(U)} human-unharmful, "
       f"base rate {len(H)/len(pairs):.1%})")
 print(f"  median harm: human-harmful {statistics.median(H):.2f}, "
       f"human-unharmful {statistics.median(U):.2f}")
 print(f"\n  {'threshold':<12}{'accuracy':>10}{'recall':>10}{'specificity':>14}")
 for t,a,r,s in scan:
  if t not in (0.7,1.0,1.3,1.4,1.5,best[0]):continue
  tag='  <- best' if t==best[0] else ''
  print(f"  {t:<12.2f}{a:>10.1%}{r:>10.1%}{s:>14.1%}{tag}")
 print(f"\n{'PASS' if report['passes'] else 'FAIL'} against the {MIN_AGREEMENT:.0%} minimum")
 print(f"written {OUT.relative_to(R)}")

if __name__=='__main__':run()
