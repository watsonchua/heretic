"""Model-free corpus diagnostics: style separability, pairing coverage, duplication, balance.

Independent of System One and kimi-k3 by design. This is the only evidence in the workflow
that does not come from an LLM, so it is the check that catches a generator and a judge
agreeing on the same mistake. Run against every build and compare the JSON.

  python diagnostics.py --pairs pairs.jsonl
"""
import argparse,json,math,statistics
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize

# G1 targets from PLAN.md section 4. Provisional until the calibration run fixes them.
GATES={'g1_lead_in_accuracy':0.60,'g1_full_text_accuracy':0.75,'g1_length_ratio':0.10,
       'g2_pair_cosine_median':0.45,'g2_pair_is_top1':0.80,'g7_max_axis_share':0.40}
LEAD_WORDS=3

def load(path):
 return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def cv_accuracy(texts,labels,**kw):
 """5-fold CV accuracy of a linear bag-of-words classifier. 0.5 is chance."""
 if len(set(labels))<2:return None
 X=TfidfVectorizer(**kw).fit_transform(texts)
 return float(cross_val_score(LogisticRegression(max_iter=2000),X,labels,cv=5).mean())

def entropy(counts):
 """Shannon entropy normalized to [0,1] against a uniform distribution of the same support."""
 total=sum(counts.values())
 if total==0 or len(counts)<2:return 0.0
 h=-sum((c/total)*math.log2(c/total) for c in counts.values())
 return float(h/math.log2(len(counts)))

def neighbours(A,B):
 """Row-wise max cosine from each row of A to any row of B, plus the argmax."""
 S=normalize(A)@normalize(B).T
 S=S.toarray() if hasattr(S,'toarray') else S
 return S.max(1),S.argmax(1)

def analyse(rows,pairs):
 harmful=[r for r in rows if r['label']=='harmful']
 harmless=[r for r in rows if r['label']=='harmless']
 texts=[r['text'] for r in rows];labels=[r['label'] for r in rows]
 out={'counts':{'harmful':len(harmful),'harmless':len(harmless),'total':len(rows)}}

 out['source_share']={l:{s:round(c/max(1,len(g)),4) for s,c in Counter(r.get('source','?') for r in g).most_common()}
                      for l,g in (('harmful',harmful),('harmless',harmless))}
 out['max_source_share']=round(max((max(v.values()) for v in out['source_share'].values() if v),default=0),4)

 # Defect A: does prompt style alone predict the label?
 lead=[' '.join(t.split()[:LEAD_WORDS]) for t in texts]
 out['g1_full_text_accuracy']=round(cv_accuracy(texts,labels,ngram_range=(1,2),min_df=2) or 0,4)
 out['g1_lead_in_accuracy']=round(cv_accuracy(lead,labels,analyzer='word',ngram_range=(1,LEAD_WORDS)) or 0,4)

 lengths={l:[len(r['text'].split()) for r in g] for l,g in (('harmful',harmful),('harmless',harmless))}
 out['length']={l:{'median':statistics.median(v),'mean':round(statistics.mean(v),1)} for l,v in lengths.items() if v}
 med=[out['length'][l]['median'] for l in ('harmful','harmless') if l in out['length']]
 out['g1_length_ratio']=round(abs(med[0]-med[1])/max(med),4) if len(med)==2 else None

 out['lead_in_entropy']={l:round(entropy(Counter(' '.join(r['text'].split()[:5]) for r in g)),4)
                         for l,g in (('harmful',harmful),('harmless',harmless)) if g}

 # Defect B and the pairing gap, both measured in one TF-IDF space.
 if harmful and harmless:
  vec=TfidfVectorizer(stop_words='english',min_df=2).fit(texts)
  A=vec.transform([r['text'] for r in harmful]);B=vec.transform([r['text'] for r in harmless])
  cross,arg=neighbours(A,B)
  out['cross_label_coverage']={f'>={t}':round(float((cross>=t).mean()),4) for t in (0.3,0.4,0.5,0.6)}
  out['cross_label_median']=round(float(np.median(cross)),4)
  for name,M in (('harmful',A),('harmless',B)):
   S=normalize(M)@normalize(M).T;S=S.toarray();np.fill_diagonal(S,0)
   out.setdefault('internal_near_duplicates',{})[name]={
    f'>={t}':round(float((S.max(1)>=t).mean()),4) for t in (0.7,0.8)}

  # G2 only applies once the corpus is paired.
  if pairs:
   index={r['id']:i for i,r in enumerate(harmless) if 'id' in r}
   want=[index.get(p.get('pair_id')) for p in pairs]
   rank=[(arg[i]==w) for i,w in enumerate(want) if w is not None]
   same=[float(normalize(A)[i]@normalize(B)[w].T) for i,w in enumerate(want) if w is not None]
   if same:
    out['g2_pair_cosine_median']=round(float(np.median(same)),4)
    out['g2_pair_is_top1']=round(float(np.mean(rank)),4)

 if pairs:
  axes=Counter(p.get('axis') for p in pairs)
  out['axis_distribution']={str(k):round(v/len(pairs),4) for k,v in axes.most_common()}
  out['g7_max_axis_share']=round(max(axes.values())/len(pairs),4)

 ttp=Counter(r['ttp'] for r in rows if r.get('ttp'))
 if ttp:out['ttp']={'distinct':len(ttp),'max_per_ttp':max(ttp.values()),
                    'entropy':round(entropy(ttp),4)}
 return out

def report(out):
 """Print the metrics, then the gates that have a target and a measured value."""
 c=out['counts']
 print(f"\ncounts            harmful {c['harmful']}  harmless {c['harmless']}  total {c['total']}")
 for label,share in out['source_share'].items():
  top=', '.join(f'{s} {v:.0%}' for s,v in list(share.items())[:4])
  print(f"  {label:<14} {top}")
 print(f"\nlength (words)    "+'  '.join(f"{l} med {v['median']} mean {v['mean']}" for l,v in out['length'].items()))
 print(f"lead-in entropy   "+'  '.join(f'{l} {v:.3f}' for l,v in out['lead_in_entropy'].items()))
 if 'cross_label_coverage' in out:
  print(f"\ncross-label nearest neighbour (harmful -> harmless)")
  for k,v in out['cross_label_coverage'].items():print(f"  cosine {k:<6} {v:>7.1%}")
  print(f"  median       {out['cross_label_median']:>7.3f}")
  print("internal near-duplicates")
  for lab,d in out.get('internal_near_duplicates',{}).items():
   print(f"  {lab:<10} "+'  '.join(f'{k} {v:.1%}' for k,v in d.items()))
 if 'ttp' in out:
  print(f"\nttp               {out['ttp']['distinct']} distinct, max {out['ttp']['max_per_ttp']} per ttp, entropy {out['ttp']['entropy']:.3f}")
 if 'axis_distribution' in out:
  print(f"axis              "+'  '.join(f'{k} {v:.0%}' for k,v in out['axis_distribution'].items()))

 print("\ngates")
 lower_is_better={'g1_lead_in_accuracy','g1_full_text_accuracy','g1_length_ratio','g7_max_axis_share'}
 for gate,target in GATES.items():
  value=out.get(gate)
  if value is None:print(f"  {gate:<28} {'n/a':>8}   (not applicable to this corpus)");continue
  ok=value<=target if gate in lower_is_better else value>=target
  arrow='<=' if gate in lower_is_better else '>='
  print(f"  {gate:<28} {value:>8.4f}   {'PASS' if ok else 'FAIL'} (target {arrow} {target})")

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--pairs',type=Path,default=Path(__file__).resolve().parent/'pairs.jsonl')
 ap.add_argument('--version',default='current',help='label for the output file')
 ap.add_argument('--out',type=Path,default=None)
 a=ap.parse_args()
 pairs=load(a.pairs)
 # Flatten the pairs into a labelled row set; the metrics below are label-based.
 rows=([{'text':p['harmful'],'label':'harmful','id':p['pair_id'],'source':'paired',
         'ttp':p.get('ttp_id')} for p in pairs]
      +[{'text':p['harmless'],'label':'harmless','id':p['pair_id'],'source':'paired',
         'ttp':p.get('ttp_id')} for p in pairs])
 out={'version':a.version,'pairs':str(a.pairs),'paired':True,**analyse(rows,pairs)}
 report(out)
 dest=a.out or Path(__file__).resolve().parent/f'diagnostics.{a.version}.json'
 dest.write_text(json.dumps(out,indent=2)+'\n')
 print(f"\nwritten {dest.relative_to(Path.cwd()) if dest.is_relative_to(Path.cwd()) else dest}")
