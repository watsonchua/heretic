"""Collapse the de-templated MITRE cores to one prompt per distinct behaviour, then sample.

Phase 1 steps 3-4. v0 counted 981 unique strings as 981 behaviours; they are 185 TTPs crossed
with ~190 wrappers. De-templating removed the wrappers but left 857 distinct cores, because the
technique wordings themselves vary. Clustering collapses those; stratified sampling then caps
MITRE's share so it cannot dominate a second time.

  embed     azure/text-embedding-3-large over the distinct cores
  cluster   average-linkage agglomerative at cosine >= --threshold
  pick      one representative per cluster, preferring a core that carries a request verb
            and sits closest to the cluster centroid
  sample    at most --per-ttp per TTP, allocated evenly across the 10 MITRE categories

  python select_cores.py --cores cores.jsonl --target 220

Embeddings are cached in embeddings.json so re-running costs nothing.
"""
import argparse,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import llm

R=Path(__file__).resolve().parent

def norm(s):return ' '.join(s.split()).casefold().rstrip('.?! ')

def load_embeddings(texts,cache,model):
 """Embed only what the cache is missing; the cache is keyed by normalized text."""
 store=json.loads(cache.read_text()) if cache.exists() else {}
 if store.get('model') not in (None,model):store={'model':model,'vectors':{}}
 store.setdefault('model',model);store.setdefault('vectors',{})
 missing=[t for t in texts if t not in store['vectors']]
 if missing:
  print(f'embedding {len(missing)} new cores ({len(texts)-len(missing)} cached)')
  for v,t in zip(llm.embed(missing,model=model,progress=True),missing):store['vectors'][t]=v
  cache.write_text(json.dumps(store))
 else:print(f'all {len(texts)} cores cached')
 return np.array([store['vectors'][t] for t in texts],dtype=np.float32)

def cluster(X,threshold):
 X=X/np.linalg.norm(X,axis=1,keepdims=True)
 model=AgglomerativeClustering(n_clusters=None,distance_threshold=1-threshold,
                               metric='cosine',linkage='average')
 return model.fit_predict(X),X

def representatives(labels,X,records):
 """One per cluster: a verb-carrying core nearest the centroid, else nearest overall."""
 out=[]
 for c in sorted(set(labels)):
  idx=[i for i,l in enumerate(labels) if l==c]
  centroid=X[idx].mean(0);centroid/=np.linalg.norm(centroid) or 1
  ranked=sorted(idx,key=lambda i:(records[i]['needs_verb'],-float(X[i]@centroid)))
  best=records[ranked[0]]
  out.append({**best,'cluster':int(c),'cluster_size':len(idx),
              'cluster_members':[records[i]['task_id'] for i in idx]})
 return out

def stratify(reps,target,per_ttp):
 """Even allocation across categories, round-robin over TTPs inside each.

 The TTP cap is global, not per category: a technique like T1055 appears under both Evasion
 and Privilege Escalation, and capping separately would let it in twice per depth. Categories
 that run out of distinct TTPs leave a shortfall, so a backfill pass tops the selection back up
 to the target from whatever clusters remain, still honouring the global cap.
 """
 by_cat=defaultdict(lambda:defaultdict(list))
 for r in reps:by_cat[r['mitre_category']][r['ttp_id']].append(r)
 cats=sorted(by_cat)
 quota={c:target//len(cats) for c in cats}
 for c in cats[:target%len(cats)]:quota[c]+=1
 picked=[];used=Counter();chosen=set()
 def take(r):
  picked.append(r);used[r['ttp_id']]+=1;chosen.add(r['cluster'])
 for c in cats:
  # Largest clusters first inside each TTP: the most frequently expressed behaviours.
  ttps={t:sorted(v,key=lambda r:-r['cluster_size']) for t,v in by_cat[c].items()}
  order=sorted(ttps,key=lambda t:-len(ttps[t]))
  taken=0
  for depth in range(per_ttp):
   for t in order:
    if taken>=quota[c]:break
    if depth<len(ttps[t]) and used[t]<per_ttp:take(ttps[t][depth]);taken+=1
   if taken>=quota[c]:break
 short=target-len(picked)
 if short>0:
  spare=sorted((r for r in reps if r['cluster'] not in chosen and used[r['ttp_id']]<per_ttp),
               key=lambda r:(used[r['ttp_id']],-r['cluster_size']))
  added=0
  for r in spare:
   if added>=short:break
   if used[r['ttp_id']]<per_ttp:take(r);added+=1
  print(f'  backfilled {added} of {short} short from {len(spare)} spare clusters')
 return picked

def run(a):
 records=[json.loads(l) for l in (R/a.cores).read_text().splitlines() if l.strip()]
 records=[r for r in records if r['core'].strip()]
 # Collapse exact repeats first so identical cores are not embedded twice.
 seen={};deduped=[]
 for r in records:
  k=norm(r['core'])
  if k in seen:seen[k]['duplicates']=seen[k].get('duplicates',0)+1
  else:seen[k]=r;r['duplicates']=0;deduped.append(r)
 print(f'{len(records)} cores -> {len(deduped)} distinct')

 X=load_embeddings([norm(r['core']) for r in deduped],R/a.cache,a.embed_model)
 labels,Xn=cluster(X,a.threshold)
 reps=representatives(labels,Xn,deduped)
 print(f'{len(deduped)} distinct -> {len(reps)} behaviour clusters at cosine {a.threshold}')
 sizes=Counter(r['cluster_size'] for r in reps)
 print(f'  singletons {sizes[1]}, largest cluster {max(r["cluster_size"] for r in reps)}')

 picked=stratify(reps,a.target,a.per_ttp)
 print(f'{len(reps)} clusters -> {len(picked)} selected '
       f'({len({p["ttp_id"] for p in picked})} TTPs, '
       f'{len({p["mitre_category"] for p in picked})} categories)')
 for c,n in sorted(Counter(p['mitre_category'] for p in picked).items()):print(f'  {n:>4}  {c}')
 over=[t for t,n in Counter(p['ttp_id'] for p in picked).items() if n>a.per_ttp]
 assert not over,f'TTP cap exceeded: {over}'

 (R/a.out).write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in picked))
 report={'source':'purpcode/CyberSecEval-MITRE','cores_in':len(records),
  'distinct_cores':len(deduped),'clusters':len(reps),'threshold':a.threshold,
  'selected':len(picked),'per_ttp_cap':a.per_ttp,'distinct_ttp':len({p['ttp_id'] for p in picked}),
  'by_category':dict(sorted(Counter(p['mitre_category'] for p in picked).items())),
  'needs_verb':sum(1 for p in picked if p['needs_verb']),
  'embed_model':a.embed_model}
 (R/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
 print(f"\nwritten {a.out} and selection.json")
 print(f"MITRE share stays under 25% of the harmful class if it totals "
       f"{math.ceil(len(picked)/0.25)} or more prompts")

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 ap.add_argument('--cores',default='cores.jsonl');ap.add_argument('--out',default='selected.mitre.jsonl')
 ap.add_argument('--cache',default='embeddings.json')
 ap.add_argument('--threshold',type=float,default=0.85)
 ap.add_argument('--target',type=int,default=220);ap.add_argument('--per-ttp',type=int,default=2)
 ap.add_argument('--embed-model',default=llm.EMBED_MODEL)
 run(ap.parse_args())
