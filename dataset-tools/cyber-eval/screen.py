"""Screen a source for cyber relevance and harm with System One, replacing the regex filter.

A classifier screens more consistently than a keyword pattern and applies the same standard to
every source. On ClearHarm it selects 30 of 179 harmful rows where a hand-written cyber regex
selected 12; on MaliciousInstruct 28 against 16, on JBB-Behaviors 23 against 10.

Records probabilities, never hard labels, so thresholds can move without re-querying. The harm
question disagreed with glm-5.2 on 28% of calibration items, concentrated in the dual-use band,
so `harm` here is a sorting aid and not a gate until calibration fixes its threshold.

  python screen.py --input sources/clearharm/data/train-00000-of-00001.parquet \\
                   --field content --name clearharm --label harmful

Writes screened/<name>.jsonl. Re-running reuses rows already screened.
"""
import argparse,csv,json
from pathlib import Path
import pyarrow.parquet as pq
import systemone

R=Path(__file__).resolve().parent
OUT=R/'screened'
NAMES=['is_cyber','harm','refuse']

def read(path,field):
 rows=pq.read_table(path).to_pylist() if path.suffix=='.parquet' else list(csv.DictReader(path.open()))
 texts=[]
 for r in rows:
  v=r[field]
  texts.append('\n'.join(v) if isinstance(v,list) else str(v))
 return rows,texts

def run(a):
 OUT.mkdir(exist_ok=True)
 dest=OUT/f'{a.name}.jsonl'
 done={}
 if dest.exists():
  done={x['text']:x for x in (json.loads(l) for l in dest.read_text().splitlines() if l.strip())}
 rows,texts=read(Path(a.input),a.field)
 if a.limit:rows,texts=rows[:a.limit],texts[:a.limit]
 url,key=systemone.credentials()
 out=[]
 for i,(r,t) in enumerate(zip(rows,texts),1):
  if t in done:out.append(done[t]);continue
  ans=systemone.ask(t,NAMES,url,key)
  out.append({'source':a.name,'source_row':i-1,'label':a.label,'text':t,
   'p_cyber':ans['is_cyber']['noul'],'harm_score':ans['harm']['score'],
   'harm_confidence':ans['harm']['confidence'],'p_refuse':ans['refuse']['noul']})
  if i%25==0 or i==len(rows):print(f'  screened {i}/{len(rows)}')
 dest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
 cyber=[x for x in out if x['p_cyber']>=a.cyber_threshold]
 print(f'\n{a.name}: {len(out)} rows, {len(cyber)} cyber at p>=%.2f'%a.cyber_threshold)
 if cyber:
  hs=sorted(x['harm_score'] for x in cyber)
  print(f'  harm score median {hs[len(hs)//2]:.2f}, '
        f'clearly harmful (>1.3) {sum(1 for h in hs if h>1.3)}, '
        f'dual-use band (0.7-1.3) {sum(1 for h in hs if 0.7<=h<=1.3)}, '
        f'benign (<0.7) {sum(1 for h in hs if h<0.7)}')
 print(f'written {dest.relative_to(R)}')

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 ap.add_argument('--input',required=True);ap.add_argument('--field',required=True)
 ap.add_argument('--name',required=True);ap.add_argument('--label',required=True)
 ap.add_argument('--cyber-threshold',type=float,default=0.5);ap.add_argument('--limit',type=int)
 run(ap.parse_args())
