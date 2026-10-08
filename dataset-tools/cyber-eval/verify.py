"""Check export schemas, source fidelity, checksums and split isolation."""
import csv,hashlib,json
from pathlib import Path
import pyarrow.parquet as pq
R=Path(__file__).resolve().parent
m=json.loads((R/'manifest.json').read_text())
for p,h in m['sha256'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
rows=[json.loads(x) for x in (R/'data/evaluation.jsonl').read_text().splitlines()]
assert len(rows)==2032
assert len({' '.join(x['text'].split()).casefold() for x in rows})==len(rows)
sources={}
groups={}
for x in rows:
 p=x['source_path']
 if p not in sources:sources[p]=pq.read_table(R/p).to_pylist() if p.endswith('.parquet') else list(csv.DictReader((R/p).open()))
 s=sources[p][x['source_row']];field=x['source_field']
 if field=='ContextString + Behavior':original=(s['ContextString']+'\n\n'+s['Behavior']).strip() if s['ContextString'] else s['Behavior']
 elif field=='messages[0].content':original=s['messages'][0]['content']
 elif field=='content':original='\n'.join(s['content'])
 else:original=s[field]
 assert original==x['text'],x['id']
 assert x['source_revision']==m['source_revisions'][x['source']]
 assert x['derivation']=='source_prompt_no_local_augmentation'
 assert groups.setdefault(x['group_id'],x['split'])==x['split']
for label,count in [('harmful',1085),('harmless',947)]:
 selected=[x for x in rows if x['label']==label];assert len(selected)==count
 expected=[{'text':x['text']} for x in selected]
 assert expected==pq.read_table(R/f'data/{label}.parquet').to_pylist()
 assert expected==[json.loads(x) for x in (R/f'data/{label}.jsonl').read_text().splitlines()]
 for split in ('train','test'):
  expected=[{'text':x['text']} for x in selected if x['split']==split]
  assert len(expected)==m['counts'][label][split]
  assert expected==pq.read_table(R/f'data/{label}/{split}.parquet').to_pylist()
  assert expected==[json.loads(x) for x in (R/f'data/{label}/{split}.jsonl').read_text().splitlines()]
print('PASS: 1085 harmful, 947 harmless; text-only schemas, source fidelity, checksums, deduplication and grouped splits.')
