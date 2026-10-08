"""Combine downloaded sources into text-only exports; no local augmentation."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
R=Path(__file__).resolve().parent
revisions={
'StrongREJECT':'f7cad6c17e624e21d8df2278e918ae1dddb4cb56',
'mlabonne/harmless_alpaca':'02c6a92cfcf11bb0c387334f8146d149d65b587f',
'HarmBench':'8e1604d1171fe8a48d8febecd22f600e462bdcdd',
'AlignmentResearch/ClearHarm':'79464fb6b3c2a8ee925184f394f9636600349f88',
'purpcode/CyberSecEval-MITRE':'70213a497f4e7d5ee5b0089eac5b8d3244eba2af',
'purpcode/CyberSecEval-FRR':'73294ca390f12ebb075d763550b942875dee5fc2'}
HARMFUL={150,151,152,153,154,161,162,164,165,166,168,169,170,172,176,179,180,181,182,184,190,191,197,199,203}
import re
pattern=re.compile(r'cyber|malware|phishing|ransomware|firewall|network security|sql injection|penetration test|password|encryption|cryptograph|data breach|computer security|internet security|information security|authentication',re.I)
exclude=re.compile(r'cyber\s*bull|rocks',re.I)
rows=[]
def add(text,label,source,path,index,field,split,**extras):
 rows.append(dict(text=text,label=label,source=source,source_revision=revisions[source],source_path=path,source_row=index,source_field=field,source_split=split,**extras))
for i,x in enumerate(csv.DictReader((R/'sources/strongreject.csv').open())):
 if i in HARMFUL:add(x['forbidden_prompt'],'harmful','StrongREJECT','sources/strongreject.csv',i,'forbidden_prompt','original')
for i,x in enumerate(csv.DictReader((R/'sources/harmbench.csv').open())):
 if x['SemanticCategory']=='cybercrime_intrusion':
  # Contextual benchmark requests must retain their supplied context.
  text=(x['ContextString']+'\n\n'+x['Behavior']).strip() if x['ContextString'] else x['Behavior']
  add(text,'harmful','HarmBench','sources/harmbench.csv',i,'ContextString + Behavior','all',source_behavior_id=x['BehaviorID'],group_text=x['Behavior'])
path='sources/clearharm/data/train-00000-of-00001.parquet'
for i,x in enumerate(pq.read_table(R/path).to_pylist()):
 if i in {2,12,28,33,44,47,54,58,97,115,123,168}:
  assert x['clf_label']==1
  add('\n'.join(x['content']),'harmful','AlignmentResearch/ClearHarm',path,i,'content','train')
for split in ('train','test'):
 path=f'sources/harmless_alpaca/data/{split}-00000-of-00001.parquet'
 for i,x in enumerate(pq.read_table(R/path).to_pylist()):
  if pattern.search(x['text']) and not exclude.search(x['text']):add(x['text'],'harmless','mlabonne/harmless_alpaca',path,i,'text',split)
for name,label,field in [('mitre','harmful','base_prompt'),('frr','harmless','messages[0].content')]:
 path=f'sources/cyberseceval-{name}/data/test-00000-of-00001.parquet'
 for i,x in enumerate(pq.read_table(R/path).to_pylist()):
  if name=='frr':assert x['is_malicious'] is False
  text=x['base_prompt'] if name=='mitre' else x['messages'][0]['content']
  add(text,label,'purpcode/CyberSecEval-'+('MITRE' if name=='mitre' else 'FRR'),path,i,field,'test',source_task_id=x['task_id'])
def norm(s):return ' '.join(s.split()).casefold()
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
unique={};duplicates=0
for x in rows:
 k=norm(x['text'])
 if k in unique:
  assert unique[k]['label']==x['label'],'Conflicting source labels'
  unique[k].setdefault('additional_provenance',[]).append({p:v for p,v in x.items() if p not in ('text','label')});duplicates+=1
 else:unique[k]=x
rows=list(unique.values())
for x in rows:
 x['id']=sha(norm(x['text']));x['group_id']=sha(norm(x.pop('group_text',x['text'])))
 x['split']='test' if int(x['group_id'][:8],16)%5==0 else 'train'
 x['expected_behavior']='refuse harmful assistance; offer safe alternatives' if x['label']=='harmful' else 'answer helpfully and accurately'
 x['derivation']='source_prompt_no_local_augmentation'
def write(path,items):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items))
for label in ('harmful','harmless'):
 selected=[x for x in rows if x['label']==label]
 write(R/f'data/{label}.jsonl',[{'text':x['text']} for x in selected]);write(R/f'data/{label}.metadata.jsonl',selected)
 pq.write_table(pa.Table.from_pylist([{'text':x['text']} for x in selected]),R/f'data/{label}.parquet')
 # Overwrite stale seed exports from the abandoned augmentation builder.
 write(R/f'data/{label}.seeds.jsonl',selected)
 for split in ('train','test'):
  out=[{'text':x['text']} for x in selected if x['split']==split]
  write(R/f'data/{label}/{split}.jsonl',out);pq.write_table(pa.Table.from_pylist(out),R/f'data/{label}/{split}.parquet')
write(R/'data/evaluation.jsonl',rows)
manifest=dict(domain='cybersecurity',source_revisions=revisions,counts={l:dict(Counter(x['split'] for x in rows if x['label']==l)) for l in ('harmful','harmless')},source_counts={l:dict(Counter(x['source'] for x in rows if x['label']==l)) for l in ('harmful','harmless')},duplicates_removed=duplicates,augmentation=False,semantic_independence_verified=False,decision_model=None)
files=[p for p in (R/'data').rglob('*') if p.is_file()]+[R/p for p in sorted({x['source_path'] for x in rows})]
manifest['sha256']={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(R/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ('counts','source_counts','duplicates_removed')},indent=2))
