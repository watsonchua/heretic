"""Client for the LiteLLM gateway: embeddings for clustering, kimi-k3 for twin generation.

Credentials come from the repository-root .env via python-dotenv. Do not use shell `source`:
two lines in that file put spaces around `=`, which the shell rejects.

Paired with systemone.py, which handles the classifier. Keeping them separate keeps the one
model-free boundary visible: diagnostics.py imports neither.
"""
import json,os,re,time
from pathlib import Path
import requests
from dotenv import dotenv_values

EMBED_MODEL='azure/text-embedding-3-large'
CHAT_MODEL='kimi-k3'      # writes the harmless twins
JUDGE_MODEL='glm-5.2'     # judges responses, and second opinion during calibration

# glm-5.2 must stay distinct from CHAT_MODEL: a judge that graded its own twins would agree
# with itself for the wrong reason.
ROOT=Path(__file__).resolve().parents[2]
FENCE=re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$')

def credentials():
 v={**dotenv_values(ROOT/'.env'),**{k:x for k,x in os.environ.items() if k.startswith('LITELLM_')}}
 missing=[k for k in ('LITELLM_BASE_URL','LITELLM_API_KEY') if not v.get(k)]
 if missing:raise SystemExit(f'missing in {ROOT/".env"}: {", ".join(missing)}')
 return v['LITELLM_BASE_URL'].rstrip('/'),v['LITELLM_API_KEY']

def _post(path,body,url=None,key=None,retries=4,timeout=120):
 if url is None:url,key=credentials()
 for attempt in range(retries):
  try:
   r=requests.post(f'{url}{path}',timeout=timeout,json=body,
                   headers={'Content-Type':'application/json','Authorization':f'Bearer {key}'})
   if r.status_code in (429,500,502,503,504):raise requests.HTTPError(str(r.status_code))
   r.raise_for_status();return r.json()
  except (requests.RequestException,ValueError):
   if attempt==retries-1:raise
   time.sleep(2**attempt)

def embed(texts,model=EMBED_MODEL,batch=256,url=None,key=None,progress=False):
 """Embed a list of strings, preserving order. Returns a list of float vectors."""
 if url is None:url,key=credentials()
 out=[]
 for i in range(0,len(texts),batch):
  chunk=texts[i:i+batch]
  d=_post('/v1/embeddings',{'model':model,'input':chunk},url,key)['data']
  out+=[e['embedding'] for e in sorted(d,key=lambda x:x['index'])]
  if progress:print(f'  embedded {len(out)}/{len(texts)}')
 return out

def chat(messages,model=CHAT_MODEL,temperature=0.7,url=None,key=None,**kw):
 r=_post('/v1/chat/completions',
         {'model':model,'temperature':temperature,'messages':messages,**kw},url,key)
 return r['choices'][0]['message']['content']

def chat_json(messages,**kw):
 """Both chat models sometimes wrap JSON in ```json fences; strip them before parsing."""
 raw=chat(messages,**kw)
 try:return json.loads(FENCE.sub('',raw.strip()))
 except json.JSONDecodeError:return None

def judge(system,user,model=JUDGE_MODEL,temperature=0,retries=3,**kw):
 """Structured judgement from glm-5.2. Returns the parsed dict, or None if it never parses."""
 for _ in range(retries):
  r=chat_json([{'role':'system','content':system},{'role':'user','content':user}],
              model=model,temperature=temperature,**kw)
  if isinstance(r,dict):return r
 return None
