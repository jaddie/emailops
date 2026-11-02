#!/usr/bin/env python3
"""Opt-in synthetic OpenAI-compatible embedding dimension/throughput benchmark.

Does not load models through management APIs or change server settings. Ordinary
inference may cause the server to auto-load/evict a model. Reports only aggregate
measurements; no model vectors or private input are stored.
"""
import argparse
from pathlib import Path
from urllib.parse import urlparse
import concurrent.futures,json,math,statistics,time,urllib.request
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base-url', default='http://localhost:1234/v1')
parser.add_argument('--model', action='append', required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if urlparse(args.base_url).hostname not in ('localhost', '127.0.0.1', '::1'):
    parser.error('this synthetic local benchmark only accepts a loopback server')
URL = args.base_url.rstrip('/') + '/embeddings'
MODELS = args.model
TEXTS=[f'Synthetic record {i}: the calibration team checked the green marker, recorded a stable reference measurement, and scheduled a follow-up inspection. '*6 for i in range(16)]
def request(model,texts,dimensions=None):
 p={'model':model,'input':texts}
 if dimensions is not None:p['dimensions']=dimensions
 t=time.perf_counter()
 try:
  r=json.load(urllib.request.urlopen(urllib.request.Request(URL,data=json.dumps(p).encode(),headers={'Content-Type':'application/json'}),timeout=90));d=r.get('data',[])
  assert len(d)==len(texts) and sorted(x['index'] for x in d)==list(range(len(texts)))
  lengths=[len(x['embedding']) for x in d]
  assert len(set(lengths))==1 and all(math.isfinite(v) for x in d for v in x['embedding'])
  return {'ok':True,'elapsed_s':time.perf_counter()-t,'dimensions':lengths[0]}
 except Exception as e:return {'ok':False,'elapsed_s':time.perf_counter()-t,'error':str(e)}
results=[]
for model in MODELS:
 for dim in [None,768,2560,4096]:
  r={'kind':'probe','model':model,'requested_dimensions':dim,**request(model,TEXTS[:1],dim)};results.append(r);print(json.dumps(r),flush=True)
 for batch,concurrency in [(1,1),(2,1),(4,1),(8,1),(1,2)]:
  warm=request(model,TEXTS[:batch]);results.append({'kind':'warmup','model':model,'batch':batch,**warm})
  for repeat in range(3):
   chunks=[TEXTS[i:i+batch] for i in range(0,len(TEXTS),batch)];t=time.perf_counter()
   with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:rows=list(pool.map(lambda x:request(model,x),chunks))
   elapsed=time.perf_counter()-t;errors=sum(not x['ok'] for x in rows)
   r={'kind':'benchmark','model':model,'batch':batch,'concurrency':concurrency,'repeat':repeat,'inputs':16,'elapsed_s':elapsed,'inputs_per_s':16/elapsed,'request_p50_s':statistics.median(x['elapsed_s'] for x in rows),'request_max_s':max(x['elapsed_s'] for x in rows),'errors':errors};results.append(r);print(json.dumps(r),flush=True)
 args.out.parent.mkdir(parents=True, exist_ok=True)
 args.out.write_text(json.dumps(results, indent=2) + '\n')
