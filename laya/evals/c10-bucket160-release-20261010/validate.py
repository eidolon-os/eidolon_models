"""HTTP soak, overload and resource test; never contacts Agent or device executors."""
import concurrent.futures
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
import subprocess

ROOT=Path('/tmp/laya-160-release-20261010')
URL='http://127.0.0.1:18774'
ROOT.mkdir(exist_ok=True)
fixtures=[json.loads(l) for l in (ROOT/'requests.jsonl').read_text().splitlines()]
conversation=[c for c in fixtures if c['set']=='conversation']
def info(port):
    return json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/info',timeout=3))
def post(body,port=18774):
    start=time.perf_counter()
    req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/systemone',json.dumps(body).encode(),{'content-type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=5) as r: code,payload=r.status,json.load(r)
    except urllib.error.HTTPError as e:code,payload=e.code,json.load(e)
    return {'status':code,'ms':(time.perf_counter()-start)*1000,'response':payload}
def resources():
    pid=int(subprocess.check_output(['systemctl','show','laya-160-validation-20261010','-p','MainPID','--value']))
    status=Path(f'/proc/{pid}/status').read_text()
    return {'at':time.time(),'pid':pid,'process':{l.split(':')[0]:l.split(':')[1].strip() for l in status.splitlines() if l.startswith(('VmRSS:','VmHWM:','Threads:'))},'meminfo':{l.split(':')[0]:l.split(':')[1].strip() for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemAvailable:','SwapFree:'))}}
def stats(rows):
    a=sorted(r['ms'] for r in rows if r['status']==200)
    return {'count':len(rows),'ok':len(a),'busy':sum(r['status']==503 for r in rows),'p50':statistics.median(a) if a else None,'p95':a[int(.95*(len(a)-1))] if a else None,'max':max(a) if a else None,'over_1000':sum(t>1000 for t in a)}
start=time.monotonic();before=info(8771);samples=[resources()]
# Warmup excluded.
for c in conversation[:3]:post(c['body'])
rows=[]
with (ROOT/'soak.jsonl').open('w') as f:
    for i in range(800):
        c=conversation[i%len(conversation)]
        r=post(c['body']);r.update(id=c['id'],index=i)
        rows.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
        if i%50==49:samples.append(resources());print('soak',i+1,flush=True)
concurrent_rows=[]
for i in range(30):
    barrier=threading.Barrier(2)
    def run():barrier.wait();return post(conversation[15]['body'])
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        pair=list(pool.map(lambda _:run(),range(2)))
    concurrent_rows.extend(pair)
# Two local model services simultaneously: synthetic classifier load only, no participation decision/actuator.
coexist=[]
body={'state':{'utterance':'这是一次隔离性能测试'},'questions':{'intent':{'type':'choice','instructions':'话语是否为测试？','criteria':{'是':'测试','否':'其他'}}}}
for i in range(30):
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        a=pool.submit(post,conversation[i%20]['body']);b=pool.submit(post,body,8773)
        coexist.append({'smart_home':a.result(),'participation_model':b.result()})
summary={'soak':stats(rows),'concurrency':stats(concurrent_rows),'coexist_smart_home':stats([r['smart_home'] for r in coexist]),'coexist_participation':stats([r['participation_model'] for r in coexist]),'seconds':time.monotonic()-start,'production_before':before,'production_after':info(8771),'candidate':info(18774),'resources':samples}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
(ROOT/'concurrency.json').write_text(json.dumps(concurrent_rows,ensure_ascii=False))
(ROOT/'coexist.json').write_text(json.dumps(coexist,ensure_ascii=False))
print(json.dumps({k:v for k,v in summary.items() if k not in ['resources','production_before','production_after','candidate']}),flush=True)

assert summary['soak']['ok'] == 800 and summary['soak']['over_1000'] == 0, 'serial deadline gate'
assert summary['concurrency']['ok'] == 30 and summary['concurrency']['busy'] == 30, 'overload gate'
assert max(r['ms'] for r in concurrent_rows if r['status'] == 503) < 100, 'busy response gate'
assert summary['coexist_smart_home']['ok'] == summary['coexist_participation']['ok'] == 30, 'coexistence gate'
assert not any(r['response'].get('truncated') for r in rows), 'truncation gate'
