"""One final label-blind reannotation; prior rejected labels remain unchanged."""
import importlib.util,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('v7_g',HERE/'generate.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
if (g.ROOT/'reviewed/adjudicated-ambiguity.json').exists():raise ValueError('already adjudicated')
source=[]
for batch,lid in [('test-0-same_name','s1'),('val-0-same_name','s2')]:
 old=next(r for r in json.loads((g.ROOT/'reviewed'/f'{batch}.json').read_text())['results'] if r['local_id']==lid)
 source.append(old)
visible=[]
for i,old in enumerate(source):
 d={k:v for k,v in old['scene'].items() if k not in ('gold','why')};d['id']=f's{i+1}';d['cast']=[dict(c,action_code=f'respond:{j}') for j,c in enumerate(d['cast'])];visible.append(d)
g.io.legacy.load_env(HERE.parent/'ip-team-v3/.env')
assert os.environ['EIDOLON_IP_DATA_BASE_URL']=='https://open.bigmodel.cn/api/paas/v4' and os.environ['EIDOLON_IP_DATA_MODEL']=='glm-5.3-flash'
client=g.ChatClient({'base_url':os.environ['EIDOLON_IP_DATA_BASE_URL'],'model':'glm-5.3-flash','api_key_env':'EIDOLON_IP_DATA_API_KEY','timeout':180,'max_tokens':6000,'thinking':{'type':'enabled'},'reasoning_effort':'low','response_format':{'type':'json_object'}})
prompt=g.RULES+'\n仅依据以下公开快照独立标注下一步，输入不含标签，也不知道其他判断。输出对象reviews数组，每项id、gold(完整动作字符串数组)、why、valid(布尔)、issue(无问题空串)。各输入id恰好一次。\n'+json.dumps(visible,ensure_ascii=False)
rr=g.io.paid(client,g.ROOT,'adjudicated-ambiguity-review',prompt,g.LIMIT)
reviews=json.loads(rr)['reviews'];byid={r['id']:r for r in reviews}
if set(byid)!={'s1','s2'} or len(reviews)!=2:raise ValueError('review count/ids')
results=[]
for i,old in enumerate(source):
 r=byid[f's{i+1}'];r['gold']=g.parse_gold(r['gold'],len(old['scene']['cast']))
 if set(r)!={'id','gold','why','valid','issue'} or type(r['valid'])!=bool or not isinstance(r['why'],str):raise ValueError('review schema')
 entry={k:v for k,v in old.items() if k not in ('scene','review','status','reviewer_sha256')}
 entry.update(id='adjudicated-'+old['id'],scene=dict(old['scene'],gold=r['gold'],why=r['why']),review=r,status='annotated_pending_audit' if r['valid'] else 'quarantined_review',reviewer_sha256=g.io.sha(rr),annotation_mode='second_blind_glm_annotation_after_semantic_disagreement; assistant_must_audit',adjudicated_from=old['id'])
 results.append(entry)
g.io.legacy.atomic_json(g.ROOT/'reviewed/adjudicated-ambiguity.json',{'plan':{'reason':'last budget call; same-name label disagreement before any training/scoring'},'results':results})
print([(r['id'],r['scene']['gold']) for r in results],flush=True)
