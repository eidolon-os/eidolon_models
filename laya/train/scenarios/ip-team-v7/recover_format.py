"""Re-parse only structural rejects; preserve raw facts and obtain new blind labels."""
import argparse,importlib.util,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('v7_generation',HERE/'generate.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('batches',nargs='+');a=ap.parse_args()
 if len(a.batches)>4:raise ValueError('max four review calls')
 g.io.legacy.load_env(HERE.parent/'ip-team-v3/.env')
 assert os.environ['EIDOLON_IP_DATA_BASE_URL']=='https://open.bigmodel.cn/api/paas/v4' and os.environ['EIDOLON_IP_DATA_MODEL']=='glm-5.3-flash'
 client=g.ChatClient({'base_url':os.environ['EIDOLON_IP_DATA_BASE_URL'],'model':'glm-5.3-flash','api_key_env':'EIDOLON_IP_DATA_API_KEY','timeout':180,'max_tokens':6000,'thinking':{'type':'enabled'},'reasoning_effort':'low','response_format':{'type':'json_object'}})
 for batch in a.batches:
  dest=g.ROOT/'reviewed'/f'recovered-{batch}.json'
  if dest.exists():continue
  previous=json.loads((g.ROOT/'reviewed'/f'{batch}.json').read_text());plan=previous['plan']
  eligible={r['local_id'] for r in previous['results'] if r['status']=='rejected_structure' and 'local_id' in r}
  raw=(g.ROOT/'responses'/f'{batch}-gen.txt').read_text();data=json.loads(raw);valid=[]
  if not isinstance(data,dict) or set(data)!={'scenes'} or not isinstance(data['scenes'],list) or not 1<=len(data['scenes'])<=3:raise ValueError('unrecoverable top schema')
  if len({d.get('id') for d in data['scenes']})!=len(data['scenes']):raise ValueError('duplicate ids')
  if len(previous['results'])==1 and previous['results'][0]['status']=='rejected_structure' and 'local_id' not in previous['results'][0]:eligible={d['id'] for d in data['scenes']}
  for d in data['scenes']:
   if d['id'] not in eligible:continue
   try:scene=g.parse_scene(d,plan)
   except (ValueError,TypeError):continue
   valid.append({'id':f"recovered-{batch}-{d['id']}",'local_id':d['id'],'batch':batch,'split':plan['split'],'slice':plan['slice'],'scene':scene,'status':'pending_review','generator_sha256':g.io.sha(raw),'recovered_from':f"{batch}-{d['id']}"})
  if not valid:print(batch,'no eligible structurally valid facts',flush=True);continue
  visible=[]
  for e in valid:
   d=dict(e['scene']);d['cast']=[dict(c,action_code=f'respond:{i}') for i,c in enumerate(d['cast'])];visible.append(d)
  prompt=g.RULES+'\n以下仅公开场景，不含标签。请首次独立标注下一步，不要虚构原标签。输出对象reviews数组，每项id、gold(完整动作字符串数组)、why(依据)、valid(布尔，输入能可靠标注则true)、issue(无则空字符串)。每个输入id恰好一次。\n'+json.dumps(visible,ensure_ascii=False)
  rr=g.io.paid(client,g.ROOT,f'recovered-{batch}-review',prompt,g.LIMIT)
  try:
   review=json.loads(rr)['reviews'];byid={r['id']:r for r in review}
   if len(review)!=len(valid) or set(byid)!={v['local_id'] for v in valid}:raise ValueError('review ids')
  except (ValueError,TypeError,KeyError):byid={}
  for e in valid:
   r=byid.get(e['local_id'])
   try:
    if not isinstance(r,dict) or set(r)!={'id','gold','why','valid','issue'} or type(r['valid'])!=bool or not isinstance(r['why'],str) or not r['why'].strip():raise ValueError('review schema')
    r['gold']=g.parse_gold(r['gold'],len(e['scene']['cast']));status='annotated_pending_audit' if r['valid'] else 'quarantined_review'
    if r['valid']:e['scene'].update(gold=r['gold'],why=r['why'])
   except (ValueError,TypeError):status='rejected_review_structure'
   e.update(review=r,status=status,reviewer_sha256=g.io.sha(rr),annotation_mode='format_only_reparse_then_blind_glm_label_then_assistant_audit')
  g.io.legacy.atomic_json(dest,{'plan':plan,'results':valid});print(batch,[e['status'] for e in valid],flush=True)
if __name__=='__main__':main()
