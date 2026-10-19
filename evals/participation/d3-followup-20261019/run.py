import copy,json,math,hashlib,time
from pathlib import Path
import httpx
from dotenv import dotenv_values
from build import question,CAST
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'jevk5/runs/d3-followup-20261019'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class BudgetClient:
 def __init__(self,key):
  self.http=httpx.Client(trust_env=False,follow_redirects=False,timeout=45)
  self.key=key;self.count={'d3':0,'reply':0};self.bad={'d3':0,'reply':0};self.records=[]
 def call(self,kind,payload,tag):
  limit={'d3':80,'reply':24}[kind]
  if self.count[kind]>=limit or self.bad[kind]>=3:raise RuntimeError('budget_or_consecutive_errors')
  self.count[kind]+=1
  entry={'kind':kind,'index':self.count[kind],'tag':tag,'request':payload,'timestamp':time.time()}
  with (OUT/'attempts.jsonl').open('a') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
  url='http://decider.yangtzeailab.com/d3/v1/systemone' if kind=='d3' else 'https://api.deepseek.com/v1/chat/completions'
  headers={'Authorization':'Bearer '+self.key} if kind=='reply' else {}
  start=time.perf_counter()
  try:
   r=self.http.post(url,json=payload,headers=headers)
   entry['status']=r.status_code
   if r.status_code!=200:raise RuntimeError('HTTP_'+str(r.status_code))
   data=r.json();entry['response']=data
   if kind=='d3':
    a=data['answers']['move'];p=a['probabilities'];keys=set(payload['questions']['move']['criteria'])
    assert set(p)==keys and a['choice'] in keys
    assert all(isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=1 for x in p.values())
    assert abs(sum(p.values())-1)<.01 and p[a['choice']]>=max(p.values())-1e-6
    value=a['choice']
   else:
    c=data['choices'][0];value=c['message']['content']
    assert isinstance(value,str) and value.strip(),'empty_reply'
    assert c.get('finish_reason')=='stop','incomplete_reply'
   entry['ok']=True;self.bad[kind]=0
  except Exception as e:
   entry['ok']=False;entry['error_type']=type(e).__name__
   self.bad[kind]+=1
   raise
  finally:
   entry['elapsed_ms']=(time.perf_counter()-start)*1000
   self.records.append(entry)
   with (OUT/'calls.jsonl').open('a') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
  return value
 def decide(self,s,q,tag):return self.call('d3',{'model':'d3','state':json.dumps(s,ensure_ascii=False),'questions':q},tag)
 def reply(self,s,action,cid,tag,clarify_instruction):
  task=clarify_instruction if action=='clarify' else '仅完成当前轮到你的一步，用一句简短自然的话回应用户原始请求及已公开的对话；不要替其他角色发言，不要抢做下一步，不要在一句中合并用户要求分次完成的内容。'
  system='你是角色回复生成器，不是决策器。只输出当前角色本次的一句中文台词，不输出角色名前缀、动作标签或旁白。只依据提供的公开内容，不编造用户未明确的意图。'
  public={'speaker_id':cid,'character':s['candidates'][cid],'public_state':s,'task':task}
  payload={'model':'deepseek-v4-flash','messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(public,ensure_ascii=False)}],
   'temperature':0,'max_tokens':320,'stream':False,'thinking':{'type':'disabled'}}
  return self.call('reply',payload,tag)

def loop_state(user,history,user_index,replies):
 latest=history[-1]
 return {'user_request':user,'scene_goal':'','candidates':copy.deepcopy(CAST),'allowed_actions':['respond','clarify','wait','finish'],
  'remaining_replies':max(1,3-replies),
  'prior_public_messages':[copy.deepcopy(m) for i,m in enumerate(history[:-1]) if i!=user_index],
  'trigger':{'same_as_user_request':True} if latest['author']=='user' else copy.deepcopy(latest)}

def episode(client,case,reverse,data):
 # Expected speakers/terminal never enter either API request.
 hist=[];results=[];decisions=0;replies=0;tag=case['id']+(':reverse' if reverse else ':original')
 for turn_index,turn in enumerate(case['turns']):
  user=turn['user'];user_index=len(hist);hist.append({'author':'user','text':user})
  events=[];terminal=None
  while decisions<4 and replies<3:
   s=loop_state(user,hist,user_index,replies);q=question(s['candidates'],reverse)
   pred=client.decide(s,q,tag+':turn'+str(turn_index)+':step'+str(decisions));decisions+=1
   event={'decision':pred,'state':s};events.append(event)
   action,_,cid=pred.partition(':')
   if action in ('wait','finish','abstain'):
    terminal=action;break
   assert action in ('respond','clarify') and cid in s['candidates']
   text=client.reply(s,action,cid,tag+':reply'+str(replies),data['clarify_instruction']);replies+=1
   event['text']=text;hist.append({'author':cid,'text':text})
   if action=='clarify':terminal='clarify';break
  if terminal is None:terminal='budget_exhausted'
  spoken=[x['decision'].split(':')[1] for x in events if 'text' in x]
  if turn['terminal']=='clarify':speaker_ok=len(spoken)==1 and any(x['decision'].startswith('clarify:') for x in events)
  else:speaker_ok=spoken==turn['speakers'] and not any(x['decision'].startswith('clarify:') for x in events)
  passed=terminal==turn['terminal'] and speaker_ok
  results.append({'turn':turn_index,'events':events,'terminal':terminal,'expected':turn,'passed':passed})
  # Do not feed a scripted answer if its prerequisite clarification/wait failed.
  if not passed:break
 return {'id':case['id'],'reverse':reverse,'turns':results,'history':hist,'decisions':decisions,'replies':replies,
   'passed':len(results)==len(case['turns']) and all(x['passed'] for x in results)}

def main():
 data=json.loads((HERE/'cases.json').read_text())
 key=dotenv_values(ROOT.parent/'eidolon_agent/config/.env').get('EIDOLON_AGENT_LLM_API_KEY')
 assert key,'reply credential unavailable'
 OUT.mkdir(parents=True,exist_ok=False)
 cfg={'cases_sha256':sha(HERE/'cases.json'),'script_sha256':sha(__file__),'build_sha256':sha(HERE/'build.py'),
      'design_sha256':sha(HERE/'DESIGN.md'),'d3_cap':80,'reply_cap':24,'retries':0,'production_agent':False,'presentation':'text_only_real_reply_generation'}
 (OUT/'config.json').write_text(json.dumps(cfg,indent=2))
 client=BudgetClient(key);static=[];rollouts=[]
 for i,r in enumerate(data['static']):
  try:
   pred=client.decide(r['state'],r['questions'],'static:'+r['id'])
   x={k:r[k] for k in ('id','family','variant','gold')};x.update(pred=pred,correct=pred in r['gold'])
  except Exception as e:x={'id':r['id'],'correct':False,'error_type':type(e).__name__}
  static.append(x)
  with (OUT/'static.jsonl').open('a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
  if (i+1)%8==0:print('static',i+1,32,flush=True)
  if client.bad['d3']>=3:break
 if client.bad['d3']<3:
  for case in data['episodes']:
   for reverse in (False,True):
    try:r=episode(client,case,reverse,data)
    except Exception as e:r={'id':case['id'],'reverse':reverse,'passed':False,'error_type':type(e).__name__}
    rollouts.append(r)
    with (OUT/'rollouts.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print('episode',case['id'],'reverse',reverse,'pass',r['passed'],flush=True)
    if max(client.bad.values())>=3:break
   if max(client.bad.values())>=3:break
 client.http.close()
 assert sha(__file__)==cfg['script_sha256'] and sha(HERE/'cases.json')==cfg['cases_sha256']
 summary={'static_completed':len(static),'static_correct':sum(x['correct'] for x in static),
   'episodes_completed':len(rollouts),'episodes_passed':sum(x['passed'] for x in rollouts),'calls':client.count,
   'failed_calls':sum(not x['ok'] for x in client.records)}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
