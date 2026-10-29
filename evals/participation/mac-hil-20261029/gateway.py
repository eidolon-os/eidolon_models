"""Loopback-only HIL adapter. No state ownership, private memory or reply generation."""
import asyncio,json,logging,math,time,os
import audit
from pathlib import Path
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI,HTTPException
from eidolon_sdk.biz.participation import DecisionRequest,DecisionResult,Proposal,Snapshot,validate_proposal
ROOT=Path(__file__).resolve().parents[3]
POLICY=json.loads((ROOT/'evals/participation/product-readiness-v1/cases.jsonl').read_text().splitlines()[0])['questions']['move']['instructions']
CLARIFY='根据本轮用户请求和公开上下文，询问一个阻碍继续的缺失信息；不要替用户猜测，不回答尚未明确的请求，问完等待用户。'
log=logging.getLogger('d3-hil')
def project(r):
 if not 1<=len(r.candidates)<=6 or r.context.summary or r.context.pending_requirements:raise ValueError('unsupported_context')
 slots={c.companion_id:f'M{i}' for i,c in enumerate(r.candidates)};back={v:k for k,v in slots.items()}
 history=r.context.recent_messages
 if not history or history[-1]!=r.trigger:raise ValueError('trigger_not_latest')
 if len({m.message_id for m in history})!=len(history):raise ValueError('duplicate_messages')
 def public(m):
  if m.author_kind=='user':author='user'
  elif m.author_kind=='companion' and m.author_id in slots:author=slots[m.author_id]
  else:raise ValueError('unsupported_author')
  return {'author':author,'text':m.text}
 if any(m.message_id==r.user_request.message_id and m!=r.user_request for m in history):raise ValueError('pinned_request_mismatch')
 cast={slots[c.companion_id]:{'name':c.display_name,'character':c.description} for c in r.candidates}
 state={'user_request':r.user_request.text,'scene_goal':r.scene_goal,'candidates':cast,
 'allowed_actions':list(r.constraints.allowed_actions),'remaining_replies':r.constraints.remaining_replies,
 'prior_public_messages':[public(m) for m in history[:-1] if m.message_id!=r.user_request.message_id],
 'trigger':{'same_as_user_request':True} if r.trigger.message_id==r.user_request.message_id else public(r.trigger)}
 choices={}
 for action,verb in [('respond','回应'),('clarify','提出一个澄清问题后等待用户')]:
  if action in r.constraints.allowed_actions:
   for k,c in cast.items():choices[f'{action}:{k}']=f"由{c['name']}（{k}）{verb}"
 for action,desc in [('wait','暂停发言，等待用户'),('finish','本轮请求已完成或用户明确结束')]:
  if action in r.constraints.allowed_actions:choices[action]=desc
 choices['abstain']='无法可靠作出允许的决策，不产生任何回复'
 return {'model':'d3','state':json.dumps(state,ensure_ascii=False),'questions':{'move':{'type':'choice','instructions':POLICY,'criteria':choices}}},back

def convert(r,body,choices,back):
 a=body['answers']['move'];pred=a['choice'];p=a['probabilities']
 if set(p)!=set(choices) or pred not in choices or not all(isinstance(x,(float,int)) and math.isfinite(x) and 0<=x<=1 for x in p.values()) or abs(sum(p.values())-1)>.01:raise ValueError('bad_probabilities')
 if p[pred]<max(p.values())-1e-6:raise ValueError('not_argmax')
 proposal=None
 if pred!='abstain':
  action,_,slot=pred.partition(':')
  proposal=Proposal(action=action,participants=(back[slot],) if slot else (),instruction=CLARIFY if action=='clarify' else '')
 result=DecisionResult(**{k:getattr(r,k) for k in Snapshot.model_fields},status='decided' if proposal else 'abstained',proposal=proposal,
 policy_version='d3-hil-original-v1',model_version='d3-http')
 validate_proposal(r,result)
 return result,float(p[pred])
@asynccontextmanager
async def lifespan(app):
 async with httpx.AsyncClient(trust_env=False,follow_redirects=False,timeout=3) as c:
  app.state.client=c;yield
app=FastAPI(lifespan=lifespan)
@app.get('/health')
async def health():return {'status':'ok','purpose':'temporary_hil','policy':'original','endpoint':'d3','audit_enabled':bool(os.environ.get('EIDOLON_PARTICIPATION_AUDIT_DIR')),'audit_write_failures':audit.failures}
@app.post('/v1/participation/decide',response_model=DecisionResult)
async def decide(r:DecisionRequest):
 start=time.perf_counter()
 identity=dict(source='gateway',decision_id=r.decision_id,session_id=r.context_ref,epoch=r.cancellation_epoch)
 audit.record('request_received',**identity,request=r.model_dump(mode='json'))
 try:
  payload,back=project(r)
  audit.record("upstream_request",**identity,payload=payload,slot_to_companion=back)
  async with asyncio.timeout(min(r.timeout_ms/1000,5)):
   async with app.state.client.stream('POST','http://decider.yangtzeailab.com/d3/v1/systemone',json=payload) as response:
    audit.record("upstream_status",**identity,status_code=response.status_code)
    response.raise_for_status();buf=bytearray()
    async for chunk in response.aiter_bytes():
     buf.extend(chunk)
     if len(buf)>262144:raise ValueError('response_too_large')
   body=json.loads(buf)
   audit.record('upstream_response',**identity,body=body)
   result,p=convert(r,body,payload['questions']['move']['criteria'],back)
  log.warning('decision=%s outcome=%s action=%s confidence=%.3f ms=%.1f',r.decision_id,result.status,result.proposal.action if result.proposal else '-',p,(time.perf_counter()-start)*1000)
  audit.record('result_returned',**identity,result=result.model_dump(mode='json'),confidence=p,elapsed_ms=(time.perf_counter()-start)*1000)
  return result
 except (ValueError,KeyError,TypeError,httpx.HTTPError,TimeoutError) as e:
  audit.record('request_failed',**identity,error_type=type(e).__name__,elapsed_ms=(time.perf_counter()-start)*1000)
  log.warning('decision=%s failed=%s',r.decision_id,type(e).__name__)
  raise HTTPException(503,'decision_unavailable') from None

 except asyncio.CancelledError:
  audit.record('request_cancelled',**identity,elapsed_ms=(time.perf_counter()-start)*1000)
  raise
