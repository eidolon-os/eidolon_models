import json
import pytest
from gateway import project,convert
from eidolon_sdk.biz.participation import DecisionRequest

def req():
 return DecisionRequest.model_validate(dict(decision_id='test',context_ref='ctx',context_version=1,membership_revision=1,cancellation_epoch=0,
 user_request=dict(message_id='u',author_kind='user',author_id='owner',text='小岚说一句'),
 trigger=dict(message_id='u',author_kind='user',author_id='owner',text='小岚说一句'),
 context=dict(recent_messages=[dict(message_id='u',author_kind='user',author_id='owner',text='小岚说一句')]),
 candidates=[dict(companion_id='real-id',display_name='小岚',description='角色')],timeout_ms=1500))
def test_mapping():
 r=req();body,back=project(r);cs=body['questions']['move']['criteria']
 for pred in cs:
  response={'answers':{'move':{'choice':pred,'probabilities':{k:float(k==pred) for k in cs}}}}
  result,p=convert(r,response,cs,back)
  assert result.context_version==r.context_version
  if pred.startswith(('respond:','clarify:')):assert result.proposal.participants==('real-id',)
  if pred.startswith('clarify:'):assert result.proposal.instruction
  if pred=='abstain':assert result.proposal is None
 assert 'owner' not in json.dumps(body)
def test_invalid_answer():
 r=req();body,back=project(r)
 with pytest.raises(ValueError):convert(r,{'answers':{'move':{'choice':'evil','probabilities':{}}}},body['questions']['move']['criteria'],back)

def test_audit_roundtrip_without_network(tmp_path, monkeypatch):
 import asyncio, httpx
 import gateway
 monkeypatch.setenv('EIDOLON_PARTICIPATION_AUDIT_DIR',str(tmp_path/'audit'))
 r=req();payload,_=project(r);choices=payload['questions']['move']['criteria']
 pred=next(k for k in choices if k.startswith('respond:'))
 body={'answers':{'move':{'choice':pred,'probabilities':{k:float(k==pred) for k in choices}}}}
 async def run():
  async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(200,json=body))) as client:
   monkeypatch.setattr(gateway.app.state,'client',client,raising=False)
   assert (await gateway.decide(r)).proposal.participants==('real-id',)
 asyncio.run(run())
 records=[json.loads(l) for p in (tmp_path/'audit').glob('*.jsonl') for l in p.read_text().splitlines()]
 assert {v['decision_id'] for v in records}=={'test'}
 assert [v['event'] for v in records]==['request_received','upstream_request','upstream_status','upstream_response','result_returned']
 assert records[1]['slot_to_companion']=={'M0':'real-id'}
 assert records[-2]['body']==body
