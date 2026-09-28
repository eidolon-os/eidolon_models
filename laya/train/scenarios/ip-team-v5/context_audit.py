"""Capacity-only stress from generated text; never used for quality or training."""
import argparse
import copy
import json
from pathlib import Path
from eidolon_sdk.biz.participation import Candidate, Context, DecisionRequest, Message
from eidolon_laya_train.records import read_jsonl
from eidolon_models_laya.sequence import Tokenizer, build_sequence, to_internal, serialize_state, render_options
from eidolon_models_laya.participation_text import project_request, check_capacity
from eidolon_models_laya.participation import ContextTooLong

p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
base=list(read_jsonl(a.dataset/'train.jsonl'))[0]
req=DecisionRequest.model_validate(base.meta['request'])
tok=Tokenizer(Path(__file__).resolve().parents[2]/'runs/r14/checkpoint/tokenizer')
measurements=[]
for members in [2,4,6,16]:
 for turns in [1,9,32,64]:
  candidates=tuple(Candidate(companion_id=f'capacity-{i}',display_name=f'角色{i}',description=req.candidates[i%len(req.candidates)].description) for i in range(members))
  # Repeat generated text to stress lengths, explicitly not realistic new conversation labels.
  messages=[req.user_request]+[Message(message_id=f'probe-{i}',author_kind='companion',author_id=f'capacity-{i%members}',text=req.user_request.text*3) for i in range(1,turns)]
  probe=req.model_copy(update={'candidates':candidates,'trigger':messages[-1],'context':Context(recent_messages=tuple(messages))})
  row={'members':members,'public_messages':turns,'quality_evidence':False}
  try:
   state,q,_=project_request(probe)
   full=len(tok.encode(serialize_state(state)))+len(tok.encode('choice question: '+q['instructions']))+sum(len(tok.encode(' '+o))+1 for o in render_options(to_internal(q)))+4
   row['full_input_tokens']=full
   row['actual_tokens']=check_capacity(tok,state,q)
   row['status']='fits'
  except ContextTooLong as exc:
   row['status']='explicit_rejection';row['reason']=str(exc)
  measurements.append(row)
a.out.write_text(json.dumps({'purpose':'capacity stress only, generated public text repeated; no semantic score or latency claim','cases':measurements},ensure_ascii=False,indent=2))
print(json.dumps(measurements,ensure_ascii=False,indent=2))
