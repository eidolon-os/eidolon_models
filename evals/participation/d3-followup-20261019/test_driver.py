import json
from pathlib import Path
import run
DATA=json.loads((Path(__file__).parent/'cases.json').read_text())
class Fake:
 def __init__(self,answers):self.answers=iter(answers);self.states=[];self.replies=0
 def decide(self,state,q,tag):self.states.append(state);return next(self.answers)
 def reply(self,state,action,cid,tag,instruction):self.replies+=1;return '已完成的公开回复。'

def test_failed_clarification_does_not_inject_answer():
 c=Fake(['wait']);case=next(x for x in DATA['episodes'] if x['id']=='clarify_resume')
 r=run.episode(c,case,False,DATA)
 assert not r['passed'] and len(r['turns'])==1 and c.replies==0
 assert len(c.states)==1 and c.states[0]['user_request']=='让他接着说。'

def test_completed_reply_is_next_trigger():
 c=Fake(['respond:M1','finish']);case=next(x for x in DATA['episodes'] if x['id']=='named_once')
 r=run.episode(c,case,False,DATA)
 assert r['passed'] and c.states[1]['trigger']=={'author':'M1','text':'已完成的公开回复。'}
 for s in c.states:assert not {'speakers','terminal','gold','expected'} & set(s)

def test_clarify_is_terminal_before_new_user():
 c=Fake(['clarify:M1','respond:M0','finish']);case=next(x for x in DATA['episodes'] if x['id']=='clarify_resume')
 r=run.episode(c,case,False,DATA)
 assert r['passed'] and len(c.states)==3
 assert c.states[1]['user_request']==case['turns'][1]['user']
 assert any(m['author']=='M1' for m in c.states[1]['prior_public_messages'])
