import copy,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
old=[json.loads(x) for x in (ROOT/'evals/participation/product-readiness-v1/cases.jsonl').read_text().splitlines()]
POLICY=old[0]['questions']['move']['instructions']
CLARIFY='根据本轮用户请求和公开上下文，询问一个阻碍继续的缺失信息；不要替用户猜测，不回答尚未明确的请求，问完等待用户。'
CAST={'M0':{'name':'阿舟','character':'喜欢散步与自然观察，表达简洁。'},'M1':{'name':'小岚','character':'喜欢美食和城市探索，愿意回应他人的具体观点。'}}

def question(cast,reverse=False):
 c={}
 for action,verb in [('respond','回应'),('clarify','提出一个澄清问题后等待用户')]:
  for cid,v in cast.items():c[f'{action}:{cid}']=f"由{v['name']}（{cid}）{verb}"
 c.update(wait='暂停发言，等待用户',finish='本轮请求已完成或用户明确结束',abstain='无法可靠作出允许的决策，不产生任何回复')
 return {'move':{'type':'choice','instructions':POLICY,'criteria':dict(reversed(list(c.items()))) if reverse else c}}

def state(user,history=(),trigger=None,cast=None):
 return {'user_request':user,'scene_goal':'','candidates':copy.deepcopy(cast or CAST),
 'allowed_actions':['respond','clarify','wait','finish'],'remaining_replies':3,
 'prior_public_messages':[{'author':a,'text':t} for a,t in history],
 'trigger':{'same_as_user_request':True} if trigger is None else {'author':trigger[0],'text':trigger[1]}}

def static_cases():
 contexts=[
 ('pronoun_missing',state('让他接着说。'),['clarify:M0','clarify:M1']),
 ('pronoun_resolved',state('我是说阿舟，继续说一个散步建议。',[('M1','你希望哪位成员继续？')]),['respond:M0']),
 ('intent_missing',state('帮我把那个安排一下。'),['clarify:M0','clarify:M1']),
 ('intent_resolved',state('阿舟，我指的是周末散步，给我一个轻松的路线建议。',[('M0','你想安排哪件事？')]),['respond:M0']),
 ('history_missing',state('请刚才介绍过路线的人继续补充。'),['clarify:M0','clarify:M1']),
 ('history_resolved',state('请刚才介绍过路线的人继续补充。',[('M1','我介绍的是河滨步道，沿途有树荫。')]),['respond:M1']),
 ('clarification_wait',state('让他接着说。',trigger=('M0','你希望哪位成员继续？')),['wait']),
 ('clarification_answer',state('让小岚补充一个餐厅建议。',[('M0','你希望哪位成员继续？')]),['respond:M1'])]
 out=[]
 for fid,s,gold in contexts:
  for variant in ('original','option_reverse','candidate_reverse','renamed'):
   t=copy.deepcopy(s)
   if variant=='candidate_reverse':t['candidates']=dict(reversed(list(t['candidates'].items())))
   if variant=='renamed':
    raw=json.dumps(t,ensure_ascii=False).replace('阿舟','青禾').replace('小岚','星野')
    t=json.loads(raw)
   out.append({'id':fid+':'+variant,'family':fid,'variant':variant,'state':t,
     'questions':question(t['candidates'],variant=='option_reverse'),'gold':gold})
 return out

def episodes():
 return [
 {'id':'named_once','turns':[{'user':'只请小岚给一个午餐建议，一句话就够，说完本轮就结束。','speakers':['M1'],'terminal':'finish'}]},
 {'id':'handoff','turns':[{'user':'阿舟先用一句话提出一个散步地点，小岚再用一句话评价他的具体建议，之后结束本轮。','speakers':['M0','M1'],'terminal':'finish'}]},
 {'id':'same_speaker','turns':[{'user':'只请阿舟分两次回答，每次一句：先建议一个散步地点，再补充要带什么。两次说完结束，不需要小岚发言。','speakers':['M0','M0'],'terminal':'finish'}]},
 {'id':'clarify_resume','turns':[{'user':'让他接着说。','speakers':None,'terminal':'clarify'},
   {'user':'我指的是阿舟，请他给一个散步建议，一句话说完就结束。','speakers':['M0'],'terminal':'finish'}]},
 {'id':'pause_resume','turns':[{'user':'先别说话，我想一想，等我说可以了再开口。','speakers':[],'terminal':'wait'},
   {'user':'可以了，只请小岚给一个午餐建议，一句就够，说完结束本轮。','speakers':['M1'],'terminal':'finish'}]},
 {'id':'explicit_finish','turns':[{'user':'这轮就到这里，不需要任何人再回复了。','speakers':[],'terminal':'finish'}]}]

def main():
 data={'policy':POLICY,'clarify_instruction':CLARIFY,'static':static_cases(),'episodes':episodes()}
 p=HERE/'cases.json';s=json.dumps(data,ensure_ascii=False,indent=2)
 if p.exists():assert p.read_text()==s
 else:p.write_text(s)
 assert len(data['static'])==32 and len(data['episodes'])==6
 print('32 stability cases; 6 episodes x 2 option orders')
if __name__=='__main__':main()
