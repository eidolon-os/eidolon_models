import copy,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];PREV=HERE.parent/'d3-followup-20261019'
sys.path.insert(0,str(PREV))
from build import question,state
ADDITION='\n补充决策原则（不改变上述动作定义）：依据原始请求、最新触发和公开历史判断下一步。信息已足以继续时正常回应，不因无关细节缺失而澄清。如果缺失的信息会影响回应者或任务选择，且用户可以补充、当前允许澄清，应主动澄清，不自行猜测。用户明确要求暂停，或公开上下文明确表明已提出澄清且尚未收到回答时，等待用户，不要重复询问；不能仅因最后一句是问句就认定正在等待澄清。原请求已经完成时结束本轮。只有无法可靠作出上述允许的决策时才弃权。'
CAST={'M0':{'name':'秋岑','character':'喜欢手工和阅读，耐心解释。'},'M1':{'name':'远帆','character':'喜欢运动与户外活动，表达简洁。'},'M2':{'name':'知夏','character':'喜欢摄影和展览，乐于讨论。'}}
def s(user,history=(),trigger=None):return state(user,history,trigger,CAST)
def main():
 old=json.loads((PREV/'cases.json').read_text())
 cases=[]
 for r in old['static']:cases.append(dict(r,suite='old_diagnostic'))
 seeds=[
 ('missing_options',s('两个方案你替我挑一个吧。'),['clarify:M0','clarify:M1','clarify:M2']),
 ('specified_options',s('知夏，周末活动两个方案：室内摄影展或户外骑车。我想避雨，请给一个选择理由。'),['respond:M2']),
 ('unknown_responsibility',s('请负责整理行程的那位，继续说明安排。'),['clarify:M0','clarify:M1','clarify:M2']),
 ('known_responsibility',s('请负责整理行程的那位，继续说明安排。',[('M2','行程我来整理，先到展馆，再去咖啡店。')]),['respond:M2']),
 ('open_chat',s('你们谁都可以，说一个让周末轻松一点的小主意。'),['respond:M0','respond:M1','respond:M2']),
 ('irrelevant_details',s('秋岑，给刚开始画画的人一个通用建议就好，不用先问预算和画材。'),['respond:M0']),
 ('explicit_silence',s('先保持安静，我要听完这一段录音，等我叫你们再说。'),['wait']),
 ('fulfilled',s('只请远帆用一句话提醒我出门带什么，说完本轮就结束。',trigger=('M1','记得带水和雨伞。')),['finish']),
 ('awaiting_answer',s('替我把需要的东西列一下。',trigger=('M0','你要为哪项活动准备东西？')),['wait']),
 ('question_not_clarification',s('秋岑先讲阅读的好处，再请知夏回应秋岑的看法。',trigger=('M0','读书能让人接触不同生活，你们觉得呢？')),['respond:M2']),
 ('clarify_disallowed',s('请刚才分享了展览信息的那位继续介绍。'),['abstain']),
 ('answer_received',s('我准备去露营，远帆只需告诉我一件最重要的装备。',[('M0','你要为哪项活动准备东西？')]),['respond:M1'])]
 for fid,st,gold in seeds:
  if fid=='clarify_disallowed':st['allowed_actions'].remove('clarify')
  for rev in (False,True):
   q=question(st['candidates'],rev)
   if fid=='clarify_disallowed':q['move']['criteria']={k:v for k,v in q['move']['criteria'].items() if not k.startswith('clarify:')}
   cases.append({'id':fid+(':reverse' if rev else ':original'),'family':fid,'variant':'reverse' if rev else 'original',
    'suite':'new_diagnostic','state':st,'questions':q,'gold':gold})
 episodes=[
 {'id':'pronoun_clarification','turns':[{'user':'麻烦让那位继续介绍吧。','speakers':None,'terminal':'clarify'},
  {'user':'我指的是知夏。请给一个参观摄影展的小建议，一句说完后结束。','speakers':['M2'],'terminal':'finish'}]},
 {'id':'missing_options_clarification','turns':[{'user':'这两个我选哪一个比较好？','speakers':None,'terminal':'clarify'},
  {'user':'我在选画画或跑步，想在室内放松。只请秋岑给一个选择和理由，一句说完结束。','speakers':['M0'],'terminal':'finish'}]},
 {'id':'normal_response','turns':[{'user':'只请远帆给一个开始跑步的通用建议，不用询问我的情况，一句话说完结束。','speakers':['M1'],'terminal':'finish'}]}]
 obj={'addition':ADDITION,'cast':CAST,'cases':cases,'episodes':episodes,'clarify_instruction':old['clarify_instruction']}
 assert len(cases)==56
 for c in cases:assert set(c['gold'])<=set(c['questions']['move']['criteria'])
 p=HERE/'cases.json';text=json.dumps(obj,ensure_ascii=False,indent=2)
 if p.exists():assert p.read_text()==text
 else:p.write_text(text)
 (HERE/'PROMPT-ADDITION.txt').write_text(ADDITION.strip()+'\n')
 print('56 static cases, two policies; three live episodes per policy')
if __name__=='__main__':main()
