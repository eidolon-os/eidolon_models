"""Small authored contract screen, not an independent generalization benchmark."""
import json,copy
from pathlib import Path
HERE=Path(__file__).resolve().parent
POLICY='选择下一步参与决策，只用公开事实。respond选择一位合适角色；开放问题可有多个合适人选，不机械轮换。用户明确要求暂不说话选wait；原请求已完成或明确结束本轮选finish，不能因角色刚说完一句就自动结束。需要补足意图/身份且允许clarify时，选择一位角色澄清；若不能可靠选择且不允许澄清则abstain，不猜人。澄清发出后应wait等用户回答；新用户输入可重新回应。clarify仅决定谁询问缺失信息，问题由其他大模型生成。'
CLARIFY='根据本轮用户请求和公开上下文，询问一个阻碍继续的缺失信息；不要替用户猜测，不回答尚未明确的请求，问完等待用户。'
solo={'M0':{'name':'小禾','character':'温和的日常陪伴者，愿意倾听和一起讨论生活。'}}
team={'M0':{'name':'阿舟','character':'喜欢散步和自然观察，讲话简洁。'},'M1':{'name':'小岚','character':'喜欢美食与城市探索，善于补充实际细节。'}}
rows=[]
def add(family,task,step,user,gold,history=(),trigger=None,cast=None,allowed=None):
 c=copy.deepcopy(cast or (solo if task=='companion' else team))
 allowed=allowed or ['respond','clarify','wait','finish']
 state={'user_request':user,'scene_goal':'','candidates':c,'allowed_actions':allowed,'remaining_replies':8,
        'prior_public_messages':[{'author':a,'text':t} for a,t in history],
        'trigger':{'same_as_user_request':True} if trigger is None else {'author':trigger[0],'text':trigger[1]}}
 choices={}
 for action in ('respond','clarify'):
  if action in allowed:
   for cid,v in c.items():choices[f'{action}:{cid}']=f"由{v['name']}（{cid}）"+('回应' if action=='respond' else '提出一个澄清问题后等待用户')
 for action,desc in [('wait','暂停发言，等待用户'),('finish','本轮请求已完成或用户明确结束')]:
  if action in allowed:choices[action]=desc
 choices['abstain']='无法可靠作出允许的决策，不产生任何回复'
 rows.append({'id':f'{family}:{step}','state':state,'questions':{'move':{'type':'choice','instructions':POLICY,'criteria':choices}},
  'labels':{'move':{'gold':gold}},'meta':{'family':family,'task':task,'step':step,'origin':'assistant_authored_contract_screen','forbid_speech':gold in (['wait'],['finish'])}})
add('c1','companion',0,'今天下班路上看到一只胖橘猫，好想和你聊聊。',['respond:M0'])
add('c1','companion',1,'它一直跟着我走，你觉得是不是想讨吃的？',['respond:M0'],[('M0','它当时在做什么呀？')])
add('c2','companion',0,'我有点累，你安静陪着我就好，可以说一句让我安心的话。',['respond:M0'])
add('c2','companion',1,'现在先别说话，我想闭眼休息一会儿。',['wait'],[('M0','我在这里，慢慢休息。')])
add('c3','companion',0,'我接个电话，等我说好了你再开口。',['wait'])
add('c3','companion',1,'好了，我们继续聊晚饭吃什么吧。',['respond:M0'])
add('c4','companion',0,'今天先聊到这儿，晚安。',['finish'])
add('c4','companion',1,'等等，我又想到一个问题，明早怎么才能不赖床？',['respond:M0'])
add('c5','companion',0,'帮我把那个安排一下。',['clarify:M0'])
add('c5','companion',1,'我说的是周末散步，想讨论一条轻松的路线。',['respond:M0'],[('M0','你想安排哪件事？')])
u='分两次说，每次一个建议：先讲如何准备睡觉，再讲如何让卧室安静。'
add('c6','companion',0,u,['respond:M0'])
add('c6','companion',1,u,['respond:M0'],trigger=('M0','睡前把手机放远一点，留几分钟放松。'))
u='只请小岚说一个周末吃饭的建议，一句就够。'
add('t1','ip_team',0,u,['respond:M1'])
add('t1','ip_team',1,u,['finish'],trigger=('M1','周末可以去河边那家面馆吃一碗热汤面。'))
u='阿舟先提出一个散步地点，小岚再评价他的具体建议。'
add('t2','ip_team',0,u,['respond:M0'])
add('t2','ip_team',1,u,['respond:M1'],trigger=('M0','我建议去湿地木栈道，下午可以看水鸟。'))
u='阿舟分两次说，先提出一个散步地点，再补充出发前需要带什么。'
add('t3','ip_team',0,u,['respond:M0'])
add('t3','ip_team',1,u,['respond:M0'],trigger=('M0','我们可以沿着湖边的步道走一圈。'))
u='你们谁来给我一个轻松的周末建议都行，只要一位说一句就够。'
add('t4','ip_team',0,u,['respond:M0','respond:M1'])
add('t4','ip_team',1,u,['finish'],trigger=('M1','可以去老街吃午饭，顺便逛逛小店。'))
add('t5','ip_team',0,'让他接着说。',['clarify:M0','clarify:M1'])
add('t5','ip_team',1,'我是说阿舟，继续补充你刚才的散步建议。',['respond:M0'],[('M0','湖边很适合散步。'),('M1','你希望哪位成员继续？')])
u='请刚才介绍过湿地路线的人，继续补充那条路线的细节。'
add('t6','ip_team',0,u,['abstain'],allowed=['respond','wait','finish'])
add('t6','ip_team',1,u,['respond:M1'],[('M1','我刚走过湿地木栈道，可以看到芦苇和水鸟。')],allowed=['respond','wait','finish'])
assert len(rows)==24 and len({r['meta']['family'] for r in rows})==12
for r in rows:assert set(r['labels']['move']['gold'])<=set(r['questions']['move']['criteria'])
p=HERE/'cases.jsonl'
text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
if p.exists():assert p.read_text()==text
else:p.write_text(text)
print('24 snapshots / 12 families frozen')
