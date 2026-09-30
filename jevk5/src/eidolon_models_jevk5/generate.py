"""Bounded synthetic-only GLM generation and blind annotation; no source uploads."""
import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from .data import digest

RULES = '''任务是智能陪伴或故事IP角色团的参与调度，不是客服、专家路由，也不生成最终陪伴回复。
允许动作respond:角色索引、wait、finish、abstain。用户向角色说话且需要回应，应选回应者；开放聊天可有多个可接受角色，完整列出。
仅提到某人不等于要求其发言。暂停等用户继续是wait；明确本轮结束是finish。要求安静陪伴但允许简短文字确认可以回应，明确完全别回复则wait。
指定某个人时只能依据公开身份线索和历史确定。多人符合且不能唯一确定时abstain；即使同名，有充分线索也应回应。不因职业匹配、排列顺序、猜测偏好来消歧。
只用公开事实，不能把别的假设快照当历史；不要求人物机械轮流。每条的可接受答案可能不唯一。'''


def plan():
    rows=[]
    for split,n in [('train',4),('dev',2)]:
        for task in ['companion','ip_team']:
            for i in range(n):
                rows.append(dict(id=f'{task}-{split}-{i}',split=split,task=task,
                    members=(1 if i%2==0 else 2) if task=='companion' else 3+i%3,
                    topic=(['失眠后的闲聊','散步回来','练习画画','收拾旧物'][i] if split=='train' else ['第一次参加读书会','修好一把旧雨伞'][i]),
                    style=('自然口语与直接表达' if split=='train' else '间接但可理解的表达，或先纠正对方再说明意思')))
    return rows


def prompt(row):
    multi = row['members'] > 1
    return RULES + f'''
原创一个{row['task']}场景家族，{row['members']}位具有人格的陪伴角色，话题{row['topic']}，表达{row['style']}。
四个独立假设情境共享同一个cast。第一条需要回应，第二条临时完全暂停，第三条本轮明确结束。
第四条{'指定人物但身份线索不足，需要澄清；第一条则补足身份线索让回应者唯一。可用同昵称或关系代词，勿只靠职业匹配。' if multi else '用户希望安静陪着但明确允许一句简短回应，用来区别于完全暂停。'}
允许1至4条相关公开历史消息，至少一条情境需要历史才能判断；单角色也可以用历史说明刚才在跟谁说话。
不要总用同一句暂停或结束模板。故事背景和人物姓名必须原创，用户原话15至100字。不要把答案写进人物资料。
返回JSON scenes数组四项，id为s0至s3，但顺序打乱；每项仅id、cast、user、before。
cast每项仅name和persona，persona含自然性格或关系；before是公开历史数组，每项仅speaker（用户为-1，角色为从0起的索引）和text。
不要输出gold、why、标签、计划或未来回复。'''


def check_scenes(value, members):
    scenes=value['scenes']
    if len(scenes)!=4 or {s['id'] for s in scenes}!={'s0','s1','s2','s3'}:
        raise ValueError('four scenes required')
    for s in scenes:
        if set(s)!={'id','cast','user','before'} or len(s['cast'])!=members:
            raise ValueError('scene schema')
        if s['cast']!=scenes[0]['cast'] or not isinstance(s['user'],str) or not 4<=len(s['user'])<=240:
            raise ValueError('shared cast or text')
        for c in s['cast']:
            if set(c)!={'name','persona'} or not all(isinstance(v,str) and v.strip() for v in c.values()):
                raise ValueError('cast schema')
        for m in s['before']:
            if set(m)!={'speaker','text'} or type(m['speaker']) is not int or not -1<=m['speaker']<members or not isinstance(m['text'],str):
                raise ValueError('history schema')
    return scenes


def call(root, name, text, key):
    calls=root/'calls';calls.mkdir(parents=True,exist_ok=True)
    marker=calls/f'{name}.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if old.get('response') is not None:return old['response']
        raise RuntimeError('ambiguous prior attempt; no automatic retry')
    if len(list(calls.glob('*.json')))>=24:raise RuntimeError('24-call cap reached')
    request={'model':'glm-5.3-flash','messages':[{'role':'user','content':text}],
             'temperature':.7,'max_tokens':5000,'thinking':{'type':'enabled'},
             'reasoning_effort':'low','response_format':{'type':'json_object'}}
    ledger={'request':request,'started':time.time(),'status':'started'}
    marker.write_text(json.dumps(ledger,ensure_ascii=False,indent=2))
    req=urllib.request.Request('https://open.bigmodel.cn/api/paas/v4/chat/completions',
        data=json.dumps(request,ensure_ascii=False).encode(),
        headers={'Content-Type':'application/json','Authorization':'Bearer '+key},method='POST')
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req,timeout=180) as response:body=json.load(response)
        result=json.loads(body['choices'][0]['message']['content'])
        ledger.update(response=result,usage=body.get('usage'),provider_id=body.get('id'),status='complete')
        return result
    except Exception as e:
        ledger.update(status='failed',error_type=type(e).__name__)
        raise
    finally:
        ledger['finished']=time.time();marker.write_text(json.dumps(ledger,ensure_ascii=False,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--env-file',type=Path);ap.add_argument('--preview',action='store_true');a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    manifest={'plan':plan(),'limit':24,'code_sha256':digest(__file__)}
    path=a.out/'plan.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('immutable generation plan changed')
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    if a.preview:
        print(RULES+'\n\n'+prompt(plan()[0]));return
    if a.env_file:
        for line in a.env_file.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                k,v=line.split('=',1)
                if k.strip()=='EIDOLON_IP_DATA_API_KEY':os.environ.setdefault(k.strip(),v.strip().strip('\"\''))
    key=os.environ['EIDOLON_IP_DATA_API_KEY']
    reviewed=a.out/'reviewed';reviewed.mkdir(exist_ok=True)
    for row in plan():
        dest=reviewed/(row['id']+'.json')
        if dest.exists():continue
        raw=call(a.out,row['id']+'-gen',prompt(row),key)
        scenes=check_scenes(raw,row['members'])
        visible=[dict(s,cast=[dict(c,action_code=f'respond:{i}') for i,c in enumerate(s['cast'])]) for s in scenes]
        text=RULES+'\n独立审核并标注下列公开快照，不知道生成计划。返回reviews数组，每项仅id、gold（完整可接受动作数组）、valid（布尔）、why（公开依据）、issue（问题或空字符串）。\n'+json.dumps(visible,ensure_ascii=False)
        review=call(a.out,row['id']+'-review',text,key)['reviews']
        if len(review)!=4 or {r['id'] for r in review}!={s['id'] for s in scenes}:raise ValueError('review ids')
        for r in review:
            if set(r)!={'id','gold','valid','why','issue'} or type(r['valid']) is not bool:raise ValueError('review schema')
            allowed={f'respond:{i}' for i in range(row['members'])}|{'wait','finish','abstain'}
            if not r['gold'] or not set(r['gold'])<=allowed:raise ValueError('review actions')
        dest.write_text(json.dumps(dict(plan=row,scenes=scenes,reviews=review,status='pending_local_audit'),ensure_ascii=False,indent=2))
        print(row['id'],'pending_local_audit',flush=True)


if __name__=='__main__':main()
