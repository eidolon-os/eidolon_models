"""One generated story pair per call, with a label-blind second annotation."""
import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse
from eidolon_laya_train.generators import ChatClient
HERE=Path(__file__).resolve().parent
TRAIN=HERE.parents[1]
spec=importlib.util.spec_from_file_location('v5_io',HERE.parent/'ip-team-v5/generate_glm.py')
v5=importlib.util.module_from_spec(spec);spec.loader.exec_module(v5)
RULES='''标注故事人物本人的文本对话调度，不是制作团队或职业顾问。
每次一个下一步；可接受集合可有多项，但每项须有公开文本依据。
acceptable只能使用respond:a、respond:b、respond:c、wait、finish、abstain，多个用英文逗号连接，绝不单独写respond。
respond:a/b/c让相应人物回应。明确点名优先；爱插嘴不等于许可；提及名字不等于点名。
wait只用于暂停等用户；finish用于已经满足或明确结束。普通新问题不能选这两项。
abstain用于无法分辨被叫到的同名人物；人物普通闲聊追问可respond。
禁止一切回复时不能respond；禁声但请求文字时可以respond。不能捏造人物经历。'''
SPECS=[
('named','只改user：x明确仅请a回应，y明确仅请b回应。自然使用人物名字，不写a/b。','user'),
('handoff','user相同，请人物一起讨论。只改peer_text：a发言后，x邀请b接话，y邀请c接话。','peer_text'),
('continue_done','user相同，请a一次讲完短故事。只改peer_text：x是a未讲完的故事，y是a已讲完整且不留追问。','peer_text'),
('pause_discuss','只改user：x要求所有人完全暂停回复等用户下一条；y用户旁听让大家继续聊。','user'),
('end_pause','只改user：x明确本轮结束且不必回话，y暂时不要回话等待用户下一条。','user'),
('same_name','a与b同名但人格关系不同。只改user：x只叫名字无法区分；y用公开的具体人格或关系线索唯一指向b。','user'),
('quiet','只改user：x禁声但请求文字陪伴；y暂时连文字也不要回复等用户下一条。','user'),
('author','user相同，要求刚找到物品的本人继续讲经过。peer_text相同，是发现者第一人称简短发言。只改peer_author：x为a，y为b；资料不能预先锁定发现者。','peer_author')]
FIELDS={**{f'name_{s}':'人物中文名字' for s in 'abc'},**{f'role_{s}':'人格与相互关系，不是职业能力分工' for s in 'abc'}}
for b in 'xy':
 FIELDS.update({f'user_{b}':'起始用户原话',f'peer_author_{b}':'最新公开人物发言作者a/b/c；无则空',f'peer_text_{b}':'公开发言原文；无则空',f'acceptable_{b}':'逗号分隔动作集合',f'reason_{b}':'选择的可见依据'})


def prompt(focus,theme):
 return RULES+f'\n原创一对{theme}中的故事快照。约束：{focus}\n两分支共用人物资料，只改指定字段。peer存在时顺序为user然后peer，之后立即决策。所有人物文本使用你生成的完整中文名字，不用昵称、缩写或字母槽位。x/y只是字段后缀，不是用户句子，必须写真实自然的原话。只输出一个平面JSON，字段恰好为：\n'+json.dumps(FIELDS,ensure_ascii=False)


def validate(raw,sid):
 d=json.loads(raw)
 if not isinstance(d,dict) or set(d)!=set(FIELDS) or not all(isinstance(v,str) for v in d.values()):raise ValueError('exact flat string fields required')
 names={s:d[f'name_{s}'] for s in 'abc'}
 if any(not 2<=len(n)<=8 or re.search('[{}a-zA-Z]',n) for n in names.values()):raise ValueError('invalid names')
 if sid=='same_name':
  if not names['a']==names['b']!=names['c']:raise ValueError('a=b !=c required')
 elif len(set(names.values()))!=3:raise ValueError('distinct names required')
 if any(x!=y and x in y for x in names.values() for y in names.values()):raise ValueError('overlapping names')
 for s in 'abc':
  if not 4<=len(d[f'role_{s}'])<=90:raise ValueError('profile length')
 for b in 'xy':
  if not 4<=len(d[f'user_{b}'])<=220 or not 4<=len(d[f'reason_{b}'])<=500:raise ValueError('user/reason length')
  au,tx=d[f'peer_author_{b}'],d[f'peer_text_{b}']
  if bool(au)!=bool(tx) or au and au not in ('a','b','c') or len(tx)>350:raise ValueError('invalid peer')
  d[f'acceptable_{b}']=v5.moves(d[f'acceptable_{b}'],'abc')
 changed=[k for k in ('user','peer_author','peer_text') if d[f'{k}_x']!=d[f'{k}_y']]
 intended=next(s[2] for s in SPECS if s[0]==sid)
 if changed!=[intended]:raise ValueError(f'changed {changed}; expected only {intended}')
 if any(re.search(r'[{}]|(?<![a-zA-Z])[abc](?![a-zA-Z])',v) for k,v in d.items() if k.startswith(('role_','user_','peer_text_'))):raise ValueError('unrendered slot in text')
 return d


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=1);a=ap.parse_args()
 if not 1<=a.count<=4:ap.error('1–4 families per invocation')
 out=TRAIN/'private/ip-team-v6-pairs';out.mkdir(parents=True,exist_ok=True)
 v5.legacy.load_env(HERE.parent/'ip-team-v3/.env')
 url=os.environ['EIDOLON_IP_DATA_BASE_URL']
 if urlparse(url).hostname!='open.bigmodel.cn' or not url.startswith('https://') or not url.endswith('/api/paas/v4'):raise ValueError('official HTTPS required')
 if os.environ['EIDOLON_IP_DATA_MODEL']!='glm-5.3-flash':raise ValueError('model mismatch')
 client=ChatClient({'base_url':url,'model':'glm-5.3-flash','api_key_env':'EIDOLON_IP_DATA_API_KEY','timeout':180,'max_tokens':4500,'thinking':{'type':'enabled'},'reasoning_effort':'high','response_format':{'type':'json_object'}})
 plan={'version':'v6-pairs-20260928','max_paid_calls':32,'specs':SPECS,'worlds':{'train':'漂浮书岛的童话旅伴','val':'沙海古城的星夜旅伴'}}
 pp=out/'plan.json'
 if pp.exists() and json.loads(pp.read_text())!=json.loads(json.dumps(plan)):raise ValueError('immutable plan mismatch')
 v5.legacy.atomic_json(pp,plan)
 done=0
 for split,theme in plan['worlds'].items():
  for sid,focus,changed in SPECS:
   fid=f'v6_{split}_{sid}';dest=out/'reviewed'/f'{fid}.json'
   if dest.exists():continue
   raw=v5.paid(client,out,fid+'-gen',prompt(focus,theme),32)
   result={'id':fid,'split':split,'slice':sid,'generator_sha256':v5.sha(raw)}
   try:d=validate(raw,sid)
   except (ValueError,TypeError) as e:result.update(status='rejected_structure',reason=str(e))
   else:
    visible={k:v for k,v in d.items() if not k.startswith(('acceptable','reason'))}
    rp=RULES+'\n独立判断两个快照。只返回JSON四字段acceptable_x,reason_x,acceptable_y,reason_y。acceptable为逗号分隔动作字符串。不得添加未见经历。\n'+json.dumps(visible,ensure_ascii=False)
    review_raw=v5.paid(client,out,fid+'-review',rp,32)
    try:
     review=json.loads(review_raw)
     if set(review)!={'acceptable_x','reason_x','acceptable_y','reason_y'}:raise ValueError('review fields')
     for b in 'xy':
      review[f'acceptable_{b}']=v5.moves(review[f'acceptable_{b}'],'abc')
      if not isinstance(review[f'reason_{b}'],str) or not review[f'reason_{b}'].strip():raise ValueError('review reason')
     status='agreed_pending_audit' if all(review[f'acceptable_{b}']==d[f'acceptable_{b}'] for b in 'xy') else 'quarantined_disagreement'
    except (ValueError,TypeError):review={};status='rejected_review_structure'
    result.update(pair=d,review=review,status=status,reviewer_sha256=v5.sha(review_raw))
   v5.legacy.atomic_json(dest,result);print(fid,result['status'],result.get('reason',''),flush=True)
   done+=1
   if done==a.count:return

if __name__=='__main__':main()
