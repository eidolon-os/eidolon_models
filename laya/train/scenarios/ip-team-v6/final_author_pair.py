"""Last bounded schema clarification: author fields are IDs, never display names."""
import importlib.util
import json
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('shared_final',HERE/'generate_shared.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
g=s.g
out=g.TRAIN/'private/ip-team-v6-pairs'
fid='v6t_val_author'
g.v5.legacy.load_env(HERE.parent/'ip-team-v3/.env')
assert os.environ['EIDOLON_IP_DATA_BASE_URL']=='https://open.bigmodel.cn/api/paas/v4'
assert os.environ['EIDOLON_IP_DATA_MODEL']=='glm-5.3-flash'
client=s.ChatClient({'base_url':os.environ['EIDOLON_IP_DATA_BASE_URL'],'model':'glm-5.3-flash','api_key_env':'EIDOLON_IP_DATA_API_KEY','timeout':180,'max_tokens':4500,'thinking':{'type':'enabled'},'reasoning_effort':'low','response_format':{'type':'json_object'}})
prompt=g.RULES+'''\n生成沙海古城旅伴的一对快照。两分支人物资料、user、peer_text完全相同；只有peer_author_x与peer_author_y不同。
user要求刚发现物品的人继续讲发现经过。peer_text是发现者的第一人称简短发现事实一句话，不自报姓名，不请别人发言，不把物品拟人化，不说明讲述已完成。随后进行下一位决策。
name_a/name_b/name_c是三位不同的中文完整名字；role_a/role_b/role_c是人格关系，不能预先指定谁发现物品。
peer_author_x必须为字符串a，peer_author_y必须为字符串b（作者ID，不能填写中文姓名）。acceptable_x/acceptable_y写完整动作代码，如respond:a，禁止只有respond或姓名；由你依据公开输入判断标签。其余字段均为字符串。
只返回平面JSON，字段恰好为：'''+json.dumps(sorted(s.fields_for('peer_author')),ensure_ascii=False)
raw=g.v5.paid(client,out,fid+'-gen',prompt,32)
row={'id':fid,'split':'val','slice':'author','schema':'shared_fields_v3_author_id_explicit','generator_sha256':g.v5.sha(raw)}
try:d=s.materialize(raw,'author','peer_author')
except (ValueError,TypeError) as e:row.update(status='rejected_structure',reason=str(e))
else:
 visible={k:v for k,v in d.items() if not k.startswith(('acceptable','reason'))}
 rp=g.RULES+'\n仅根据公开输入判断两个快照，不看生成标签。返回四个字符串字段acceptable_x,reason_x,acceptable_y,reason_y。\n'+json.dumps(visible,ensure_ascii=False)
 rr=g.v5.paid(client,out,fid+'-review',rp,32)
 try:
  r=json.loads(rr)
  if set(r)!=s.ANNOT:raise ValueError('review fields')
  for b in 'xy':
   r['acceptable_'+b]=g.v5.moves(r['acceptable_'+b],'abc')
   if not isinstance(r['reason_'+b],str) or not r['reason_'+b].strip():raise ValueError('review reason')
  status='agreed_pending_audit' if all(d['acceptable_'+b]==r['acceptable_'+b] for b in 'xy') else 'quarantined_disagreement'
 except (ValueError,TypeError):r={};status='rejected_review_structure'
 row.update(pair=d,review=r,status=status,reviewer_sha256=g.v5.sha(rr))
g.v5.legacy.atomic_json(out/'reviewed'/f'{fid}.json',row)
print(fid,row['status'],flush=True)
