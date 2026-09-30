"""Recorded structural repair before blind review; no new generation or gold editing."""
import json, os, copy
from pathlib import Path
from eidolon_models_jevk5 import generate as g
root=Path('jevk5/private/companion-ip-v1')
for line in Path('laya/train/scenarios/ip-team-v3/.env').read_text().splitlines():
 if line.startswith('EIDOLON_IP_DATA_API_KEY='):
  os.environ.setdefault('EIDOLON_IP_DATA_API_KEY',line.split('=',1)[1].strip().strip('\"\''))
row=next(r for r in g.plan() if r['id']=='companion-dev-0')
original=json.loads((root/'calls/companion-dev-0-gen.json').read_text())['response']
fixed=copy.deepcopy(original)
for scene in fixed['scenes']:scene['cast']=copy.deepcopy(fixed['scenes'][0]['cast'])
scenes=g.check_scenes(fixed,row['members'])
(root/'cast-repair.json').write_text(json.dumps({'id':row['id'],'operation':'copy first returned cast to all snapshots before blind annotation','original':original,'repaired':fixed},ensure_ascii=False,indent=2))
visible=[dict(s,cast=[dict(c,action_code=f'respond:{i}') for i,c in enumerate(s['cast'])]) for s in scenes]
review=g.call(root,row['id']+'-review',g.RULES+'\n独立审核并标注下列公开快照。返回reviews数组，每项仅id、gold（完整可接受动作数组）、valid（布尔）、why（公开依据）、issue（问题或空字符串）。\n'+json.dumps(visible,ensure_ascii=False),os.environ['EIDOLON_IP_DATA_API_KEY'])['reviews']
(root/'reviewed'/f"{row['id']}.json").write_text(json.dumps(dict(plan=row,scenes=scenes,reviews=review,status='pending_local_audit',structural_repair='cast-repair.json'),ensure_ascii=False,indent=2))
print('cast repaired and independently annotated',flush=True)
