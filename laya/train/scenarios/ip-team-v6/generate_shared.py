"""Shared fields occur once on the wire; projection duplicates no generated facts."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
from eidolon_laya_train.generators import ChatClient
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('pair_rules',HERE/'generate_pairs.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
BASE={f'{k}_{c}' for k in ('name','role') for c in 'abc'}
ANNOT={f'{k}_{b}' for k in ('acceptable','reason') for b in 'xy'}

def fields_for(changed):
    shared={'user','peer_author','peer_text'}-{changed}
    # User-triggered pair has no generated peer turn at all.
    if changed=='user':shared=set()
    return BASE|ANNOT|shared|{f'{changed}_x',f'{changed}_y'}


def materialize(raw,sid,changed):
    d=json.loads(raw)
    if not isinstance(d,dict) or set(d)!=fields_for(changed):raise ValueError('shared-schema fields mismatch')
    full={k:v for k,v in d.items() if k in BASE|ANNOT}
    for key in ('user','peer_author','peer_text'):
        for b in 'xy':full[f'{key}_{b}']=d.get(f'{key}_{b}',d.get(key,''))
    return g.validate(json.dumps(full,ensure_ascii=False),sid)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=1);a=ap.parse_args()
    if not 1<=a.count<=4:ap.error('1–4 families')
    out=g.TRAIN/'private/ip-team-v6-pairs'
    if any(json.loads(p.read_text())['status']=='pending' for p in (out/'calls').glob('*.json')):raise ValueError('pending request')
    g.v5.legacy.load_env(HERE.parent/'ip-team-v3/.env')
    url=os.environ['EIDOLON_IP_DATA_BASE_URL']
    if url!='https://open.bigmodel.cn/api/paas/v4' or os.environ['EIDOLON_IP_DATA_MODEL']!='glm-5.3-flash':raise ValueError('official endpoint/model required')
    client=ChatClient({'base_url':url,'model':'glm-5.3-flash','api_key_env':'EIDOLON_IP_DATA_API_KEY','timeout':180,'max_tokens':4500,'thinking':{'type':'enabled'},'reasoning_effort':'low','response_format':{'type':'json_object'}})
    queue=[('train','handoff'),('val','handoff'),('val','same_name'),('train','pause_discuss'),('val','pause_discuss'),('train','end_pause'),('val','end_pause'),('train','quiet'),('val','quiet'),('val','author'),('train','continue_done'),('val','continue_done')]
    plan={'schema':'shared_fields_v2','same_total_call_budget':32,'queue':queue,'reason':'avoid asking GLM to reproduce identical fields twice'}
    pp=out/'shared-plan.json'
    if pp.exists() and json.loads(pp.read_text())!=json.loads(json.dumps(plan)):raise ValueError('plan mismatch')
    g.v5.legacy.atomic_json(pp,plan)
    done=0
    for split,sid in queue:
        fid=f'v6s_{split}_{sid}';dest=out/'reviewed'/f'{fid}.json'
        if dest.exists():continue
        if len(list((out/'calls').glob('*.json')))>30:
            print('Insufficient remaining budget for a generation/annotation pair; no call.',flush=True);return
        _,focus,changed=next(x for x in g.SPECS if x[0]==sid)
        theme='漂浮书岛的童话旅伴' if split=='train' else '沙海古城的星夜旅伴'
        instructions=g.RULES+f'\n原创{theme}故事人物的一对快照。{focus}\n人物正文只使用完整名字，不写昵称或字母槽位。共享字段只提供一次，变化字段才有_x和_y后缀。x/y只是字段名，绝不能作为用户原话。'
        if changed=='user':instructions+='\n两个快照都处于用户刚输入、尚无人回应的时刻，不生成任何peer字段。'
        else:instructions+='\nuser之后是peer发言，随后决策；共享的user和作者或发言只能写一次。'
        instructions+='\n每个字段都是字符串。只返回平面JSON，字段恰好为：'+json.dumps(sorted(fields_for(changed)),ensure_ascii=False)+'\nacceptable用完整动作代码例如respond:b，多个用英文逗号连接。role为人格关系；name为真实中文名。'
        raw=g.v5.paid(client,out,fid+'-gen',instructions,32)
        row={'id':fid,'split':split,'slice':sid,'schema':'shared_fields_v2','generator_sha256':g.v5.sha(raw)}
        try:d=materialize(raw,sid,changed)
        except (ValueError,TypeError) as e:row.update(status='rejected_structure',reason=str(e))
        else:
            visible={k:v for k,v in d.items() if not k.startswith(('acceptable','reason'))}
            rp=g.RULES+'\n独立判断，不看原标签，只依据下列公开输入。返回四个字符串字段acceptable_x,reason_x,acceptable_y,reason_y。动作必须为完整respond:a/b/c或wait/finish/abstain。\n'+json.dumps(visible,ensure_ascii=False)
            rr=g.v5.paid(client,out,fid+'-review',rp,32)
            try:
                r=json.loads(rr)
                if set(r)!=ANNOT:raise ValueError('review fields')
                for b in 'xy':
                    r['acceptable_'+b]=g.v5.moves(r['acceptable_'+b],'abc')
                    if not isinstance(r['reason_'+b],str) or not r['reason_'+b].strip():raise ValueError('review reason')
                status='agreed_pending_audit' if all(r['acceptable_'+b]==d['acceptable_'+b] for b in 'xy') else 'quarantined_disagreement'
            except (ValueError,TypeError):r={};status='rejected_review_structure'
            row.update(status=status,pair=d,review=r,reviewer_sha256=g.v5.sha(rr))
        g.v5.legacy.atomic_json(dest,row);print(fid,row['status'],row.get('reason',''),flush=True)
        done+=1
        if done==a.count:return
if __name__=='__main__':main()
