"""Bounded, resumable GLM generation of three-state decision families."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
from eidolon_laya_train.generators import ChatClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / 'private/ip-team-v8-boundaries'
spec = importlib.util.spec_from_file_location('v7_generation', HERE.parent / 'ip-team-v7/generate.py')
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)
io = v7.io
LIMIT = 128
RULES = v7.RULES + '''
同名消歧必须有用户给出的明确关系/特征线索，或公开对话唯一确定的指代；不能因为问题提到某物就按职业选专家。两位同名且普通问题没有身份线索时，只能abstain，不能擅自都叫来或随选一位。
用户要求暂时完全停止直至其再次发话是wait；明确说本轮结束是finish。用户以后还会回来不否定本轮已经finish；两种停止都合理时完整保留。用户在一旁听但明确让人物继续时不是暂停。实际可接受动作仅按公开事实判断，不推断不存在的用户偏好。'''
WORLDS = ['苔藓古城里结伴的精灵', '群岛渡船上的旅伴', '会移动的沙漠集市居民', '云端图书馆里的故事人物', '山间小村的异兽朋友', '星海旅途中归家的伙伴', '老戏院里的木偶朋友', '冰湖边相伴过冬的邻居']


def plan():
    result = []
    for split, count, offset in [('train', 32, 0), ('val', 16, 3), ('test', 16, 5)]:
        for i in range(count):
            result.append(dict(id=f'{split}-{i:02d}', split=split, members=2+i%5,
                               world=WORLDS[(i+offset)%len(WORLDS)], stop='临时暂停等用户' if i%2==0 else '明确本轮结束',
                               wording=['轻松口语', '自然委婉但意思明确', '先否定误解再说明真实意图', '简短自然请求'][i%4]))
    # Interleave splits so pilot catches split-specific quality issues before spending.
    return sorted(result, key=lambda r: (int(r['id'].split('-')[1]), ['train','val','test'].index(r['split'])))


def prompt(row):
    return RULES + f'''
原创一个故事家族，背景是{row['world']}，共{row['members']}名角色，措辞{row['wording']}。恰好两位角色完整姓名相同，性格/关系特征不同；其余角色不同名。
生成三个共享完全相同cast的公开快照，各自是独立假设情境，不能把另一个情境的原话视为历史。三个快照的话题背景相同，改变用户的关键意图或身份线索，让下一步决策有实际区别：
一份通过用户明确的性格/关系线索唯一指定同名者之一说话；一份用户要求{row['stop']}且不让任何角色回应；一份只叫两位的同名问普通生活话题，无任何身份线索、无指代历史，也没有要求大家一起回答。
三情境before和after都为空；不使用职业匹配作消歧。缺少身份线索那份必须明确向叫此名字的一个本人说“你”，不能用“两位/你们/大家”，也不能改成第三人称询问旁人该人的情况。不要在人物资料使用“上面/下面/第一位/第二位”等数组位置指代。
三个user长度尽量接近，均40–90字，最长与最短差不超过20字；可以有自然背景细节，但不能额外泄露身份或改变动作。不要总是早餐/晚饭/打发时间，使用不同生活话题及表达。每个人必须是故事角色，有自然人格或公开关系，不是工作团队。避免总是cast第一位被指明。
返回JSON对象scenes数组，恰好s1/s2/s3三项，排列顺序任意（不要把情境类型写入id）。每项只含id、cast、user、before、after；cast每项只有name(完整中文名2–8字)、persona(4–80字)；user为4–240字原话；before与after都是[]。
不要输出gold、why、预期动作或点评，不要未来回复，不使用占位符。完整复制共享cast，不用“同上”。具体名字、人物性格与话题须原创。'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', type=int, default=1)
    args = ap.parse_args()
    if not 1 <= args.batches <= 16:
        ap.error('1–16 batches')
    ROOT.mkdir(parents=True, exist_ok=True)
    cfg = {'limit': LIMIT, 'queue': plan()}
    path = ROOT / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != cfg:
        raise ValueError('immutable plan changed')
    io.legacy.atomic_json(path, cfg)
    io.legacy.load_env(HERE.parent / 'ip-team-v3/.env')
    if os.environ['EIDOLON_IP_DATA_BASE_URL'] != 'https://open.bigmodel.cn/api/paas/v4' or os.environ['EIDOLON_IP_DATA_MODEL'] != 'glm-5.3-flash':
        raise ValueError('endpoint/model')
    client = ChatClient({'base_url': os.environ['EIDOLON_IP_DATA_BASE_URL'], 'model': 'glm-5.3-flash', 'api_key_env': 'EIDOLON_IP_DATA_API_KEY', 'timeout': 180, 'max_tokens': 6000, 'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low', 'response_format': {'type': 'json_object'}})
    completed = 0
    for row in cfg['queue']:
        dest = ROOT / 'reviewed' / (row['id']+'.json')
        if dest.exists():
            continue
        if len(list((ROOT/'calls').glob('*.json'))) > LIMIT-2:
            print('budget exhausted', flush=True)
            return
        raw = io.paid(client, ROOT, row['id']+'-gen', prompt(row), LIMIT)
        result = {'id': row['id'], 'split': row['split'], 'batch': row['id'], 'plan': row,
                  'generator_sha256': io.sha(raw), 'status': 'rejected_structure'}
        try:
            data = json.loads(raw)['scenes']
            if len(data) != 3 or {d['id'] for d in data} != {'s1','s2','s3'}:
                raise ValueError('three unique snapshots required')
            # Reuse public-only validation, allowing 2–6 candidates for every split.
            scenes = [v7.parse_scene(d, {'split':'test'}) for d in data]
            if any(d['cast'] != scenes[0]['cast'] for d in scenes) or any(d['before'] or d['after'] for d in scenes):
                raise ValueError('shared cast and empty histories required')
            if len(scenes[0]['cast']) != row['members'] or len({d['user'] for d in scenes}) != 3:
                raise ValueError('members or duplicate user')
        except (ValueError, TypeError, KeyError) as e:
            result['reason'] = str(e)
        else:
            # Each annotation input contains only public facts, not the planned groups.
            visible = [dict(d, cast=[dict(c, action_code=f'respond:{i}') for i,c in enumerate(d['cast'])]) for d in scenes]
            review_prompt = RULES + '\n独立标注下面三个公开快照。各自独立，不能从另一个快照借用身份线索；没有原gold或预设答案。完整列出可接受动作。返回JSON reviews数组，每项恰好id、gold(动作数组)、why(公开依据)、valid(布尔)、issue(无问题空字符串)。\n' + json.dumps(visible, ensure_ascii=False)
            rr = io.paid(client, ROOT, row['id']+'-review', review_prompt, LIMIT)
            result.update(scenes=scenes, reviewer_sha256=io.sha(rr), status='rejected_review_structure')
            try:
                reviews = json.loads(rr)['reviews']
                if len(reviews) != 3 or {r['id'] for r in reviews} != {'s1','s2','s3'}:
                    raise ValueError('review ids/count')
                byid = {r['id']:r for r in reviews}
                for d in scenes:
                    r = byid[d['id']]
                    if set(r) != {'id','gold','why','valid','issue'} or type(r['valid']) is not bool or not isinstance(r['why'],str) or not r['why'].strip():
                        raise ValueError('review schema')
                    d.update(gold=v7.parse_gold(r['gold'],len(d['cast'])), why=r['why'])
                result.update(reviews=reviews, status='annotated_pending_audit' if all(r['valid'] for r in reviews) else 'quarantined_review')
            except (ValueError, TypeError, KeyError) as e:
                result['reason'] = str(e)
        io.legacy.atomic_json(dest, result)
        print(row['id'], result['status'], flush=True)
        completed += 1
        if completed >= args.batches:
            return


if __name__ == '__main__':
    main()
