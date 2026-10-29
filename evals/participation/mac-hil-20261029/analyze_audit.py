"""Join local HIL audit by exact decision_id; never infer absent playback receipts."""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def collect(directory, session_id=None):
    decisions = defaultdict(list)
    scenes = {}
    malformed = []
    for path in sorted(Path(directory).glob('*.jsonl')):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
            except ValueError:
                malformed.append(f'{path.name}:{n}')
                continue
            if session_id and row.get('session_id') != session_id:
                continue
            if row.get('event') == 'scene_opened':
                scenes[row['session_id']] = row
            if row.get('decision_id'):
                decisions[(row.get('session_id'), row['decision_id'])].append(row)
    results = []
    for (session, decision), records in decisions.items():
        records.sort(key=lambda r: r['timestamp'])
        def last(kind, source=None):
            return next((r for r in reversed(records) if r['event'] == kind and (not source or r['source'] == source)), None)
        request = last('request_received', 'gateway') or last('decision_requested', 'agent')
        result = last('result_returned', 'gateway')
        reply = last('reply_terminal', 'agent')
        accepted = last('decision_validated', 'agent')
        rejected = last('decision_not_applied', 'agent')
        results.append(dict(session_id=session, decision_id=decision, timestamp=records[0]['timestamp'],
                            epoch=records[0].get('epoch'), request=request.get('request') if request else None,
                            upstream_request=last('upstream_request', 'gateway'),
                            upstream_response=last('upstream_response', 'gateway'),
                            result=result, applied=bool(accepted),
                            application_status='validated' if accepted else ('not_applied' if rejected else 'unknown'),
                            reply=reply, events=records))
    results.sort(key=lambda r:r['timestamp'])
    return dict(schema_version=1, scenes=scenes, decisions=results, malformed_lines=malformed,
                limitations=['generated_text is not a word-level audible transcript',
                             'missing terminal event means unknown, not success',
                             'no automatic semantic correctness score'])


def write_private(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--session')
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    result=collect(args.directory,args.session)
    args.output.mkdir(parents=True,exist_ok=True,mode=0o700)
    write_private(args.output/'joined.json',json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 逐次决策审计索引','','详情见 joined.json；不自动把接口成功标为语义正确。', '',
           '|时间 UTC|decision_id|epoch|动作|是否采纳|回复终态|播放完成回执|',
           '|---|---|---:|---|---|---|---|']
    for d in result['decisions']:
        r=d['result'] or {}; reply=d['reply'] or {}; proposal=r.get('result',{}).get('proposal') or {}
        lines.append(f"|{d['timestamp']}|{d['decision_id']}|{d['epoch']}|{proposal.get('action','未知')}|{d['application_status']}|{reply.get('outcome','未知/无回复')}|{reply.get('playback_completed','未知/无回复')}|")
    lines+=['',f"无法解析的行：{len(result['malformed_lines'])}。正在写入的末行可稍后重读。",'',
            '完整数据包含候选角色、模型原始输入与概率、用户请求、公共历史、生成文本、回复与设备 ID。',
            '取消时生成文本可能只包含一部分；任何生成文本均不保证已完整被用户听到。']
    write_private(args.output/'INDEX.md','\n'.join(lines)+'\n')
    print(f"Joined {len(result['decisions'])} decisions; malformed lines: {len(result['malformed_lines'])}; output: {args.output}")

if __name__=='__main__':
    main()
