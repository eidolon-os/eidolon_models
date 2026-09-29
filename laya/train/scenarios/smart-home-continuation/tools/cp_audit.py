"""cp-absent（“用户点名的那台不在候选里”）对照副本的缺陷审计：剩下的候选里只要有一台的房间 / 名字二字片段出现在话里，
这条副本的金标（重新理解）就不可靠——用户可能正是在指那一台。c-test 里唯一一条错误执行就是这种副本（“主卧那盏”，床头灯也在主卧）。

    .venv/bin/python train/scenarios/smart-home-continuation/tools/cp_audit.py <eval 报告.json> <记录.jsonl> [--tau-exec x --tau-cancel y]

打印：有缺陷的副本 id 数，以及剔除它们之后的 cmetrics 汇总（评测集文件本身不改）。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmetrics import summarize  # noqa: E402


def flawed(rec: dict) -> bool:
    if "cp-absent" not in rec["tags"]:
        return False
    u = rec["state"]["utterance"]
    for lab, desc in rec["questions"]["pick"]["criteria"].items():
        if lab in ("取消", "重新理解"):
            continue
        text = lab.split("（")[0] + desc.split("·")[0]
        if any(text[k:k + 2] in u for k in range(len(text) - 1)):
            return True
    return False


ap = argparse.ArgumentParser()
ap.add_argument("report")
ap.add_argument("records")
ap.add_argument("--tau-exec", type=float, default=0.95)
ap.add_argument("--tau-cancel", type=float, default=0.5)
a = ap.parse_args()
bad = {json.loads(l)["id"] for l in open(a.records) if l.strip() and flawed(json.loads(l))}
rows = [r for r in json.load(open(a.report))["rows"] if r["qid"] in ("pick", "follow")]
kept = [r for r in rows if r["record_id"] not in bad]
print(f"flawed cp-absent {len(bad)}; rows {len(rows)} -> {len(kept)}")
for name, rs in (("as registered", rows), ("flawed removed", kept)):
    s = summarize(rs, a.tau_exec, a.tau_cancel)
    print(f"  {name}: n={s['n']} argmax={s['argmax_acc']} takeover={s['takeover']} counts={s['counts']} "
          f"wrong_exec_ub95={s['wrong_exec_upper95_of_auto']} wrong_cancel={s['wrong_cancel_rate']}")
