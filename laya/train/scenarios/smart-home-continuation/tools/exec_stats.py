"""单句“误执行”：按 Agent 的实际规则，从服务回复（三题都答）逐条判定会不会改动设备、改得对不对。

    python3 train/scenarios/smart-home-continuation/tools/exec_stats.py \
        --records train/scenarios/smart-home/eval/locked-accept.jsonl --responses <dir>/locked-accept.responses.jsonl \
        [--min-confidence 0.8] [--out stats.json]

回复来自 `service_check.py --save`（同 Agent 适配器：state + 三题原样发，不带 ask_if），所以金标为“无关”的句子也有
设备 / 动作的真实选择与置信度，不再需要假设。规则（eidolon_agent LayaInterpreter._result + SmartHomeCommand._confident）：

- 意图 无关 → 不动设备；设备 = 多个设备或整屋 → 交还；意图 查询 → 只回答，不改状态；
- 意图 控制 且设备 = 没有对应的设备 → 不针对任何设备（Agent 回“没有这台”或交还）；
- 意图 控制、设备是具体一台、意图 / 设备 / 动作三个置信度都 ≥ 阈值 → **执行**（Agent 词表拼不出命令时还会交还，
  所以这里是模型侧上界）。

执行的每一条按金标归类：正确；错设备（金标是控制，设备不在金标里，含金标“没有对应的设备”）；错动作（设备对、动作不在金标里）；
误触发（金标不是控制：无关或查询，却会改动设备）。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

MULTI, NONE = "多个设备或整屋", "没有对应的设备"


def gold_has(gold, choice) -> bool:
    return choice in gold if isinstance(gold, list) else choice == gold


def judge(ans: dict, gold: dict, t: float) -> str | None:
    """None when the Agent would not change a device state; otherwise the category of that execution."""
    intent, device, action = (ans[q]["choice"] for q in ("intent", "device", "action"))
    if intent != "控制" or device in (MULTI, NONE):
        return None
    if min(ans[q]["probabilities"][ans[q]["choice"]] for q in ("intent", "device", "action")) < t:
        return None
    if not gold_has(gold["intent"], "控制"):
        return "误触发"
    if not gold_has(gold.get("device"), device):
        return "错设备"
    if not gold_has(gold.get("action"), action):
        return "错动作"
    return "正确"


def stats(records: str, responses: str, t: float) -> dict:
    gold = {}
    utt = {}
    for line in open(records, encoding="utf-8"):
        r = json.loads(line)
        gold[r["id"]] = {q: v["gold"] for q, v in r["labels"].items()}
        utt[r["id"]] = r["state"]["utterance"]
    c = Counter()
    wrong = []
    n = 0
    for line in open(responses, encoding="utf-8"):
        r = json.loads(line)
        n += 1
        k = judge(r["answers"], gold[r["id"]], t)
        if k is None:
            continue
        c[k] += 1
        if k != "正确":
            a = r["answers"]
            wrong.append({"id": r["id"], "kind": k, "utterance": utt[r["id"]], "gold": gold[r["id"]],
                          "pred": {q: [a[q]["choice"], round(a[q]["probabilities"][a[q]["choice"]], 3)]
                                   for q in ("intent", "device", "action")}})
    executed = sum(c.values())
    return {"records": n, "threshold": t, "executed": executed, "by_kind": dict(c),
            "wrong_total": executed - c["正确"], "wrong": wrong}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--min-confidence", type=float, default=0.8)
    ap.add_argument("--out")
    a = ap.parse_args()
    s = stats(a.records, a.responses, a.min_confidence)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in s.items() if k != "wrong"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
