"""单句“误执行”按 Agent 实际规则重放（Laya 这一层），逐条输出决策，可比较两种 logits 来源（torch / NPU）。

规则照搬 eidolon_agent：
- `infra/interpretation/adapters/laya.py::_result`：无关 → unrelated；设备“多个设备或整屋” → 弃权（None）；
  查询 → query（“没有对应的设备”时无目标）；控制 → control（“没有对应的设备”时 target_status none，回“家里没有…”，不操作设备）。
- `domain/smarthome/command.py::_confident`：intent_p ≥ τ；非无关再要 device_p ≥ τ；控制再要 action_p ≥ τ。不满足 → 交给 LLM。
- 只有“控制 + 真实设备 + 三题都 ≥ τ”才会下发设备命令（这里称“执行”）。

**这里没有重放的 Agent 门**（由 Agent 侧按决策 ID 对齐）：严格规则一致性（independent interpreter）、有上下文时走 LLM、
lexicon 能否把动作拼成命令（拼不出只会减少执行，不会增加误执行）。所以这里的“执行”是 Laya 层的上限。

    .venv/bin/python train/scenarios/smart-home-continuation/tools/agent_replay.py \
        --records train/scenarios/smart-home/eval/locked-v2-dev.jsonl \
        --items <dir>/locked-v2-dev.items.jsonl [--logits <dir>/locked-v2-dev.logits.jsonl] \
        --config models/laya-smart-home/7b695ba8/torch/rl_agent_config.json --out <dir>/locked-v2-dev.replay.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from eidolon_laya_train.evaluate import temperature_for

MULTIPLE, NO_DEVICE = "多个设备或整屋", "没有对应的设备"


def softmax(z: list[float], t: float) -> list[float]:
    z = [x / t for x in z]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def as_set(v) -> set[str]:
    if v is None:
        return set()
    return set(v) if isinstance(v, list) else {v}


def decide(ans: dict, tau: float) -> tuple[str, str | None, str | None]:
    """(kind, device, action)。kind ∈ execute / unrelated / query / not_found / abstain_multiple / llm。"""
    (i, pi), (d, pd), (a, pa) = ans["intent"], ans["device"], ans["action"]
    if i == "无关":
        return ("unrelated", None, None) if pi >= tau else ("llm", None, None)
    if d == MULTIPLE:
        return ("abstain_multiple", None, None)
    if i == "查询":
        return ("query", None if d == NO_DEVICE else d, None) if pi >= tau and pd >= tau else ("llm", None, None)
    if not (pi >= tau and pd >= tau and pa >= tau):
        return ("llm", None, None)
    if d == NO_DEVICE:
        return ("not_found", None, None)
    return ("execute", d, a)


def outcome(kind: str, dev: str | None, act: str | None, gold: dict) -> str:
    gi, gd, ga = gold["intent"], as_set(gold.get("device")), as_set(gold.get("action"))
    if kind == "execute":
        if gi != "控制":
            return f"误执行·误触发（金标{gi}）"
        if dev not in gd:
            return "误执行·错设备" + ("（金标是出口）" if gd & {MULTIPLE, NO_DEVICE} else "")
        if not ga:
            return "执行·动作未标"
        return "正确执行" if act in ga else "误执行·错动作"
    if kind == "llm" or kind == "abstain_multiple":
        return f"交给LLM（金标{gi}）"
    if kind == "unrelated":
        return "正确·无关" if gi == "无关" else f"不执行·判成无关（金标{gi}）"
    if kind == "query":
        if gi != "查询":
            return f"不执行·判成查询（金标{gi}）"
        return "正确·查询" if (dev in gd or (dev is None and NO_DEVICE in gd)) else "查询答错设备"
    if kind == "not_found":
        if gi == "控制" and NO_DEVICE in gd:
            return "正确·没有设备"
        return f"不执行·报没有设备（金标{gi}）"
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--items", required=True, help="eidolon-laya-train items 的输出（含 ref_logits、选项名）")
    ap.add_argument("--logits", help="laya_npu.py run 的输出；给了就用它，否则用 items 里的 ref_logits")
    ap.add_argument("--config", required=True, help="rl_agent_config.json（校准温度）")
    ap.add_argument("--tau", type=float, nargs="+", default=[0.8, 0.9])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text("utf-8"))
    items = {it["key"]: it for it in map(json.loads, Path(a.items).read_text("utf-8").splitlines())}
    if a.logits:
        for row in map(json.loads, Path(a.logits).read_text("utf-8").splitlines()):
            items[row["key"]]["logits"] = row["logits"]
    recs = [json.loads(x) for x in Path(a.records).read_text("utf-8").splitlines() if x.strip()]
    out, summary = [], {str(t): Counter() for t in a.tau}
    for r in recs:
        ans, full = {}, {}
        for q in ("intent", "device", "action"):
            it = items[f"{r['id']}/{q}"]
            z = it.get("logits", it["ref_logits"])
            p = softmax(z, temperature_for(cfg, it["qtype"], len(z)))
            order = sorted(range(len(p)), key=lambda k: -p[k])
            ans[q] = (it["names"][order[0]], p[order[0]])
            full[q] = [[it["names"][k], round(p[k], 4)] for k in order[:3]]
        gold = {q: (r["labels"].get(q) or {}).get("gold") for q in ("intent", "device", "action")}
        gold["intent"] = gold["intent"][0] if isinstance(gold["intent"], list) else gold["intent"]
        row = {"decision_id": r["id"], "utterance": r["state"]["utterance"],
               "home": next((t[5:] for t in r["tags"] if t.startswith("home:")), None), "tags": r["tags"][:1],
               "gold": gold, "top3": full, "decisions": {}}
        for t in a.tau:
            kind, dev, act = decide(ans, t)
            o = outcome(kind, dev, act, gold)
            row["decisions"][str(t)] = {"kind": kind, "device": dev, "action": act, "outcome": o}
            summary[str(t)][o] += 1
        out.append(row)
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), "utf-8")
    Path(a.out).with_suffix(".summary.json").write_text(
        json.dumps({t: dict(c) for t, c in summary.items()}, ensure_ascii=False, indent=1) + "\n", "utf-8")
    for t, c in summary.items():
        ex = sum(v for k, v in c.items() if k.startswith(("正确执行", "误执行", "执行·")))
        wrong = sum(v for k, v in c.items() if k.startswith("误执行"))
        print(f"τ={t}: 执行 {ex}，误执行 {wrong}：" + "，".join(f"{k} {v}" for k, v in sorted(c.items()) if k.startswith("误执行")))


if __name__ == "__main__":
    main()
