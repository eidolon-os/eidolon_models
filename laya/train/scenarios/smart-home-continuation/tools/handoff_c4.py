"""c4 交接给 Agent（Codex）的两份清单：按 1 s 预算重数已有并发记录；已知错误案例与期望结果。

    python3 laya/train/scenarios/smart-home-continuation/tools/handoff_c4.py --out evals/smart-home-continuation/c4-handoff

从 eidolon_models 根目录运行，只读已提交的证据（evals/smart-home-continuation/c4-npu、laya/train/runs/c4 的续接报告），不跑模型。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

C4 = Path("evals/smart-home-continuation/c4-npu")
SINGLE = Path("laya/train/scenarios/smart-home/eval")
CONT = Path("laya/evals/smart-home-continuation")
CONT_REPORTS = {"c-dev": Path("laya/train/runs/c4/eval-c/c-dev.json"), "c-test": Path("laya/train/runs/c4/eval-test/c-test.json")}
TAU_EXEC, TAU_CANCEL = 0.95, 0.5
SETS = ("locked-accept", "locked-182", "locked-v2-dev", "locked-v3-dev")


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text("utf-8").splitlines() if x.strip()]


def pct(ms: list[float], q: float) -> float:
    return round(ms[min(len(ms) - 1, int(q * len(ms)))], 1)


def budget(out: Path) -> None:
    res: dict = {"budgets_ms": [800, 1000], "note": "家居请求 = 单句（三题）/ 续接（一题）；http 非 200 的另计，不进时延分布", "runs": {}}
    for model, d in (("c4", C4 / "concurrency"), ("r14", C4 / "concurrency-r14")):
        for f in sorted(d.glob("*.jsonl")):
            groups: dict[str, list[dict]] = defaultdict(list)
            for r in jl(f):
                kind = r["kind"]
                if kind == "home":
                    kind = "home_continuation" if r["id"].startswith("smart-home-continuation/") else "home_single"
                groups[kind].append(r)
            for kind, rows in groups.items():
                ms = sorted(r["ms"] for r in rows if r["http"] == 200 and r["ms"] is not None)
                res["runs"].setdefault(model, {}).setdefault(f.stem, {})[kind] = {
                    "n": len(rows), "non_200": sum(r["http"] != 200 for r in rows),
                    "p50": pct(ms, .5), "p95": pct(ms, .95), "max": ms[-1],
                    "over_800": sum(m > 800 for m in ms), "over_1000": sum(m > 1000 for m in ms),
                    "over_1500": sum(m > 1500 for m in ms)}
    (out / "budget-1s.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", "utf-8")
    for model, runs in res["runs"].items():
        for run, kinds in runs.items():
            for kind, s in kinds.items():
                print(f"{model:4s} {run:11s} {kind:18s} n={s['n']:4d} p50/p95/max {s['p50']}/{s['p95']}/{s['max']}"
                      f"  >800 {s['over_800']}  >1000 {s['over_1000']}  non200 {s['non_200']}")


def top(ans: dict) -> tuple[str, float]:
    return ans["choice"], round(ans["probabilities"][ans["choice"]], 4)


def single_cases() -> list[dict]:
    cases = []
    for s in SETS:
        recs = {r["id"]: r for r in jl(SINGLE / f"{s}.jsonl")}
        npu = {r["id"]: r for r in jl(C4 / "service/npu" / f"{s}.responses.jsonl")}
        for w in json.loads((C4 / "exec/torch/c4" / f"{s}.exec-0.8.json").read_text("utf-8"))["wrong"]:
            rec, g = recs[w["id"]], w["gold"]
            n = npu[w["id"]]["answers"]
            if w["kind"] == "误触发":
                expect = f"不改动任何设备（金标{g['intent']}：{'不是对助手的请求' if g['intent'] == '无关' else '只回答状态'}）"
            elif w["kind"] == "错设备":
                expect = ("回“家里没有这台”或澄清，不操作任何设备" if g.get("device") == "没有对应的设备"
                          else f"交 LLM（金标多台 / 整屋）；不能只操作 {w['pred']['device'][0]}" if g.get("device") == "多个设备或整屋"
                          else f"操作 {g['device']}，或澄清 / 交 LLM；不能操作 {w['pred']['device'][0]}")
            else:
                expect = f"对 {g['device']} 执行 {g['action']}，或澄清 / 交 LLM；不能执行 {w['pred']['action'][0]}"
            cases.append({
                "task": "single", "set": s, "record_file": str(SINGLE / f"{s}.jsonl"), "id": w["id"],
                "home": next((t[5:] for t in rec.get("tags", []) if t.startswith("home:")), None),
                "utterance": w["utterance"], "gold": g, "error": w["kind"],
                "laya_torch": w["pred"], "laya_npu": {q: list(top(n[q])) for q in ("intent", "device", "action")},
                "laya_layer_decision_at_0.8": "执行（三题 ≥ 0.8、具体设备）", "expected_full_chain": expect})
    return cases


def cont_outcome(pred: str, p: float, gold: str) -> str:
    if pred not in ("取消", "重新理解") and p >= TAU_EXEC:
        return "正确执行" if pred == gold else "错误执行"
    if pred == "取消" and p >= TAU_CANCEL:
        return "正确取消" if gold == "取消" else "错误取消"
    return "正确交还" if gold == "重新理解" else "漏掉续接"


def cont_cases() -> list[dict]:
    deployed = {r["id"]: r for r in jl(C4 / "deployed-rk3588-laya-home-c4-20260930-1/c-dev.responses.jsonl")}
    cases = []
    for name, rep in CONT_REPORTS.items():
        recs = {r["id"]: r for r in jl(CONT / f"{name}.jsonl")}
        for row in json.loads(rep.read_text("utf-8"))["rows"]:
            gold = row["gold"][0]
            o = cont_outcome(row["pred"], row["p_top"], gold)
            if o not in ("错误执行", "错误取消"):
                continue
            rec, q = recs[row["record_id"]], row["qid"]
            case = {
                "task": q, "set": name, "record_file": str(CONT / f"{name}.jsonl"), "id": row["record_id"],
                "state": rec["state"], "options": list(rec["questions"][q]["criteria"]), "gold": gold, "error": o,
                "laya_torch": [row["pred"], row["p_top"]], "tags": row["tags"]}
            if row["record_id"] in deployed:
                case["laya_npu_deployed"] = list(top(deployed[row["record_id"]]["answers"][q]))
            if o == "错误取消":
                case["expected_full_chain"] = ("对 " + gold + " 执行待确认动作，或交 LLM；不能撤销待确认提案"
                                               if gold not in ("取消", "重新理解")
                                               else "交 LLM 重新理解（改口 / 新要求 / 没选定）；不能按“取消”了结而吞掉用户的话")
            else:
                case["expected_full_chain"] = "见 note"
            cases.append(case)
    for c in cases:
        if c["set"] == "c-test" and c["error"] == "错误执行":
            c["note"] = ("金标缺陷（LABELING 裁决 C15）：cp-absent 规则没检查剩下的候选，按 C15 这条的正确答案就是模型选的设备；"
                         "冻结文件未改，报告同时给登记口径与剔除后结果。Agent 链路按 C15 验收：执行模型所选设备不算错。")
            c["expected_full_chain"] = "按 C15：执行所选设备即正确"
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    budget(out)
    cases = single_cases() + cont_cases()
    (out / "known-errors.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), "utf-8")
    for c in cases:
        print(f"{c['set']:14s} {c['task']:7s} {c['error']:5s} {c['id']:48s} {c.get('utterance') or c['state']['utterance']}")


if __name__ == "__main__":
    main()
