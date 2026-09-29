"""c 系列续接指标（PLAN.md §4–5）：从 `eidolon-laya-train eval` 报告的 rows 算六类结局、接管率和阈值网格。

    .venv/bin/python train/scenarios/smart-home-continuation/tools/cmetrics.py \
        --report train/runs/c1/eval/c-dev.json [--tau-exec 0.95 --tau-cancel 0.9] [--out metrics.json]

不给阈值时按登记规则在网格上选：τ_exec 取错误执行为 0 的最小值，τ_cancel 取错误取消 ≤ 1% 的最小值。
给了阈值（冻结后跑 c-test）就只用给的。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

CANCEL, REDO = "取消", "重新理解"
GRID = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99)
OUTCOMES = ("正确执行", "错误执行", "正确取消", "错误取消", "漏掉续接", "正确交还")


def outcome(row: dict, t_exec: float, t_cancel: float) -> str:
    top, p, gold = row["pred"], row["p_top"], row["gold"][0]
    if top not in (CANCEL, REDO) and p >= t_exec:
        return "正确执行" if top == gold else "错误执行"
    if top == CANCEL and p >= t_cancel:
        return "正确取消" if gold == CANCEL else "错误取消"
    return "正确交还" if gold == REDO else "漏掉续接"


def upper95(k: int, n: int) -> float | None:
    """单侧 95% Clopper–Pearson 上界（二分求解）。"""
    if n == 0:
        return None
    if k == 0:
        return 1 - 0.05 ** (1 / n)
    from math import comb

    def tail(p: float) -> float:  # P(X <= k)
        return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))

    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tail(mid) > 0.05 else (lo, mid)
    return hi


def summarize(rows: list[dict], t_exec: float, t_cancel: float) -> dict:
    c = Counter(outcome(r, t_exec, t_cancel) for r in rows)
    n = len(rows)
    takeable = sum(1 for r in rows if r["gold"][0] != REDO)
    taken = c["正确执行"] + c["正确取消"]
    auto_exec = c["正确执行"] + c["错误执行"]
    return {
        "tau_exec": t_exec, "tau_cancel": t_cancel, "n": n,
        "counts": {k: c[k] for k in OUTCOMES},
        "takeover": round(taken / takeable, 4) if takeable else None,
        "takeover_of_all": round(taken / n, 4) if n else None,
        "wrong_exec_rate_of_auto": round(c["错误执行"] / auto_exec, 4) if auto_exec else None,
        "wrong_exec_upper95_of_auto": round(upper95(c["错误执行"], auto_exec), 4) if auto_exec else None,
        "wrong_cancel_rate": round(c["错误取消"] / n, 4) if n else None,
        "argmax_acc": round(sum(r["correct"] for r in rows) / n, 4) if n else None,
    }


def select(rows: list[dict]) -> tuple[float, float]:
    n = len(rows)
    t_exec = next((t for t in GRID if summarize(rows, t, 1.01)["counts"]["错误执行"] == 0), 1.01)
    t_cancel = next((t for t in GRID if summarize(rows, 1.01, t)["counts"]["错误取消"] <= 0.01 * n), 1.01)
    return t_exec, t_cancel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--tau-exec", type=float)
    ap.add_argument("--tau-cancel", type=float)
    ap.add_argument("--out")
    args = ap.parse_args()
    rep = json.loads(Path(args.report).read_text("utf-8"))
    rows = [r for r in rep["rows"] if r["qid"] in ("pick", "follow")]
    if args.tau_exec is None or args.tau_cancel is None:
        t_exec, t_cancel = select(rows)
        chosen = "selected"
    else:
        t_exec, t_cancel, chosen = args.tau_exec, args.tau_cancel, "frozen"
    out = {"report": args.report, "thresholds": chosen, "overall": summarize(rows, t_exec, t_cancel),
           "grid": [summarize(rows, t, t) | {"tau": t} for t in GRID], "by_question": {}, "by_family": {},
           "errors": []}
    for q in ("pick", "follow"):
        sub = [r for r in rows if r["qid"] == q]
        if sub:
            out["by_question"][q] = summarize(sub, t_exec, t_cancel)
    fam = defaultdict(list)
    for r in rows:
        fam[r["tags"][2] if len(r["tags"]) > 2 else "?"].append(r)
    for f, sub in sorted(fam.items()):
        s = summarize(sub, t_exec, t_cancel)
        out["by_family"][f] = {"n": s["n"], "argmax_acc": s["argmax_acc"], **{k: v for k, v in s["counts"].items() if v}}
    for r in rows:
        o = outcome(r, t_exec, t_cancel)
        if o in ("错误执行", "错误取消") or not r["correct"]:
            out["errors"].append({"record_id": r["record_id"], "outcome": o, "tags": r["tags"], "gold": r["gold"][0],
                                  "pred": r["pred"], "p_top": r["p_top"]})
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", "utf-8")
    o = out["overall"]
    print(f"{chosen} tau_exec={t_exec} tau_cancel={t_cancel} n={o['n']} argmax={o['argmax_acc']} "
          f"takeover={o['takeover']} counts={o['counts']} wrong_exec_ub95={o['wrong_exec_upper95_of_auto']} "
          f"wrong_cancel={o['wrong_cancel_rate']}")
    for q, s in out["by_question"].items():
        print(f"  {q}: n={s['n']} argmax={s['argmax_acc']} takeover={s['takeover']} counts={s['counts']}")


if __name__ == "__main__":
    main()
