"""Summarize existing eval outputs using c-series and Agent replay semantics.

No inference, new policy or target resolution. The execution counts are the
Laya-layer upper bound; Agent validation and physical execution are not simulated.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from agent_replay import decide
from agent_replay import outcome as single_outcome
from cmetrics import outcome, summarize

from eidolon_laya_train.evaluate import gate


def summarize_report(report):
    rows = report["policy_rows"]
    cont = [r for r in rows if r["qid"] in ("pick", "follow")]
    by_id = defaultdict(dict)
    for row in rows:
        by_id[row["record_id"]][row["qid"]] = row
    result = {"n_records": report["n_records"], "overall": report["overall"],
              "continuation": summarize(cont, .95, .5) if cont else None,
              "by_tag": {}, "errors": [], "single": {}}
    for tag in sorted({tag for r in cont for tag in r["tags"]}):
        result["by_tag"][tag] = summarize([r for r in cont if tag in r["tags"]], .95, .5)
    for row in cont:
        o = outcome(row, .95, .5)
        if not row["correct"] or o in ("错误执行", "错误取消"):
            result["errors"].append({"record_id": row["record_id"], "qid": row["qid"],
                                     "gold": row["gold"], "pred": row["pred"], "p_top": row["p_top"],
                                     "outcome": o, "tags": row["tags"]})
    result["wrong_exec_confidence"] = {
        str(threshold): sum(not r["correct"] and r["pred"] not in ("取消", "重新理解")
                           and r["p_top"] >= threshold for r in cont)
        for threshold in (.8, .9, .95, .97, .99)
    }
    for tau in (.8, .9):
        outcomes = Counter()
        decisions = []
        n_control = 0
        for rid, qs in by_id.items():
            if not {"intent", "device", "action"} <= qs.keys():
                continue
            answers = {q: (qs[q]["pred"], qs[q]["p_top"]) for q in ("intent", "device", "action")}
            gold = {q: qs[q]["gold"] for q in ("intent", "device", "action")}
            gold["intent"] = gold["intent"][0]
            n_control += gold["intent"] == "控制"
            kind, dev, act = decide(answers, tau)
            o = single_outcome(kind, dev, act, gold)
            outcomes[o] += 1
            decisions.append({"record_id": rid, "kind": kind, "outcome": o,
                              "pred": answers, "gold": gold})
        result["single"][str(tau)] = {
            "n": len(decisions), "n_control": n_control, "outcomes": dict(outcomes),
            "execute": sum(d["kind"] == "execute" for d in decisions),
            "wrong_execute": sum(d["outcome"].startswith("误执行") for d in decisions),
            "correct_control_coverage": outcomes["正确执行"] / n_control if n_control else None,
            "decisions": decisions,
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    rep = json.loads(a.report.read_text())
    result = summarize_report(rep)
    result["source_report"] = str(a.report)
    result["checkpoint_sha256"] = rep.get("checkpoint_sha256")
    result["checkpoint_config_sha256"] = rep.get("checkpoint_config_sha256")
    if a.baseline:
        base = json.loads(a.baseline.read_text())
        b = summarize_report(base)
        result["paired_gate"] = gate(rep, base, alpha=.05)
        result["single_deltas"] = {}
        for tau in (.8, .9):
            old = {r["record_id"]: r for r in b["single"][str(tau)]["decisions"]}
            new = result["single"][str(tau)]
            new_errors = [r for r in new["decisions"] if r["outcome"].startswith("误执行")
                          and not old[r["record_id"]]["outcome"].startswith("误执行")]
            fixed_errors = [r for r in new["decisions"] if not r["outcome"].startswith("误执行")
                            and old[r["record_id"]]["outcome"].startswith("误执行")]
            result["single_deltas"][str(tau)] = {
                "new_errors": new_errors, "fixed_errors": fixed_errors,
                "wrong_execute": [b["single"][str(tau)]["wrong_execute"], new["wrong_execute"]],
                "coverage": [b["single"][str(tau)]["correct_control_coverage"], new["correct_control_coverage"]],
            }
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("n_records", "overall", "continuation", "wrong_exec_confidence")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
