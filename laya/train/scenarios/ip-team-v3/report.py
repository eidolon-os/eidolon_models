"""Record-, family- and context-level metrics for text-only IP Team move choices."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from eidolon_laya_train.records import read_jsonl


def fraction(values: list[bool]) -> dict:
    return {"correct": sum(values), "n": len(values),
            "accuracy": round(sum(values) / len(values), 4) if values else None}


def summarize(report: dict, records: dict) -> dict:
    rows = {row["record_id"]: row for row in report["rows"]}
    if len(report["rows"]) != len(records) or set(rows) != set(records):
        raise ValueError("report and frozen eval records do not match one-to-one")
    if any(row["qid"] != "move" for row in rows.values()):
        raise ValueError("unexpected question")
    correct, action_correct, speaker_correct, silent_false_speech = [], [], [], []
    by_action, by_mode, by_stress, by_members, by_family = (defaultdict(list) for _ in range(5))
    pairs = defaultdict(dict)
    for rid, record in records.items():
        row = rows[rid]
        gold = record.labels["move"]["gold"]
        pred = row["pred"]
        ga, pa = gold.split(":", 1)[0], pred.split(":", 1)[0]
        ok = pred == gold
        correct.append(ok)
        action_correct.append(ga == pa)
        if ga in {"respond", "clarify"}:
            speaker_correct.append(":" in pred and pred.split(":", 1)[1] == gold.split(":", 1)[1])
        if ga in {"wait", "finish"}:
            silent_false_speech.append(pa in {"respond", "clarify"})
        by_action[ga].append(ok)
        by_mode[record.meta["page_mode"]].append(ok)
        by_stress["long_public_history" if record.meta.get("history_stress") else "short_public_history"].append(ok)
        by_members[len(record.state["candidates"])].append(ok)
        by_family[record.meta["family"]].append(ok)
        pair = record.meta.get("pair_id")
        if pair:
            variant = rid.split("~v", 1)[1].split("-", 1)[0]
            pairs[(pair, variant)][record.meta["branch"]] = ok
    pair_results = [len(branches) == 2 and all(branches.values()) for branches in pairs.values()]
    return {
        "move": fraction(correct), "action_only": fraction(action_correct),
        "speaker_when_needed": fraction(speaker_correct),
        "silent_wrong_speech": {"count": sum(silent_false_speech), "n": len(silent_false_speech)},
        "by_gold_action": {k: fraction(v) for k, v in sorted(by_action.items())},
        "by_page_mode": {k: fraction(v) for k, v in sorted(by_mode.items())},
        "by_context": {k: fraction(v) for k, v in sorted(by_stress.items())},
        "by_candidate_count": {k: fraction(v) for k, v in sorted(by_members.items())},
        "all_variants_and_steps_correct_per_family": fraction([all(v) for v in by_family.values()]),
        "counterfactual_branch_pairs_both_correct": fraction(pair_results),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    records = {r.id: r for r in read_jsonl(args.eval_set)}
    result = json.dumps(summarize(json.loads(args.report.read_text()), records), ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
