"""Summarize the frozen Ensemble comparison at the record and episode levels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from eidolon_laya_train.records import read_jsonl


def ratio(items: list[bool]) -> dict:
    return {"correct": sum(items), "n": len(items), "accuracy": round(sum(items) / len(items), 4) if items else None}


def summarize(report: dict, records: dict) -> dict:
    answers = defaultdict(dict)
    for row in report["rows"]:
        answers[row["record_id"]][row["qid"]] = row
    if set(answers) != set(records) or any(set(q) != {"move", "task"} for q in answers.values()):
        raise ValueError("eval report lacks exactly one move and task per frozen record")
    move, task, full, speaker, silent_wrong_speech = [], [], [], [], []
    by_action, by_mode, by_family_variant = defaultdict(list), defaultdict(list), defaultdict(list)
    for rid, r in records.items():
        q = answers[rid]
        m, t = q["move"], q["task"]
        move.append(m["correct"])
        task.append(t["correct"])
        both = m["correct"] and t["correct"]
        full.append(both)
        gold_move = r.labels["move"]["gold"]
        action = gold_move.split(":", 1)[0]
        if action in {"respond", "clarify"}:
            speaker.append(m["pred"].split(":", 1)[-1] == gold_move.split(":", 1)[-1] if ":" in m["pred"] else False)
        else:
            silent_wrong_speech.append(m["pred"].startswith(("respond:", "clarify:")))
        by_action[action].append(both)
        by_mode[r.meta["page_mode"]].append(both)
        by_family_variant[(r.meta["family"], rid.split("~v", 1)[1].split("-", 1)[0])].append(both)
    return {
        "move": ratio(move), "task": ratio(task), "full_step": ratio(full),
        "speaker_when_needed": ratio(speaker),
        "silent_wrong_speech": {"count": sum(silent_wrong_speech), "n": len(silent_wrong_speech)},
        "by_gold_action_full_step": {k: ratio(v) for k, v in sorted(by_action.items())},
        "by_page_mode_full_step": {k: ratio(v) for k, v in sorted(by_mode.items())},
        "whole_episode_variant": ratio([all(v) for v in by_family_variant.values()]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-set", required=True, type=Path)
    p.add_argument("--report", action="append", required=True, nargs=2, metavar=("NAME", "PATH"))
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    records = {r.id: r for r in read_jsonl(args.eval_set)}
    summary = {name: summarize(json.loads(Path(path).read_text()), records) for name, path in args.report}
    result = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
