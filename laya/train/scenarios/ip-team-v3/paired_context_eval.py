"""Development-only paired test of irrelevant old public history."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

from eidolon_laya_train.evaluate import evaluate, write_report
from eidolon_laya_train.model import load_checkpoint, record_items
from eidolon_laya_train.records import read_jsonl, write_jsonl


BACKGROUND_MESSAGES = 11  # episodes.BACKGROUND, frozen for this v3 probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    long = [r for r in read_jsonl(args.val) if r.meta.get("history_stress")]
    if not long:
        raise ValueError("no history-stress development records")
    short = []
    for original in long:
        changed = copy.deepcopy(original)
        history = changed.state["prior_public_messages"]
        if len(history) < BACKGROUND_MESSAGES:
            raise ValueError(f"missing frozen background in {changed.id}")
        changed.state["prior_public_messages"] = history[BACKGROUND_MESSAGES:]
        changed.id += ":without-background"
        short.append(changed)

    loaded = load_checkpoint(args.checkpoint, "mps")
    lengths = {}
    for variant, records in (("with_background", long), ("without_background", short)):
        lengths[variant] = [len(record_items(r, loaded.tok, loaded.cfg)[0]["ids"]) for r in records]
        if max(lengths[variant]) > int(loaded.cfg["max_len"]):
            raise ValueError(f"{variant} exceeds configured window")
    args.out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for variant, records in (("with_background", long), ("without_background", short)):
        path = args.out / f"{variant}.jsonl"
        write_jsonl(path, records)
        paths[variant] = path
    reports = {variant: evaluate(loaded, path, batch_size=4) for variant, path in paths.items()}
    for variant, report in reports.items():
        write_report(report, args.out / f"{variant}.report.json")

    rows = {variant: {row["record_id"].split(":without-background")[0]: row
                      for row in report["rows"]} for variant, report in reports.items()}
    by_family = defaultdict(lambda: {"n": 0, "with_background": 0, "without_background": 0})
    both, long_only, short_only, neither = 0, 0, 0, 0
    for original in long:
        rid = original.id
        gold = original.labels["move"]["gold"]
        a = rows["with_background"][rid]["pred"] == gold
        b = rows["without_background"][rid]["pred"] == gold
        family = by_family[original.meta["family"]]
        family["n"] += 1
        family["with_background"] += a
        family["without_background"] += b
        if a and b:
            both += 1
        elif a:
            long_only += 1
        elif b:
            short_only += 1
        else:
            neither += 1
    result = {
        "split": "val_development_only", "paired_records": len(long),
        "independent_families": len(by_family), "removed_old_public_messages": BACKGROUND_MESSAGES,
        "token_range": {k: [min(v), max(v)] for k, v in lengths.items()},
        "with_background_correct": both + long_only,
        "without_background_correct": both + short_only,
        "paired_outcomes": {"both_correct": both, "long_only_correct": long_only,
                            "short_only_correct": short_only, "neither_correct": neither},
        "by_family": dict(by_family),
    }
    (args.out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
