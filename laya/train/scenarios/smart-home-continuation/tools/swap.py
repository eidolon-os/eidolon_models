"""换序一致率（PLAN.md §4）：选项打乱 k 次后 argmax 的选项名与原顺序一致的比例。

    .venv/bin/python train/scenarios/smart-home-continuation/tools/swap.py \
        --checkpoint train/runs/c1/checkpoint --eval-set ../evals/smart-home-continuation/c-dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import random

from eidolon_laya_train.model import load_checkpoint, record_items, score_items
from eidolon_laya_train.records import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--device")
    args = ap.parse_args()
    loaded = load_checkpoint(args.checkpoint, args.device)
    recs = list(read_jsonl(args.eval_set))

    def argmax_names(shuffle):
        items = [it for r in recs for it in record_items(r, loaded.tok, loaded.cfg, shuffle=shuffle)]
        scored = score_items(loaded, items)
        return {(it["record_id"], it["qid"]): it["names"][max(range(len(it["logits"])), key=it["logits"].__getitem__)]
                for it in scored}

    base = argmax_names(None)
    same = total = 0
    flips = []
    for k in range(args.k):
        other = argmax_names(random.Random(1000 + k))
        for key, name in base.items():
            total += 1
            if other[key] == name:
                same += 1
            else:
                flips.append({"record_id": key[0], "base": name, "shuffled": other[key]})
    print(json.dumps({"eval_set": args.eval_set, "k": args.k, "n": total, "consistent": round(same / total, 4),
                      "flips": flips[:50]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
