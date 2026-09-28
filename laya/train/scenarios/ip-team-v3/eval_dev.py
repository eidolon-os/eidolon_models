"""Memory-bounded development evaluation; leaves frozen comparison unopened."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eidolon_laya_train.assemble import stable_file_hash
from eidolon_laya_train.evaluate import evaluate, write_report
from eidolon_laya_train.model import load_checkpoint
from eidolon_laya_train.records import read_jsonl

from report import summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    loaded = load_checkpoint(args.checkpoint, "mps")
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "val"):
        path = args.dataset / f"{split}.jsonl"
        report = evaluate(loaded, path, batch_size=args.batch_size)
        report["checkpoint_sha256"] = stable_file_hash(args.checkpoint / "model.safetensors")
        write_report(report, args.out / f"{split}.json")
        records = {r.id: r for r in read_jsonl(path)}
        summary[split] = summarize(report, records)
    train, val = summary["train"], summary["val"]
    summary["gate"] = {
        "train_move_at_least_85_percent": train["move"]["accuracy"] >= .85,
        "val_move_at_least_65_percent": val["move"]["accuracy"] >= .65,
        "val_silent_wrong_speech_at_most_one": val["silent_wrong_speech"]["count"] <= 1,
    }
    summary["gate"]["passed"] = all(summary["gate"].values())
    result = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    (args.out / "dev-summary.json").write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
