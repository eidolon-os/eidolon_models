"""Add reviewed GLM train families to the frozen v3 train split only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eidolon_laya_train.records import read_jsonl, write_jsonl


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--generated", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expected-families", type=int, default=18)
    ap.add_argument("--expected-records", type=int, default=92)
    args = ap.parse_args()
    base = {part: list(read_jsonl(args.base / f"{part}.jsonl")) for part in ("train", "val", "calib")}
    new = list(read_jsonl(args.generated))
    if len(new) != args.expected_records or len({r.meta["family"] for r in new}) != args.expected_families:
        raise ValueError("reviewed GLM expansion has unexpected family/record count")
    families = {part: {r.meta["family"] for r in rows} for part, rows in base.items()}
    new_families = {r.meta["family"] for r in new}
    if any(new_families & old for old in families.values()):
        raise ValueError("generated families overlap a pre-existing split")
    all_rows = [*base["train"], *new, *base["val"], *base["calib"]]
    if len({r.id for r in all_rows}) != len(all_rows):
        raise ValueError("duplicate record ID")
    if any(r.split != "train" or r.scenario != "ip-team-v3" for r in new):
        raise ValueError("generated rows have unexpected scenario or split")
    args.out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for part, rows in (("train", [*base["train"], *new]), ("val", base["val"]),
                       ("calib", base["calib"])):
        path = args.out / f"{part}.jsonl"
        if path.exists():
            raise ValueError(f"refusing to overwrite existing split: {path}")
        write_jsonl(path, rows)
        paths[part] = {"records": len(rows), "families": len({r.meta["family"] for r in rows}),
                       "sha256": sha(path)}
    for part in ("val", "calib"):
        if sha(args.base / f"{part}.jsonl") != paths[part]["sha256"]:
            raise ValueError(f"{part} changed unexpectedly")
    (args.out / "lineage.json").write_text(json.dumps({"base": str(args.base.resolve()),
        "base_train_sha256": sha(args.base / "train.jsonl"),
        "generated_sha256": sha(args.generated), "splits": paths}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(paths, ensure_ascii=False))


if __name__ == "__main__":
    main()
