#!/usr/bin/env python3
"""从撰写训练数据里删掉与锁定评测集（v1 + v2）完全相同或近似重复的行。只依赖标准库。

    python3 evals/smart-home-v2/purge_training_overlap.py          # 只报告
    python3 evals/smart-home-v2/purge_training_overlap.py --apply  # 真删，被删的行写进 purged.jsonl

规则与 check_leakage.py 相同：去标点后相同，或编辑距离 ≤ 2（句长 ≥ 8）/ ≤ 1（句长 4–7）。
选择删训练行而不是改评测句：评测句要保持自然说法，训练数据多，删几十行没有代价。
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_leakage import LAYA, lev_le, norm  # noqa: E402

HERE = Path(__file__).resolve().parent


def locked_texts() -> list[tuple[str, str]]:
    out = []
    for root in (LAYA / "evals/smart-home", LAYA / "evals/smart-home-v2"):
        for f in sorted(root.glob("scenarios/*/cases.jsonl")):
            for line in f.read_text("utf-8").splitlines():
                if line.strip():
                    c = json.loads(line)
                    out.append((c["id"], norm(c["text"])))
    return out


def hit(n: str, locked: list[tuple[str, str]]) -> str | None:
    for lid, ln in locked:
        if n == ln:
            return lid
        if len(n) >= 4 and len(ln) >= 4 and lev_le(n, ln, 2 if min(len(n), len(ln)) >= 8 else 1):
            return lid
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    locked = locked_texts()
    purged, total = [], 0
    for f in sorted(glob.glob(str(LAYA / "train/data/authored/**/*.jsonl"), recursive=True)):
        lines = Path(f).read_text("utf-8").splitlines()
        keep = []
        for n, line in enumerate(lines, 1):
            if not line.strip():
                continue
            total += 1
            r = json.loads(line)
            lid = hit(norm(r["text"]), locked)
            if lid:
                purged.append({"file": f"{Path(f).name}:{n}", "locked": lid, **r})
            else:
                keep.append(line)
        if args.apply and len(keep) != len([x for x in lines if x.strip()]):
            Path(f).write_text("\n".join(keep) + "\n", encoding="utf-8")
    by = {}
    for p in purged:
        by.setdefault(p["locked"][:5] if p["locked"].startswith("v2-") else "v1", 0)
        by[p["locked"][:5] if p["locked"].startswith("v2-") else "v1"] += 1
    print(f"authored {total} rows; overlapping a locked case: {len(purged)} {by}")
    if args.apply:
        out = HERE / "purged.jsonl"
        with out.open("a", encoding="utf-8") as fh:
            for p in purged:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"removed; record appended to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
