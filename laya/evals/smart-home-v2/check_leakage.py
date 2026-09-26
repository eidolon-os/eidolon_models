#!/usr/bin/env python3
"""v2 锁定集 vs 全部训练文本：完全相同 / 去标点后相同 / 编辑距离 ≤ 2（撰写数据）。只依赖标准库。

    python3 evals/smart-home-v2/check_leakage.py [--fix-report out.json]

训练文本来源：train/data/authored/**/*.jsonl（text）、train/data/external 转换后的记录（如果存在
train/data/external/external-records.jsonl）、以及 v1 锁定集（v2 也不该和 v1 撞）。
"""

import argparse
import glob
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYA = HERE.parent.parent
PUNCT = re.compile(r"[\s，。！？、,.!?~…：:；;\"'“”‘’（）()【】\[\]]+")


def norm(s: str) -> str:
    return PUNCT.sub("", s)


def lev_le(a: str, b: str, k: int) -> bool:
    if abs(len(a) - len(b)) > k:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        lo = i
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            lo = min(lo, cur[j])
        if lo > k:
            return False
        prev = cur
    return prev[-1] <= k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", help="write hits as JSON")
    args = ap.parse_args()
    v2 = []
    for f in sorted((HERE / "scenarios").glob("*/cases.jsonl")):
        for line in f.read_text("utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                v2.append((c["id"], c["text"]))
    authored = []
    for f in glob.glob(str(LAYA / "train/data/authored/**/*.jsonl"), recursive=True):
        for n, line in enumerate(open(f, encoding="utf-8"), 1):
            if line.strip():
                authored.append((f"{Path(f).name}:{n}", json.loads(line)["text"]))
    v1 = []
    for f in (LAYA / "evals/smart-home/scenarios").glob("*/cases.jsonl"):
        for line in f.read_text("utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                v1.append((f"v1:{c['id']}", c["text"]))
    external = []
    ext = LAYA / "train/data/external/external-records.jsonl"
    if ext.exists():
        for line in open(ext, encoding="utf-8"):
            r = json.loads(line)
            external.append((r["source"], r["state"]["utterance"]))
    exact_ext = {norm(t): w for w, t in external}
    near_pool = [(w, t, norm(t)) for w, t in authored + v1]
    hits = []
    for vid, text in v2:
        n = norm(text)
        found = None
        if n in exact_ext:
            found = ("exact-external", exact_ext[n], None)
        else:
            for w, t, tn in near_pool:
                if tn == n:
                    found = ("exact", w, t)
                    break
                if len(n) >= 4 and lev_le(n, tn, 2 if len(n) >= 8 else 1):
                    found = ("near", w, t)
                    break
        if found:
            hits.append({"id": vid, "text": text, "kind": found[0], "where": found[1], "match": found[2]})
    print(f"v2 {len(v2)} cases vs authored {len(authored)} + v1 {len(v1)} + external {len(external)}: {len(hits)} hits")
    for h in hits:
        print(f"  {h['id']} [{h['kind']}] {h['text']}  <->  {h['match'] or ''} ({h['where']})")
    if args.report:
        Path(args.report).write_text(json.dumps(hits, ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
