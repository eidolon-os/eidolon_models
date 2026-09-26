#!/usr/bin/env python3
"""双人标注抽检：抽样 → 第二位标注者盲标 → 算一致率 → 列出不一致待裁决。只依赖标准库。

    python3 train/data/qa/qa.py sample b6-to-others b6-outside-keep --rate 0.1   # 写 blind-<batch>.jsonl（不含金标）
    #  … 第二位标注者按 LABELING.md 填写 blind 文件里的 intent / device / action …
    python3 train/data/qa/qa.py score <batch>                                   # 一致率 + disagreements-<batch>.jsonl

一致的判定：intent 相同；device / action 若双方都给了，两边的"可接受集合"有交集即算一致
（列表表示其中任一都算对，所以只要有一个共同答案，二者就不矛盾）。
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUTHORED = HERE.parent / "authored" / "smart-home"
EXITS = {"多个设备或整屋", "没有对应的设备"}


def as_set(v):
    if v is None:
        return None
    return set(v) if isinstance(v, list) else {v}


def cmd_sample(args) -> int:
    rows = []
    for name in args.files:
        f = AUTHORED / f"{name}.jsonl"
        for n, line in enumerate(f.read_text("utf-8").splitlines(), 1):
            if line.strip():
                r = json.loads(line)
                key = hashlib.sha256(f"qa:{name}:{n}".encode()).hexdigest()
                if int(key[:8], 16) / 0xFFFFFFFF < args.rate:
                    rows.append({"ref": f"{name}:{n}", "home": r["home"], "text": r["text"]})
    batch = args.batch or args.files[0]
    (HERE / f"blind-{batch}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    print(f"wrote blind-{batch}.jsonl: {len(rows)} rows")
    return 0


def cmd_score(args) -> int:
    blind = [json.loads(x) for x in (HERE / f"blind-{args.batch}.jsonl").read_text("utf-8").splitlines() if x.strip()]
    gold = {}
    for ref in {b["ref"].rsplit(":", 1)[0] for b in blind}:
        for n, line in enumerate((AUTHORED / f"{ref}.jsonl").read_text("utf-8").splitlines(), 1):
            if line.strip():
                gold[f"{ref}:{n}"] = json.loads(line)
    agree = {"intent": [0, 0], "device": [0, 0], "action": [0, 0]}
    dis = []
    for b in blind:
        g = gold[b["ref"]]
        if "intent" not in b:
            print(f"unlabelled: {b['ref']}", file=sys.stderr)
            continue
        row_ok = True
        for q in ("intent", "device", "action"):
            gv, bv = as_set(g.get(q)), as_set(b.get(q))
            if q == "intent":
                ok = gv == bv
            elif gv is None and bv is None:
                continue
            elif q == "action" and (gv is None or bv is None) and (
                (as_set(g.get("device")) or set()) | (as_set(b.get("device")) or set())
            ) <= EXITS:
                continue  # 出口行的动作可填可不填（LABELING §4、§8），一方没填不算分歧
            elif gv is None or bv is None:
                ok = False
            else:
                ok = bool(gv & bv)
            agree[q][0] += ok
            agree[q][1] += 1
            row_ok &= ok
        if not row_ok:
            dis.append({"ref": b["ref"], "home": b["home"], "text": b["text"],
                        "gold": {k: g.get(k) for k in ("intent", "device", "action") if g.get(k) is not None},
                        "second": {k: b.get(k) for k in ("intent", "device", "action") if b.get(k) is not None}})
    report = {q: (round(a / n, 4) if n else None, n) for q, (a, n) in agree.items()}
    (HERE / f"disagreements-{args.batch}.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in dis), "utf-8")
    (HERE / f"report-{args.batch}.json").write_text(json.dumps({"agreement": report, "rows": len(blind), "disagreements": len(dis)}, ensure_ascii=False, indent=1), "utf-8")
    print(f"agreement {report}; disagreements {len(dis)} -> disagreements-{args.batch}.jsonl")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample"); p.add_argument("files", nargs="+"); p.add_argument("--rate", type=float, default=0.1); p.add_argument("--batch"); p.set_defaults(func=cmd_sample)
    p = sub.add_parser("score"); p.add_argument("batch"); p.set_defaults(func=cmd_score)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
