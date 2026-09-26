#!/usr/bin/env python3
"""评测集的全量双人标注：撰写者的金标（cases.jsonl）对第二位标注者的盲标（qa/blind-*.jsonl）。只依赖标准库。

    python3 evals/double_annotation.py evals/smart-home-v3                # 一致率 + qa/disagreements.jsonl
    python3 evals/double_annotation.py evals/smart-home-v3 --union-lists  # 双方都给列表且有交集时，把金标改成并集（LABELING §8）

一致的判定与 train/data/qa/qa.py 相同：意图相同；设备 / 动作两边的"可接受集合"有交集；出口行（多个设备或整屋 /
没有对应的设备）的动作一方没填不算分歧。意图或设备不一致的逐条裁决后改 cases.jsonl，并在行上记 `adjudicated`。
"""

import argparse
import json
import sys
from pathlib import Path

EXITS = {"多个设备或整屋", "没有对应的设备"}


def as_set(v):
    if v is None:
        return None
    return set(v) if isinstance(v, list) else {v}


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def compare(a: dict, b: dict) -> dict[str, bool | None]:
    out = {"intent": a.get("intent") == b.get("intent")}
    for q in ("device", "action"):
        av, bv = as_set(a.get(q)), as_set(b.get(q))
        if av is None and bv is None:
            out[q] = None
        elif q == "action" and (av is None or bv is None) and (
            (as_set(a.get("device")) or set()) | (as_set(b.get("device")) or set())
        ) <= EXITS:
            out[q] = None
        elif av is None or bv is None:
            out[q] = False
        else:
            out[q] = bool(av & bv)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--union-lists", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    blind = {}
    for f in sorted((root / "qa").glob("blind-*.jsonl")):
        for r in load(f):
            blind[r["id"]] = r
    agree = {q: [0, 0] for q in ("intent", "device", "action")}
    dis, strict = [], {"device": [0, 0], "action": [0, 0]}
    for f in sorted(root.glob("scenarios/*/cases.jsonl")):
        rows = load(f)
        changed = False
        for c in rows:
            b = blind.get(c["id"])
            if b is None or "gold" not in b:
                print(f"no second annotation for {c['id']}", file=sys.stderr)
                continue
            if b["text"] != c["text"] or b["home"] != c["home"]:
                sys.exit(f"{c['id']}: blind row does not match the case")
            res = compare(c["gold"], b["gold"])
            for q, ok in res.items():
                if ok is not None:
                    agree[q][0] += ok
                    agree[q][1] += 1
            for q in ("device", "action"):
                if res[q] is not None and res["intent"]:
                    strict[q][0] += as_set(c["gold"].get(q)) == as_set(b["gold"].get(q))
                    strict[q][1] += 1
            if not all(v is not False for v in res.values()):
                dis.append({"id": c["id"], "home": c["home"], "text": c["text"], "first": c["gold"],
                            "second": b["gold"], "notes": [x for x in (c.get("note"), b.get("note")) if x]})
            elif args.union_lists and res["intent"]:
                for q in ("device", "action"):
                    a, bb = as_set(c["gold"].get(q)), as_set(b["gold"].get(q))
                    if a and bb and a != bb:
                        order = ([c["gold"][q]] if isinstance(c["gold"][q], str) else c["gold"][q])
                        extra = [x for x in (b["gold"][q] if isinstance(b["gold"][q], list) else [b["gold"][q]]) if x not in a]
                        c["gold"][q] = order + extra
                        c.setdefault("adjudicated", []).append(f"{q}: union with second annotator")
                        changed = True
        if changed:
            f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    report = {q: {"agree": round(a / n, 4) if n else None, "n": n} for q, (a, n) in agree.items()}
    report["strict_same_set"] = {q: round(a / n, 4) if n else None for q, (a, n) in strict.items()}
    report["disagreements"] = len(dis)
    (root / "qa" / "disagreements.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in dis), "utf-8")
    (root / "qa" / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
