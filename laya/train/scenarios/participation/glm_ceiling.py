"""Ceiling test: GLM answers p-dev decision points zero-shot with the full LABELING.md in the prompt.

    uv run --extra torch --extra train python train/scenarios/participation/glm_ceiling.py \
        --snapshots <snap-*.jsonl>... --out <dir> --per-call 6 --calls 100

Answers land in <dir>/answers.jsonl in the blind-snapshots format (a top-level "gold"), so
`episodes.py compare <episodes> <dir>/answers.jsonl --snapshots` scores them. Calls share the ledger and the
cap of glm_generate.py; nothing is retried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))

from glm_generate import BASE, CAP, LEDGER, MODEL, load_env  # noqa: E402

ASK = """你在为"智能陪伴 / IP 角色团"做参与决策。下面是完整的标注规范，严格按它判断。

{rules}

对下面每个决策点，只看给出的公开信息（场景、候选、到这一步为止的事件），判断在**最后一个事件**之后下一步应该怎样。
只输出 JSON：{{"answers": [{{"id": 决策点id, "action": "respond|clarify|wait|finish", "speaker": [候选id…], "clarify_about": "指代不明|要求不明|对象不在场"}}]}}
（wait / finish 不写 speaker；只有 clarify 写 clarify_about；respond / clarify 的 speaker 写所有合适的候选 id。）

决策点：
{items}"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-call", type=int, default=6)
    ap.add_argument("--calls", type=int, required=True)
    a = ap.parse_args()
    from eidolon_laya_train.generators import ChatClient

    load_env()
    client = ChatClient({"base_url": BASE, "model": MODEL, "api_key_env": "EIDOLON_IP_DATA_API_KEY", "timeout": 240,
                         "max_tokens": 4000, "thinking": {"type": "enabled"}, "reasoning_effort": "low",
                         "response_format": {"type": "json_object"}})
    rules = (HERE / "LABELING.md").read_text("utf-8")
    snaps = [json.loads(x) for p in a.snapshots for x in Path(p).read_text("utf-8").splitlines() if x.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (LEDGER / "calls").mkdir(parents=True, exist_ok=True)
    done = set()
    if (out / "answers.jsonl").exists():
        done = {json.loads(x)["id"] for x in (out / "answers.jsonl").read_text("utf-8").splitlines() if x.strip()}
    todo = [s for s in snaps if s["id"] not in done]
    made = 0
    while todo and made < a.calls:
        used = len(list((LEDGER / "calls").glob("*.json")))
        if used >= CAP:
            print(f"cap reached: {used}/{CAP}", flush=True)
            break
        batch, todo = todo[: a.per_call], todo[a.per_call:]
        items = "\n".join(json.dumps({k: v for k, v in s.items() if k != "gold"}, ensure_ascii=False) for s in batch)
        text = ASK.format(rules=rules, items=items)
        cid = f"{time.strftime('%Y%m%dT%H%M%S')}-{hashlib.sha1(text.encode()).hexdigest()[:8]}"
        rec = {"id": cid, "model": MODEL, "purpose": "ceiling", "snapshots": [s["id"] for s in batch], "status": "sent"}
        (LEDGER / "calls" / f"{cid}.json").write_text(json.dumps(rec, ensure_ascii=False))
        made += 1
        try:
            meta = client.complete_with_meta(text, temperature=0.0)
            rec.update(status="ok", usage=meta.get("usage"))
            got = {x.get("id"): x for x in json.loads(meta["text"]).get("answers", [])}
        except Exception as exc:  # noqa: BLE001 - counted, recorded, not retried
            rec.update(status="error", error=f"{type(exc).__name__}: {exc}"[:300])
            got = {}
        (LEDGER / "calls" / f"{cid}.json").write_text(json.dumps(rec, ensure_ascii=False))
        with (out / "answers.jsonl").open("a", encoding="utf-8") as f:
            for s in batch:
                g = got.get(s["id"]) or {}
                gold = {"action": g.get("action")}
                if g.get("speaker"):
                    gold["speaker"] = list(g["speaker"])
                if g.get("clarify_about"):
                    gold["clarify_about"] = g["clarify_about"]
                f.write(json.dumps({"id": s["id"], "gold": gold, "answered": s["id"] in got}, ensure_ascii=False) + "\n")
        print(f"call {cid} {rec['status']} answered {sum(s['id'] in got for s in batch)}/{len(batch)} (ledger {used + 1}/{CAP})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
