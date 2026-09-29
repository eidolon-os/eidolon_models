"""单句补充数据（smart-home-c3）的盲标：`make` 抽 20% 写盲标文件（不含金标），`report` 对比盲标与金标。

    .venv/bin/python train/scenarios/smart-home-continuation/tools/blind_single.py make
    .venv/bin/python train/scenarios/smart-home-continuation/tools/blind_single.py report

对比 intent 与 device（金标是列表时，盲标落在列表里算一致）；action 只在两边都是控制时比。
"""
import json
import random
import sys
from pathlib import Path

LAYA = Path(__file__).resolve().parents[4]
ROOT = LAYA / "train/data/authored/smart-home-c3"
QA = ROOT / "qa"
HOMES = LAYA / "train/data/homes/smart-home"


def rows():
    out = []
    for f in sorted(ROOT.glob("b*.jsonl")):
        for n, line in enumerate(f.read_text("utf-8").splitlines(), 1):
            if line.strip():
                c = json.loads(line)
                c["qid"] = f"{f.stem}:{n}"
                out.append(c)
    return out


def as_set(v):
    return set(v) if isinstance(v, list) else ({v} if v else set())


if sys.argv[1] == "make":
    rs = rows()
    rng = random.Random(3)
    pick = rng.sample(rs, max(1, len(rs) // 5))
    QA.mkdir(exist_ok=True)
    with (QA / "blind.jsonl").open("w", encoding="utf-8") as fh:
        for c in pick:
            home = json.loads((HOMES / f"{c['home']}.json").read_text("utf-8"))
            devs = [f"{d['name']}（{d['room']}·{d['type']}）" for d in home["devices"]]
            fh.write(json.dumps({"qid": c["qid"], "home": c["home"], "设备清单": devs, "text": c["text"]},
                                ensure_ascii=False) + "\n")
    print("blind", len(pick))
else:
    key = {c["qid"]: c for c in rows()}
    n = agree = 0
    dis = []
    for line in (QA / "labels.jsonl").read_text("utf-8").splitlines():
        b = json.loads(line)
        g = key.get(b["qid"])
        if g is None:
            continue
        n += 1
        ok = b["intent"] == g["intent"]
        if ok and g["intent"] in ("控制", "查询"):
            ok = bool(as_set(b.get("device")) & as_set(g.get("device")))
        if ok and g["intent"] == "控制" and b.get("action") and g.get("action"):
            ok = bool(as_set(b["action"]) & as_set(g["action"]))
        agree += ok
        if not ok:
            dis.append({"qid": b["qid"], "text": g["text"], "gold": {k: g.get(k) for k in ("intent", "device", "action")},
                        "blind": {k: b.get(k) for k in ("intent", "device", "action")}, "note": b.get("note")})
    (QA / "disagreements.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in dis), "utf-8")
    print(f"agree {agree}/{n} = {agree / max(n, 1):.3f}")
    for d in dis:
        print(" ", d)
