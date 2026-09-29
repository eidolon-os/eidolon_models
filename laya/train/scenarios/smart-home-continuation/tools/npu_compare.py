"""NPU（laya_npu.py run 的 logits）对 torch（items 里的 ref_logits）逐题比较：同选项率、校准后 |Δp|、翻转的题。

    .venv/bin/python train/scenarios/smart-home-continuation/tools/npu_compare.py \
        --items-dir train/runs/c4/agent-replay/items-c4 --logits-dir train/runs/c4/npu/npu-out \
        --config models/laya-smart-home/7b695ba8/torch/rl_agent_config.json --records-dir <集合名 → 记录文件>...

指标层（续接六类结局、单句准确率 / 自动执行）用 `eidolon-laya-train eval --logits` 与 cmetrics；决策层用 agent_replay.py --logits。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from eidolon_laya_train.evaluate import temperature_for


def softmax(z, t):
    z = [x / t for x in z]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-dir", required=True)
    ap.add_argument("--logits-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--records", nargs="+", required=True, help="记录文件；按文件名（去掉 .jsonl）找 items / logits")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text("utf-8"))
    out = {}
    for rec_path in a.records:
        name = Path(rec_path).stem
        utt = {json.loads(x)["id"]: json.loads(x)["state"]["utterance"]
               for x in Path(rec_path).read_text("utf-8").splitlines() if x.strip()}
        items = {it["key"]: it for it in map(json.loads, (Path(a.items_dir) / f"{name}.items.jsonl").read_text("utf-8").splitlines())}
        npu = {r["key"]: r for r in map(json.loads, (Path(a.logits_dir) / f"{name}.logits.jsonl").read_text("utf-8").splitlines())}
        by_q: dict[str, list] = {}
        flips, max_dp, max_dl = [], 0.0, 0.0
        for key, it in items.items():
            if key not in npu:
                continue
            t = temperature_for(cfg, it["qtype"], len(it["ref_logits"]))
            pt, pn = softmax(it["ref_logits"], t), softmax(npu[key]["logits"], t)
            at, an = max(range(len(pt)), key=pt.__getitem__), max(range(len(pn)), key=pn.__getitem__)
            q = key.rsplit("/", 1)[1]
            by_q.setdefault(q, []).append(at == an)
            max_dp = max(max_dp, max(abs(x - y) for x, y in zip(pt, pn, strict=True)))
            max_dl = max(max_dl, max(abs(x - y) for x, y in zip(it["ref_logits"], npu[key]["logits"], strict=True)))
            if at != an:
                rid = key.rsplit("/", 1)[0]
                flips.append({"key": key, "utterance": utt.get(rid), "torch": [it["names"][at], round(pt[at], 4)],
                              "npu": [it["names"][an], round(pn[an], 4)], "bucket": npu[key].get("bucket")})
        n = sum(len(v) for v in by_q.values())
        out[name] = {"items": n, "missing": len(items) - n,
                     "same_choice": round(sum(sum(v) for v in by_q.values()) / n, 5) if n else None,
                     "by_question": {q: {"n": len(v), "same": sum(v)} for q, v in sorted(by_q.items())},
                     "max_abs_dp_calibrated": round(max_dp, 4), "max_abs_dlogit": round(max_dl, 4), "flips": flips}
        print(f"{name:14s} items {n} same {out[name]['same_choice']} flips {len(flips)} max|Δp| {max_dp:.4f} max|Δlogit| {max_dl:.4f}")
        for f in flips:
            print(f"    {f['key']} | {f['utterance']} | torch {f['torch']} → npu {f['npu']} (L{f['bucket']})")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", "utf-8")


if __name__ == "__main__":
    main()
