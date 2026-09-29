"""分任务看温度（PLAN.md §3）：同一份 calib 上，联合拟合的温度与按任务分别拟合的温度，各自的 ECE / NLL。

    .venv/bin/python train/scenarios/smart-home-continuation/tools/calib_split.py \
        --checkpoint train/runs/c1/checkpoint --calib train/runs/c1/dataset/calib.jsonl

按 (任务, 选项数分档) 分组；“联合”用 checkpoint 里 calibrate 写好的温度（temperature_for），“分别”在每组上单独拟合。
分别拟合后某任务 ECE 降超过 0.03，才考虑在 Agent 适配器里按题重新调温。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

from eidolon_laya_train.calibrate import _nll, fit_temperature
from eidolon_laya_train.evaluate import temperature_for
from eidolon_laya_train.model import load_checkpoint, record_items, score_items
from eidolon_laya_train.records import read_jsonl
from eidolon_models_laya.vendor.laya.common import temp_bucket


def ece(items: list[dict], T: float, bins: int = 10) -> float:
    acc = defaultdict(list)
    for it in items:
        z = [x / T for x in it["logits"]]
        m = max(z)
        e = [math.exp(x - m) for x in z]
        s = sum(e)
        p = [x / s for x in e]
        top = max(range(len(p)), key=p.__getitem__)
        acc[min(int(p[top] * bins), bins - 1)].append((p[top], it["target"][top] > 0))
    n = sum(len(v) for v in acc.values())
    return sum(abs(sum(c for _p, c in v) / len(v) - sum(p for p, _c in v) / len(v)) * len(v) / n
               for v in acc.values() if v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--device")
    args = ap.parse_args()
    loaded = load_checkpoint(args.checkpoint, args.device)
    recs = list(read_jsonl(args.calib))
    scen = {r.id: r.scenario for r in recs}
    items = [it for r in recs for it in record_items(r, loaded.tok, loaded.cfg)]
    scored = score_items(loaded, items)
    groups: dict[tuple, list] = defaultdict(list)
    for it in scored:
        groups[(scen[it["record_id"]], temp_bucket(it["qtype"], len(it["markers"])))].append(it)
        groups[(scen[it["record_id"]], "all")].append(it)
    out = {}
    for (task, bucket), its in sorted(groups.items()):
        k = len(its[0]["markers"]) if bucket != "all" else None
        t_joint = temperature_for(loaded.cfg, its[0]["qtype"], k) if k else None
        t_own = fit_temperature(its)
        row = {"n": len(its), "T_own": round(t_own, 3), "ece_own": round(ece(its, t_own), 4),
               "nll_own": round(_nll(its, t_own), 4), "ece_T1": round(ece(its, 1.0), 4)}
        if t_joint is not None:
            row |= {"T_joint": round(t_joint, 3), "ece_joint": round(ece(its, t_joint), 4),
                    "nll_joint": round(_nll(its, t_joint), 4)}
        out[f"{task} {bucket}"] = row
    out["checkpoint_temperatures"] = {"temperature": loaded.cfg.get("temperature"),
                                      "temperature_by_options": loaded.cfg.get("temperature_by_options")}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
