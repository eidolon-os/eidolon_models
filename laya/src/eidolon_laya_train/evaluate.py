"""``eval``: score a checkpoint on record-shaped eval sets, slice the results, gate against a baseline.

Per question: predicted option, its calibrated probability, whether it is among the gold
answers. Aggregates per scenario / question type / tag: accuracy, ECE (15 bins), and
coverage/accuracy at confidence thresholds. ``gate`` compares two eval reports and fails if any
scenario's accuracy dropped by more than ``tolerance``.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from eidolon_models_laya.vendor.laya.common import (
    QTYPE_NAMES,
    clamp_temperature,
    ece_score,
    temp_bucket,
)

from .model import Loaded, record_items, score_items
from .records import Record, gold_names, read_jsonl

THRESHOLDS = (0.5, 0.7, 0.9)


def _softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def temperature_for(cfg: dict, qtype: int, k: int) -> float:
    by = cfg.get("temperature_by_options") or {}
    b = temp_bucket(qtype, k)
    if b in by:
        return clamp_temperature(by[b])
    temps = cfg.get("temperature") or [1.0, 1.0, 1.0]
    return clamp_temperature(temps[qtype])


def score_records(loaded: Loaded, records: list[Record], batch_size: int = 16) -> list[dict]:
    by_id = {r.id: r for r in records}
    items = [it for r in records for it in record_items(r, loaded.tok, loaded.cfg)]
    rows = []
    for it in score_items(loaded, items, batch_size):
        r = by_id[it["record_id"]]
        q = r.questions[it["qid"]]
        T = temperature_for(loaded.cfg, it["qtype"], len(it["markers"]))
        p = _softmax([x / T for x in it["logits"]])
        j = max(range(len(p)), key=p.__getitem__)
        golds = gold_names(q, r.labels[it["qid"]])
        rows.append(
            {
                "record_id": r.id,
                "scenario": r.scenario,
                "qid": it["qid"],
                "qtype": QTYPE_NAMES[it["qtype"]],
                "tags": r.tags,
                "pred": it["names"][j],
                "p_top": round(p[j], 4),
                "probabilities": {n: round(x, 4) for n, x in zip(it["names"], p, strict=True)},
                "gold": sorted(golds),
                "correct": it["names"][j] in golds,
                "n_options": len(p),
            }
        )
    return rows


def _agg(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    conf = np.array([r["p_top"] for r in rows])
    ok = np.array([r["correct"] for r in rows], dtype=float)
    out = {"n": len(rows), "acc": round(float(ok.mean()), 4), "ece": round(ece_score(conf, ok), 4)}
    for th in THRESHOLDS:
        sel = conf >= th
        out[f"p>={th}"] = {
            "coverage": round(float(sel.mean()), 4),
            "acc": round(float(ok[sel].mean()), 4) if sel.any() else None,
        }
    return out


def evaluate(loaded: Loaded, eval_path: Path, *, batch_size: int = 16) -> dict:
    records = list(read_jsonl(eval_path))
    rows = score_records(loaded, records, batch_size)
    by_scn, by_qt, by_qid, by_tag = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for r in rows:
        by_scn[r["scenario"]].append(r)
        by_qt[r["qtype"]].append(r)
        by_qid[f"{r['scenario']}/{r['qid']}"].append(r)
        for t in r["tags"]:
            by_tag[t].append(r)
    return {
        "eval_set": str(eval_path),
        "n_records": len(records),
        "overall": _agg(rows),
        "by_scenario": {k: _agg(v) for k, v in sorted(by_scn.items())},
        "by_qtype": {k: _agg(v) for k, v in sorted(by_qt.items())},
        "by_question": {k: _agg(v) for k, v in sorted(by_qid.items())},
        "by_tag": {k: _agg(v) for k, v in sorted(by_tag.items()) if len(v) >= 5},
        "rows": rows,
    }


def gate(candidate: dict, baseline: dict, tolerance: float = 0.0, min_n: int = 10) -> dict:
    """Every scenario, question and tagged slice (with at least ``min_n`` items) must hold its
    accuracy within ``tolerance``. Slices are where regressions hide: a checkpoint can lift the
    overall number while learning to call every utterance a command."""
    checks = []
    for key in ("by_scenario", "by_question", "by_tag"):
        for name, base in baseline.get(key, {}).items():
            cand = candidate.get(key, {}).get(name)
            if not cand or base.get("n", 0) < min_n:
                continue
            ok = cand["acc"] >= base["acc"] - tolerance
            checks.append(
                {
                    "slice": f"{key}:{name}",
                    "baseline": base["acc"],
                    "candidate": cand["acc"],
                    "ok": ok,
                }
            )
    return {"passed": all(c["ok"] for c in checks), "tolerance": tolerance, "checks": checks}


def write_report(report: dict, path: Path, *, keep_rows: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(report)
    if not keep_rows:
        data.pop("rows", None)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
