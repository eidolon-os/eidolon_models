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


def item_key(it: dict) -> str:
    return f"{it['record_id']}/{it['qid']}"


def score_records(
    loaded: Loaded, records: list[Record], batch_size: int = 16, logits: dict[str, list[float]] | None = None
) -> list[dict]:
    """``logits`` (item key → raw logits) replaces the model forward: that is how a platform runner's
    output (``export`` → device → ``*.logits.jsonl``) is scored with exactly the same calibration and metrics."""
    by_id = {r.id: r for r in records}
    items = [it for r in records for it in record_items(r, loaded.tok, loaded.cfg)]
    if logits is None:
        scored = score_items(loaded, items, batch_size)
    else:
        missing = [item_key(it) for it in items if item_key(it) not in logits]
        if missing:
            raise ValueError(f"{len(missing)} items have no logits, e.g. {missing[0]}")
        scored = [dict(it, logits=logits[item_key(it)]) for it in items]
    rows = []
    for it in scored:
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


def _bootstrap_ci(values: list[float], n: int = 1000, seed: int = 7) -> list[float]:
    """95% percentile bootstrap interval of the mean (resampling units, not questions)."""
    if not values:
        return [None, None]
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def decision_metrics(rows: list[dict], threshold: float = 0.9) -> dict:
    """Record-level numbers that decide whether the model may act on its own.

    - ``control_e2e``: gold-控制 records with every scored question right.
    - ``auto_execute``: the record the model would execute without asking — intent predicted
      控制 and every question's top probability ≥ ``threshold``. ``precision`` = all its answers
      right; ``coverage`` = share of gold-控制 records executed this way.
    - ``false_trigger``: gold 无关 (and gold 查询) records predicted 控制.
    Intervals are 95% bootstrap over records.
    """
    by: dict[str, dict] = defaultdict(dict)
    for r in rows:
        by[r["record_id"]][r["qid"]] = r
    recs = [q for q in by.values() if "intent" in q]
    gold_ctrl = [q for q in recs if q["intent"]["gold"] == ["控制"]]
    e2e = [float(all(x["correct"] for x in q.values())) for q in gold_ctrl]
    auto = [
        q
        for q in recs
        if q["intent"]["pred"] == "控制" and min(x["p_top"] for x in q.values()) >= threshold
    ]
    auto_ok = [float(all(x["correct"] for x in q.values())) for q in auto]
    auto_ctrl = [q for q in auto if q["intent"]["gold"] == ["控制"]]
    out = {
        "records": len(recs),
        "control_e2e": {
            "n": len(gold_ctrl),
            "acc": round(float(np.mean(e2e)), 4) if e2e else None,
            "ci95": _bootstrap_ci(e2e),
        },
        "auto_execute": {
            "threshold": threshold,
            "n": len(auto),
            "precision": round(float(np.mean(auto_ok)), 4) if auto_ok else None,
            "ci95": _bootstrap_ci(auto_ok),
            "coverage": round(len(auto_ctrl) / len(gold_ctrl), 4) if gold_ctrl else None,
            "wrong": [
                next(iter(q.values()))["record_id"]
                for q in auto
                if not all(x["correct"] for x in q.values())
            ],
        },
    }
    for g in ("无关", "查询"):
        gold = [q for q in recs if q["intent"]["gold"] == [g]]
        ft = [float(q["intent"]["pred"] == "控制") for q in gold]
        out[f"false_trigger_{g}"] = {
            "n": len(gold),
            "rate": round(float(np.mean(ft)), 4) if ft else None,
            "ci95": _bootstrap_ci(ft),
        }
    return out


def dump_items(loaded: Loaded, eval_path: Path, *, batch_size: int = 16) -> list[dict]:
    """Model inputs for every question of an eval set, with the reference (this backend's) logits:
    what a platform runner consumes, and what it is checked against."""
    records = list(read_jsonl(eval_path))
    items = [it for r in records for it in record_items(r, loaded.tok, loaded.cfg)]
    return [
        {"key": item_key(it), "qtype": it["qtype"], "ids": list(it["ids"]), "markers": list(it["markers"]),
         "ref_logits": [round(x, 5) for x in it["logits"]]}
        for it in score_items(loaded, items, batch_size)
    ]


def evaluate(
    loaded: Loaded, eval_path: Path, *, batch_size: int = 16, logits: dict[str, list[float]] | None = None
) -> dict:
    records = list(read_jsonl(eval_path))
    rows = score_records(loaded, records, batch_size, logits)
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
    by_record: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_record[r["record_id"]].append(float(r["correct"]))
    overall = _agg(rows)
    overall["ci95"] = _bootstrap_ci(
        [float(np.mean(v)) for v in by_record.values()]
    )  # record-level resampling
    return {
        "eval_set": str(eval_path),
        "n_records": len(records),
        "overall": overall,
        "decision": decision_metrics(rows),
        "by_scenario": {k: _agg(v) for k, v in sorted(by_scn.items())},
        "by_qtype": {k: _agg(v) for k, v in sorted(by_qt.items())},
        "by_question": {k: _agg(v) for k, v in sorted(by_qid.items())},
        "by_tag": {k: _agg(v) for k, v in sorted(by_tag.items()) if len(v) >= 5},
        "rows": rows,
    }


def _sign_test_p(broke: int, fixed: int) -> float:
    """One-sided exact binomial p that ``broke`` flips out of ``broke + fixed`` arise by chance."""
    n = broke + fixed
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(broke, n + 1)) / 2**n


def gate(
    candidate: dict,
    baseline: dict,
    tolerance: float = 0.0,
    min_n: int = 10,
    alpha: float | None = None,
) -> dict:
    """Every scenario, question and tagged slice (with at least ``min_n`` items) must hold its
    accuracy within ``tolerance``. Slices are where regressions hide: a checkpoint can lift the
    overall number while learning to call every utterance a command.

    When both reports carry per-question rows the check is paired: each slice reports how many
    questions the candidate broke and fixed, and with ``alpha`` set a slice only fails if the drop
    exceeds ``tolerance`` *and* the broke/fixed split is significant (one-sided sign test), so a
    couple of low-confidence flips on a 20-case slice do not block a release."""
    paired = "rows" in candidate and "rows" in baseline
    if paired:
        base_rows = {(r["record_id"], r["qid"]): r for r in baseline["rows"]}
        cand_rows = {(r["record_id"], r["qid"]): r for r in candidate["rows"]}

    def members(key: str, name: str):
        for k, r in base_rows.items():
            if k not in cand_rows:
                continue
            if key == "by_scenario" and r["scenario"] == name:
                yield r, cand_rows[k]
            elif key == "by_question" and f"{r['scenario']}/{r['qid']}" == name:
                yield r, cand_rows[k]
            elif key == "by_tag" and name in r["tags"]:
                yield r, cand_rows[k]

    checks = []
    for key in ("by_scenario", "by_question", "by_tag"):
        for name, base in baseline.get(key, {}).items():
            cand = candidate.get(key, {}).get(name)
            if not cand or base.get("n", 0) < min_n:
                continue
            drop_ok = cand["acc"] >= base["acc"] - tolerance
            check = {"slice": f"{key}:{name}", "baseline": base["acc"], "candidate": cand["acc"]}
            if paired:
                pairs = list(members(key, name))
                broke = sum(b["correct"] and not c["correct"] for b, c in pairs)
                fixed = sum(c["correct"] and not b["correct"] for b, c in pairs)
                p = _sign_test_p(broke, fixed)
                check.update(broke=broke, fixed=fixed, p=round(p, 4))
                ok = drop_ok or (alpha is not None and p >= alpha)
            else:
                ok = drop_ok
            check["ok"] = ok
            checks.append(check)
    return {
        "passed": all(c["ok"] for c in checks),
        "tolerance": tolerance,
        "alpha": alpha,
        "paired": paired,
        "checks": checks,
    }


def write_report(report: dict, path: Path, *, keep_rows: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(report)
    if not keep_rows:
        data.pop("rows", None)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
