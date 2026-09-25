"""``calibrate``: fit temperatures on ``calib.jsonl`` and write them into the checkpoint.

laya applies one temperature per question type (``temperature`` list) and, when present, a
finer one per (type, option-count) bucket (``temperature_by_options``). Both are fitted here
by minimising NLL against the calibration targets, with a golden-section search on log T,
and clamped to laya's [0.5, 5] so a fit can never sharpen a coin flip into a certainty.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from eidolon_models_laya.vendor.laya.common import QTYPE_NAMES, clamp_temperature, temp_bucket

from .model import Loaded, record_items, score_items
from .records import read_jsonl


def _nll(scored: list[dict], T: float) -> float:
    total = 0.0
    for it in scored:
        z = [x / T for x in it["logits"]]
        m = max(z)
        lse = m + math.log(sum(math.exp(x - m) for x in z))
        total += -sum(t * (x - lse) for t, x in zip(it["target"], z, strict=True))
    return total / max(len(scored), 1)


def fit_temperature(scored: list[dict], lo: float = 0.5, hi: float = 5.0) -> float:
    """Golden-section search on log T for the NLL minimum."""
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = _nll(scored, math.exp(c)), _nll(scored, math.exp(d))
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = _nll(scored, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = _nll(scored, math.exp(d))
    return clamp_temperature(math.exp((a + b) / 2), lo, hi)


def calibrate(
    loaded: Loaded, calib_path: Path, *, min_bucket: int = 30, batch_size: int = 16, log=print
) -> dict:
    records = list(read_jsonl(calib_path))
    items = [it for r in records for it in record_items(r, loaded.tok, loaded.cfg)]
    scored = score_items(loaded, items, batch_size)
    by_type: dict[int, list] = defaultdict(list)
    by_bucket: dict[str, list] = defaultdict(list)
    for it in scored:
        by_type[it["qtype"]].append(it)
        by_bucket[temp_bucket(it["qtype"], len(it["markers"]))].append(it)
    temps = [1.0, 1.0, 1.0]
    report = {"n_items": len(scored), "by_type": {}, "by_bucket": {}}
    for qt, its in by_type.items():
        T = fit_temperature(its)
        temps[qt] = T
        report["by_type"][QTYPE_NAMES[qt]] = {
            "n": len(its),
            "T": round(T, 4),
            "nll_before": round(_nll(its, 1.0), 4),
            "nll_after": round(_nll(its, T), 4),
        }
    by_options = {}
    for bucket, its in by_bucket.items():
        if len(its) < min_bucket:
            report["by_bucket"][bucket] = {"n": len(its), "skipped": f"< {min_bucket}"}
            continue
        T = fit_temperature(its)
        by_options[bucket] = round(T, 4)
        report["by_bucket"][bucket] = {
            "n": len(its),
            "T": round(T, 4),
            "nll_before": round(_nll(its, 1.0), 4),
            "nll_after": round(_nll(its, T), 4),
        }
    loaded.cfg["temperature"] = [round(t, 4) for t in temps]
    loaded.cfg["temperature_by_options"] = by_options
    log(json.dumps(report, ensure_ascii=False))
    return report


def write_temperatures(checkpoint_dir: Path, cfg: dict) -> None:
    p = checkpoint_dir / "rl_agent_config.json"
    cur = json.loads(p.read_text(encoding="utf-8"))
    cur["temperature"] = cfg["temperature"]
    cur["temperature_by_options"] = cfg["temperature_by_options"]
    p.write_text(json.dumps(cur, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
