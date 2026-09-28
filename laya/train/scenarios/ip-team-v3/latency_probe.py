"""Warm Mac MPS forward latency for complete text-only Ensemble inputs."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from context_probe import case, measure
from eidolon_laya_train.model import load_checkpoint, record_items, score_items
from eidolon_laya_train.records import Record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    start = time.perf_counter()
    loaded = load_checkpoint(args.checkpoint, "mps")
    load_s = time.perf_counter() - start
    cfg = dict(loaded.cfg, max_len=2048, head_max_len=256)
    scenarios = (
        ("short", 2, 2, 80, 60),
        ("normal", 4, 4, 120, 100),
        ("default_budget", 4, 9, 160, 120),
        ("many_members", 6, 9, 160, 120),
    )
    out = {"checkpoint": args.checkpoint, "device": "mps", "load_s": round(load_s, 3), "cases": []}
    for spec in scenarios:
        name, _, state, q = case(*spec)
        budget = measure(loaded.tok, state, q, 2048, 256)
        if not budget["state_complete"] or not budget["options_complete"]:
            raise ValueError(f"{name}: truncated benchmark input")
        record = Record(id=f"probe:{name}", scenario="context-probe", source="synthetic-probe",
                        state=state, questions={"move": q})
        items = record_items(record, loaded.tok, cfg, require_label=False)
        for _ in range(5):
            score_items(loaded, items)
            torch.mps.synchronize()
        elapsed = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            score_items(loaded, items)
            torch.mps.synchronize()
            elapsed.append(1000 * (time.perf_counter() - start))
        out["cases"].append({"case": name, "input_tokens": len(items[0]["ids"]),
                             "median_ms": round(statistics.median(elapsed), 1),
                             "p95_ms": round(sorted(elapsed)[int(.95 * len(elapsed)) - 1], 1),
                             "min_ms": round(min(elapsed), 1), "max_ms": round(max(elapsed), 1)})
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
