#!/usr/bin/env python3
"""laya on the RK3588 NPU: convert what ``eidolon-laya export-npu`` wrote, then run eval items on it.

Runs on the board. ``convert`` needs rknn-toolkit2 (≥ 2.3), ``run`` needs rknn-toolkit-lite2; both need
only numpy besides, so this file does not import the eidolon packages (``score_hidden`` / ``npu_inputs``
mirror ``eidolon_models_laya.export_npu``; ``tests/test_npu.py`` keeps them equal).

    python laya_npu.py convert <npu_dir>                              # hidden_l<L>.onnx → hidden_l<L>.rknn (fp16)
    python laya_npu.py run <npu_dir> <items_dir> <out_dir> [--core 0|012]

``run`` reads ``<set>.items.jsonl`` (from ``eidolon-laya-train items``) and writes ``<set>.logits.jsonl``
(score it on the Mac with ``eidolon-laya-train eval --logits <out_dir>``) plus ``run.json``: latency per
bucket and agreement with the reference logits the items file carries.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def _erf(x: np.ndarray) -> np.ndarray:  # Abramowitz–Stegun 7.1.26, |error| < 1.5e-7
    s = np.sign(x)
    x = np.abs(x)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def score_hidden(h: np.ndarray, markers: list[int], sc: dict[str, np.ndarray]) -> np.ndarray:
    m = h[np.asarray(markers)]
    mu, var = m.mean(-1, keepdims=True), m.var(-1, keepdims=True)
    x = (m - mu) / np.sqrt(var + 1e-5) * sc["ln_w"] + sc["ln_b"]
    x = x @ sc["w1"].T + sc["b1"]
    x = 0.5 * x * (1 + _erf(x / np.sqrt(2)))
    return (x @ sc["w2"].T + sc["b2"])[:, 0]


def npu_inputs(ids: list[int], qtype: int, L: int, emb: np.ndarray, type_emb: np.ndarray) -> list[np.ndarray]:
    n = len(ids)
    x = np.zeros((1, L, emb.shape[1]), np.float32)
    x[0, :n] = emb[np.asarray(ids)]
    mask = np.zeros((1, L), np.int64)
    mask[0, :n] = 1
    return [x, mask, type_emb[qtype][None, None, :].astype(np.float32)]


def buckets_in(npu_dir: Path, ext: str) -> dict[int, Path]:
    found = {int(re.search(r"hidden_l(\d+)", p.name).group(1)): p for p in npu_dir.glob(f"hidden_l*.{ext}")}
    return dict(sorted(found.items()))


def cmd_convert(args) -> int:
    from rknn.api import RKNN

    npu_dir = Path(args.npu_dir)
    for L, onnx_path in buckets_in(npu_dir, "onnx").items():
        out = onnx_path.with_suffix(".rknn")
        if out.exists() and not args.force:
            print(f"L={L}: {out.name} exists (--force to redo)")
            continue
        r = RKNN(verbose=False)
        r.config(target_platform="rk3588", optimization_level=3)
        t = time.perf_counter()
        assert r.load_onnx(model=str(onnx_path), inputs=["inputs_embeds", "attention_mask", "type_vec"],
                           input_size_list=[[1, L, 768], [1, L], [1, 1, 768]]) == 0, "load_onnx"
        assert r.build(do_quantization=False) == 0, "build"
        assert r.export_rknn(str(out)) == 0, "export_rknn"
        r.release()
        print(f"L={L}: {out.name} in {time.perf_counter() - t:.0f}s, {out.stat().st_size / 1e6:.0f} MB", flush=True)
    return 0


def cmd_run(args) -> int:
    from rknnlite.api import RKNNLite

    npu_dir, items_dir, out_dir = Path(args.npu_dir), Path(args.items_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    core = {"0": RKNNLite.NPU_CORE_0, "012": RKNNLite.NPU_CORE_0_1_2, "auto": RKNNLite.NPU_CORE_AUTO}[args.core]
    emb = np.load(npu_dir / "tok_emb_fp16.npy", mmap_mode="r")
    type_emb = np.load(npu_dir / "type_emb.npy")
    sc = dict(np.load(npu_dir / "scorer.npz"))
    runtimes = {}
    for L, path in buckets_in(npu_dir, "rknn").items():
        r = RKNNLite(verbose=False)
        assert r.load_rknn(str(path)) == 0, f"load_rknn {path}"
        assert r.init_runtime(core_mask=core) == 0, "init_runtime"
        warm = npu_inputs([0, 1], 0, L, emb, type_emb)
        for _ in range(3):
            r.inference(inputs=warm)
        runtimes[L] = r
    if not runtimes:
        sys.exit(f"no hidden_l*.rknn in {npu_dir}; run convert first")
    summary = {"core": args.core, "buckets": list(runtimes), "sets": {}}
    for items_path in sorted(items_dir.glob("*.items.jsonl")):
        name = items_path.name[: -len(".items.jsonl")]
        items = [json.loads(x) for x in items_path.read_text("utf-8").splitlines() if x.strip()]
        lat = {L: [] for L in runtimes}
        total, agree, worst, over = [], 0, 0.0, 0
        with (out_dir / f"{name}.logits.jsonl").open("w", encoding="utf-8") as f:
            for it in items:
                L = next((b for b in runtimes if len(it["ids"]) <= b), None)
                if L is None:
                    over += 1  # longer than the largest bucket: cannot run on this export
                    continue
                t0 = time.perf_counter()
                inputs = npu_inputs(it["ids"], it["qtype"], L, emb, type_emb)
                t1 = time.perf_counter()
                h = np.asarray(runtimes[L].inference(inputs=inputs)[0], np.float32)[0]
                t2 = time.perf_counter()
                lo = score_hidden(h, it["markers"], sc)
                t3 = time.perf_counter()
                lat[L].append((t2 - t1) * 1000)
                total.append((t3 - t0) * 1000)
                ref = np.asarray(it["ref_logits"], np.float32)
                agree += int(lo.argmax() == ref.argmax())
                worst = max(worst, float(np.abs(lo - ref).max()))
                f.write(json.dumps({"key": it["key"], "logits": [round(float(x), 5) for x in lo], "bucket": L,
                                    "npu_ms": round((t2 - t1) * 1000, 2)}) + "\n")
        n = sum(len(v) for v in lat.values())
        summary["sets"][name] = {
            "items": len(items), "ran": n, "over_largest_bucket": over,
            "argmax_agree_with_ref": round(agree / n, 4) if n else None,
            "max_abs_logit_diff": round(worst, 4),
            "npu_ms_by_bucket": {L: {"n": len(v), "p50": round(statistics.median(v), 1),
                                     "p95": round(sorted(v)[int(0.95 * (len(v) - 1))], 1)} for L, v in lat.items() if v},
            "item_ms_p50": round(statistics.median(total), 1) if total else None,
        }
        print(name, json.dumps(summary["sets"][name], ensure_ascii=False), flush=True)
    for r in runtimes.values():
        r.release()
    summary["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    load = Path("/sys/kernel/debug/rknpu/load")
    summary["npu_load"] = load.read_text().strip() if load.exists() else None
    summary["host"] = os.uname().nodename
    (out_dir / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", "utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("convert")
    p.add_argument("npu_dir")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_convert)
    p = sub.add_parser("run")
    p.add_argument("npu_dir")
    p.add_argument("items_dir")
    p.add_argument("out_dir")
    p.add_argument("--core", choices=("0", "012", "auto"), default="0")
    p.set_defaults(func=cmd_run)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
