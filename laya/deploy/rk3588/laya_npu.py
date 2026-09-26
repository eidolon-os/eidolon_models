#!/usr/bin/env python3
"""laya on the RK3588 NPU: convert what ``eidolon-laya export-npu`` wrote, then run eval items on it.

Runs on the board. ``convert`` needs rknn-toolkit2 (≥ 2.3), ``run`` needs rknn-toolkit-lite2; both need
only numpy besides, so this file does not import the eidolon packages (``score_hidden`` / ``npu_inputs``
mirror ``eidolon_models_laya.export_npu``; ``tests/test_npu.py`` keeps them equal).

    python laya_npu.py convert <npu_dir>                              # hidden_l<L>.onnx → hidden_l<L>.rknn (fp16)
    python laya_npu.py run <npu_dir> <items_dir> <out_dir> [--core 0|012]
    python laya_npu.py bench <npu_dir> <set>.items.jsonl [--n 60]    # one decision = all its questions: 3 schedules

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


def cmd_bench(args) -> int:
    """End-to-end latency of one decision (all questions of a record) under three schedules:
    sequential on core 0, sequential with each model on cores 0-1-2, and one question per core in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    from rknnlite.api import RKNNLite

    npu_dir = Path(args.npu_dir)
    emb = np.load(npu_dir / "tok_emb_fp16.npy", mmap_mode="r")
    type_emb = np.load(npu_dir / "type_emb.npy")
    sc = dict(np.load(npu_dir / "scorer.npz"))
    paths = buckets_in(npu_dir, "rknn")
    items = [json.loads(x) for x in Path(args.items).read_text("utf-8").splitlines() if x.strip()]
    by_rec: dict[str, list[dict]] = {}
    for it in items:
        by_rec.setdefault(it["key"].rsplit("/", 1)[0], []).append(it)
    recs = [v for v in by_rec.values() if len(v) == 3][: args.n]
    if args.all:  # early exit is worth most on non-commands, which have no device/action item
        recs = list(by_rec.values())
    cores = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]

    def runtime(L, mask):
        r = RKNNLite(verbose=False)
        assert r.load_rknn(str(paths[L])) == 0 and r.init_runtime(core_mask=mask) == 0
        for _ in range(2):
            r.inference(inputs=npu_inputs([0, 1], 0, L, emb, type_emb))
        return r

    def one(r, it, L):
        h = np.asarray(r.inference(inputs=npu_inputs(it["ids"], it["qtype"], L, emb, type_emb))[0], np.float32)[0]
        return score_hidden(h, it["markers"], sc)

    def bucket(it):
        return next(b for b in paths if len(it["ids"]) <= b)

    out = {}
    for label, mask in (("seq_core0", RKNNLite.NPU_CORE_0), ("seq_core012", RKNNLite.NPU_CORE_0_1_2)):
        rt = {L: runtime(L, mask) for L in paths}
        ts = []
        for rec in recs:
            t = time.perf_counter()
            for it in rec:
                one(rt[bucket(it)], it, bucket(it))
            ts.append((time.perf_counter() - t) * 1000)
        out[label] = ts
        for r in rt.values():
            r.release()
    # Placement used for both parallel schedules (and what a deployment would load): each question id gets
    # its own core and only the buckets it actually needs — intent on core 0 (128), device on core 1
    # (all), action on core 2 (≤ 256). One runtime per core per bucket for every bucket OOMs the NPU.
    qids = sorted({it["key"].rsplit("/", 1)[1] for rec in recs for it in rec}, key=["intent", "device", "action"].index)
    core_of = {q: cores[n] for n, q in enumerate(qids)}
    need = {q: sorted({bucket(it) for rec in recs for it in rec if it["key"].endswith("/" + q)}) for q in qids}
    rt = {(q, L): runtime(L, core_of[q]) for q in qids for L in need[q]}
    res_place = {q: need[q] for q in qids}

    def run_q(q, it):
        return one(rt[(q, bucket(it))], it, bucket(it))

    ts = []
    with ThreadPoolExecutor(3) as pool:
        for rec in recs:
            t = time.perf_counter()
            list(pool.map(lambda it: run_q(it["key"].rsplit("/", 1)[1], it), rec))
            ts.append((time.perf_counter() - t) * 1000)
    out["parallel_3cores"] = ts
    # early exit (what options.ask_if does in the service): intent first; 无关 stops there,
    # 查询 adds device, 控制 adds device + action in parallel
    ts, kinds = [], []
    with ThreadPoolExecutor(2) as pool:
        for rec in recs:
            q = {it["key"].rsplit("/", 1)[1]: it for it in rec}
            t = time.perf_counter()
            lo = run_q("intent", q["intent"])
            intent = q["intent"]["names"][int(lo.argmax())]
            rest = [x for x in {"无关": [], "查询": ["device"], "控制": ["device", "action"]}[intent] if x in q]
            # (a record whose gold has no device/action — a gold 无关 — cannot time those; rare, underestimates)
            list(pool.map(lambda x: run_q(x, q[x]), rest))
            ts.append((time.perf_counter() - t) * 1000)
            kinds.append(intent)
    out["early_exit"] = ts
    # speculative: all questions start at once; the answer is ready as soon as intent says 无关 (the other
    # two finish in the background and are dropped), otherwise when the needed ones are done
    ts = []
    with ThreadPoolExecutor(3) as pool:
        for rec in recs:
            q = {it["key"].rsplit("/", 1)[1]: it for it in rec}
            t = time.perf_counter()
            fut = {x: pool.submit(run_q, x, it) for x, it in q.items()}
            intent = q["intent"]["names"][int(fut["intent"].result().argmax())]
            for x in {"无关": [], "查询": ["device"], "控制": ["device", "action"]}[intent]:
                if x in fut:
                    fut[x].result()
            ts.append((time.perf_counter() - t) * 1000)
            for f in fut.values():  # let the dropped work finish before the next utterance (not timed)
                f.result()
    out["speculative"] = ts
    for r in rt.values():
        r.release()
    longest = [max(bucket(it) for it in rec) for rec in recs]
    res = {"records": len(recs), "by_longest_bucket": {L: longest.count(L) for L in sorted(set(longest))},
           "placement_buckets": res_place, "runtimes_loaded_parallel": sum(len(v) for v in res_place.values())}
    for sched in ("parallel_3cores", "early_exit", "speculative"):
        for k in ("无关", "查询", "控制"):
            sub = [t for t, kk in zip(out[sched], kinds) if kk == k]
            if sub:
                res[f"{sched}_pred_{k}"] = {"n": len(sub), "p50": round(statistics.median(sub), 1),
                                            "p95": round(sorted(sub)[int(0.95 * (len(sub) - 1))], 1)}
    for label, ts in out.items():
        res[label] = {"p50": round(statistics.median(ts), 1), "p95": round(sorted(ts)[int(0.95 * (len(ts) - 1))], 1)}
        for L in sorted(set(longest)):
            sub = [t for t, lb in zip(ts, longest) if lb == L]
            res[label][f"p50_longest_{L}"] = round(statistics.median(sub), 1)
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    print(json.dumps(res, ensure_ascii=False, indent=1))
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
    p = sub.add_parser("bench")
    p.add_argument("npu_dir")
    p.add_argument("items", help="one <set>.items.jsonl; records with all three questions are timed")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--all", action="store_true", help="every record (incl. 无关 ones with only an intent item)")
    p.set_defaults(func=cmd_bench)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
