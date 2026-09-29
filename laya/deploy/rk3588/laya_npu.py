#!/usr/bin/env python3
"""laya on the RK3588 NPU: convert what ``eidolon-laya export-npu`` wrote, then run eval items on it.

Runs on the board. ``convert`` needs rknn-toolkit2 (≥ 2.3), ``run`` needs rknn-toolkit-lite2; both need
only numpy besides, so this file does not import the eidolon packages (``score_hidden`` / ``npu_inputs``
mirror ``eidolon_models_laya.export_npu``; ``tests/test_npu.py`` keeps them equal).

    python laya_npu.py convert <npu_dir>                              # hidden_l<L>.onnx → hidden_l<L>.rknn (fp16)
    python laya_npu.py run <npu_dir> <items_dir> <out_dir> [--core 0|012]
    python laya_npu.py bench <npu_dir> <set>.items.jsonl [--n 60] [--gate participation]    # one decision: 3 schedules
    python laya_npu.py bench-sets <npu_dir> <set>.items.jsonl --sets 2 --buckets 384,512,640  # a fixed footprint

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


# Which question decides what else is asked: {"first": qid, "then": {answer of first: [other qids]}}.
GATES = {
    "home": {"first": "intent", "then": {"无关": [], "查询": ["device"], "控制": ["device", "action"]}},
    "participation": {"first": "action", "then": {"respond": ["speaker"], "clarify": ["speaker", "clarify_about"],
                                                  "wait": [], "finish": []}},
}


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
    if args.buckets:  # NPU address space is limited (the parallel placement loads one runtime per question
        keep = {int(x) for x in args.buckets.split(",")}  # per bucket): time only decisions that fit these
        paths = {L: p for L, p in paths.items() if L in keep}
    items = [json.loads(x) for x in Path(args.items).read_text("utf-8").splitlines() if x.strip()]
    by_rec: dict[str, list[dict]] = {}
    for it in items:
        by_rec.setdefault(it["key"].rsplit("/", 1)[0], []).append(it)
    by_rec = {k: v for k, v in by_rec.items() if all(len(it["ids"]) <= max(paths) for it in v)}
    gate = GATES[args.gate]
    first, then = gate["first"], gate["then"]
    order = [first] + [q for qs in then.values() for q in qs]
    order = list(dict.fromkeys(order))
    full = max(len(v) for v in by_rec.values())
    recs = [v for v in by_rec.values() if len(v) == full][: args.n]
    if args.all:  # early exit is worth most on decisions that need only the first question
        vals = list(by_rec.values())  # evenly spaced, so the mix of decisions matches the set
        recs = vals[:: max(1, len(vals) // args.n)][: args.n] if args.n else vals
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
    qids = sorted({it["key"].rsplit("/", 1)[1] for rec in recs for it in rec}, key=order.index)
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
    # early exit (what options.ask_if does in the service): the first question alone, then only what its
    # answer needs, in parallel (home: 无关 stops there, 查询 adds device, 控制 adds device + action)
    ts, kinds = [], []
    with ThreadPoolExecutor(2) as pool:
        for rec in recs:
            q = {it["key"].rsplit("/", 1)[1]: it for it in rec}
            t = time.perf_counter()
            lo = run_q(first, q[first])
            intent = q[first]["names"][int(lo.argmax())]
            rest = [x for x in then[intent] if x in q]
            # (a record whose gold does not ask a needed question cannot time it; rare, underestimates)
            list(pool.map(lambda x: run_q(x, q[x]), rest))
            ts.append((time.perf_counter() - t) * 1000)
            kinds.append(intent)
    out["early_exit"] = ts
    # speculative: all questions start at once; the answer is ready as soon as the first question needs
    # nothing else (the rest finish in the background and are dropped), otherwise when the needed ones are done
    ts = []
    with ThreadPoolExecutor(3) as pool:
        for rec in recs:
            q = {it["key"].rsplit("/", 1)[1]: it for it in rec}
            t = time.perf_counter()
            fut = {x: pool.submit(run_q, x, it) for x, it in q.items()}
            intent = q[first]["names"][int(fut[first].result().argmax())]
            for x in then[intent]:
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
        for k in then:
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


def cmd_bench_sets(args) -> int:
    """One decision on a fixed deployment footprint: ``--sets`` runtime sets (set k on core k), each holding
    ``--buckets``. Every question runs on the same model, so any free set can take any question.

    staged:      the first question alone, then what its answer needs, spread over the sets
    speculative: the first and the next ``--sets - 1`` questions of the gate start at once; the answer is ready
                 when the first says nothing else is needed, else when the needed ones are done (a needed question
                 that was not started runs on the first free set)
    """
    from concurrent.futures import ThreadPoolExecutor

    from rknnlite.api import RKNNLite

    npu_dir = Path(args.npu_dir)
    emb = np.load(npu_dir / "tok_emb_fp16.npy", mmap_mode="r")
    type_emb = np.load(npu_dir / "type_emb.npy")
    sc = dict(np.load(npu_dir / "scorer.npz"))
    paths = buckets_in(npu_dir, "rknn")
    keep = sorted(int(x) for x in args.buckets.split(","))
    paths = {L: paths[L] for L in keep}
    gate = GATES[args.gate]
    first, then = gate["first"], gate["then"]
    order = list(dict.fromkeys([first] + [q for qs in then.values() for q in qs]))
    items = [json.loads(x) for x in Path(args.items).read_text("utf-8").splitlines() if x.strip()]
    by_rec: dict[str, dict[str, dict]] = {}
    for it in items:
        by_rec.setdefault(it["key"].rsplit("/", 1)[0], {})[it["key"].rsplit("/", 1)[1]] = it
    vals = [v for v in by_rec.values() if all(len(it["ids"]) <= keep[-1] for it in v.values())]
    recs = vals[:: max(1, len(vals) // args.n)][: args.n]
    cores = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]

    def bucket(it):
        return next(b for b in keep if len(it["ids"]) <= b)

    sets = []
    for k in range(args.sets):
        rt = {}
        for L, path in paths.items():
            r = RKNNLite(verbose=False)
            if r.load_rknn(str(path)) != 0 or r.init_runtime(core_mask=cores[k]) != 0:
                print(json.dumps({"error": f"could not load set {k} bucket {L}", "loaded": k * len(paths) + len(rt)}))
                return 1
            for _ in range(2):
                r.inference(inputs=npu_inputs([0, 1], 0, L, emb, type_emb))
            rt[L] = r
        sets.append(rt)

    def run_on(k, it):
        L = bucket(it)
        h = np.asarray(sets[k][L].inference(inputs=npu_inputs(it["ids"], it["qtype"], L, emb, type_emb))[0],
                       np.float32)[0]
        return score_hidden(h, it["markers"], sc)

    def answer(it, lo):
        return it["names"][int(lo.argmax())]

    out = {"staged": [], "speculative": []}
    kinds = []
    cpu = {"staged": 0.0, "speculative": 0.0}  # process CPU seconds (user + sys) spent in each schedule

    def cpu_now():
        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_utime + r.ru_stime

    with ThreadPoolExecutor(args.sets) as pool:
        for q in recs:
            c0 = cpu_now()
            # staged: first alone on set 0, then the needed ones on sets 0.. in parallel
            t = time.perf_counter()
            kind = answer(q[first], run_on(0, q[first]))
            need = [x for x in then[kind] if x in q]
            list(pool.map(lambda kx: run_on(kx[0] % args.sets, q[kx[1]]), enumerate(need)))
            out["staged"].append((time.perf_counter() - t) * 1000)
            kinds.append(kind)
            cpu["staged"] += cpu_now() - c0
            c0 = cpu_now()
            # speculative: first on set 0 and the next sets-1 questions of the gate on sets 1..
            spec = [x for x in order[1:args.sets] if x in q]
            t = time.perf_counter()
            fut = {first: pool.submit(run_on, 0, q[first])}
            for k, x in enumerate(spec, start=1):
                fut[x] = pool.submit(run_on, k, q[x])
            kind = answer(q[first], fut[first].result())
            rest = []
            for x in [x for x in then[kind] if x in q]:
                if x in fut:
                    fut[x].result()
                else:
                    rest.append(x)  # not started: set 0 is free now that the first is done
            for x in rest:
                run_on(0, q[x])
            out["speculative"].append((time.perf_counter() - t) * 1000)
            for f in fut.values():  # dropped work finishes before the next decision (not timed)
                f.result()
            cpu["speculative"] += cpu_now() - c0
    for rt in sets:
        for r in rt.values():
            r.release()
    longest = [max(bucket(it) for it in q.values()) for q in recs]
    res = {"records": len(recs), "sets": args.sets, "buckets": keep, "runtimes_loaded": args.sets * len(keep),
           "by_longest_bucket": {L: longest.count(L) for L in sorted(set(longest))},
           "by_first_answer": {k: kinds.count(k) for k in then if kinds.count(k)}}
    for label, ts in out.items():
        res[label] = {"p50": round(statistics.median(ts), 1), "p95": round(sorted(ts)[int(0.95 * (len(ts) - 1))], 1),
                      "mean": round(statistics.mean(ts), 1), "cpu_ms_per_decision": round(cpu[label] / len(ts) * 1000, 1)}
        for k in then:
            sub = [x for x, kk in zip(ts, kinds) if kk == k]
            if sub:
                res[label][f"p50_{k}"] = round(statistics.median(sub), 1)
        for L in sorted(set(longest)):
            sub = [x for x, lb in zip(ts, longest) if lb == L]
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
    p.add_argument("--all", action="store_true",
                   help="every record (incl. ones with only the first question), --n of them evenly spaced")
    p.add_argument("--gate", choices=sorted(GATES), default="home", help="which question decides the others")
    p.add_argument("--buckets", help="comma-separated buckets to load (default all); longer decisions are skipped")
    p.set_defaults(func=cmd_bench)
    p = sub.add_parser("bench-sets", help="one decision on a fixed footprint: N runtime sets (set k on core k)")
    p.add_argument("npu_dir")
    p.add_argument("items")
    p.add_argument("--sets", type=int, default=2, choices=(1, 2, 3))
    p.add_argument("--buckets", required=True, help="comma-separated buckets each set holds; longer decisions are skipped")
    p.add_argument("--n", type=int, default=150, help="decisions, evenly spaced over the set")
    p.add_argument("--gate", choices=sorted(GATES), default="participation")
    p.set_defaults(func=cmd_bench_sets)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
