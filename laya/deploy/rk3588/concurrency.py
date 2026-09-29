"""Home and participation Laya services side by side, as the Agent would see them.

    # on the Mac: build the request bodies (home dev sets as the Agent adapter sends them; p-dev as SDK requests)
    uv run python deploy/rk3588/concurrency.py prepare --model-dir models/laya-participation/ae6718a4 --out <dir>
    # on the board (stdlib only): one scenario per call, results appended to <dir>/<scenario>.jsonl
    python3 concurrency.py run <dir> home-alone | part-alone | both | collide
    # anywhere: summary against the Agent's budgets
    python3 concurrency.py report <dir> [--service-log <journal of eidolon-laya-participation>]

The Agent's budgets are taken as they are in eidolon_agent: home interpretation 800 ms and confidence >= 0.8 on
intent (+ device unless 无关, + action for 控制), otherwise the LLM answers; participation primary 1,500 ms with
the LLM fallback off, so an abstention, a 504 deadline or a 503 busy ends the turn. Participation bodies carry
timeout_ms 1,500 so the service enforces the deadline itself (and keeps a timed-out inference counted as pending,
which is what would make the next call 503). Home calls wait up to 5 s so the true time is recorded; above 800 ms
counts as a timeout. Nothing is restarted or reconfigured.

Scenarios: home-alone (home dev sets one at a time); part-alone (p-dev one at a time); both (p-dev back to back,
as a team deciding every turn, while home commands arrive every 0.5-2 s); collide (a home command and a
participation decision released at the same instant).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HOME_BUDGET_MS, HOME_MIN_P = 800, 0.8
PART_BUDGET_MS = 1500
HOME_URL = "http://127.0.0.1:8771/v1/systemone"
PART_URL = "http://127.0.0.1:8773/v1/participation/decide"
MULTI, NONE = "多个设备或整屋", "没有对应的设备"


def prepare(a) -> int:
    here = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(here / "src"))
    sys.path.insert(0, str(here / "train/scenarios/participation"))
    from service_check import EVALS, load, requests_for

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "home.jsonl").open("w", encoding="utf-8") as f:
        for name in ("locked-v2-dev", "locked-v3-dev"):  # dev sets only; test / accept stay sealed
            for line in (here / f"train/scenarios/smart-home/eval/{name}.jsonl").read_text("utf-8").splitlines():
                r = json.loads(line)
                body = {"state": r["state"], "questions": r["questions"]}  # the adapter sends no ask_if
                gold = {q: v["gold"] for q, v in r["labels"].items()}
                f.write(json.dumps({"id": r["id"], "set": name, "body": body, "gold": gold}, ensure_ascii=False) + "\n")
    tasks = json.loads((Path(a.model_dir) / "participation.json").read_text("utf-8"))["clarify_instructions"]
    with (out / "participation.jsonl").open("w", encoding="utf-8") as f:
        for ep in load([str(p) for p in sorted((EVALS / "dev").glob("*.jsonl"))]):
            for step, req, _ in requests_for(ep):
                body = req.model_dump(mode="json") | {"timeout_ms": PART_BUDGET_MS}
                g = ep["events"][step]["gold"]
                gold = {k: g[k] for k in ("action", "speaker", "clarify_about") if k in g}
                f.write(json.dumps({"id": req.decision_id, "body": body, "gold": gold}, ensure_ascii=False) + "\n")
    (out / "clarify_instructions.json").write_text(json.dumps(tasks, ensure_ascii=False), "utf-8")
    print(sum(1 for _ in open(out / "home.jsonl")), "home,", sum(1 for _ in open(out / "participation.jsonl")), "participation")
    return 0


class Clock:
    def __init__(self):
        self.t0 = time.monotonic()

    def now(self) -> float:
        return round((time.monotonic() - self.t0) * 1000, 1)


def post(url: str, body: dict, timeout: float) -> tuple[int, dict | None, str | None]:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read()).get("error", {}).get("code")
        except ValueError:
            err = None
        return e.code, None, err
    except (urllib.error.URLError, TimeoutError) as e:
        return 0, None, type(getattr(e, "reason", e)).__name__


def call(kind: str, item: dict, clock: Clock, sink: list, lock: threading.Lock, tag: str) -> None:
    url, timeout = (HOME_URL, 5.0) if kind == "home" else (PART_URL, 5.0)  # 5 s: record the true time
    start = clock.now()
    status, payload, err = post(url, item["body"], timeout)
    end = clock.now()
    row = {"kind": kind, "id": item["id"], "tag": tag, "start": start, "end": end, "ms": round(end - start, 1),
           "http": status, "error": err}
    if payload is not None:
        if kind == "home":
            row["answers"] = {q: {"choice": v.get("choice"), "p": (v.get("probabilities") or {}).get(v.get("choice"))}
                              for q, v in (payload.get("answers") or {}).items()}
            row["truncated"] = bool(payload.get("truncated"))
            row["server_ms"] = (payload.get("timing_ms") or {}).get("total")
        else:
            p = payload.get("proposal") or {}
            row["status"] = payload.get("status")
            row["proposal"] = {"action": p.get("action"), "participants": p.get("participants", []),
                               "instruction": p.get("instruction", "")}
    with lock:
        sink.append(row)


def cpu_ticks(pids: list[int]) -> dict:
    out = {}
    for pid in pids:
        try:
            f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            out[str(pid)] = int(f[11]) + int(f[12])  # utime + stime
        except (OSError, IndexError, ValueError):
            pass
    return out


def npu_sampler(stop: threading.Event, samples: list, clock: Clock) -> None:
    path = Path("/sys/kernel/debug/rknpu/load")
    while not stop.wait(0.5):
        try:
            samples.append({"t": clock.now(), "load": path.read_text().strip()})
        except OSError:
            return


def run(a) -> int:
    global HOME_URL, PART_URL
    HOME_URL, PART_URL = a.home_url, a.part_url
    d = Path(a.dir)
    home = [json.loads(line) for line in (d / "home.jsonl").read_text("utf-8").splitlines()][: a.limit]
    part = [json.loads(line) for line in (d / "participation.jsonl").read_text("utf-8").splitlines()][: a.limit]
    rng = random.Random(a.seed)
    clock, lock, rows, npu = Clock(), threading.Lock(), [], []
    stop = threading.Event()
    sampler = threading.Thread(target=npu_sampler, args=(stop, npu, clock), daemon=True)
    sampler.start()
    pids = [int(p) for p in a.pids.split(",")] if a.pids else []
    cpu0, wall0 = cpu_ticks(pids), time.monotonic()

    if a.scenario == "home-alone":
        for it in home:
            call("home", it, clock, rows, lock, "alone")
            time.sleep(a.gap)
    elif a.scenario == "part-alone":
        for it in part:
            call("participation", it, clock, rows, lock, "alone")
            time.sleep(a.gap)
    elif a.scenario == "both":
        done = threading.Event()

        def team():
            for it in part:
                call("participation", it, clock, rows, lock, "both")
                time.sleep(a.gap)
            done.set()

        t = threading.Thread(target=team)
        t.start()
        order = home[:]
        rng.shuffle(order)
        i = 0
        while not done.is_set():
            if done.wait(rng.uniform(0.5, 2.0)):
                break
            call("home", order[i % len(order)], clock, rows, lock, "both")
            i += 1
        t.join()
    elif a.scenario == "collide":
        order = home[:]
        rng.shuffle(order)
        step = max(1, len(part) // a.pairs)
        for k in range(a.pairs):
            gate = threading.Barrier(2)

            def go(kind, it, gate=gate, tag=f"collide-{k}"):
                gate.wait()
                call(kind, it, clock, rows, lock, tag)

            ts = [threading.Thread(target=go, args=("home", order[k % len(order)])),
                  threading.Thread(target=go, args=("participation", part[(k * step) % len(part)]))]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            time.sleep(a.gap_pairs)
    stop.set()
    cpu1, wall = cpu_ticks(pids), time.monotonic() - wall0
    rows.sort(key=lambda r: r["start"])
    with (d / f"{a.scenario}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"scenario": a.scenario, "wall_s": round(wall, 1), "gap": a.gap, "seed": a.seed,
            "cpu_ticks": {p: cpu1.get(p, 0) - cpu0.get(p, 0) for p in cpu0}, "clk_tck": 100,
            "npu_load": npu, "started_unix": time.time() - wall}
    (d / f"{a.scenario}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    print(a.scenario, len(rows), "calls", round(wall), "s")
    return 0


def home_outcome(r: dict, gold: dict) -> dict:
    """What the Agent's LayaInterpreter + SmartHomeCommand would do with this reply."""
    if r["http"] != 200:
        return {"outcome": "error", "reason": f"http_{r['http']}_{r['error']}"}
    if r["ms"] > HOME_BUDGET_MS:
        return {"outcome": "timeout", "reason": "over_800ms"}
    ans = r["answers"]
    if r.get("truncated"):
        return {"outcome": "abstain", "reason": "truncated"}
    intent, device, action = (ans.get(q, {}).get("choice") for q in ("intent", "device", "action"))
    keys = ["intent"]
    if intent != "无关":
        if device == MULTI:
            return {"outcome": "abstain", "reason": "multiple_devices"}
        keys.append("device")
        if intent == "控制":
            keys.append("action")
    if any((ans.get(q, {}).get("p") or 0) < HOME_MIN_P for q in keys):
        return {"outcome": "abstain", "reason": "low_confidence"}
    ok = all(_is_gold(ans[q]["choice"], gold.get(q)) for q in keys)
    return {"outcome": "decided", "correct": ok}


def _is_gold(choice, gold) -> bool:
    return choice in gold if isinstance(gold, list) else choice == gold  # some items accept several answers


def part_outcome(r: dict, gold: dict, tasks: dict) -> dict:
    if r["http"] != 200:
        return {"outcome": "error" if r["http"] not in (503, 504) else ("busy" if r["http"] == 503 else "timeout"),
                "reason": f"http_{r['http']}_{r['error']}"}
    if r["status"] != "decided":
        return {"outcome": "abstain", "reason": "abstained"}
    p = r["proposal"]
    act_ok = p["action"] == gold["action"]
    e2e = act_ok and (p["action"] not in ("respond", "clarify") or p["participants"][0] in set(gold.get("speaker", []))) \
        and (p["action"] != "clarify" or p["instruction"] == tasks.get(gold.get("clarify_about"), ""))
    stop_violation = gold["action"] in ("wait", "finish") and p["action"] in ("respond", "clarify")
    return {"outcome": "decided", "correct": act_ok, "e2e": e2e, "stop_violation": stop_violation}


def q(xs: list[float], p: float):
    return round(sorted(xs)[min(len(xs) - 1, int(p * len(xs)))], 1) if xs else None


def summarize(rows: list[dict], outcomes: list[dict]) -> dict:
    ms = [r["ms"] for r in rows]
    by = {}
    for o in outcomes:
        by[o["outcome"]] = by.get(o["outcome"], 0) + 1
    reasons = {}
    for o in outcomes:
        if o["outcome"] != "decided":
            reasons[o["reason"]] = reasons.get(o["reason"], 0) + 1
    dec = [o for o in outcomes if o["outcome"] == "decided"]
    s = {"n": len(rows), "p50": q(ms, .5), "p95": q(ms, .95), "p99": q(ms, .99), "max": max(ms) if ms else None,
         "mean": round(statistics.fmean(ms), 1) if ms else None, "outcomes": by, "not_decided_reasons": reasons,
         "decided_accuracy": round(sum(o["correct"] for o in dec) / len(dec), 4) if dec else None}
    if dec and "e2e" in dec[0]:
        s["decided_e2e"] = round(sum(o["e2e"] for o in dec) / len(dec), 4)
        s["stop_violations"] = sum(o["stop_violation"] for o in dec)
    return s


def overlapped(r: dict, others: list[dict]) -> bool:
    return any(o["start"] < r["end"] and r["start"] < o["end"] for o in others)


def report(a) -> int:
    d = Path(a.dir)
    gold = {}
    for name in ("home", "participation"):
        for line in (d / f"{name}.jsonl").read_text("utf-8").splitlines():
            it = json.loads(line)
            gold[it["id"]] = it["gold"]
    tasks = json.loads((d / "clarify_instructions.json").read_text("utf-8"))
    out = {"budgets": {"home_ms": HOME_BUDGET_MS, "home_min_p": HOME_MIN_P, "participation_ms": PART_BUDGET_MS}}
    for sc in ("home-alone", "part-alone", "both", "collide"):
        f = d / f"{sc}.jsonl"
        if not f.is_file():
            continue
        rows = [json.loads(line) for line in f.read_text("utf-8").splitlines()]
        meta = json.loads((d / f"{sc}.meta.json").read_text("utf-8"))
        res = {"wall_s": meta["wall_s"]}
        for kind in ("home", "participation"):
            mine = [r for r in rows if r["kind"] == kind]
            if not mine:
                continue
            other = [r for r in rows if r["kind"] != kind]
            oc = [home_outcome(r, gold[r["id"]]) if kind == "home" else part_outcome(r, gold[r["id"]], tasks) for r in mine]
            res[kind] = summarize(mine, oc)
            if other:
                both_ = [(r, o) for r, o in zip(mine, oc, strict=True) if overlapped(r, other)]
                res[kind]["overlapped_with_other"] = len(both_)
                if both_:
                    res[kind]["overlapped"] = summarize([r for r, _ in both_], [o for _, o in both_])
            if kind == "home":
                sm = [r["server_ms"] for r in mine if r.get("server_ms") is not None]
                res[kind]["server_ms_p50"], res[kind]["server_ms_p95"] = q(sm, .5), q(sm, .95)
        ticks = meta.get("cpu_ticks") or {}
        if ticks:
            res["cpu_s_by_pid"] = {p: t / meta["clk_tck"] for p, t in ticks.items()}
        out[sc] = res
    if a.service_log:
        import re
        pattern = re.compile(r"participation decision=(\S+) status=(\S+) action=\S+ reason=(\S+) .*? ms=([\d.]+)")
        logged = {}
        for line in Path(a.service_log).read_text("utf-8", errors="replace").splitlines():
            m = pattern.search(line)
            if m:
                logged.setdefault(m[1], []).append({"status": m[2], "reason": m[3], "ms": float(m[4])})
        out["participation_log_reasons"] = {}
        for sc in ("part-alone", "both", "collide"):
            f = d / f"{sc}.jsonl"
            if f.is_file():
                ids = [json.loads(line)["id"] for line in f.read_text("utf-8").splitlines() if '"participation"' in line]
                c = {}
                for i in ids:
                    for e in logged.get(i, [])[-1:]:
                        if e["status"] == "abstained":
                            c[e["reason"]] = c.get(e["reason"], 0) + 1
                out["participation_log_reasons"][sc] = c
    (d / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", "utf-8")
    for sc, res in out.items():
        if not isinstance(res, dict) or "wall_s" not in res:
            continue
        for kind in ("home", "participation"):
            if kind in res:
                s = res[kind]
                print(f"{sc:10s} {kind:13s} n={s['n']:4d} p50={s['p50']} p95={s['p95']} p99={s['p99']} max={s['max']} "
                      f"{s['outcomes']} acc={s['decided_accuracy']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("dir")
    p.add_argument("scenario", choices=("home-alone", "part-alone", "both", "collide"))
    p.add_argument("--gap", type=float, default=0.1, help="seconds between calls of one stream")
    p.add_argument("--pairs", type=int, default=100)
    p.add_argument("--gap-pairs", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--pids", help="comma-separated service PIDs whose CPU time is recorded")
    p.add_argument("--limit", type=int, help="first N items of each stream (smoke test)")
    p.add_argument("--home-url", default=HOME_URL)
    p.add_argument("--part-url", default=PART_URL)
    p = sub.add_parser("report")
    p.add_argument("dir")
    p.add_argument("--service-log")
    a = ap.parse_args()
    return {"prepare": prepare, "run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
