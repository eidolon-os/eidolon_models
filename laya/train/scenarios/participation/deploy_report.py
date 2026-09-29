"""p-dev through a running participation service, decision point by decision point, against gold and PyTorch.

    uv run --extra torch --extra train python train/scenarios/participation/deploy_report.py \
        --model-dir models/laya-participation/<rev> --url http://127.0.0.1:<port>/v1/participation/decide \
        --service-log <file with the service's "participation decision=..." lines> --out <dir>

Each variant-0 snapshot of p-dev is sent as an SDK DecisionRequest (candidates in that variant's slot order,
as service_check.py does, so the service sees the evaluated state byte for byte). The same request is decided
locally by the PyTorch adapter as the reference. The service's reason / confidence / time for each decision
come from its log (it sends none of them). Writes <out>/items.jsonl (one row per decision point) and
<out>/summary.json (coverage, accuracy on what it decided, abstention reasons, differences from PyTorch,
latency, and the hashes of every input).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))

from service_check import EVALS, load, requests_for  # noqa: E402

LOG_LINE = re.compile(
    r"participation decision=(?P<id>\S+) status=(?P<status>\S+) action=(?P<action>\S+) reason=(?P<reason>\S+) "
    r"confidence=(?P<confidence>\S+) detail=(?P<detail>.*?) ms=(?P<ms>[\d.]+) policy=(?P<policy>\S+) model=(?P<model>\S+)"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gold_of(ep: dict, step: int) -> dict:
    g = ep["events"][step]["gold"]
    return {k: g[k] for k in ("action", "speaker", "clarify_about") if k in g}


def decision_of(result: dict) -> dict:
    p = result.get("proposal") or {}
    return {"status": result["status"], "action": p.get("action"), "participants": list(p.get("participants", [])),
            "instruction": p.get("instruction", "")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="the packaged model the service runs (for the PyTorch reference)")
    ap.add_argument("--url", required=True)
    ap.add_argument("--service-log", help="text containing the service's decision log lines (fetched after the run)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap", type=float, default=0.2, help="seconds between requests (one decision at a time)")
    ap.add_argument("--responses", help="reuse a previous run's responses.jsonl instead of sending again")
    a = ap.parse_args()
    from eidolon_models_laya.config import Settings
    from eidolon_models_laya.engine import load_engine
    from eidolon_models_laya.participation import load_participation_adapter, participation_profile_digest

    model = Path(a.model_dir)
    tasks = json.loads((model / "participation.json").read_text("utf-8"))["clarify_instructions"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    engine, manifest = load_engine(Settings(model_dir=model, device="cpu"), log=lambda *_: None)
    adapter = load_participation_adapter(
        model / "participation.json", engine, model_dir=model, model_revision=manifest.revision,
        expected_sha256=participation_profile_digest(manifest.raw))
    files = sorted((EVALS / "dev").glob("*.jsonl"))
    eps = load([str(p) for p in files])
    cases = [(ep, step, req) for ep in eps for step, req, _ in requests_for(ep)]

    responses = {}
    if a.responses:
        for line in Path(a.responses).read_text("utf-8").splitlines():
            r = json.loads(line)
            responses[r["id"]] = r
    else:
        with (out / "responses.jsonl").open("w", encoding="utf-8") as f:
            for _, _, req in cases:
                body = json.dumps(req.model_dump(mode="json")).encode()
                t = time.perf_counter()
                got = json.loads(urllib.request.urlopen(urllib.request.Request(
                    a.url, body, {"content-type": "application/json"}), timeout=15).read())
                r = {"id": req.decision_id, "http_ms": round((time.perf_counter() - t) * 1000, 1), "result": got}
                responses[req.decision_id] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                time.sleep(a.gap)
    logged = {}
    if a.service_log:
        for line in Path(a.service_log).read_text("utf-8", errors="replace").splitlines():
            m = LOG_LINE.search(line)
            if m:
                logged[m["id"]] = m.groupdict()

    rows = []
    for ep, step, req in cases:
        ref, why = adapter.decide_explained(req)
        got = responses[req.decision_id]
        tgt = decision_of(got["result"])
        log = logged.get(req.decision_id, {})
        gold = gold_of(ep, step)
        speakers = set(gold.get("speaker", []))
        if tgt["status"] == "abstained":
            action_ok = e2e_ok = None
        else:
            action_ok = tgt["action"] == gold["action"]
            e2e_ok = action_ok and (tgt["action"] not in ("respond", "clarify") or tgt["participants"][0] in speakers) \
                and (tgt["action"] != "clarify" or tgt["instruction"] == tasks.get(gold.get("clarify_about"), ""))
        rows.append({
            "id": req.decision_id, "episode": ep["episode"], "step": step, "family": ep["family"], "mode": ep["mode"],
            "gold": gold,
            "deployed": tgt | {"reason": log.get("reason"), "confidence": _num(log.get("confidence")),
                               "detail": (log.get("detail") or "").strip(" -") or None,
                               "service_ms": _num(log.get("ms")), "http_ms": got.get("http_ms")},
            "pytorch": decision_of(ref.model_dump(mode="json")) | {"reason": why["reason"], "confidence": why.get("confidence"),
                                                                  "detail": why.get("detail")},
            "action_correct": action_ok, "e2e_correct": e2e_ok,
            "same_as_pytorch": decision_of(got["result"]) == decision_of(ref.model_dump(mode="json")),
        })
    with (out / "items.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    decided = [r for r in rows if r["deployed"]["status"] == "decided"]
    abstained = [r for r in rows if r["deployed"]["status"] == "abstained"]
    ms = [r["deployed"]["service_ms"] for r in rows if r["deployed"]["service_ms"] is not None]
    q = (lambda xs, p: round(sorted(xs)[int(p * (len(xs) - 1))], 1)) if ms else (lambda xs, p: None)
    by_kind = {}
    for r in rows:
        if r["deployed"]["service_ms"] is not None:
            by_kind.setdefault(r["deployed"]["action"] or "abstained", []).append(r["deployed"]["service_ms"])
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(HERE.parents[2] / "src")], cwd=HERE,
                           capture_output=True, text=True).stdout.strip()
    model_files = {f: (model / "torch" / f).is_file() and sha(model / "torch" / f)
                   for f in ("rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json")}
    summary = {
        "decision_points": len(rows),
        "decided": len(decided), "abstained": len(abstained), "coverage": round(len(decided) / len(rows), 4),
        "action_accuracy_on_decided": round(sum(r["action_correct"] for r in decided) / max(1, len(decided)), 4),
        "e2e_accuracy_on_decided": round(sum(r["e2e_correct"] for r in decided) / max(1, len(decided)), 4),
        "stop_violations_on_decided": sum(1 for r in decided if r["gold"]["action"] in ("wait", "finish")
                                          and r["deployed"]["action"] in ("respond", "clarify")),
        "abstention_reasons": dict(Counter(r["deployed"]["reason"] or "unlogged" for r in abstained)),
        "abstained_ids": [r["id"] for r in abstained],
        "abstained_gold_actions": dict(Counter(r["gold"]["action"] for r in abstained)),
        "same_as_pytorch": sum(r["same_as_pytorch"] for r in rows),
        "differences_from_pytorch": [
            {"id": r["id"], "deployed": {k: r["deployed"][k] for k in ("status", "action", "participants", "confidence")},
             "pytorch": {k: r["pytorch"][k] for k in ("status", "action", "participants", "confidence")}}
            for r in rows if not r["same_as_pytorch"]],
        "service_ms": {"p50": q(ms, .5), "p95": q(ms, .95),
                       "by_action_p50": {k: (len(v), q(v, .5)) for k, v in sorted(by_kind.items())}},
        "policy_version": adapter.policy_version, "model_version": adapter.model_version,
        "min_confidence": adapter.min_confidence,
        "hashes": {
            "p_dev_files": {str(p.relative_to(EVALS.parent.parent)): sha(p) for p in files},
            "participation.json": sha(model / "participation.json"), "manifest.json": sha(model / "manifest.json"),
            "model_files": model_files,
            "responses.jsonl": sha(Path(a.responses) if a.responses else out / "responses.jsonl"),
            "items.jsonl": sha(out / "items.jsonl"),
        },
        "eidolon_models_revision": head, "laya_src_uncommitted_changes": bool(dirty),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print(json.dumps({k: summary[k] for k in ("decision_points", "decided", "coverage", "action_accuracy_on_decided",
                                              "e2e_accuracy_on_decided", "stop_violations_on_decided", "abstention_reasons",
                                              "same_as_pytorch", "service_ms")}, ensure_ascii=False))
    return 0


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
