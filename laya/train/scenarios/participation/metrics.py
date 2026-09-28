"""Participation decision metrics from an eval report (rows needed: evaluate without --no-rows).

    python3 train/scenarios/participation/metrics.py <report.json> [--json out.json] [--errors out.jsonl --records eval.jsonl]

Per record (one decision point): action right; speaker right when the gold action speaks; clarify_about right
when it clarifies; end-to-end = all of those. Also the two costly errors — speaking when it should stay silent
(gold wait / finish, predicted respond / clarify) and staying silent when it should speak — the confusion matrix,
slices by mode / trigger / family / source, variant consistency, and whole-episode correctness.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

SPEAK = {"respond", "clarify"}
SILENT = {"wait", "finish"}


def tag(tags: list[str], key: str) -> str | None:
    return next((t.split(":", 1)[1] for t in tags if t.startswith(key + ":")), None)


def records(report: dict) -> list[dict]:
    by: dict[str, dict] = defaultdict(dict)
    for r in report["rows"]:
        by[r["record_id"]][r["qid"]] = r
    out = []
    for rid, q in by.items():
        a = q["action"]
        gold_a = a["gold"]
        ok_a = a["correct"]
        speaks = any(g in SPEAK for g in gold_a)
        ok_s = True
        if ok_a and a["pred"] in SPEAK and "speaker" in q and q["speaker"]["gold"] is not None:
            ok_s = q["speaker"]["correct"]
        ok_c = True
        if ok_a and a["pred"] == "clarify" and "clarify_about" in q and q["clarify_about"]["gold"] is not None:
            ok_c = q["clarify_about"]["correct"]
        t = a["tags"]
        out.append({
            "id": rid, "gold": gold_a, "pred": a["pred"], "p": a["p_top"],
            "action_ok": ok_a, "speaker_ok": ok_s if speaks and ok_a else None, "e2e": bool(ok_a and ok_s and ok_c),
            "stop_violation": all(g in SILENT for g in gold_a) and a["pred"] in SPEAK,
            "missed_speech": all(g in SPEAK for g in gold_a) and a["pred"] in SILENT,
            "mode": tag(t, "mode"), "trigger": tag(t, "trigger"), "family": tag(t, "family"), "source": tag(t, "source"),
            "episode": tag(t, "episode"), "step": tag(t, "step"), "variant": tag(t, "variant"),
        })
    return out


def pct(xs: list[bool]) -> float | None:
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def summary(recs: list[dict]) -> dict:
    silent = [r for r in recs if all(g in SILENT for g in r["gold"])]
    speak = [r for r in recs if all(g in SPEAK for g in r["gold"])]
    spk = [r["speaker_ok"] for r in recs if r["speaker_ok"] is not None]
    out = {"n": len(recs), "action": pct([r["action_ok"] for r in recs]), "speaker": pct(spk), "e2e": pct([r["e2e"] for r in recs]),
           "stop_violation": pct([r["stop_violation"] for r in silent]), "missed_speech": pct([r["missed_speech"] for r in speak])}
    by_ep = defaultdict(list)
    for r in recs:
        if r["variant"] in (None, "0"):
            by_ep[r["episode"]].append(r["e2e"])
    out["episodes_all_right"] = pct([all(v) for v in by_ep.values()])
    pairs = defaultdict(dict)
    for r in recs:
        pairs[(r["episode"], r["step"])][r["variant"]] = r
    multi = [v for v in pairs.values() if len(v) > 1]
    out["variant_same_action"] = pct([len({r["pred"] for r in v.values()}) == 1 for v in multi])
    out["variant_all_e2e"] = pct([all(r["e2e"] for r in v.values()) for v in multi])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--json", help="write the numbers here")
    ap.add_argument("--errors", help="write every wrong record (variant 0) with its state here; needs --records")
    ap.add_argument("--records", help="the eval records file the report was made from")
    a = ap.parse_args()
    recs = records(json.loads(Path(a.report).read_text("utf-8")))
    res = {"all": summary(recs)}
    for key in ("mode", "trigger", "source"):
        for v in sorted({r[key] for r in recs if r[key]}):
            res[f"{key}:{v}"] = summary([r for r in recs if r[key] == v])
    fam = {f: summary([r for r in recs if r["family"] == f]) for f in sorted({r["family"] for r in recs if r["family"]})}
    confusion = Counter((("|".join(r["gold"])), r["pred"]) for r in recs)
    for k, v in res.items():
        print(f"{k:22s} " + "  ".join(f"{m} {v[m]}" for m in ("n", "action", "speaker", "e2e", "stop_violation", "missed_speech",
                                                             "episodes_all_right", "variant_same_action", "variant_all_e2e")))
    print("families (e2e):", "  ".join(f"{f} {v['e2e']}" for f, v in fam.items()))
    print("confusion gold→pred:", dict(sorted(confusion.items(), key=lambda kv: -kv[1])))
    if a.json:
        Path(a.json).write_text(json.dumps({"slices": res, "families": fam,
                                            "confusion": {f"{g}->{p}": n for (g, p), n in confusion.items()}},
                                           ensure_ascii=False, indent=1), "utf-8")
    if a.errors:
        states = {}
        for line in Path(a.records).read_text("utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                states[r["id"]] = r["state"]
        with Path(a.errors).open("w", encoding="utf-8") as f:
            for r in recs:
                if not r["e2e"] and r["variant"] in (None, "0"):
                    f.write(json.dumps({**r, "state": states.get(r["id"])}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
