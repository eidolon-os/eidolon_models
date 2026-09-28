"""Summarize IP Team one-step correctness from the ordinary Laya eval rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _rate(ok: int, n: int) -> float | None:
    return round(ok / n, 4) if n else None


def summarize(report: dict, records: list[dict]) -> dict:
    source = {r["id"]: r for r in records}
    if len(source) != len(records):
        raise ValueError("duplicate record ids")
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in report["rows"]:
        rows[row["record_id"]][row["qid"]] = row
    if set(rows) != set(source) or any(set(q) != {"action", "speaker"} for q in rows.values()):
        raise ValueError("eval rows do not cover exactly two questions per source record")

    action_ok = speaker_ok = step_ok = 0
    speaking_ok = speaking_n = 0
    by_action: dict[str, Counter] = defaultdict(Counter)
    by_tag: dict[str, Counter] = defaultdict(Counter)
    by_family: dict[str, Counter] = defaultdict(Counter)
    confidence = {threshold: Counter() for threshold in (0.5, 0.7, 0.9)}
    pairs: dict[str, list[bool]] = defaultdict(list)
    invalid_proposals = 0
    errors = []
    for rid, r in source.items():
        q = rows[rid]
        a, s = q["action"], q["speaker"]
        gold_action = r["labels"]["action"]["gold"]
        ao = bool(a["correct"])
        so = bool(s["correct"])
        full = ao and so
        action_ok += ao
        speaker_ok += so
        step_ok += full
        if gold_action in ("respond", "clarify"):
            speaking_n += 1
            speaking_ok += so
        elif s["pred"] != "NONE":
            invalid_proposals += 1
        by_action[gold_action]["n"] += 1
        by_action[gold_action]["ok"] += full
        tag = r["tags"][0]
        by_tag[tag]["n"] += 1
        by_tag[tag]["ok"] += full
        family = r["meta"]["family"]
        by_family[family]["n"] += 1
        by_family[family]["ok"] += full
        pair = rid.rsplit("-", 1)[0]
        if tag == "counterfactual":
            pairs[pair].append(full)
        for threshold, count in confidence.items():
            if min(a["p_top"], s["p_top"]) >= threshold:
                count["n"] += 1
                count["ok"] += full
        if not full:
            errors.append({
                "id": rid, "tag": tag,
                "gold_action": gold_action, "pred_action": a["pred"],
                "gold_speaker": r["labels"]["speaker"]["gold"], "pred_speaker": s["pred"],
            })

    n = len(source)
    pair_values = [all(v) for v in pairs.values() if len(v) == 2]
    return {
        "n": n,
        "n_families": len(by_family),
        "action_accuracy": _rate(action_ok, n),
        "speaker_accuracy": _rate(speaker_ok, n),
        "speaker_accuracy_when_speaking": _rate(speaking_ok, speaking_n),
        "full_step_accuracy": _rate(step_ok, n),
        "invalid_non_none_on_silent_gold": invalid_proposals,
        "by_action": {k: {"n": v["n"], "full_step_accuracy": _rate(v["ok"], v["n"])} for k, v in sorted(by_action.items())},
        "by_tag": {k: {"n": v["n"], "full_step_accuracy": _rate(v["ok"], v["n"])} for k, v in sorted(by_tag.items())},
        "mean_family_accuracy": _rate(sum(v["ok"] / v["n"] for v in by_family.values()), len(by_family)),
        "by_family": {k: {"n": v["n"], "full_step_accuracy": _rate(v["ok"], v["n"])} for k, v in sorted(by_family.items())},
        "counterfactual_pairs": {"n": len(pair_values), "both_correct": sum(pair_values), "rate": _rate(sum(pair_values), len(pair_values))},
        "confidence": {str(k): {"coverage": _rate(v["n"], n), "selected_n": v["n"], "full_step_accuracy": _rate(v["ok"], v["n"])} for k, v in confidence.items()},
        "error_count": len(errors),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    records = [json.loads(line) for line in Path(args.records).read_text(encoding="utf-8").splitlines() if line.strip()]
    result = summarize(report, records)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("errors", "by_family")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
