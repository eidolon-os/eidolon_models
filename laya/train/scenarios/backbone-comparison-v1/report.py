"""Summarize paired DEV results; keep tokenizer damage separate from accuracy."""
import importlib.util
import itertools
import json
import copy
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
s = importlib.util.spec_from_file_location("comparison", HERE / "compare.py")
c = importlib.util.module_from_spec(s)
s.loader.exec_module(c)


def aggregate(rows, raw):
    valid = [r for r in rows if "pred" in r]
    if not valid:
        return {"n": len(rows), "valid": 0}
    def count(rs):
        return {"n": len(rs), "correct": sum(r["correct"] for r in rs)}
    speak = [r for r in valid if all(g.startswith("respond:") for g in r["gold"])]
    stop = [r for r in valid if set(r["gold"]) <= {"wait", "finish"}]
    abstain = [r for r in valid if r["gold"] == ["abstain"]]
    families, snapshots = defaultdict(list), defaultdict(list)
    for r in valid:
        m = raw[r["split"], r["id"]]["meta"]
        families[r["split"], m["family"]].append(r)
        snapshots[r["split"], m.get("snapshot", m["family"])].append(r)
    full_family_sizes = defaultdict(int)
    full_snapshot_sizes = defaultdict(int)
    for (split, _), r in raw.items():
        full_family_sizes[split, r["meta"]["family"]] += 1
        full_snapshot_sizes[split, r["meta"].get("snapshot", r["meta"]["family"])] += 1
    complete_families = {k: rs for k, rs in families.items() if len(rs) == full_family_sizes[k]}
    complete_snapshots = {k: rs for k, rs in snapshots.items() if len(rs) == full_snapshot_sizes[k]}
    def canon(r):
        p = r["pred"]
        return "respond:" + str(raw[r["split"], r["id"]]["meta"]["slot_to_source"][p.split(":")[1]]) if p.startswith("respond:") else p
    times = [r["elapsed_ms"] for r in valid if not r["first_call"]]
    return {"n": len(rows), "valid": len(valid), "correct": sum(r["correct"] for r in valid),
            "accuracy": statistics.mean(r["correct"] for r in valid),
            "must_speak": {**count(speak), "missed": sum(not r["pred"].startswith("respond:") for r in speak)},
            "must_stop": {**count(stop), "wrong_speech": sum(r["pred"].startswith("respond:") for r in stop)},
            "must_abstain": count(abstain),
            "complete_families": len(complete_families),
            "all_variants_correct_families": sum(all(r["correct"] for r in rs) for rs in complete_families.values()),
            "complete_snapshots": len(complete_snapshots),
            "equivariant_snapshots": sum(len({canon(r) for r in rs}) == 1 for rs in complete_snapshots.values()),
            "slices": {sl: count([r for r in valid if r["slice"] == sl]) for sl in sorted({r["slice"] for r in valid})},
            "confidence": {str(t): count([r for r in valid if r["p_top"] >= t and r["pred"] != "abstain"])
                           for t in (.5, .7, .8, .9)},
            "latency_ms": {"median": statistics.median(times), "p95": float(np.quantile(times, .95))} if times else None}


def main():
    raw = {(split, r["id"]): r for split, r in c.records()}
    cap = {(r["split"], r["id"]): r for r in json.loads((c.OUT / "preflight-v2.json").read_text())["rows"]}
    expected = set(raw)
    models, predictions = {}, {}
    for alias in ["laya", "laya-r6", "laya-r7", "openjev", "decider", "jevk5"]:
        p = c.OUT / f"{alias}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        assert len(rows) == len({(r["split"], r["id"]) for r in rows})
        if set((r["split"], r["id"]) for r in rows) != expected:
            continue
        predictions[alias] = {(r["split"], r["id"]): r for r in rows}
        key = "laya" if alias.startswith("laya") else alias
        models[alias] = {
            "by_split": {split: aggregate([r for r in rows if r["split"] == split], raw) for split in c.DATA},
            "common_untruncated_diagnostic": aggregate([r for r in rows if cap[r["split"], r["id"]]["common_untruncated"]], raw),
            "own_full_information": aggregate([r for r in rows if cap[r["split"], r["id"]]["models"][key]["full_information"]], raw),
            "untruncated_count": sum(cap[r["split"], r["id"]]["models"][key]["untruncated"] for r in rows),
            "state_unknown_tokens": sum(cap[r["split"], r["id"]]["models"][key].get("state_unknown_tokens", 0) for r in rows),
            "raw_sha256": c.digest(p),
        }
        # Supplementary audit of the PRE-EXISTING host threshold, not a threshold search.
        filtered = copy.deepcopy(rows)
        for r in filtered:
            if "pred" in r and r["p_top"] < .8:
                r["pred"] = "abstain"
                r["correct"] = "abstain" in r["gold"]
        models[alias]["existing_threshold_0_8_exploratory"] = {
            split: aggregate([r for r in filtered if r["split"] == split], raw) for split in c.DATA}
    paired = {}
    for a, b in itertools.combinations(predictions, 2):
        for split in c.DATA:
            ids = [k for k in sorted(expected) if k[0] == split and
                   all(cap[k]["models"]["laya" if m.startswith("laya") else m]["full_information"] for m in (a, b))]
            if not ids:
                continue
            differences = defaultdict(list)
            for k in ids:
                differences[raw[k]["meta"]["family"]].append(int(predictions[a][k]["correct"]) - int(predictions[b][k]["correct"]))
            # Equal family weight; descriptive interval, no release/significance claim.
            values = np.array([np.mean(v) for v in differences.values()])
            rng = np.random.default_rng(71)
            boot = [np.mean(rng.choice(values, len(values), replace=True)) for _ in range(2000)]
            paired[f"{a} minus {b}/{split}"] = {
                "n": len(ids), "a_only_correct": sum(predictions[a][k]["correct"] and not predictions[b][k]["correct"] for k in ids),
                "b_only_correct": sum(predictions[b][k]["correct"] and not predictions[a][k]["correct"] for k in ids),
                "family_mean_difference": float(np.mean(values)), "descriptive_family_interval95": np.quantile(boot, [.025, .975]).tolist()}
    result = {"models": models, "paired": paired, "four_way_lossless_common": sum(r["common_full_information"] for r in cap.values()),
              "four_way_untruncated_common": sum(r["common_untruncated"] for r in cap.values()),
              "limitations": ["DEV only; previously used for Laya selection", "same synthetic teacher", "not independent real-world generalization", "unknown Chinese tokens damage open-jev input", "runtime and precision differ; no optimized speed ranking"]}
    (HERE / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    def fraction(x):
        return f"{x.get('correct', 0)}/{x['n']}"
    lines = ["# 四模型开发集比较", "", "自动汇总；完整解释见 RESULTS.md。open-jev 全量成绩包含截断及未知词元损失，只作实际接入诊断。", "",
             "| 模型 | v7正确/42 | v8正确/72 | v8选对回应者 | v8漏回应 | v8应停误发言 | 20条共同未截断诊断 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for a, m in models.items():
        v7, v8 = m["by_split"]["v7"], m["by_split"]["v8"]
        lines.append(f"| {a} | {fraction(v7)} | {fraction(v8)} | {fraction(v8['must_speak'])} | {v8['must_speak']['missed']}/{v8['must_speak']['n']} | {v8['must_stop']['wrong_speech']}/{v8['must_stop']['n']} | {fraction(m['common_untruncated_diagnostic'])} |")
    (HERE / "TABLE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
