"""Offline open-set and discovery experiment, not a production identity service."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DURATIONS = (0.6, 1.0, 1.5, 2.0, 3.0, "full")


def unit(x):
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x)
    if not np.isfinite(x).all() or norm == 0:
        raise ValueError("invalid vector")
    return x / norm


def split_samples(manifest):
    groups = defaultdict(list)
    hashes = set()
    ids = set()
    for row in manifest["samples"]:
        if row["sha256"] in hashes or row["id"] in ids:
            raise ValueError("duplicate source recording")
        hashes.add(row["sha256"])
        ids.add(row["id"])
        groups[row["speaker"]].append(row["id"])
    speakers = sorted(groups)
    if len(speakers) != 12 or any(len(v) != 12 for v in groups.values()):
        raise ValueError("frozen experiment requires 12 speakers x 12 recordings")
    return groups, speakers[:6], speakers[6:]


def vectors_for(data, groups, speakers, duration):
    index = {(x["source_id"], str(x["duration"])): x for x in data["rows"]}
    enrollment, probes = {}, []
    failures = 0
    for speaker in speakers:
        enrollment_rows = [index[(i, "full")] for i in groups[speaker][:3]]
        if any(x["status"] != "ok" for x in enrollment_rows):
            raise ValueError("enrollment extraction failed; model cannot be evaluated")
        enrollment[speaker] = unit(np.mean([x["vector"] for x in enrollment_rows], axis=0))
        for sid in groups[speaker][3:]:
            row = index[(sid, str(duration))]
            failures += row["status"] != "ok"
            probes.append(dict(id=sid, speaker=speaker,
                               vector=unit(row["vector"]) if row["status"] == "ok" else None))
    return enrollment, probes, failures


def calibrate(enrollment, probes):
    same, different, pair_same, pair_different = [], [], [], []
    for row in probes:
        if row["vector"] is None:
            continue
        for person, vector in enrollment.items():
            (same if person == row["speaker"] else different).append(float(row["vector"] @ vector))
    for a, b in itertools.combinations(probes, 2):
        if a["vector"] is not None and b["vector"] is not None:
            (pair_same if a["speaker"] == b["speaker"] else pair_different).append(
                float(a["vector"] @ b["vector"]))
    if not all([same, different, pair_same, pair_different]):
        raise ValueError("calibration must contain valid positive and negative recordings")
    known = float(max(different) + 1e-6)
    join = float(max(pair_different) + 1e-6)
    return dict(known=known, join=join,
                new=float(min(join - 0.05, np.percentile(pair_same, 5))), margin=0.05,
                criterion="zero observed calibration pair false accepts, not a FAR guarantee",
                same_pairs=len(pair_same), different_pairs=len(pair_different),
                calibration_pair_frr=float(np.mean(np.asarray(pair_same) < join)))


def open_set(enrollment, probes, policy):
    # Two registered identities; other four people are absent from the gallery.
    known = dict(list(enrollment.items())[:2])
    counts = Counter(known_probes=0, unknown_probes=0, correct_known=0,
                     wrong_identity=0, unknown_false_accept=0, known_unresolved=0,
                     extraction_failures=0)
    for row in probes:
        is_known = row["speaker"] in known
        counts["known_probes" if is_known else "unknown_probes"] += 1
        if row["vector"] is None:
            counts["extraction_failures"] += 1
            if is_known:
                counts["known_unresolved"] += 1
            continue
        scores = sorted(((float(row["vector"] @ v), k) for k, v in known.items()), reverse=True)
        accepted = scores[0][0] >= policy["known"] and scores[0][0] - scores[1][0] >= policy["margin"]
        if accepted:
            if not is_known:
                counts["unknown_false_accept"] += 1
            elif scores[0][1] == row["speaker"]:
                counts["correct_known"] += 1
            else:
                counts["wrong_identity"] += 1
        elif is_known:
            counts["known_unresolved"] += 1
    return dict(counts)


def discover(enrollment, probes, policy, *, seed, can_create=True):
    """A bounded, intentionally simple offline policy with no persistent Person side effects."""
    known = list(enrollment)[:2]
    clusters = [dict(known=k, templates=[enrollment[k]], members=[], established=True)
                for k in known]
    counts = Counter(unresolved=0, duplicate_events=0, cross_person_assignments=0,
                     promotions=0, impure_promotions=0, extraction_failures=0)
    order = np.random.default_rng(seed).permutation(len(probes))
    seen = set()
    for i in order:
        row = probes[i]
        if row["id"] in seen:
            counts["duplicate_events"] += 1
            continue
        seen.add(row["id"])
        vector = row["vector"]
        if vector is None:
            counts["unresolved"] += 1
            counts["extraction_failures"] += 1
            continue
        scores = sorted(((float(vector @ unit(np.mean(c["templates"], axis=0))), j)
                         for j, c in enumerate(clusters)), reverse=True)
        best, j = scores[0]
        cluster = clusters[j]
        threshold = policy["known"] if cluster["known"] else policy["join"]
        consistent = cluster["known"] or min(float(vector @ x) for x in cluster["templates"]) >= policy["join"]
        if best >= threshold and best - scores[1][0] >= policy["margin"] and consistent:
            existing_truth = {m["speaker"] for m in cluster["members"]}
            if cluster["known"]:
                existing_truth.add(cluster["known"])
            counts["cross_person_assignments"] += bool(existing_truth - {row["speaker"]})
            cluster["members"].append(row)
            if not cluster["known"] and len(cluster["templates"]) < 5:
                cluster["templates"].append(vector)
            if not cluster["established"] and len(cluster["members"]) >= 3:
                cluster["established"] = True
                counts["promotions"] += 1
                counts["impure_promotions"] += len({m["speaker"] for m in cluster["members"]}) > 1
        elif can_create and max(float(vector @ t) for c in clusters for t in c["templates"]) < policy["new"] and len(clusters) < 22:
            clusters.append(dict(known=None, templates=[vector], members=[row], established=False))
        else:
            counts["unresolved"] += 1
    candidates = [c for c in clusters if c["known"] is None]
    speaker_clusters = Counter()
    established_unknown = set()
    for cluster in candidates:
        people = {m["speaker"] for m in cluster["members"]}
        speaker_clusters.update(people)
        if cluster["established"] and len(people) == 1:
            established_unknown.update(people - set(known))
    return dict(counts, candidate_clusters_created=len(candidates),
                extra_clusters=sum(max(0, n - 1) for n in speaker_clusters.values()),
                clusters_containing_known_people=sum(bool({m["speaker"] for m in c["members"]} & set(known)) for c in candidates),
                pure_unknown_people_promoted=len(established_unknown),
                unknown_people_expected=len(enrollment) - 2,
                impure_final_clusters=sum(len({m["speaker"] for m in c["members"]}) > 1 for c in candidates))


def analyze(root):
    manifest = json.loads((root / "assets.json").read_text())
    groups, calibration, test = split_samples(manifest)
    results = []
    for path in sorted((root / "embeddings").glob("*.json")):
        data = json.loads(path.read_text())
        by_duration = {}
        for duration in DURATIONS:
            enroll_cal, probes_cal, _ = vectors_for(data, groups, calibration, duration)
            enroll_test, probes_test, failures = vectors_for(data, groups, test, duration)
            policy = calibrate(enroll_cal, probes_cal)
            times = [r["elapsed_ms"] for r in data["rows"] if r["duration"] == duration]
            by_duration[str(duration)] = dict(
                policy=policy, open_set=open_set(enroll_test, probes_test, policy),
                discovery=[discover(enroll_test, probes_test, policy, seed=seed,
                                    can_create=duration == "full" or duration >= 1.5)
                           for seed in range(5)],
                extraction_failures=failures,
                timing_ms=dict(p50=float(np.percentile(times, 50)), p95=float(np.percentile(times, 95)), n=len(times)))
        results.append(dict(model=data["model"], metadata=data["metadata"],
                            source_embeddings_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            load_ms=data["load_ms"], first_inference_ms=data["first_inference_ms"],
                            peak_process_rss_bytes=data["peak_process_rss_bytes"],
                            by_duration=by_duration))
    parity = None
    native, onnx = root / "embeddings/campplus_native.json", root / "embeddings/campplus_onnx.json"
    if native.exists() and onnx.exists():
        a, b = [json.loads(p.read_text())["rows"] for p in [native, onnx]]
        lookup = {(r["source_id"], str(r["duration"])): r for r in b}
        cosine, maxabs = [], []
        for row in a:
            other = lookup[(row["source_id"], str(row["duration"]))]
            if row["status"] == other["status"] == "ok":
                x, y = np.asarray(row["vector"]), np.asarray(other["vector"])
                cosine.append(float(x @ y))
                maxabs.append(float(np.max(np.abs(x-y))))
        parity = dict(n=len(cosine), cosine_min=min(cosine), cosine_mean=float(np.mean(cosine)),
                      max_abs_component_error=max(maxabs),
                      note="Numeric sample agreement; not universal equivalence or license to reuse thresholds")
    return dict(production_ready=False, dataset="AISHELL-1 public subset", calibration_speakers=calibration,
                test_speakers=test, recordings=144, independent_capture_sessions_known=False,
                model_training_overlap_excluded=False, seeds=list(range(5)),
                quality_gate="clean read speech assumed; duration gate only, no VAD/overlap validation",
                models=results, campplus_parity=parity)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = analyze(args.root)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("models", [x["model"] for x in report["models"]])
