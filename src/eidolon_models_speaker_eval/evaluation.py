"""Evaluate frozen embeddings without hiding failed extractions or data leakage."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict


def _unit(vector):
    if not isinstance(vector, list) or not vector:
        raise ValueError("nonempty vector required")
    if any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        for x in vector
    ):
        raise ValueError("finite numeric vector required")
    norm = math.hypot(*vector)
    if not norm or not math.isfinite(norm):
        raise ValueError("zero or unrepresentable vector norm")
    return [x / norm for x in vector]


def _fingerprint(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def _validate(data):
    if data.get("schema_version") != 1:
        raise ValueError("unsupported dataset schema")
    model = data.get("model", {})
    if (
        not model.get("id")
        or not model.get("preprocessing")
        or not re.fullmatch("[a-f0-9]{64}", model.get("weights_sha256", ""))
    ):
        raise ValueError("versioned model and weights digest required")
    samples, hashes, sessions, dimensions = {}, set(), {}, set()
    for row in data["samples"]:
        for key in ("id", "speaker", "device", "capture_session"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"sample {key} required")
        if row["id"] in samples:
            raise ValueError("duplicate sample id")
        digest = row.get("audio_sha256", "")
        if not re.fullmatch("[a-f0-9]{64}", digest) or digest in hashes:
            raise ValueError("missing or duplicated audio digest")
        hashes.add(digest)
        partition = row["partition"]
        if partition not in {"enrollment", "calibration", "test"}:
            raise ValueError("invalid partition")
        # Session ids are dataset-global, not local counters on each device.
        previous = sessions.setdefault(row["capture_session"], partition)
        if previous != partition:
            raise ValueError("capture session leaks between partitions")
        if row["status"] not in {"ok", "insufficient", "unavailable"}:
            raise ValueError("invalid extraction status")
        duration = row.get("duration_ms")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError("positive audio duration required")
        if partition == "enrollment" and (row["device"] != "mobile" or row["status"] != "ok"):
            raise ValueError("enrollment must be successful mobile audio")
        if row["status"] == "ok":
            dimensions.add(len(_unit(row.get("vector"))))
        elif "vector" in row:
            raise ValueError("failed extraction cannot carry a usable vector")
        samples[row["id"]] = row
    if len(dimensions) != 1:
        raise ValueError("embedding dimensions must match")
    ids = set()
    for trial in data["trials"]:
        if not trial.get("id") or trial["id"] in ids:
            raise ValueError("unique trial id required")
        ids.add(trial["id"])
        probe = samples.get(trial["probe_id"])
        enrollment = trial["enrollment_ids"]
        if not probe or probe["partition"] == "enrollment":
            raise ValueError("probe must belong to calibration or test")
        if not enrollment or len(set(enrollment)) != len(enrollment):
            raise ValueError("unique enrollment references required")
        if any(x not in samples or samples[x]["partition"] != "enrollment" for x in enrollment):
            raise ValueError("invalid enrollment reference")
    return samples


def _ratio(a, b):
    return a / b if b else None


def _metrics(rows):
    positive = [r for r in rows if r["expected_match"]]
    negative = [r for r in rows if not r["expected_match"]]
    scored_positive = [r for r in positive if r["score"] is not None]
    scored_negative = [r for r in negative if r["score"] is not None]
    return {
        "positive_trials": len(positive),
        "negative_trials": len(negative),
        "far": _ratio(sum(r["accepted"] for r in negative), len(negative)),
        "scored_negative_far": _ratio(
            sum(r["accepted"] for r in scored_negative), len(scored_negative)
        ),
        "end_to_end_frr": _ratio(sum(not r["accepted"] for r in positive), len(positive)),
        "scored_positive_frr": _ratio(
            sum(not r["accepted"] for r in scored_positive), len(scored_positive)
        ),
        "inconclusive": sum(r["status"] == "insufficient" for r in rows),
        "unavailable": sum(r["status"] == "unavailable" for r in rows),
        "extraction_failure_rate": _ratio(sum(r["score"] is None for r in rows), len(rows)),
    }


def evaluate(data, *, partition, threshold):
    if partition not in {"calibration", "test"}:
        raise ValueError("choose calibration or test")
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("threshold must be a finite cosine value")
    samples = _validate(data)
    rows = []
    for trial in data["trials"]:
        probe = samples[trial["probe_id"]]
        if probe["partition"] != partition:
            continue
        grouped = defaultdict(list)
        for sid in trial["enrollment_ids"]:
            item = samples[sid]
            grouped[item["speaker"]].append(_unit(item["vector"]))
        score = None
        if probe["status"] == "ok":
            vector = _unit(probe["vector"])
            templates = [
                _unit([math.fsum(xs) for xs in zip(*vectors, strict=True)])
                for vectors in grouped.values()
            ]
            score = max(
                math.fsum(a * b for a, b in zip(template, vector, strict=True))
                for template in templates
            )
            score = max(-1.0, min(1.0, score))
        rows.append(
            dict(
                id=trial["id"],
                probe_id=probe["id"],
                device=probe["device"],
                duration_ms=probe["duration_ms"],
                allowed_speakers=len(grouped),
                quality=probe.get("quality"),
                expected_match=probe["speaker"] in grouped,
                status=probe["status"],
                reason=probe.get("reason"),
                score=score,
                accepted=score is not None and score >= threshold,
            )
        )
    if not rows:
        raise ValueError(f"no {partition} trials")
    devices = sorted({r["device"] for r in rows})
    probes = {r["probe_id"] for r in rows}
    return {
        "schema_version": 1,
        "dataset_sha256": _fingerprint(data),
        "model": data["model"],
        "scoring": "normalized-mean-per-speaker-max-allowlist-v1",
        "partition": partition,
        "threshold": threshold,
        "production_ready": False,
        "metrics": _metrics(rows),
        "by_device": {d: _metrics([r for r in rows if r["device"] == d]) for d in devices},
        "by_allowlist_size": {
            str(n): _metrics([r for r in rows if r["allowed_speakers"] == n])
            for n in sorted({r["allowed_speakers"] for r in rows})
        },
        "independence": {
            "unique_probe_recordings": len(probes),
            "unique_probe_speakers": len({samples[p]["speaker"] for p in probes}),
            "unique_probe_sessions": len({samples[p]["capture_session"] for p in probes}),
            "confidence_interval": None,
            "note": (
                "Trials may share people, sessions or recordings; "
                "counts do not prove independent low-FAR accuracy."
            ),
        },
        "trials": rows,
    }


def calibration_curve(data, *, thresholds):
    """Explore calibration data only; never select a production policy."""
    reports = [evaluate(data, partition="calibration", threshold=t) for t in thresholds]
    if not reports:
        raise ValueError("at least one calibration threshold required")
    return dict(
        schema_version=1,
        dataset_sha256=reports[0]["dataset_sha256"],
        model=data["model"],
        production_ready=False,
        curve=[dict(threshold=r["threshold"], metrics=r["metrics"]) for r in reports],
    )
