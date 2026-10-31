"""Acceptance set for calibrated recognition (design §3.3, stage 1 exit).

Held-out public speakers (never in the cohort) form two-person households.
Each Person registers the way the product does: three samples plus a trial,
evaluated by Channel's own evaluate_enrollment, which produces the template
and the registration windows. Every other recording is then a probe: the
Person's own later recordings are genuine, everyone outside the household is
a stranger. Each probe is decided by Channel's own household_tables + decide,
and by the fixed-threshold policy it replaces, for comparison.

    eidolon_channel/.venv/bin/python -I \
        eidolon_models/evals/speaker-verification/cohort-20261031/evaluate.py \
        --held-out .eidolon/speaker-cohort-20261031/build/held-out.json \
        --model-dir <dir holding the policy and impostor-cohort.npz>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from eidolon_sdk.biz.person import PersonProfile
from eidolon_sdk.biz.speaker_identity import SpeakerCandidate, SpeakerRecognitionPolicy
from eidolon_sdk.biz.voiceprint import EnrollmentEvaluationRequest, EnrollmentSample, VoiceTemplate

from eidolon.livekit.plugins.speaker_verification.calibration import (
    decide,
    household_tables,
    load_cohort,
)
from eidolon.livekit.plugins.speaker_verification.enrollment import evaluate_enrollment

POLICY_FILE = "recognition-controlled-trial.json"


def households(speakers: list[str]) -> list[list[str]]:
    groups = [speakers[i:i + 2] for i in range(0, len(speakers) - len(speakers) % 2, 2)]
    if len(speakers) % 2:
        groups[-1].append(speakers[-1])
    return groups


def register(rows) -> tuple[VoiceTemplate, list]:
    samples = [EnrollmentSample.model_validate(r["sample"]) for r in rows[:4]]
    evaluation = evaluate_enrollment(EnrollmentEvaluationRequest(
        samples={str(i): s for i, s in enumerate(samples[:3])}, trial=samples[3],
        recording_sha256="0" * 64))
    template = VoiceTemplate(profile_id="v", revision=1, sample=evaluation.template,
                             calibration=evaluation.calibration)
    return template, rows[4:]


def probes(rows, floor):
    for row in rows:
        for w in [row["sample"]] + row["sample"]["crops"]:
            if w["speech_ms"] >= floor:
                yield w["speech_ms"], np.asarray(w["vector"])


def band_name(policy, speech_ms):
    return policy.calibration.band(speech_ms).name


def legacy(policy, scores):
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, score = ranked[0]
    if score < policy.threshold:
        return "below", None
    if len(ranked) > 1 and score - ranked[1][1] < policy.min_person_margin:
        return "ambiguous", None
    return "identified", best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    policy = SpeakerRecognitionPolicy.model_validate_json((args.model_dir / POLICY_FILE).read_text())
    cohort = load_cohort(args.model_dir, policy)
    held = json.loads(args.held_out.read_text())["speakers"]
    speakers = sorted(s for s, rows in held.items() if len(rows) >= 6)

    tally = {"calibrated": defaultdict(Counter), "legacy": defaultdict(Counter)}
    thresholds = defaultdict(list)
    # Stranger false accepts per (Person, stranger) voice pair: a few look-alike
    # voices behave very differently from errors spread over everyone.
    pairs = {"calibrated": Counter(), "legacy": Counter()}
    stranger_trials = Counter()
    for group in households(speakers):
        persons, later = [], {}
        for speaker in group:
            template, rest = register(held[speaker])
            persons.append(SpeakerCandidate(person_id=speaker, person_revision=1, display_name=speaker,
                                            profile=PersonProfile(), templates=[template]))
            later[speaker] = rest
        tables = household_tables(policy, cohort, persons)
        for person, table in tables.items():
            for band, entry in table.items():
                thresholds[band].append((entry.threshold, entry.direct, entry.genuine_acceptance))
        vectors = {p.person_id: np.asarray(p.templates[0].sample.vector) for p in persons}
        strangers = [s for s in speakers if s not in group]
        trials = [(s, s, later[s]) for s in group] + [(None, s, held[s]) for s in strangers]
        for truth, voice, rows in trials:
            for speech_ms, vector in probes(rows, policy.min_speech_ms):
                if truth is None:
                    stranger_trials[tuple(group), voice] += 1
                scores = {p: float(vector @ t) for p, t in vectors.items()}
                band = policy.calibration.band(speech_ms)
                decision = decide(scores, tables, band)
                for name, (outcome, who) in (("calibrated", (decision.outcome, decision.person_id)),
                                             ("legacy", legacy(policy, scores))):
                    kind = "genuine" if truth else "stranger"
                    if outcome == "identified":
                        outcome = "correct" if who == truth else "wrong_person"
                        if truth is None:
                            pairs[name][who, voice] += 1
                    for key in (band.name, "ge_1s" if speech_ms >= 1000 else "lt_1s", "all"):
                        tally[name][(kind, key)][outcome] += 1

    report = {"policy": policy.version, "speakers": len(speakers),
              "households": len(households(speakers)), "cohort_vectors": int(len(cohort.vectors))}
    for name, rows in tally.items():
        out = {}
        for (kind, key), counts in sorted(rows.items()):
            total = sum(counts.values())
            out[f"{kind}/{key}"] = {"n": total, **{k: round(v / total, 4) for k, v in sorted(counts.items())}}
        report[name] = out
    for name, counted in pairs.items():
        total = sum(counted.values())
        top = counted.most_common(5)
        report[name]["stranger_false_accepts"] = {
            "trials": total, "voice_pairs": len(counted),
            "pairs_compared": len(stranger_trials) * 2,
            "top_pairs_share": round(sum(n for _, n in top) / total, 3) if total else 0,
            "top_pairs": [[f"{p}<-{v}", n] for (p, v), n in top],
        }
    report["thresholds"] = {
        band: {"median": round(float(np.median([t for t, _, _ in v])), 3),
               "p90": round(float(np.percentile([t for t, _, _ in v], 90)), 3),
               "direct": round(sum(d for _, d, _ in v) / len(v), 3)}
        for band, v in thresholds.items()}
    text = json.dumps(report, indent=1, ensure_ascii=False)
    print(text)
    if args.json:
        args.json.write_text(text)


if __name__ == "__main__":
    main()
