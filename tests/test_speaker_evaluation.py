"""Speaker evaluation must count abstentions and reject leaked test audio."""

import copy
import json
import os
import sys

import pytest

from eidolon_models_speaker_eval.evaluation import calibration_curve, evaluate


def dataset():
    def sample(i, speaker, device, partition, vector):
        return dict(
            id=i,
            speaker=speaker,
            device=device,
            partition=partition,
            capture_session=i,
            audio_sha256=f"{len(i):064x}",
            duration_ms=3000,
            status="ok",
            vector=vector,
        )

    rows = [
        sample("enroll", "alice", "mobile", "enrollment", [1, 0]),
        sample("positive", "alice", "box3", "test", [0.8, 0.6]),
        sample("negative-long", "bob", "box3", "test", [0, 1]),
    ]
    return dict(
        schema_version=1,
        model=dict(id="test", weights_sha256="a" * 64, preprocessing="reference-v1"),
        samples=rows,
        trials=[dict(id=r["id"], probe_id=r["id"], enrollment_ids=["enroll"]) for r in rows[1:]],
    )


def test_positive_and_negative_are_scored_without_identity_probability():
    report = evaluate(dataset(), partition="test", threshold=0.65)
    assert report["metrics"]["far"] == 0
    assert report["metrics"]["end_to_end_frr"] == 0
    assert report["trials"][0]["score"] == pytest.approx(0.8)
    assert "vector" not in str(report)
    assert report["production_ready"] is False


def test_missing_quality_counts_as_failed_positive_not_excluded():
    data = dataset()
    data["samples"][1].update(status="insufficient", reason="too_short")
    del data["samples"][1]["vector"]
    report = evaluate(data, partition="test", threshold=0.65)
    assert report["metrics"]["end_to_end_frr"] == 1
    assert report["metrics"]["scored_positive_frr"] is None
    assert report["metrics"]["inconclusive"] == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["samples"][1].update(audio_sha256=d["samples"][0]["audio_sha256"]),
        lambda d: d["samples"][1].update(capture_session="enroll"),
        lambda d: d["samples"][0].update(device="box3"),
        lambda d: d["samples"][1].update(vector=[float("nan"), 1]),
        lambda d: d["samples"][1].update(vector=[0, 0]),
        lambda d: d["samples"][1].update(vector=[1, 0, 0]),
        lambda d: d["trials"][0].update(enrollment_ids=["enroll", "enroll"]),
    ],
)
def test_invalid_or_leaked_evidence_is_not_evaluated(mutate):
    data = dataset()
    mutate(data)
    with pytest.raises(ValueError):
        evaluate(data, partition="test", threshold=0.65)


def test_no_negatives_is_unknown_far():
    data = dataset()
    data["trials"] = data["trials"][:1]
    assert evaluate(data, partition="test", threshold=0.65)["metrics"]["far"] is None


def test_calibration_never_uses_test_partition():
    with pytest.raises(ValueError, match="calibration"):
        calibration_curve(dataset(), thresholds=[0.3, 0.65])


def test_correlated_trials_are_reported_not_claimed_as_independent():
    data = dataset()
    duplicate = copy.deepcopy(data["trials"][1])
    duplicate["id"] = "another-allowlist"
    data["trials"].append(duplicate)
    report = evaluate(data, partition="test", threshold=0.65)
    assert report["metrics"]["negative_trials"] == 2
    assert report["independence"]["unique_probe_recordings"] == 2
    assert report["independence"]["confidence_interval"] is None


def test_multiple_enrollments_are_normalized_mean_per_person():
    data = dataset()
    second = copy.deepcopy(data["samples"][0])
    second.update(id="enroll-2", capture_session="enroll-2", audio_sha256="b" * 64, vector=[0, 1])
    data["samples"].append(second)
    data["trials"][0]["enrollment_ids"].append("enroll-2")
    report = evaluate(data, partition="test", threshold=0.65)
    assert report["trials"][0]["score"] == pytest.approx(1.4 / 2**0.5)


def test_cli_creates_private_report_and_never_overwrites_evidence(tmp_path, monkeypatch):
    from eidolon_models_speaker_eval.cli import main

    source, output = tmp_path / "input.json", tmp_path / "report.json"
    source.write_text(json.dumps(dataset()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            str(source),
            "--partition",
            "test",
            "--threshold",
            "0.65",
            "--output",
            str(output),
        ],
    )
    main()
    assert json.loads(output.read_text())["production_ready"] is False
    assert os.stat(output).st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        main()


def test_calibration_session_cannot_reappear_in_test():
    data = dataset()
    row = copy.deepcopy(data["samples"][1])
    row.update(id="calibration", partition="calibration", audio_sha256="c" * 64)
    data["samples"].append(row)
    with pytest.raises(ValueError, match="session leaks"):
        evaluate(data, partition="test", threshold=0.65)
