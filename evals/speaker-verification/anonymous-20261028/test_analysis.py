"""Behavioral checks for experimental accounting, not production identity tests."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("anonymous_analysis", Path(__file__).with_name("analyze.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def setup():
    vectors = np.eye(4)
    gallery = {"a": vectors[0], "b": vectors[1], "c": vectors[2], "d": vectors[3]}
    policy = dict(known=0.8, join=0.8, new=0.6, margin=0.05)
    return gallery, policy


def probe(i, speaker, vector):
    return dict(id=i, speaker=speaker, vector=vector)


def test_three_distinct_recordings_promote_one_pure_unknown():
    gallery, policy = setup()
    rows = [probe(str(i), "c", gallery["c"]) for i in range(3)]
    result = analysis.discover(gallery, rows, policy, seed=0)
    assert result["candidate_clusters_created"] == 1
    assert result["pure_unknown_people_promoted"] == 1
    assert result["promotions"] == 1
    assert result["impure_promotions"] == 0


def test_repeated_event_never_counts_as_three_turns():
    gallery, policy = setup()
    row = probe("same-source", "c", gallery["c"])
    result = analysis.discover(gallery, [row, row, row], policy, seed=0)
    assert result["promotions"] == 0
    assert result["duplicate_events"] == 2


def test_two_unknown_people_remain_separate():
    gallery, policy = setup()
    rows = [probe(f"{s}-{i}", s, gallery[s]) for s in ["c", "d"] for i in range(3)]
    result = analysis.discover(gallery, rows, policy, seed=2)
    assert result["candidate_clusters_created"] == 2
    assert result["cross_person_assignments"] == 0
    assert result["pure_unknown_people_promoted"] == 2


def test_short_evidence_does_not_create_a_person():
    gallery, policy = setup()
    result = analysis.discover(gallery, [probe("short", "c", gallery["c"])],
                               policy, seed=0, can_create=False)
    assert result["candidate_clusters_created"] == 0
    assert result["unresolved"] == 1


def test_quality_or_extraction_failure_is_not_a_new_person():
    gallery, policy = setup()
    rows = [probe("failed", "a", None)]
    result = analysis.discover(gallery, rows, policy, seed=0)
    assert result["candidate_clusters_created"] == 0
    assert result["extraction_failures"] == 1
    known = analysis.open_set(gallery, rows, policy)
    assert known["known_probes"] == known["known_unresolved"] == 1


def test_equal_scores_abstain_instead_of_assigning_arbitrary_person():
    gallery, policy = setup()
    ambiguous = analysis.unit([1, 1, 0, 0])
    policy["known"] = 0.6
    result = analysis.open_set(gallery, [probe("ambiguous", "a", ambiguous)], policy)
    assert result["known_unresolved"] == 1
    assert result["correct_known"] == 0


def test_source_overlap_is_rejected_before_calibration():
    samples = [dict(speaker=f"p{s:02}", id=f"r{s}-{i}", sha256=f"{s}-{i}")
               for s in range(12) for i in range(12)]
    _, cal, test = analysis.split_samples(dict(samples=samples))
    assert not set(cal) & set(test)
    samples[-1]["sha256"] = samples[0]["sha256"]
    with pytest.raises(ValueError, match="duplicate"):
        analysis.split_samples(dict(samples=samples))


def test_zero_and_nonfinite_vectors_rejected():
    for vector in [[0, 0], [1, float("nan")]]:
        with pytest.raises(ValueError):
            analysis.unit(vector)
