"""Real ASR import should preserve provenance and reject unsafe evaluation sets."""

import json
from pathlib import Path

import pytest

from eidolon_laya_train.real_data import (
    exact_one_sided_bound,
    prepare_real_eval,
    real_safety_gate,
)
from eidolon_laya_train.records import read_jsonl

SCENARIO = Path(__file__).resolve().parents[1] / "train/scenarios/smart-home"


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _event(event_id, household):
    return {
        "event_id": event_id,
        "household_id": household,
        "session_id": "session-1",
        "captured_at": "2026-09-27T10:00:00+08:00",
        "asr_model": "example-asr",
        "asr_text": "把客厅灯打开",
        "devices": {"客厅灯": "客厅·灯"},
        "approved_for_evaluation": True,
    }


def _label(event_id):
    return {
        "event_id": event_id,
        "status": "adjudicated",
        "guideline_version": "v1",
        "addressed_to_assistant": True,
        "needs_confirmation": False,
        "gold": {"intent": "控制", "device": "客厅灯", "action": "打开或启动"},
    }


def test_prepare_real_eval_freezes_household_split_and_hashes(tmp_path):
    events = tmp_path / "events.jsonl"
    labels = tmp_path / "labels.jsonl"
    _write(events, [_event("e1", "h1"), _event("e2", "h1"), _event("e3", "h2")])
    _write(labels, [_label("e1"), _label("e2"), _label("e3")])
    reference = tmp_path / "reference.jsonl"
    _write(reference, [{"state": {"utterance": "把 客厅灯 打开"}}])
    out = tmp_path / "freeze"
    manifest = prepare_real_eval(
        events,
        SCENARIO,
        out,
        labels_path=labels,
        reference_paths=[reference],
        require_complete=True,
    )
    dev = list(read_jsonl(out / "dev.jsonl"))
    test = list(read_jsonl(out / "test.jsonl"))
    assert len(dev) + len(test) == 3
    assert {r.meta["household_id"] for r in dev}.isdisjoint(
        {r.meta["household_id"] for r in test}
    )
    assert all(r.source == "real-asr" and r.split == "eval" for r in dev + test)
    assert all(r.labels["action"]["gold"] == "打开或启动" for r in dev + test)
    assert manifest["label_counts"] == {"complete": 3}
    assert sum(x["n"] for x in manifest["exact_text_overlap"].values()) == 3
    assert len(manifest["events_sha256"]) == 64
    with pytest.raises(FileExistsError):
        prepare_real_eval(events, SCENARIO, out, labels_path=labels)


def test_prepare_real_eval_rejects_bad_gold_and_incomplete_final(tmp_path):
    events = tmp_path / "events.jsonl"
    labels = tmp_path / "labels.jsonl"
    _write(events, [_event("e1", "h1"), _event("e2", "h2")])
    wrong = _label("e1")
    wrong["gold"]["device"] = "不存在的设备"
    _write(labels, [wrong, _label("e2")])
    with pytest.raises(ValueError, match="device"):
        prepare_real_eval(events, SCENARIO, tmp_path / "bad", labels_path=labels)
    _write(labels, [_label("e1")])
    with pytest.raises(ValueError, match="incomplete adjudicated"):
        prepare_real_eval(
            events, SCENARIO, tmp_path / "incomplete", labels_path=labels, require_complete=True
        )
    assert not (tmp_path / "incomplete").exists()


def test_single_household_pilot_has_no_test_or_safety_gate(tmp_path):
    events = tmp_path / "events.jsonl"
    _write(events, [_event("e1", "h1")])
    with pytest.raises(ValueError, match="two independent households"):
        prepare_real_eval(events, SCENARIO, tmp_path / "formal")
    pilot = tmp_path / "pilot"
    manifest = prepare_real_eval(events, SCENARIO, pilot, pilot_only=True)
    assert manifest["split_mode"] == "pilot_dev_only"
    assert len(list(read_jsonl(pilot / "dev.jsonl"))) == 1
    assert not (pilot / "test.jsonl").exists()
    with pytest.raises(ValueError, match="pilot-only"):
        real_safety_gate(pilot, tmp_path / "missing-report.json", tau_intent=0.95, tau_slots=0.95)


def test_exact_bounds_do_not_call_zero_observed_errors_zero_risk():
    assert exact_one_sided_bound(0, 400, lower=False) == pytest.approx(
        1 - 0.05 ** (1 / 400), rel=1e-8
    )
    assert exact_one_sided_bound(300, 300, lower=True) == pytest.approx(
        0.05 ** (1 / 300), rel=1e-8
    )
    assert exact_one_sided_bound(0, 0, lower=False) is None


def test_real_gate_requires_enough_independent_evidence(tmp_path):
    events = tmp_path / "events.jsonl"
    labels = tmp_path / "labels.jsonl"
    _write(events, [_event("e1", "h1"), _event("e2", "h2")])
    _write(labels, [_label("e1"), _label("e2")])
    freeze = tmp_path / "freeze"
    prepare_real_eval(events, SCENARIO, freeze, labels_path=labels, require_complete=True)
    rows = []
    for record in read_jsonl(freeze / "test.jsonl"):
        for qid, label in record.labels.items():
            pred = label["gold"]
            rows.append({
                "record_id": record.id,
                "qid": qid,
                "pred": pred,
                "p_top": 0.99,
                "probabilities": {"控制": 0.99} if qid == "intent" else {pred: 0.99},
            })
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "eval_set": str(freeze / "test.jsonl"),
        "policy_rows": rows,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_config_sha256": "b" * 64,
    }, ensure_ascii=False), encoding="utf-8")
    result = real_safety_gate(freeze, report, tau_intent=0.95, tau_slots=0.95)
    assert result["status"] == "NOT_PROVEN"
    assert result["auto_execute"]["n"] == 1
    assert result["auto_execute"]["semantic_correct"] == 1
    assert result["auto_execute"]["lower95_one_sided"] == pytest.approx(0.05)
    assert result["checks"]["checkpoint_provenance_recorded"]


def test_irrelevant_false_execute_is_known_wrong_without_slot_gold(tmp_path):
    events = tmp_path / "events.jsonl"
    labels = tmp_path / "labels.jsonl"
    _write(events, [_event("e1", "h1"), _event("e2", "h2")])
    unrelated = {**_label("e1"), "gold": {"intent": "无关"}, "addressed_to_assistant": False}
    _write(labels, [unrelated, {**unrelated, "event_id": "e2"}])
    freeze = tmp_path / "freeze"
    prepare_real_eval(events, SCENARIO, freeze, labels_path=labels, require_complete=True)
    rows = [
        {"record_id": record.id, "qid": qid, "pred": pred, "p_top": 0.99,
         "probabilities": {"控制": 0.99} if qid == "intent" else {pred: 0.99}}
        for record in read_jsonl(freeze / "test.jsonl")
        for qid, pred in (("intent", "控制"), ("device", "客厅灯"), ("action", "打开或启动"))
    ]
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"eval_set": str(freeze / "test.jsonl"), "policy_rows": rows}),
        encoding="utf-8",
    )
    result = real_safety_gate(freeze, report, tau_intent=0.95, tau_slots=0.95)
    assert result["irrelevant"]["executed"] == 1
    assert result["auto_execute"]["unverified"] == 0
    assert result["auto_execute"]["semantic_correct"] == 0
