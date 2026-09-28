"""Protect evaluation boundaries and family accounting in the backbone pilot."""
import copy
import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parents[1] / "train/scenarios/backbone-comparison-v1"


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_payload_is_independent_of_gold_and_audit_metadata():
    c = load("compare")
    _, r = c.records()[0]
    before = c.payload(r)
    modified = copy.deepcopy(r)
    modified["labels"] = {"move": {"gold": ["abstain"]}}
    modified["meta"] = {"reason": "SECRET_ANSWER_MUST_NOT_ENTER_PROMPT"}
    assert c.payload(modified) == before
    assert "SECRET_ANSWER" not in str(before)


def test_only_fixed_development_files_and_unique_option_mapping():
    c = load("compare")
    rs = c.records()
    assert len(rs) == 114
    assert all(p.name == "val.jsonl" and "sealed" not in p.parts for p in c.DATA.values())
    for _, r in rs:
        _, _, names, texts = c.payload(r)
        assert len(set(texts)) == len(names)
        assert all(text.startswith(name + ": ") for name, text in zip(names, texts))
        assert set(r["labels"]["move"]["gold"]) <= set(names)


def test_laya_tensor_inputs_do_not_depend_on_gold():
    from transformers import AutoTokenizer
    from eidolon_laya_train.model import record_items
    from eidolon_laya_train.records import Record
    c = load("compare")
    _, r = c.records()[0]
    tok = AutoTokenizer.from_pretrained(c.CHECKPOINTS["laya"] / "tokenizer", local_files_only=True)
    other = copy.deepcopy(r)
    other["labels"] = {"move": {"gold": ["abstain"]}}
    cfg = {"max_len": 2048, "head_max_len": 256}
    a = record_items(Record.from_dict(r), tok, cfg)[0]
    b = record_items(Record.from_dict(other), tok, cfg)[0]
    assert a["target"] != b["target"]
    assert a["ids"] == b["ids"]
    assert a["markers"] == b["markers"]
    assert a["qtype"] == b["qtype"]


def test_partial_family_is_not_reported_as_all_variants_correct():
    report = load("report")
    raw = {}
    for i in range(2):
        raw["v7", f"r{i}"] = {"meta": {"family": "same", "slot_to_source": {"M0": 0}}}
    row = dict(split="v7", id="r0", pred="respond:M0", gold=["respond:M0"], correct=True,
               slice="respond", elapsed_ms=1, first_call=False, p_top=.9)
    partial = report.aggregate([row], raw)
    assert partial["complete_families"] == 0
    assert partial["all_variants_correct_families"] == 0
    complete = report.aggregate([row, dict(row, id="r1", pred="abstain", correct=False)], raw)
    assert complete["complete_families"] == 1
    assert complete["all_variants_correct_families"] == 0
    assert complete["must_speak"]["missed"] == 1
