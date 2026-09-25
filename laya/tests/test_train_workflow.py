"""The torch-free half of the training workflow: records, scenarios, generators, augment, assemble."""

import json
import random
from pathlib import Path

import pytest

from eidolon_laya_train.assemble import assemble
from eidolon_laya_train.augment import augment_records
from eidolon_laya_train.generators import generate
from eidolon_laya_train.records import (
    Record,
    gold_names,
    option_names,
    read_jsonl,
    target_vector,
    write_jsonl,
)
from eidolon_laya_train.scenario import Scenario

HERE = Path(__file__).resolve().parent
SMART_HOME = HERE.parent / "train" / "scenarios" / "smart-home"


def _q(crit=None):
    return {
        "type": "choice",
        "instructions": "x",
        "criteria": crit or {"a": "A", "b": "B", "c": None},
    }


def test_target_vector_from_gold_target_and_lists():
    q = _q()
    assert target_vector(q, {"gold": "b"}) == [0.0, 1.0, 0.0]
    assert target_vector(q, {"gold": ["a", "c"]}) == [0.5, 0.0, 0.5]
    assert target_vector(q, {"target": {"a": 3, "b": 1}}) == [0.75, 0.25, 0.0]
    with pytest.raises(ValueError):
        target_vector(q, {"gold": "zzz"})
    assert option_names({"type": "noul", "instructions": "x"}) == ["false", "true"]
    assert target_vector({"type": "noul", "instructions": "x"}, {"gold": True}) == [0.0, 1.0]
    assert option_names({"type": "score", "instructions": "x", "criteria": ["l", "m", "h"]}) == [
        "0",
        "1",
        "2",
    ]
    assert gold_names(q, {"target": {"a": 0.2, "b": 0.8}}) == {"b"}


def test_records_roundtrip(tmp_path):
    r = Record(
        id="s/1",
        scenario="s",
        source="t",
        state={"utterance": "开灯"},
        questions={"q": _q()},
        labels={"q": {"gold": "a"}},
    )
    p = tmp_path / "x.jsonl"
    assert write_jsonl(p, [r]) == 1
    back = list(read_jsonl(p))
    assert back[0] == r and "split" not in json.loads(p.read_text())


def test_scenario_materializes_dynamic_criteria_with_exits_last():
    scn = Scenario.load(SMART_HOME)
    qs = scn.build_questions({"devices": {"客厅灯": "客厅·灯", "主卧空调": "主卧·空调"}})
    assert list(qs["device"]["criteria"]) == [
        "客厅灯",
        "主卧空调",
        "多个设备或整屋",
        "没有对应的设备",
    ]
    assert set(qs) == {"intent", "device", "action"}
    with pytest.raises(ValueError):
        scn.build_questions()  # device needs its dynamic slot


def test_import_evals_cases_yields_gold_labels():
    scn = Scenario.load(SMART_HOME)
    recs = list(
        generate(
            scn,
            {
                "kind": "import",
                "adapter": "evals_cases",
                "path": "../../../evals/smart-home",
                "name": "locked",
            },
        )
    )
    assert len(recs) == 182
    r = next(r for r in recs if r.id.endswith("/02-01"))
    assert r.labels["intent"] == {"gold": "控制"} and isinstance(r.labels["device"]["gold"], list)
    assert "没有对应的设备" in r.questions["device"]["criteria"]


def test_template_generator_covers_every_slice():
    scn = Scenario.load(SMART_HOME)
    recs = list(
        generate(
            scn,
            {"kind": "template", "module": "generators.rules:generate", "per_slice": 20},
            seed=1,
        )
    )
    tags = {t for r in recs for t in r.tags if not t.startswith("home:")}
    assert {
        "explicit-control",
        "implicit-intent",
        "status-query",
        "non-command",
        "multi-device",
        "device-not-in-home",
    } <= tags
    for r in recs:  # every gold is an option of its question
        for qid, lab in r.labels.items():
            target_vector(r.questions[qid], lab)


def test_augment_exit_injection_and_subset():
    scn = Scenario.load(SMART_HOME)
    recs = list(
        generate(
            scn,
            {"kind": "template", "module": "generators.rules:generate", "per_slice": 10},
            seed=2,
        )
    )
    out = list(
        augment_records(
            recs,
            scn,
            [
                {"name": "exit_injection", "question": "device", "exit": "没有对应的设备"},
                {"name": "option_subset", "question": "device", "keep": 3},
            ],
            seed=3,
        )
    )
    derived = [r for r in out if "derived_from" in r.meta]
    assert derived
    for d in derived:
        if "aug:exit_injection" in d.tags:
            assert d.labels["device"]["gold"] == "没有对应的设备"
            orig = next(r for r in recs if r.id == d.meta["derived_from"])
            for g in gold_names(orig.questions["device"], orig.labels["device"]):
                assert g not in d.questions["device"]["criteria"]
        if "aug:option_subset" in d.tags:
            orig = next(r for r in recs if r.id == d.meta["derived_from"])
            golds = gold_names(orig.questions["device"], orig.labels["device"])
            crit = d.questions["device"]["criteria"]
            assert len(crit) <= len(golds) + 3 + 2
            assert list(crit)[-2:] == ["多个设备或整屋", "没有对应的设备"]
            assert golds <= set(crit)
            target_vector(d.questions["device"], d.labels["device"])


def test_assemble_excludes_locked_eval_and_splits_deterministically(tmp_path):
    scn = Scenario.load(SMART_HOME)
    gen = list(
        generate(
            scn,
            {"kind": "template", "module": "generators.rules:generate", "per_slice": 15},
            seed=4,
        )
    )
    locked = list(
        generate(
            scn, {"kind": "import", "adapter": "evals_cases", "path": "../../../evals/smart-home"}
        )
    )
    # smuggle one eval utterance into the generated set under a fresh id
    leak = Record(
        id="smart-home/leak",
        scenario="smart-home",
        source="t",
        state=locked[0].state,
        questions=locked[0].questions,
        labels=locked[0].labels,
    )
    write_jsonl(tmp_path / "gen.jsonl", gen + [leak])
    cfg = {
        "sources": [{"path": "gen.jsonl"}],
        "locked_eval": [str(SMART_HOME / "../../../evals/smart-home")],
        "split": {"val": 0.2, "calib": 0.2},
        "seed": 1,
    }
    m = assemble(cfg, tmp_path / "ds", tmp_path)
    assert m["dropped"].get("locked") == 1
    assert sum(m["counts"].values()) == m["unique"]
    splits = {
        r.id: r.split
        for name in ("train", "val", "calib")
        for r in read_jsonl(tmp_path / "ds" / f"{name}.jsonl")
    }
    m2 = assemble(cfg, tmp_path / "ds2", tmp_path)
    assert m2["counts"] == m["counts"]
    for name in ("train", "val", "calib"):
        for r in read_jsonl(tmp_path / "ds2" / f"{name}.jsonl"):
            assert splits[r.id] == name


def test_option_shuffle_keeps_target_aligned():
    pytest.importorskip("torch")
    from eidolon_laya_train.model import record_items

    tok = (
        pytest.importorskip("transformers").AutoTokenizer.from_pretrained(
            str(HERE.parent / "models" / "laya-multilingual" / "1c5edc17" / "torch" / "tokenizer")
        )
        if (
            HERE.parent / "models" / "laya-multilingual" / "1c5edc17" / "torch" / "tokenizer"
        ).exists()
        else None
    )
    if tok is None:
        pytest.skip("checkpoint not fetched")
    r = Record(
        id="s/1",
        scenario="s",
        source="t",
        state={"utterance": "开灯"},
        questions={"q": _q()},
        labels={"q": {"gold": "c"}},
    )
    for seed in range(5):
        it = record_items(
            r, tok, {"max_len": 256, "head_max_len": 128}, shuffle=random.Random(seed)
        )[0]
        assert it["names"][it["label"]] == "c" and it["target"][it["label"]] == 1.0


def test_package_writes_manifest_that_artifacts_can_verify(tmp_path):
    from eidolon_laya_train.package import package
    from eidolon_models_laya.artifacts import Manifest, verify_torch

    ck = tmp_path / "ck"
    (ck / "encoder").mkdir(parents=True)
    (ck / "tokenizer").mkdir()
    (ck / "model.safetensors").write_bytes(b"w")
    (ck / "rl_agent_config.json").write_text(json.dumps({"encoder": "e", "max_len": 8, "head_max_len": 4, "fine_tuned_from": "x"}))
    (ck / "encoder" / "config.json").write_text("{}")
    (ck / "tokenizer" / "tokenizer.json").write_text("{}")
    out = package(ck, tmp_path / "models", "demo", run_id="r1")
    m = Manifest.load(out)
    assert m.hub == "local" and verify_torch(m) == [] and out.name == m.revision
    assert set(m.torch_files) == {"model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json"}
