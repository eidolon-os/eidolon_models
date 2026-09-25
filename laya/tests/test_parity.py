"""Against the real checkpoint: our torch-free port must match upstream laya exactly."""

import json

import pytest

from eidolon_models_laya.sequence import ModelConfig, Tokenizer, build_sequence, to_internal

pytestmark = pytest.mark.model

TEXTS = [
    "悟空，前面那座山有没有妖怪？",
    "Refund the duplicate charge please",
    "混合 mixed 文本 🐒 <mask> tokens",
    "  leading spaces\nand newlines\t",
]
QUESTIONS = {
    "addressee": {
        "type": "choice",
        "instructions": "`utterance` 这句话是说给谁听的？",
        "criteria": {
            "唐僧": "师父",
            "孙悟空": "悟空、猴哥",
            "猪八戒": None,
            "沙僧": {"alias": ["悟净"]},
            "所有人": "",
        },
    },
    "needs_reply": {
        "type": "noul",
        "instructions": "需要回应吗？",
        "criteria": {True: "要回应", "false": None},
    },
    "urgency": {"type": "score", "instructions": "有多急？", "criteria": ["不急", "尽快", "马上"]},
    "tier": {
        "type": "choice",
        "instructions": ["structured", "instructions"],
        "criteria": ["small", "large"],
    },
}


def _state(n_units: int):
    return {
        "speaker": "主人",
        "utterance": "八戒，你刚才说的那户人家在哪儿？",
        "history": [{"speaker": "猪八戒", "text": "村东头有户人家。"}] * n_units,
    }


@pytest.fixture(scope="module")
def tok(torch_files):
    return Tokenizer(torch_files.tokenizer_dir)


def test_tokenizer_matches_transformers(tok, upstream_agent):
    for text in TEXTS:
        assert tok.encode(text) == upstream_agent.tok(text, add_special_tokens=False)["input_ids"]
    t = upstream_agent.tok
    assert (tok.mask_token_id, tok.cls_token_id, tok.sep_token_id, tok.pad_token_id) == (
        t.mask_token_id,
        t.cls_token_id,
        t.sep_token_id,
        t.pad_token_id,
    )


@pytest.mark.parametrize("n_units", [1, 20, 400])
@pytest.mark.parametrize("truncate_left", [False, True])
def test_build_sequence_matches_upstream(tok, upstream_agent, n_units, truncate_left):
    from eidolon_models_laya.vendor.laya.common import build_sequence as upstream_build

    for qdef in QUESTIONS.values():
        q = to_internal(qdef)
        ours, markers, _ = build_sequence(tok, _state(n_units), q, 1024, 256, truncate_left)
        ref, ref_markers = upstream_build(
            upstream_agent.tok, _state(n_units), q, 1024, 256, truncate_left=truncate_left
        )
        assert ours == ref and markers == ref_markers


@pytest.fixture(scope="module")
def torch_engine(torch_files):
    from eidolon_models_laya.backends import TorchBackend
    from eidolon_models_laya.engine import DecisionEngine

    backend = TorchBackend(torch_files.torch_dir, device="cpu")
    return DecisionEngine(
        backend,
        Tokenizer(torch_files.tokenizer_dir),
        ModelConfig.from_dict(torch_files.model_config()),
    )


@pytest.mark.parametrize("n_units", [1, 20, 400])
def test_torch_engine_answers_exactly_like_upstream(torch_engine, upstream_agent, n_units):
    ours = torch_engine.predict(_state(n_units), QUESTIONS)
    ref = upstream_agent.predict(_state(n_units), QUESTIONS)
    assert ours.answers == ref["answers"]
    assert ours.input_tokens == ref["usage"]["input_tokens"]
    assert bool(ours.truncated) == (n_units == 400)


def test_onnx_engine_agrees_with_torch(torch_engine, onnx_files):
    from eidolon_models_laya.backends import OnnxBackend
    from eidolon_models_laya.engine import DecisionEngine

    onnx_engine = DecisionEngine(
        OnnxBackend(onnx_files.onnx_path),
        Tokenizer(onnx_files.tokenizer_dir),
        ModelConfig.from_dict(onnx_files.model_config()),
    )
    for n_units in (1, 20):
        a = onnx_engine.predict(_state(n_units), QUESTIONS).answers
        b = torch_engine.predict(_state(n_units), QUESTIONS).answers
        assert a["addressee"]["choice"] == b["addressee"]["choice"]
        for qid in QUESTIONS:
            for key in ("probabilities",):
                if key in a[qid]:
                    for label, p in a[qid][key].items():
                        assert p == pytest.approx(b[qid][key][label], abs=2e-3), (qid, label)
        assert a["needs_reply"]["noul"] == pytest.approx(b["needs_reply"]["noul"], abs=2e-3)


def test_example_request_is_valid(torch_engine):
    from pathlib import Path

    example = json.loads(
        (Path(__file__).parents[1] / "examples" / "xiyouji-addressee.json").read_text(
            encoding="utf-8"
        )
    )
    result = torch_engine.predict(example["state"], example["questions"])
    assert set(result.answers) == {"addressee", "needs_reply"}
