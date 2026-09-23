"""Sequence building and decoding without a model: a character-level fake tokenizer."""

import numpy as np
import pytest

from eidolon_models_laya.sequence import (
    ModelConfig,
    build_sequence,
    check_question,
    clamp_temperature,
    collate,
    decode_answers,
    render_options,
    to_internal,
)


class CharTokenizer:
    mask_token = "<mask>"
    mask_token_id, cls_token_id, sep_token_id, pad_token_id = 4, 1, 2, 0

    def encode(self, text):
        return [100 + ord(c) % 1000 for c in text]


TOK = CharTokenizer()
CFG = ModelConfig(
    max_len=64, head_max_len=32, temperature=(1.0, 1.0, 1.0), temperature_by_options={}
)


def test_render_options():
    assert render_options(to_internal({"type": "noul", "instructions": "x"})) == [
        "false: no, the statement does not hold",
        "true: yes, the statement holds",
    ]
    assert render_options(
        to_internal({"type": "choice", "instructions": "x", "criteria": ["a", "b"]})
    ) == ["a", "b"]
    assert render_options(
        to_internal({"type": "choice", "instructions": "x", "criteria": {"a": "desc", "b": None}})
    ) == ["a: desc", "b"]
    assert render_options(
        to_internal({"type": "score", "instructions": "x", "criteria": ["low", "high"]})
    ) == ["level 0: low", "level 1: high"]


@pytest.mark.parametrize(
    "qdef,match",
    [
        ("nope", "must be a dict"),
        ({"type": "rank", "instructions": "x"}, "unknown type"),
        ({"type": "noul"}, "no 'instructions'"),
        ({"type": "choice", "instructions": "x", "criteria": {}}, "at least one criterion"),
        ({"type": "score", "instructions": "x", "criteria": {"a": 1}}, "list of level"),
    ],
)
def test_check_question_names_the_problem(qdef, match):
    with pytest.raises(ValueError, match=match):
        check_question("q1", qdef)


def test_markers_point_at_mask_tokens_and_state_follows():
    q = to_internal({"type": "choice", "instructions": "who", "criteria": ["a", "b", "c"]})
    ids, markers, truncated = build_sequence(TOK, "hello", q, 64, 32)
    assert len(markers) == 3 and all(ids[m] == TOK.mask_token_id for m in markers)
    assert ids[0] == TOK.cls_token_id and ids[-1] == TOK.sep_token_id
    assert not truncated


def test_truncation_is_reported_and_left_keeps_the_end():
    q = to_internal({"type": "noul", "instructions": "x"})
    state = "A" * 100 + "Z"
    ids_head, _, cut_head = build_sequence(TOK, state, q, 64, 32)
    ids_tail, _, cut_tail = build_sequence(TOK, state, q, 64, 32, truncate_left=True)
    assert cut_head and cut_tail and len(ids_head) == len(ids_tail) == 64
    z = TOK.encode("Z")[0]
    assert ids_head[-2] != z and ids_tail[-2] == z  # the last state token before [SEP]


def test_collate_pads_and_masks():
    b = collate(
        [
            {"ids": [1, 5, 2], "markers": [1], "qtype": 2},
            {"ids": [1, 5, 6, 7, 2], "markers": [1, 2], "qtype": 0},
        ],
        pad_id=0,
    )
    assert b["input_ids"].shape == (2, 5) and b["input_ids"].dtype == np.int64
    assert b["attention_mask"].sum() == 8
    assert b["marker_mask"].tolist() == [[True, False], [True, True]]
    assert b["qtype"].tolist() == [2, 0]


def test_decode_answers_shapes():
    internals = [
        to_internal({"type": "choice", "instructions": "x", "criteria": ["a", "b", "c"]}),
        to_internal({"type": "noul", "instructions": "y"}),
        to_internal({"type": "score", "instructions": "z", "criteria": ["l", "m", "h"]}),
    ]
    items = [{"markers": [1, 2, 3]}, {"markers": [1, 2]}, {"markers": [1, 2, 3]}]
    logits = np.array([[0, 5, 0], [0, 3, -1e4], [0, 0, 9]], dtype=np.float32)
    answers = decode_answers(
        ["c", "n", "s"], internals, items, logits, np.zeros((3, 2), np.float32), CFG
    )
    assert answers["c"]["choice"] == "b" and set(answers["c"]["probabilities"]) == {"a", "b", "c"}
    assert answers["n"]["noul"] > 0.9 and answers["n"]["confidence"] == answers["n"]["noul"]
    assert 1.9 < answers["s"]["score"] <= 2.0
    assert answers["c"]["action"] == {"act_probability": 0.5}


def test_temperature_clamp():
    assert clamp_temperature(0.1) == 0.5 and clamp_temperature(9) == 5.0
    assert clamp_temperature("x") == 1.0 and clamp_temperature(float("nan")) == 1.0
