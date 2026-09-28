"""Regression fixtures only: none of these cases enter the training dataset."""
from dataclasses import dataclass
import pytest
from eidolon_sdk.biz.participation import Candidate, Context, DecisionRequest, Message, Constraints, validate_proposal
from eidolon_models_laya.participation import ContextTooLong, ParticipationAdapter
from eidolon_models_laya.participation_text import project_request, CompactTextPredictor, check_capacity


def request(order=("id-a", "id-b"), *, duplicate=False):
    user = Message(message_id="u", author_kind="user", author_id="owner", text="小禾，请你接着说。")
    peer = Message(message_id="p", author_kind="companion", author_id="id-a", text="我说到一半。")
    return DecisionRequest(decision_id="d", context_ref="c", context_version=2,
                           membership_revision=4, cancellation_epoch=5, timeout_ms=1000,
                           user_request=user, trigger=peer,
                           context=Context(recent_messages=(user, peer)),
                           candidates=tuple(Candidate(companion_id=i, display_name="小禾" if duplicate or i=="id-a" else "青芽",
                                                      description="喜欢讲故事" if i=="id-a" else "话少，爱听故事") for i in order))


def test_projection_identity_permutation_and_ambiguous_display_names():
    s, q, ids = project_request(request(("id-b", "id-a"), duplicate=True))
    assert ids == {"M0": "id-b", "M1": "id-a"}
    assert s["trigger"]["author"] == "M1"
    assert s["candidates"]["M1"]["name"] == s["candidates"]["M0"]["name"]
    assert s["prior_public_messages"] == []
    assert "喜欢讲故事" not in str(q)
    assert not any(k.startswith("clarify") for k in q["criteria"])
    assert s["user_request"] == "小禾，请你接着说。"


def test_allowed_actions_and_capacity_fail_closed():
    req = request().model_copy(update={"constraints": Constraints(allowed_actions=("wait", "finish"))})
    s, q, ids = project_request(req)
    assert set(q["criteria"]) == {"wait", "finish", "abstain"}
    bad = request().model_copy(update={"context": Context(summary="unsupported summary")})
    with pytest.raises(ContextTooLong):
        project_request(bad)


class TinyTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 3
    cls_token_id = 1
    sep_token_id = 2
    def encode(self, value):
        return list(range(len(value)))


def test_no_head_or_context_truncation():
    state, question, _ = project_request(request())
    tok = TinyTokenizer()
    assert check_capacity(tok, state, question, 2048, 512) > 0
    with pytest.raises(ContextTooLong):
        check_capacity(tok, state, question, 100, 512)
    with pytest.raises(ContextTooLong):
        check_capacity(tok, state, question, 2048, 32)
    state["user_request"] = "[MASK]"
    with pytest.raises(ContextTooLong):
        check_capacity(tok, state, question, 2048, 512)


def test_sdk_result_maps_real_id_and_preserves_snapshot():
    class Engine:
        tokenizer = TinyTokenizer()
        max_len, head_max_len = 2048, 512
        def predict(self, state, questions, truncate_left):
            assert not truncate_left
            @dataclass
            class Prediction:
                truncated: bool = False
                answers = {"move": {"choice": "respond:M1", "answer_confidence": 0.9}}
            return Prediction()
    req = request(("id-b", "id-a"))
    result = ParticipationAdapter(CompactTextPredictor(Engine()), model_version="test", policy_version="test", min_confidence=.8).decide(req)
    validate_proposal(req, result)
    assert result.proposal.participants == ("id-a",)
    assert result.cancellation_epoch == 5
    assert result.proposal.instruction == ""  # legal for respond; no fabricated instruction
