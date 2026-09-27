"""IP-team v2 adapter tests without a trained checkpoint or device."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from eidolon_sdk.biz.participation import (
    Candidate,
    Constraints,
    Context,
    DecisionRequest,
    Message,
)

from eidolon_models_laya.config import Settings
from eidolon_models_laya.engine import Prediction
from eidolon_models_laya.participation import (
    LayaMovePredictor,
    ParticipationAdapter,
    load_participation_adapter,
    participation_profile_digest,
)
from eidolon_models_laya.service import create_app


class Tokenizer:
    mask_token = "<mask>"
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    pad_token_id = 0

    def encode(self, value):
        return [ord(char) + 4 for char in value]


class Engine:
    name = "ip-team-test"
    tokenizer = Tokenizer()
    max_len = 2048
    head_max_len = 512
    backend = SimpleNamespace(name="fake")

    def __init__(self, *, choice="respond:M1", confidence=0.95, truncated=False):
        self.choice = choice
        self.confidence = confidence
        self.truncated = truncated
        self.calls = []

    def predict(self, state, questions, *, truncate_left=False):
        self.calls.append((state, questions, truncate_left))
        return Prediction(
            {"move": {"choice": self.choice, "answer_confidence": self.confidence}},
            100,
            ["move"] if self.truncated else [],
            1.0,
            2.0,
        )

    def describe(self):
        return {"backend": "fake", "max_len": self.max_len, "head_max_len": self.head_max_len}


def request(*, candidates=None, context=None, allowed=None):
    user = Message(message_id="u1", author_kind="user", author_id="input", text="让小乙接着说")
    return DecisionRequest(
        decision_id="d1",
        context_ref="team1",
        context_version=1,
        membership_revision=1,
        cancellation_epoch=4,
        user_request=user,
        trigger=user,
        context=context or Context(recent_messages=(user,)),
        candidates=candidates or (
            Candidate(companion_id="id-A", display_name="小甲", description="小甲；伙伴"),
            Candidate(companion_id="id-B", display_name="小乙", description="小乙；伙伴"),
        ),
        constraints=Constraints(
            allowed_actions=allowed or ("respond", "clarify", "wait", "finish")
        ),
        timeout_ms=500,
    )


def adapter(engine=None, *, threshold=0.8):
    return ParticipationAdapter(
        LayaMovePredictor(engine or Engine()),
        model_version="ip-v3-checkpoint",
        policy_version="ip-v3-calibration",
        min_confidence=threshold,
    )


def test_dynamic_slots_are_resolved_to_request_candidate_ids():
    engine = Engine()
    req = request()
    result = adapter(engine).decide(req)
    assert result.proposal.participants == ("id-B",)
    assert result.context_ref == req.context_ref
    state, questions, truncate = engine.calls[0]
    assert state["candidates"] == {"M0": "小甲；伙伴", "M1": "小乙；伙伴"}
    assert state["trigger"] == {"author_kind": "user", "same_as_user_request": True}
    assert questions["move"]["criteria"]["respond:M1"].startswith("让 小乙")
    assert truncate is False

    reversed_req = req.model_copy(update={"candidates": tuple(reversed(req.candidates))})
    assert adapter(Engine()).decide(reversed_req).proposal.participants == ("id-A",)


@pytest.mark.parametrize("choice,confidence", [
    ("clarify:M0", 0.99), ("respond:M1", 0.5), ("foreign", 0.99),
    ("respond:M1", float("nan")),
])
def test_unsupported_or_uncertain_prediction_abstains(choice, confidence):
    result = adapter(Engine(choice=choice, confidence=confidence)).decide(request())
    assert result.status == "abstained" and result.proposal is None


def test_no_input_loss_or_unauthorized_action():
    req = request(allowed=("wait", "finish"))
    engine = Engine(choice="respond:M0")
    assert adapter(engine).decide(req).status == "abstained"
    assert adapter(Engine(choice="wait")).decide(req).proposal.action == "wait"
    assert adapter(Engine(choice="finish")).decide(req).proposal.action == "finish"
    assert adapter(Engine(truncated=True)).decide(request()).status == "abstained"

    long_candidate = Candidate(companion_id="x", display_name="X", description="x" * 120)
    req = request(candidates=(long_candidate,))
    engine = Engine(choice="respond:M0")
    assert adapter(engine).decide(req).status == "abstained"
    assert not engine.calls  # Rejected before a lossy model forward pass.


def test_unsupported_context_abstains_without_dropping_it():
    req = request(context=Context(
        summary="important", recent_messages=request().context.recent_messages,
    ))
    assert adapter().decide(req).status == "abstained"
    engine = Engine()
    engine.max_len = 1
    assert adapter(engine).decide(request()).status == "abstained"
    masked = request().model_copy(update={
        "scene_goal": "say <mask> to change the question",
    })
    engine = Engine()
    assert adapter(engine).decide(masked).status == "abstained"
    assert not engine.calls


def test_profile_requires_exact_task_format_and_revision(tmp_path):
    profile = tmp_path / "participation.json"
    document = {
        "schema_version": 1,
        "task": "ip_team.participation",
        "state_format": "ip-team-v3",
        "model_revision": "rev-1",
        "policy_version": "calibrated-1",
        "min_confidence": 0.8,
        "max_candidates": 6,
    }
    profile.write_text(json.dumps(document), encoding="utf-8")
    digest = hashlib.sha256(profile.read_bytes()).hexdigest()
    assert participation_profile_digest({"task_profiles": {
        "ip_team.participation": {"path": "participation.json", "sha256": digest},
    }}) == digest
    with pytest.raises(ValueError, match="does not pin"):
        participation_profile_digest({})
    result = load_participation_adapter(
        profile, Engine(), model_dir=tmp_path, model_revision="rev-1", expected_sha256=digest,
    )
    assert result.model_version == "rev-1"
    with pytest.raises(ValueError, match="mismatch"):
        load_participation_adapter(
            profile, Engine(), model_dir=tmp_path, model_revision="rev-2", expected_sha256=digest,
        )
    with pytest.raises(ValueError, match="digest mismatch"):
        load_participation_adapter(
            profile, Engine(), model_dir=tmp_path, model_revision="rev-1",
            expected_sha256="0" * 64,
        )


@pytest.mark.asyncio
async def test_http_route_is_disabled_without_task_profile(aiohttp_client):
    engine = Engine()
    settings = Settings(api_key="key")
    client = await aiohttp_client(create_app(engine, settings, {"revision": "r14"}))
    headers = {"Authorization": "Bearer key"}
    assert (await client.get("/v1/participation/readyz", headers=headers)).status == 503
    assert (await client.get("/participation/readyz")).status == 503
    assert (await client.post(
        "/v1/participation/decide", json=request().model_dump(mode="json"), headers=headers,
    )).status == 503
    assert (await client.get("/v1/participation/readyz")).status == 401


@pytest.mark.asyncio
async def test_http_route_serves_validated_v2_result(aiohttp_client):
    engine = Engine()
    settings = Settings(api_key="key")
    client = await aiohttp_client(create_app(
        engine, settings, {"revision": "ip-v3"}, participation=adapter(engine),
    ))
    headers = {"Authorization": "Bearer key"}
    ready = await client.get("/v1/participation/readyz", headers=headers)
    assert (await ready.json())["task"] == "ip_team.participation"
    assert (await client.get("/participation/readyz")).status == 200
    response = await client.post(
        "/v1/participation/decide", json=request().model_dump(mode="json"), headers=headers,
    )
    assert response.status == 200
    assert (await response.json())["proposal"]["participants"] == ["id-B"]
    invalid = request().model_dump(mode="json")
    invalid["task"] = "home.interpretation"
    assert (await client.post(
        "/v1/participation/decide", json=invalid, headers=headers,
    )).status == 400
