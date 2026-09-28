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


# participation-laya-v1: action / speaker / clarify_about, as the participation release was trained

QUESTIONS = {
    "action": {"type": "choice", "instructions": "下一步应该怎样？", "criteria": {
        "respond": "要人说", "clarify": "先问清", "wait": "先等", "finish": "结束"}},
    "speaker": {"type": "choice", "instructions": "由谁来说？"},
    "clarify_about": {"type": "choice", "instructions": "问清什么？", "criteria": {
        "指代不明": "哪一位", "要求不明": "做什么", "对象不在场": "不在"}},
}
CLARIFY = {"指代不明": "请用户说清指谁。", "要求不明": "请用户说清要什么。", "对象不在场": "告诉用户这位不在。"}


class V1Engine(Engine):
    def __init__(self, answers, *, truncated=False):
        super().__init__()
        self.answers = answers
        self.truncated = truncated

    def predict(self, state, questions, *, truncate_left=False, ask_if=None):
        self.calls.append((state, questions, ask_if))
        asked = {"action"} | {q for q, deps in (ask_if or {}).items()
                              if self.answers["action"]["choice"] in deps["action"]}
        return Prediction({q: a for q, a in self.answers.items() if q in asked}, 100,
                          ["action"] if self.truncated else [], 1.0, 2.0)


def v1_adapter(engine, *, threshold=0.8):
    from eidolon_models_laya.participation import LayaParticipationPredictor
    return ParticipationAdapter(
        LayaParticipationPredictor(engine, questions=QUESTIONS, clarify_instructions=CLARIFY),
        model_version="p2", policy_version="participation-laya-v1/p2", min_confidence=threshold,
    )


def team_request(*, allowed=None):
    user = Message(message_id="u1", author_kind="user", author_id="input", text="你俩都说说")
    said = Message(message_id="c1", author_kind="companion", author_id="id-A", text="我先说一个。")
    return request(allowed=allowed, context=Context(recent_messages=(user, said))).model_copy(
        update={"user_request": user, "trigger": said})


def answers(action, conf=0.95, speaker=None, about=None):
    out = {"action": {"choice": action, "answer_confidence": conf}}
    if speaker:
        out["speaker"] = {"choice": speaker, "answer_confidence": 0.6}
    if about:
        out["clarify_about"] = {"choice": about, "answer_confidence": 0.9}
    return out


def test_v1_state_matches_training_snapshot_format():
    engine = V1Engine(answers("respond", speaker="M1"))
    result = v1_adapter(engine).decide(team_request())
    assert result.proposal.action == "respond" and result.proposal.participants == ("id-B",)
    state, questions, ask_if = engine.calls[0]
    assert state == {
        "场景目标": "",
        "候选": [{"编号": "M0", "名字": "小甲", "角色": "小甲；伙伴"},
                 {"编号": "M1", "名字": "小乙", "角色": "小乙；伙伴"}],
        "本轮用户请求": "你俩都说说",
        "公开记录": [{"说话": "用户", "内容": "你俩都说说"}, {"说话": "M0 小甲", "内容": "我先说一个。"}],
        "最新一条": "M0 小甲：我先说一个。",
    }
    assert questions["speaker"]["criteria"] == {"M0": "小甲：小甲；伙伴", "M1": "小乙：小乙；伙伴"}
    assert ask_if == {"speaker": {"action": ["respond", "clarify"]}, "clarify_about": {"action": ["clarify"]}}


def test_v1_clarify_carries_the_reason_task_and_silent_actions_have_none():
    result = v1_adapter(V1Engine(answers("clarify", speaker="M0", about="指代不明"))).decide(team_request())
    assert result.proposal.action == "clarify"
    assert result.proposal.participants == ("id-A",)
    assert result.proposal.instruction == CLARIFY["指代不明"]
    for silent in ("wait", "finish"):
        proposal = v1_adapter(V1Engine(answers(silent))).decide(team_request()).proposal
        assert proposal.action == silent and proposal.participants == () and proposal.instruction == ""


@pytest.mark.parametrize("model_answers,allowed", [
    (answers("respond", conf=0.5, speaker="M1"), None),        # below the calibrated threshold
    (answers("respond", speaker="M7"), None),                  # speaker outside the candidates
    (answers("respond"), None),                                # no speaker answer
    (answers("clarify", speaker="M0", about="别的"), None),    # reason without a task
    (answers("respond", speaker="M1"), ("wait", "finish")),    # action not allowed now
])
def test_v1_uncertain_or_illegal_choices_abstain(model_answers, allowed):
    result = v1_adapter(V1Engine(model_answers)).decide(team_request(allowed=allowed))
    assert result.status == "abstained" and result.proposal is None


def test_v1_rejects_unsupported_context_before_the_model():
    engine = V1Engine(answers("respond", speaker="M1"))
    no_name = team_request().model_copy(update={"candidates": (
        Candidate(companion_id="id-A", display_name="", description="伙伴"),
        Candidate(companion_id="id-B", display_name="小乙", description="伙伴"),
    )})
    assert v1_adapter(engine).decide(no_name).status == "abstained"
    stranger = Message(message_id="s1", author_kind="system", author_id="sys", text="系统提示")
    odd = team_request().model_copy(update={
        "context": Context(recent_messages=(stranger,)), "trigger": stranger})
    assert v1_adapter(engine).decide(odd).status == "abstained"
    assert not engine.calls
    assert v1_adapter(V1Engine(answers("finish"), truncated=True)).decide(team_request()).status == "abstained"


def test_v1_keeps_the_last_sixteen_public_messages():
    msgs = tuple(Message(message_id=f"m{i}", author_kind="user", author_id="input", text=f"第{i}句")
                 for i in range(20))
    req = request(context=Context(recent_messages=msgs)).model_copy(update={"trigger": msgs[-1]})
    engine = V1Engine(answers("finish"))
    v1_adapter(engine).decide(req)
    public = engine.calls[0][0]["公开记录"]
    assert len(public) == 16 and public[0]["内容"] == "第4句" and public[-1]["内容"] == "第19句"


def test_v2_profile_loads_the_three_question_predictor(tmp_path):
    from eidolon_models_laya.participation import LayaParticipationPredictor
    document = {
        "schema_version": 2, "task": "ip_team.participation", "state_format": "participation-laya-v1",
        "model_revision": "rev-2", "policy_version": "participation-laya-v1/rev-2/min0.9",
        "min_confidence": 0.9, "max_candidates": 5, "questions": QUESTIONS, "clarify_instructions": CLARIFY,
    }
    profile = tmp_path / "participation.json"
    profile.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    digest = hashlib.sha256(profile.read_bytes()).hexdigest()
    loaded = load_participation_adapter(
        profile, Engine(), model_dir=tmp_path, model_revision="rev-2", expected_sha256=digest,
    )
    assert isinstance(loaded.predictor, LayaParticipationPredictor)
    assert loaded.min_confidence == 0.9
    for bad in ({**document, "state_format": "ip-team-v3"}, {**document, "schema_version": 1},
                {**document, "questions": {**QUESTIONS, "action": {**QUESTIONS["action"], "criteria": {"respond": "x"}}}}):
        profile.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
        digest = hashlib.sha256(profile.read_bytes()).hexdigest()
        with pytest.raises(ValueError):
            load_participation_adapter(
                profile, Engine(), model_dir=tmp_path, model_revision="rev-2", expected_sha256=digest,
            )
