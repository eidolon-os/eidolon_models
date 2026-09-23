"""HTTP surface against a fake engine: auth, validation, back-pressure. No model."""

import pytest

from eidolon_models_laya.config import Settings
from eidolon_models_laya.engine import Prediction
from eidolon_models_laya.service import STATE, create_app

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
QUESTIONS = {"who": {"type": "choice", "instructions": "who?", "criteria": ["a", "b"]}}


class FakeBackend:
    name = "fake"

    def describe(self):
        return {"backend": "fake"}


class FakeEngine:
    name = "laya-test"
    backend = FakeBackend()

    def __init__(self):
        self.calls = []

    def describe(self):
        return {"backend": "fake", "max_len": 1024, "head_max_len": 256}

    def predict(self, state, questions, *, truncate_left=False):
        self.calls.append((state, questions, truncate_left))
        if "bad" in questions:
            raise ValueError("question 'bad': unknown type")
        answers = {
            qid: {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 1.0},
                "confidence": 1.0,
                "action": {"act_probability": 0.5},
            }
            for qid in questions
        }
        return Prediction(answers, 42, ["who"] if truncate_left else [], 1.0, 2.0)


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
async def client(aiohttp_client, engine):
    settings = Settings(api_key=KEY, max_pending=2, max_questions=3, max_body_bytes=4096)
    app = create_app(engine, settings, {"model": "laya-test", "revision": "abc"})
    return await aiohttp_client(app)


async def test_health_and_ready_need_no_auth(client):
    assert (await client.get("/healthz")).status == 200
    body = await (await client.get("/readyz")).json()
    assert body == {"status": "ready", "backend": "fake"}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": KEY}])
async def test_v1_requires_the_bearer_key(client, headers):
    resp = await client.post(
        "/v1/systemone", json={"state": "x", "questions": QUESTIONS}, headers=headers
    )
    assert resp.status == 401
    assert resp.headers["WWW-Authenticate"] == "Bearer"


async def test_info(client):
    body = await (await client.get("/v1/info", headers=AUTH)).json()
    assert body["model"] == "laya-test" and body["engine"]["backend"] == "fake"
    assert body["limits"]["max_questions"] == 3


async def test_systemone_round_trip(client, engine):
    resp = await client.post(
        "/v1/systemone",
        headers=AUTH,
        json={
            "state": {"utterance": "悟空"},
            "questions": QUESTIONS,
            "options": {"truncate_left": True},
            "model": "jev-1",
        },
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["answers"]["who"]["choice"] == "a"
    assert body["usage"] == {"input_tokens": 42, "output_tokens": 0}
    assert body["truncated"] == ["who"] and body["backend"] == "fake"
    assert engine.calls == [({"utterance": "悟空"}, QUESTIONS, True)]


@pytest.mark.parametrize(
    "payload,code",
    [
        ({"questions": QUESTIONS}, "bad_request"),
        ({"state": "x", "questions": ["not", "a", "dict"]}, "bad_request"),
        (
            {"state": "x", "questions": {f"q{i}": QUESTIONS["who"] for i in range(4)}},
            "too_many_questions",
        ),
        ({"state": "x", "questions": {"bad": {}}}, "invalid_question"),
        ({"state": "x", "questions": QUESTIONS, "options": "fast"}, "bad_request"),
    ],
)
async def test_validation(client, payload, code):
    resp = await client.post("/v1/systemone", headers=AUTH, json=payload)
    assert resp.status == 400
    assert (await resp.json())["error"]["code"] == code


async def test_chinese_is_sent_as_utf8_not_escapes(client, engine):
    engine.predict = lambda state, questions, truncate_left=False: Prediction(
        {
            "who": {
                "type": "choice",
                "choice": "猪八戒",
                "probabilities": {"猪八戒": 1.0},
                "confidence": 1.0,
                "action": {"act_probability": 0.5},
            }
        },
        1,
        [],
        1.0,
        1.0,
    )
    resp = await client.post(
        "/v1/systemone", headers=AUTH, json={"state": "x", "questions": QUESTIONS}
    )
    raw = await resp.read()
    assert "猪八戒".encode() in raw and b"\\u" not in raw
    assert resp.headers["Content-Type"] == "application/json; charset=utf-8"


async def test_bad_json(client):
    resp = await client.post("/v1/systemone", headers=AUTH, data=b"{not json")
    assert resp.status == 400 and (await resp.json())["error"]["code"] == "bad_json"


async def test_body_limit(client):
    resp = await client.post(
        "/v1/systemone", headers=AUTH, json={"state": "x" * 8000, "questions": QUESTIONS}
    )
    assert resp.status == 413


async def test_busy_is_rejected_not_queued(client, engine):
    client.server.app[STATE]["pending"] = 2
    resp = await client.post(
        "/v1/systemone", headers=AUTH, json={"state": "x", "questions": QUESTIONS}
    )
    assert resp.status == 503 and resp.headers["Retry-After"] == "1"
    assert engine.calls == []
