import json

import httpx
import pytest
from eidolon_sdk.biz.participation import DecisionRequest

from eidolon_models_laya.config import Settings
from eidolon_models_laya.remote_participation import RemoteParticipationAdapter, convert, project
from eidolon_models_laya.service import create_app


def request(count=2):
    user = dict(message_id="u", author_kind="user", author_id="owner", text="请你们分别介绍自己")
    return DecisionRequest.model_validate(
        dict(
            decision_id="decision",
            context_ref="team-run",
            context_version=1,
            membership_revision=1,
            cancellation_epoch=3,
            user_request=user,
            trigger=user,
            context=dict(recent_messages=[user]),
            timeout_ms=1500,
            candidates=[
                dict(companion_id=f"real-{i}", display_name=f"角色{i}", description="伙伴")
                for i in range(count)
            ],
        )
    )


def answer(payload, choice="respond:M0"):
    return {
        "answers": {
            "move": {
                "choice": choice,
                "probabilities": {
                    key: float(key == choice) for key in payload["questions"]["move"]["criteria"]
                },
            }
        }
    }


def test_all_16_members_remain_addressable_with_snapshot_and_budget():
    req = request(16)
    payload, mapping = project(req)
    assert len(mapping) == 16
    result, _ = convert(
        req, answer(payload, "respond:M15"), payload["questions"]["move"]["criteria"], mapping
    )
    assert result.proposal.participants == ("real-15",)
    assert result.cancellation_epoch == 3 and result.context_ref == "team-run"
    assert json.loads(payload["state"])["remaining_replies"] == 8
    assert "real-15" not in json.dumps(payload)


@pytest.mark.parametrize(
    "updates",
    [
        {"context": {"summary": "must not silently drop", "recent_messages": []}},
        {"candidates": []},
    ],
)
def test_unsupported_inputs_refused_without_calling_upstream(updates):
    raw = request().model_dump()
    raw.update(updates)
    adapter = RemoteParticipationAdapter(
        "https://model.example/d3/v1/systemone",
        transport=httpx.MockTransport(lambda _: pytest.fail("unexpected external request")),
    )
    try:
        with pytest.raises(ValueError):
            adapter.decide_explained(DecisionRequest.model_validate(raw))
    finally:
        adapter.close()


async def test_existing_service_readiness_checks_upstream_and_recovery(aiohttp_client):
    healthy = False
    calls = []

    def upstream(req):
        calls.append((req.method, req.url.path))
        if req.method == "GET":
            return httpx.Response(
                200, json={"status": "ready" if healthy else "loading", "model": "d3"}
            )
        return httpx.Response(200, json=answer(json.loads(req.content)))

    adapter = RemoteParticipationAdapter(
        "https://model.example/d3/v1/systemone", transport=httpx.MockTransport(upstream)
    )
    settings = Settings(service="participation", backend="remote", enable_participation=True)
    client = await aiohttp_client(create_app(None, settings, {}, participation=adapter))
    assert (await client.get("/healthz")).status == 200
    assert (await client.get("/v1/participation/readyz")).status == 503
    assert calls == [("GET", "/d3/health")]
    healthy = True
    response = await client.get("/v1/participation/readyz")
    assert response.status == 200 and (await response.json())["max_candidates"] == 16
    response = await client.post("/v1/participation/decide", json=request().model_dump(mode="json"))
    assert response.status == 200
    assert (await response.json())["proposal"]["participants"] == ["real-0"]
    assert (await client.post("/v1/systemone", json={})).status == 404


def test_remote_settings_require_explicit_endpoint_and_no_model_weights():
    settings = Settings.for_service("participation", backend="remote")
    with pytest.raises(ValueError, match="explicit upstream_url"):
        settings.require_model()
    settings.with_overrides(upstream_url="https://model.example/d3/v1/systemone").require_model()


def test_remote_audit_retains_correlation_without_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("EIDOLON_PARTICIPATION_AUDIT_DIR", str(tmp_path / "audit"))
    adapter = RemoteParticipationAdapter(
        "https://models.example/d3/v1/systemone",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=answer(json.loads(req.content)))
        ),
    )
    try:
        adapter.decide_explained(request())
    finally:
        adapter.close()
    records = [
        json.loads(line)
        for file in (tmp_path / "audit").glob("*.jsonl")
        for line in file.read_text().splitlines()
    ]
    assert [item["event"] for item in records] == [
        "request_received",
        "upstream_request",
        "upstream_status",
        "upstream_response",
        "result_returned",
    ]
    assert all(
        item["session_id"] == "team-run" and item["decision_id"] == "decision" for item in records
    )
    assert records[1]["slot_to_companion"] == {"M0": "real-0", "M1": "real-1"}


def test_original_hil_mapping_and_invalid_answer_contract():
    req = request(1)
    payload, back = project(req)
    choices = payload["questions"]["move"]["criteria"]
    for choice in choices:
        result, _ = convert(req, answer(payload, choice), choices, back)
        assert result.context_version == req.context_version
        if choice.startswith(("respond:", "clarify:")):
            assert result.proposal.participants == ("real-0",)
        if choice.startswith("clarify:"):
            assert result.proposal.instruction
        if choice == "abstain":
            assert result.proposal is None
    with pytest.raises(ValueError):
        convert(req, {"answers": {"move": {"choice": "evil", "probabilities": {}}}}, choices, back)
