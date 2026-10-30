"""Remote d3 implementation of the existing participation v2 contract.

Only public decision context leaves this boundary. This adapter owns no TeamRun,
media, private memory, role reply or device authority.
"""

import json
import math
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from eidolon_sdk.biz.participation import DecisionResult, Proposal, Snapshot, validate_proposal

from . import participation_audit as audit

POLICY = Path(__file__).with_name("participation_d3_policy.txt").read_text(encoding="utf-8")
CLARIFY = (
    "根据本轮用户请求和公开上下文，询问一个阻碍继续的缺失信息；"
    "不要替用户猜测，不回答尚未明确的请求，问完等待用户。"
)


def project(r):
    if not 1 <= len(r.candidates) <= 16 or r.context.summary or r.context.pending_requirements:
        raise ValueError("unsupported_context")
    slots = {c.companion_id: f"M{i}" for i, c in enumerate(r.candidates)}
    back = {v: k for k, v in slots.items()}
    history = r.context.recent_messages
    if not history or history[-1] != r.trigger:
        raise ValueError("trigger_not_latest")
    if len({m.message_id for m in history}) != len(history):
        raise ValueError("duplicate_messages")

    def public(m):
        if m.author_kind == "user":
            author = "user"
        elif m.author_kind == "companion" and m.author_id in slots:
            author = slots[m.author_id]
        else:
            raise ValueError("unsupported_author")
        return {"author": author, "text": m.text}

    if any(m.message_id == r.user_request.message_id and m != r.user_request for m in history):
        raise ValueError("pinned_request_mismatch")
    cast = {
        slots[c.companion_id]: {"name": c.display_name, "character": c.description}
        for c in r.candidates
    }
    state = {
        "user_request": r.user_request.text,
        "team_background": r.scene_goal,
        "candidates": cast,
        "allowed_actions": list(r.constraints.allowed_actions),
        "remaining_replies": r.constraints.remaining_replies,
        "prior_public_messages": [
            public(m) for m in history[:-1] if m.message_id != r.user_request.message_id
        ],
        "trigger": {"same_as_user_request": True}
        if r.trigger.message_id == r.user_request.message_id
        else public(r.trigger),
    }
    choices = {}
    for action, verb in [("respond", "回应"), ("clarify", "提出一个澄清问题后等待用户")]:
        if action in r.constraints.allowed_actions:
            for k, c in cast.items():
                choices[f"{action}:{k}"] = f"由{c['name']}（{k}）{verb}"
    for action, desc in [
        ("wait", "暂停发言，等待用户"),
        ("finish", "本轮请求已完成或用户明确结束"),
    ]:
        if action in r.constraints.allowed_actions:
            choices[action] = desc
    choices["abstain"] = "无法可靠作出允许的决策，不产生任何回复"
    return {
        "model": "d3",
        "state": json.dumps(state, ensure_ascii=False),
        "questions": {"move": {"type": "choice", "instructions": POLICY, "criteria": choices}},
    }, back


def convert(r, body, choices, back):
    a = body["answers"]["move"]
    pred = a["choice"]
    p = a["probabilities"]
    if (
        set(p) != set(choices)
        or pred not in choices
        or not all(type(x) in (float, int) and math.isfinite(x) and 0 <= x <= 1 for x in p.values())
        or abs(sum(p.values()) - 1) > 0.01
    ):
        raise ValueError("bad_probabilities")
    if p[pred] < max(p.values()) - 1e-6:
        raise ValueError("not_argmax")
    proposal = None
    if pred != "abstain":
        action, _, slot = pred.partition(":")
        proposal = Proposal(
            action=action,
            participants=(back[slot],) if slot else (),
            instruction=CLARIFY if action == "clarify" else "",
        )
    result = DecisionResult(
        **{k: getattr(r, k) for k in Snapshot.model_fields},
        status="decided" if proposal else "abstained",
        proposal=proposal,
        policy_version="d3-team-discussion-v3",
        model_version="d3-http",
    )
    validate_proposal(r, result)
    return result, float(p[pred])


class RemoteParticipationAdapter:
    policy_version = "d3-team-discussion-v3"
    model_version = "d3-http"
    max_candidates = 16

    def __init__(self, url: str, *, transport=None):
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not parsed.path.endswith("/v1/systemone")
        ):
            raise ValueError("remote participation requires an explicit SystemOne endpoint")
        self.url = url
        self.health_url = url.removesuffix("/v1/systemone") + "/health"
        self.client = httpx.Client(
            trust_env=False, follow_redirects=False, timeout=3, transport=transport
        )

    def close(self):
        self.client.close()

    def readiness(self):
        response = self.client.get(self.health_url, timeout=2)
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "ready" or body.get("model") != "d3":
            raise ValueError("remote participation model is not ready")

    def decide_explained(self, request):
        identity = dict(
            source="gateway",
            decision_id=request.decision_id,
            session_id=request.context_ref,
            epoch=request.cancellation_epoch,
        )
        started = time.perf_counter()
        audit.record("request_received", **identity, request=request.model_dump(mode="json"))
        try:
            result, why = self._decide(request, identity)
            audit.record(
                "result_returned",
                **identity,
                result=result.model_dump(mode="json"),
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
            return result, why
        except Exception as exc:
            audit.record("request_failed", **identity, error_type=type(exc).__name__)
            raise

    def _decide(self, request, identity):
        payload, back = project(request)
        audit.record("upstream_request", **identity, payload=payload, slot_to_companion=back)
        with self.client.stream(
            "POST", self.url, json=payload, timeout=min(request.timeout_ms / 1000, 5)
        ) as response:
            audit.record("upstream_status", **identity, status_code=response.status_code)
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > 262144:
                    raise ValueError("decision_response_too_large")
        parsed = json.loads(body)
        audit.record("upstream_response", **identity, body=parsed)
        result, confidence = convert(
            request, parsed, payload["questions"]["move"]["criteria"], back
        )
        return result, {"reason": "remote_decision", "confidence": confidence}
