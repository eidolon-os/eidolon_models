"""HTTP surface.

    GET  /healthz          liveness, no auth
    GET  /readyz           model loaded, no auth
    GET  /participation/readyz  task-qualified readiness, no auth
    GET  /v1/info          model, backend, limits                       (auth)
    GET  /v1/participation/readyz  IP-team v2 adapter ready, if configured (auth)
    POST /v1/participation/decide  SDK participation v2, if configured (auth)
    POST /v1/systemone     {"state": ..., "questions": {...}, "options": {"truncate_left": bool, "ask_if": {...}}}
                           ask_if: {"device": {"intent": ["控制", "查询"]}} asks device only when intent is one of those
                           Jev / laya wire format; extra fields in the reply only  (auth)

Inference runs on one worker thread: a forward pass already uses every core it
was given, so two at once only interleave. Callers beyond ``max_pending`` get
503 immediately instead of joining an unbounded queue.
"""

from __future__ import annotations

import asyncio
import functools
import hmac
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from aiohttp import web
from eidolon_sdk.biz.participation import DecisionRequest
from pydantic import ValidationError

from . import __version__
from .config import Settings
from .engine import DecisionEngine
from .participation import ParticipationAdapter

log = logging.getLogger("eidolon_laya")

ENGINE = web.AppKey("engine", DecisionEngine)
SETTINGS = web.AppKey("settings", Settings)
INFO = web.AppKey("info", dict)
STATE = web.AppKey("state", dict)
PARTICIPATION = web.AppKey("participation", ParticipationAdapter | None)

# Labels and messages are mostly Chinese: send them as UTF-8, not \uXXXX escapes.
_json = functools.partial(
    web.json_response, dumps=functools.partial(json.dumps, ensure_ascii=False)
)


def _error(status: int, code: str, message: str, **headers: str) -> web.Response:
    return _json(
        {"error": {"code": code, "message": message}}, status=status, headers=headers or None
    )


@web.middleware
async def _auth(request: web.Request, handler):
    key = request.app[SETTINGS].api_key
    if key and request.path.startswith("/v1/"):
        header = request.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip(), key):
            return _error(
                401,
                "unauthorized",
                "missing or wrong bearer token",
                **{"WWW-Authenticate": "Bearer"},
            )
    return await handler(request)


async def healthz(_: web.Request) -> web.Response:
    return _json({"status": "ok"})


async def readyz(request: web.Request) -> web.Response:
    return _json({"status": "ready", "backend": request.app[ENGINE].backend.name})


async def info(request: web.Request) -> web.Response:
    settings, state = request.app[SETTINGS], request.app[STATE]
    return _json(
        {
            **request.app[INFO],
            "service": {
                "version": __version__,
                "uptime_s": round(time.time() - state["started"]),
                "served": state["served"],
            },
            "engine": request.app[ENGINE].describe(),
            "limits": {
                "max_pending": settings.max_pending,
                "max_questions": settings.max_questions,
                "max_body_bytes": settings.max_body_bytes,
            },
        }
    )


async def participation_readyz(request: web.Request) -> web.Response:
    adapter = request.app[PARTICIPATION]
    if adapter is None:
        return _error(503, "not_configured", "IP-team decision model is not configured")
    return _json({
        "status": "ready",
        "task": "ip_team.participation",
        "schema_version": 2,
        "policy_version": adapter.policy_version,
        "model_version": adapter.model_version,
    })


async def participation_decide(request: web.Request) -> web.Response:
    adapter = request.app[PARTICIPATION]
    if adapter is None:
        return _error(503, "not_configured", "IP-team decision model is not configured")
    try:
        payload = await request.json()
        decision = DecisionRequest.model_validate(payload)
    except web.HTTPRequestEntityTooLarge:
        raise
    except (ValueError, ValidationError):
        return _error(400, "invalid_request", "expected a valid participation v2 request")

    state, settings = request.app[STATE], request.app[SETTINGS]
    if state["pending"] >= settings.max_pending:
        return _error(503, "busy", "decision service is busy", **{"Retry-After": "1"})
    state["pending"] += 1
    loop = asyncio.get_running_loop()
    started = time.perf_counter()
    future = loop.run_in_executor(state["executor"], adapter.decide_explained, decision)
    # A request deadline cannot stop a running inference thread. Count it as
    # pending until the worker actually exits, so timed-out callers cannot
    # enqueue work without bound.
    def completed(work: asyncio.Future) -> None:
        state["pending"] -= 1
        if not work.cancelled():
            work.exception()  # Observe a failure after the HTTP deadline.

    future.add_done_callback(completed)
    try:
        response = await asyncio.wait_for(
            asyncio.shield(future),
            timeout=decision.timeout_ms / 1000,
        )
    except TimeoutError:
        return _error(504, "deadline", "decision deadline exceeded")
    except Exception:
        log.exception("participation prediction failed")
        return _error(500, "internal", "prediction failed; see server log")
    state["served"] += 1
    response, why = response
    # Correlation id, versions, action, reason and time only — never the public dialogue.
    log.info(
        "participation decision=%s status=%s action=%s reason=%s confidence=%s detail=%s ms=%.0f policy=%s model=%s",
        decision.decision_id,
        response.status,
        response.proposal.action if response.proposal else "-",
        why.get("reason", "-"),
        why.get("confidence", "-"),
        why.get("detail", "-") or "-",
        (time.perf_counter() - started) * 1000,
        adapter.policy_version,
        adapter.model_version,
    )
    return _json(response.model_dump(mode="json"))


async def systemone(request: web.Request) -> web.Response:
    settings, state, engine = request.app[SETTINGS], request.app[STATE], request.app[ENGINE]
    try:
        body = await request.json()
    except web.HTTPRequestEntityTooLarge:
        raise
    except Exception:
        return _error(400, "bad_json", "request body must be a JSON object")
    if not isinstance(body, dict) or "state" not in body or "questions" not in body:
        return _error(400, "bad_request", "body needs 'state' and 'questions'")
    questions = body["questions"]
    if not isinstance(questions, dict):
        return _error(400, "bad_request", "'questions' must be an object of id -> definition")
    if len(questions) > settings.max_questions:
        return _error(
            400,
            "too_many_questions",
            f"{len(questions)} questions > limit {settings.max_questions}",
        )
    options = body.get("options") or {}
    if not isinstance(options, dict):
        return _error(400, "bad_request", "'options' must be an object")
    truncate_left = bool(options.get("truncate_left", False))
    ask_if = options.get("ask_if")

    if state["pending"] >= settings.max_pending:
        return _error(
            503, "busy", f"{state['pending']} requests already pending", **{"Retry-After": "1"}
        )
    state["pending"] += 1
    try:
        loop = asyncio.get_running_loop()
        prediction = await loop.run_in_executor(
            state["executor"],
            lambda: engine.predict(
                body["state"], questions, truncate_left=truncate_left, ask_if=ask_if
            ),
        )
    except ValueError as exc:
        return _error(400, "invalid_question", str(exc))
    except Exception:
        log.exception("prediction failed")
        return _error(500, "internal", "prediction failed; see server log")
    finally:
        state["pending"] -= 1
    state["served"] += 1
    log.info(
        "systemone q=%d skipped=%d tokens=%d forward=%.0fms total=%.0fms truncated=%s",
        len(questions),
        len(prediction.skipped),
        prediction.input_tokens,
        prediction.forward_ms,
        prediction.total_ms,
        prediction.truncated or "-",
    )
    return _json({
        "contract_version": "eidolon.models.laya.systemone.v1",
        "features": {"question_state": True},
        "revision": request.app[INFO].get("revision"),
        **prediction.as_response(engine.name, engine.backend.name),
    })


def create_app(
    engine: DecisionEngine,
    settings: Settings,
    model_info: dict,
    *,
    participation: ParticipationAdapter | None = None,
) -> web.Application:
    app = web.Application(client_max_size=settings.max_body_bytes, middlewares=[_auth])
    app[ENGINE] = engine
    app[SETTINGS] = settings
    app[INFO] = model_info
    app[PARTICIPATION] = participation
    app[STATE] = {
        "pending": 0,
        "served": 0,
        "started": time.time(),
        "executor": ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya"),
    }

    async def _shutdown(app: web.Application) -> None:
        app[STATE]["executor"].shutdown(wait=False, cancel_futures=True)

    app.on_cleanup.append(_shutdown)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/readyz", readyz)
    # The Host's release gate probes this loopback-only surface without a
    # credential. It reports task readiness, never an inference or secret.
    app.router.add_get("/participation/readyz", participation_readyz)
    app.router.add_get("/v1/info", info)
    app.router.add_get("/v1/participation/readyz", participation_readyz)
    app.router.add_post("/v1/participation/decide", participation_decide)
    app.router.add_post("/v1/systemone", systemone)
    return app
