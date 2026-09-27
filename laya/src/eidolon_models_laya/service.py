"""HTTP surface.

    GET  /healthz          liveness, no auth
    GET  /readyz           model loaded, no auth
    GET  /v1/info          model, backend, limits                       (auth)
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

from . import __version__
from .config import Settings
from .engine import DecisionEngine

log = logging.getLogger("eidolon_laya")

ENGINE = web.AppKey("engine", DecisionEngine)
SETTINGS = web.AppKey("settings", Settings)
INFO = web.AppKey("info", dict)
STATE = web.AppKey("state", dict)

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
        "revision": request.app[INFO].get("revision"),
        **prediction.as_response(engine.name, engine.backend.name),
    })


def create_app(engine: DecisionEngine, settings: Settings, model_info: dict) -> web.Application:
    app = web.Application(client_max_size=settings.max_body_bytes, middlewares=[_auth])
    app[ENGINE] = engine
    app[SETTINGS] = settings
    app[INFO] = model_info
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
    app.router.add_get("/v1/info", info)
    app.router.add_post("/v1/systemone", systemone)
    return app
