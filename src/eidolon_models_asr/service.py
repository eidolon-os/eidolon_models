"""HTTP health endpoints and the streaming WebSocket service."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from aiohttp import WSMsgType, web

from . import __version__
from .backend import StreamingBackend, payload_json, result_payload
from .config import Settings, detect_host_kind
from .protocol import PROTOCOL_VERSION, ProtocolError, parse_start

logger = logging.getLogger("eidolon.asr")
BACKEND_KEY: web.AppKey[StreamingBackend] = web.AppKey("backend", object)
SETTINGS_KEY: web.AppKey[Settings] = web.AppKey("settings", Settings)


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": "eidolon-asr", "version": __version__})


async def ready(request: web.Request) -> web.Response:
    backend = request.app[BACKEND_KEY]
    return web.json_response(
        {
            "ok": True,
            "service": "eidolon-asr",
            "backend": backend.name,
            "model_id": backend.model_id,
            "offline_model_id": backend.offline_model_id,
            "punctuation_model_id": backend.punctuation_model_id,
            "host_kind": detect_host_kind(),
            "protocol_version": PROTOCOL_VERSION,
        }
    )


async def info(request: web.Request) -> web.Response:
    backend = request.app[BACKEND_KEY]
    settings = request.app[SETTINGS_KEY]
    return web.json_response(
        {
            "service": "eidolon-asr",
            "version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "audio": {"sample_rate": 16000, "channels": 1, "format": "pcm_s16le"},
            "endpoint_owner": "upstream",
            "backend": backend.name,
            "model_id": backend.model_id,
            "offline_enabled": backend.offline_model_id is not None,
            "offline_model_id": backend.offline_model_id,
            "punctuation_enabled": backend.punctuation_model_id is not None,
            "punctuation_model_id": backend.punctuation_model_id,
            "chunk_size": list(settings.chunk_size),
        }
    )


async def stream(request: web.Request) -> web.WebSocketResponse:
    backend = request.app[BACKEND_KEY]
    settings = request.app[SETTINGS_KEY]
    ws = web.WebSocketResponse(max_msg_size=settings.max_binary_message_bytes)
    await ws.prepare(request)
    await ws.send_str(
        payload_json(
            {
                "type": "connected",
                "protocol_version": PROTOCOL_VERSION,
                "backend": backend.name,
                "model_id": backend.model_id,
                "offline_model_id": backend.offline_model_id,
                "punctuation_model_id": backend.punctuation_model_id,
            }
        )
    )

    session = None
    start = None
    try:
        async for message in ws:
            if message.type == WSMsgType.BINARY:
                if session is None or start is None:
                    raise ProtocolError("start an utterance before sending audio")
                results = await asyncio.to_thread(session.feed_pcm16, message.data)
                for result in results:
                    await ws.send_str(
                        payload_json(
                            result_payload(
                                result,
                                stream_id=start.stream_id,
                                utterance_id=start.utterance_id,
                                backend=backend,
                            )
                        )
                    )
                continue
            if message.type != WSMsgType.TEXT:
                if message.type in {WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED}:
                    break
                raise ProtocolError("unsupported WebSocket message type")
            try:
                payload: dict[str, Any] = json.loads(message.data)
            except json.JSONDecodeError as exc:
                raise ProtocolError("text messages must be JSON objects") from exc
            message_type = payload.get("type")
            if message_type in {"start", "start_utterance"}:
                if session is not None:
                    raise ProtocolError("finish the active utterance before starting another")
                start = parse_start(payload)
                session = backend.new_session()
                await ws.send_str(
                    payload_json(
                        {
                            "type": "utterance_started",
                            "stream_id": start.stream_id,
                            "utterance_id": start.utterance_id,
                        }
                    )
                )
            elif message_type == "end_utterance":
                if session is None or start is None:
                    raise ProtocolError("no active utterance")
                result = await asyncio.to_thread(session.finish)
                await ws.send_str(
                    payload_json(
                        result_payload(
                            result,
                            stream_id=start.stream_id,
                            utterance_id=start.utterance_id,
                            backend=backend,
                        )
                    )
                )
                session = None
                start = None
            elif message_type == "ping":
                await ws.send_str(payload_json({"type": "pong"}))
            elif message_type == "close_stream":
                await ws.close(code=1000, message=b"normal closure")
            else:
                raise ProtocolError(f"unsupported message type: {message_type!r}")
    except (ProtocolError, ValueError, RuntimeError) as exc:
        logger.info("ASR stream rejected: %s", exc)
        if not ws.closed:
            error_payload = {"type": "error", "code": "bad_request", "message": str(exc)}
            await ws.send_str(payload_json(error_payload))
            await ws.close(code=1008, message=b"protocol error")
    except Exception:
        logger.exception("ASR stream failed")
        if not ws.closed:
            await ws.send_str(
                payload_json(
                    {"type": "error", "code": "internal_error", "message": "inference failed"}
                )
            )
            await ws.close(code=1011, message=b"inference error")
    return ws


def create_app(settings: Settings, backend: StreamingBackend) -> web.Application:
    app = web.Application()
    app[BACKEND_KEY] = backend
    app[SETTINGS_KEY] = settings
    app.add_routes(
        [
            web.get("/healthz", health),
            web.get("/readyz", ready),
            web.get("/v1/info", info),
            web.get("/v1/stream", stream),
        ]
    )
    return app
