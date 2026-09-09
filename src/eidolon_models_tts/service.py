"""The WebSocket surface: `eidolon_sdk.biz.contracts.local_tts`, served.

Nothing here knows the engine is a subprocess — `engine.py` beside this file is
the only thing that does. What this owns is the contract: the greeting, one
request at a time, audio as binary frames, and a refusal that says whether it
is worth sending again.
"""

from __future__ import annotations

import asyncio
import json
import logging

from aiohttp import WSMsgType, web

from eidolon_models_tts import protocol
from eidolon_models_tts.config import Settings
from eidolon_models_tts.engine import Engine, EngineUnavailable, SynthesisFailed, UtteranceReport

logger = logging.getLogger(__name__)

SETTINGS_KEY = web.AppKey("settings", Settings)
ENGINE_KEY = web.AppKey("engine", Engine)

SERVICE_NAME = "eidolon-tts"


def _audio_description() -> dict[str, object]:
    return {
        "sample_rate": protocol.AUDIO_SAMPLE_RATE,
        "channels": protocol.AUDIO_CHANNELS,
        "format": protocol.AUDIO_FORMAT,
    }


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": SERVICE_NAME})


async def ready(request: web.Request) -> web.Response:
    engine = request.app[ENGINE_KEY]
    settings = request.app[SETTINGS_KEY]
    document = {
        "ok": engine.ready,
        "service": SERVICE_NAME,
        protocol.PROTOCOL_VERSION_FIELD: protocol.PROTOCOL_VERSION,
        "voice_id": settings.voice,
        "audio": _audio_description(),
        "init_warmup_ms": engine.init_warmup_ms,
    }
    # Not ready is a 503, so a release's readiness gate waits instead of
    # calling the Host activated: the engine binds this port before its NPU
    # graphs are warm, and on this board that gap is about three seconds.
    return web.json_response(document, status=200 if engine.ready else 503)


async def info(request: web.Request) -> web.Response:
    engine = request.app[ENGINE_KEY]
    settings = request.app[SETTINGS_KEY]
    return web.json_response(
        {
            "service": SERVICE_NAME,
            protocol.PROTOCOL_VERSION_FIELD: protocol.PROTOCOL_VERSION,
            "capability": protocol.CAPABILITY,
            "voice_id": settings.voice,
            "audio": _audio_description(),
            "max_text_characters": protocol.MAX_TEXT_CHARACTERS,
            "concurrent_requests": 1,
            "init_warmup_ms": engine.init_warmup_ms,
            "npu_cores": {
                "head": settings.head_core,
                "encoder": settings.encoder_core,
                "flow": settings.flow_core,
                "hift": settings.hift_core,
            },
        }
    )


async def _refuse(socket: web.WebSocketResponse, code: str, message: str,
                  request_id: str | None = None) -> None:
    payload: dict[str, object] = {
        "type": protocol.ERROR,
        "code": code,
        "message": message,
        "retryable": code in protocol.RETRYABLE_ERROR_CODES,
    }
    if request_id is not None:
        payload[protocol.REQUEST_ID_FIELD] = request_id
    await socket.send_json(payload)


async def stream(request: web.Request) -> web.StreamResponse:
    engine = request.app[ENGINE_KEY]
    settings = request.app[SETTINGS_KEY]
    socket = web.WebSocketResponse(heartbeat=30)
    await socket.prepare(request)

    await socket.send_json(
        {
            "type": protocol.CONNECTED,
            protocol.PROTOCOL_VERSION_FIELD: protocol.PROTOCOL_VERSION,
            "voice_id": settings.voice,
            **_audio_description(),
        }
    )

    #: The request being synthesized, so a `cancel` can name it and anything
    #: else can be told it is talking about the wrong one.
    active: str | None = None
    cancelled: set[str] = set()

    async def deliver(request_id: str, text: str) -> None:
        nonlocal active
        active = request_id
        await socket.send_json(
            {"type": protocol.SYNTHESIS_STARTED, protocol.REQUEST_ID_FIELD: request_id}
        )
        report: UtteranceReport | None = None
        try:
            async for piece in engine.synthesize(text):
                if request_id in cancelled:
                    # Delivery stops here. The engine keeps the NPU until this
                    # utterance ends — see the contract's note on what cancel
                    # promises — so the next request may wait a moment.
                    await socket.send_json(
                        {
                            "type": protocol.SYNTHESIS_CANCELLED,
                            protocol.REQUEST_ID_FIELD: request_id,
                        }
                    )
                    return
                if isinstance(piece, UtteranceReport):
                    report = piece
                    break
                await socket.send_bytes(piece)
        except SynthesisFailed as error:
            await _refuse(socket, protocol.ERROR_INTERNAL, str(error), request_id)
            return
        except EngineUnavailable as error:
            await _refuse(socket, protocol.ERROR_ENGINE_UNAVAILABLE, str(error), request_id)
            return
        finally:
            active = None
            cancelled.discard(request_id)

        finished: dict[str, object] = {
            "type": protocol.SYNTHESIS_FINISHED,
            protocol.REQUEST_ID_FIELD: request_id,
            "pcm_bytes": report.pcm_bytes if report else 0,
        }
        if report is not None:
            # The engine's own measurements, passed through rather than
            # recomputed: a client that sees underruns here is being told the
            # Host could not keep ahead of playback, which is the one quality
            # fact it cannot observe for itself.
            for wire, key in (
                ("audio_seconds", "audio_seconds"),
                ("ttft_ms", "ttft_first_pcm_ms"),
                ("steady_rtf", "steady_pcm_rtf"),
                ("underruns", "underrun_count"),
                ("profile_ms", "profile_precompute_ms"),
            ):
                value = report.number(key)
                if value is not None:
                    finished[wire] = value
        await socket.send_json(finished)

    async for message in socket:
        if message.type is WSMsgType.BINARY:
            await _refuse(
                socket, protocol.ERROR_BAD_REQUEST, "this stream carries text, not audio"
            )
            continue
        if message.type is not WSMsgType.TEXT:
            break
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            await _refuse(socket, protocol.ERROR_BAD_REQUEST, "malformed JSON")
            continue
        if not isinstance(payload, dict):
            await _refuse(socket, protocol.ERROR_BAD_REQUEST, "a message is an object")
            continue

        kind = payload.get("type")
        if kind == protocol.PING:
            await socket.send_json({"type": protocol.PONG})
            continue
        if kind == protocol.CLOSE_STREAM:
            break
        if kind == protocol.CANCEL:
            try:
                target = protocol.parse_cancel(payload)
            except protocol.ProtocolError as error:
                await _refuse(socket, protocol.ERROR_BAD_REQUEST, str(error))
                continue
            # Recorded even when it names nothing in flight: a cancel that
            # crosses the finish of what it meant to stop is not an error, and
            # answering one with a refusal would make a client retry a turn the
            # user already interrupted.
            cancelled.add(target)
            continue
        if kind != protocol.SYNTHESIZE:
            await _refuse(socket, protocol.ERROR_BAD_REQUEST, f"unknown message type: {kind!r}")
            continue

        try:
            synthesize_request = protocol.parse_synthesize(payload)
        except protocol.ProtocolError as error:
            code = (
                protocol.ERROR_TEXT_TOO_LONG
                if "characters" in str(error)
                else protocol.ERROR_BAD_REQUEST
            )
            await _refuse(socket, code, str(error), payload.get(protocol.REQUEST_ID_FIELD))
            continue
        if active is not None:
            await _refuse(
                socket,
                protocol.ERROR_BUSY,
                f"this Host says one thing at a time; {active} is still speaking",
                synthesize_request.request_id,
            )
            continue
        await deliver(synthesize_request.request_id, synthesize_request.text)

    if not socket.closed:
        await socket.close()
    return socket


def create_app(settings: Settings, engine: Engine) -> web.Application:
    app = web.Application()
    app[SETTINGS_KEY] = settings
    app[ENGINE_KEY] = engine
    app.router.add_get("/healthz", health)
    app.router.add_get(protocol.READY_PATH, ready)
    app.router.add_get(protocol.INFO_PATH, info)
    app.router.add_get(protocol.STREAM_PATH, stream)

    async def _start(_: web.Application) -> None:
        await engine.start()

    async def _stop(_: web.Application) -> None:
        await engine.stop()

    app.on_startup.append(_start)
    app.on_cleanup.append(_stop)
    return app


def serve(settings: Settings) -> None:
    engine = Engine(settings)
    web.run_app(
        create_app(settings, engine),
        host=settings.host,
        port=settings.port,
        print=None,
        loop=asyncio.new_event_loop(),
    )
