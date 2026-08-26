"""HTTP health endpoints and the streaming WebSocket service."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from typing import Any

from aiohttp import WSMsgType, web

from . import __version__
from .backend import StreamingBackend, StreamingSession, payload_json, result_payload
from .capacity import CapacityError, CapacityManager, UtteranceLease
from .config import Settings, detect_host_kind
from .protocol import PROTOCOL_VERSION, ProtocolError, StartMessage, parse_start

logger = logging.getLogger("eidolon.asr")
BACKEND_KEY: web.AppKey[StreamingBackend] = web.AppKey("backend", object)
SETTINGS_KEY: web.AppKey[Settings] = web.AppKey("settings", Settings)
CAPACITY_KEY: web.AppKey[CapacityManager] = web.AppKey("capacity", CapacityManager)


class UtteranceLimitError(RuntimeError):
    pass


class UtteranceContext:
    def __init__(
        self,
        *,
        start: StartMessage,
        backend: StreamingBackend,
        capacity: CapacityManager,
        lease: UtteranceLease,
        settings: Settings,
        ws: web.WebSocketResponse,
        send_lock: asyncio.Lock,
    ) -> None:
        self.start = start
        self._backend = backend
        self._capacity = capacity
        self._lease = lease
        self._settings = settings
        self._ws = ws
        self._send_lock = send_lock
        self._state_lock = asyncio.Lock()
        self._buffer = bytearray()
        self._audio_bytes = 0
        self._session: StreamingSession | None = backend.new_session() if lease.active else None
        self._failure: CapacityError | None = None
        self._closed = False
        self._activation_task: asyncio.Task[None] | None = None

    def start_activation(self) -> None:
        if not self._lease.active and self._activation_task is None:
            self._activation_task = asyncio.create_task(self._activate())

    @property
    def queued(self) -> bool:
        return self._lease.queued

    @property
    def queue_position(self) -> int:
        return self._lease.queue_position

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._ws.closed:
            return
        async with self._send_lock:
            if not self._ws.closed:
                await self._ws.send_str(payload_json(payload))

    async def _send_results(self, results: list[Any]) -> None:
        for result in results:
            await self._send(
                result_payload(
                    result,
                    stream_id=self.start.stream_id,
                    utterance_id=self.start.utterance_id,
                    backend=self._backend,
                )
            )

    async def _activate(self) -> None:
        try:
            await self._capacity.wait_ready(
                self._lease,
                self._settings.max_queue_wait_seconds,
            )
            await self._send(
                {
                    "type": "utterance_active",
                    "stream_id": self.start.stream_id,
                    "utterance_id": self.start.utterance_id,
                    "queue_wait_ms": self._lease.queue_wait_ms,
                }
            )
            async with self._state_lock:
                if self._closed:
                    return
                self._session = self._backend.new_session()
                buffered = bytes(self._buffer)
                self._buffer.clear()
                for offset in range(0, len(buffered), 3200):
                    results = await asyncio.to_thread(
                        self._session.feed_pcm16,
                        buffered[offset : offset + 3200],
                    )
                    await self._send_results(results)
        except CapacityError as exc:
            self._failure = exc
            await self._send(
                {
                    "type": "error",
                    "code": exc.code,
                    "message": str(exc),
                    "retryable": True,
                }
            )
            if not self._ws.closed:
                await self._ws.close(code=1013, message=b"capacity timeout")

    async def feed(self, pcm: bytes) -> None:
        if len(pcm) % 2:
            raise ValueError("PCM16 payload length must be even")
        self._audio_bytes += len(pcm)
        max_audio_bytes = round(self._settings.max_utterance_seconds * 32000)
        if self._audio_bytes > max_audio_bytes:
            raise UtteranceLimitError(
                f"utterance exceeds {self._settings.max_utterance_seconds:g} seconds"
            )
        if self._failure is not None:
            raise self._failure
        async with self._state_lock:
            if self._session is None:
                self._buffer.extend(pcm)
                return
            results = await asyncio.to_thread(self._session.feed_pcm16, pcm)
            await self._send_results(results)

    async def finish(self) -> Any:
        if self._activation_task is not None:
            await self._activation_task
        if self._failure is not None:
            raise self._failure
        async with self._state_lock:
            if self._session is None:
                raise RuntimeError("utterance did not acquire a realtime slot")
            return await asyncio.to_thread(self._session.finish)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._activation_task is not None and not self._activation_task.done():
            self._activation_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._activation_task
        self._buffer.clear()
        await self._capacity.release(self._lease)


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": "eidolon-asr", "version": __version__})


async def ready(request: web.Request) -> web.Response:
    backend = request.app[BACKEND_KEY]
    capacity = await request.app[CAPACITY_KEY].snapshot()
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
            "capacity": capacity,
        }
    )


async def info(request: web.Request) -> web.Response:
    backend = request.app[BACKEND_KEY]
    settings = request.app[SETTINGS_KEY]
    capacity = await request.app[CAPACITY_KEY].snapshot()
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
            "capacity": capacity,
            "max_utterance_seconds": settings.max_utterance_seconds,
            "max_queue_wait_seconds": settings.max_queue_wait_seconds,
        }
    )


async def stream(request: web.Request) -> web.StreamResponse:
    backend = request.app[BACKEND_KEY]
    settings = request.app[SETTINGS_KEY]
    capacity = request.app[CAPACITY_KEY]
    if not await capacity.open_connection():
        return web.json_response(
            {
                "type": "error",
                "code": "connection_capacity_exceeded",
                "message": f"maximum of {settings.max_connections} connections reached",
                "retryable": True,
            },
            status=503,
            headers={"Retry-After": "1"},
        )
    ws = web.WebSocketResponse(max_msg_size=settings.max_binary_message_bytes)
    send_lock = asyncio.Lock()

    async def send(payload: dict[str, Any]) -> None:
        if ws.closed:
            return
        async with send_lock:
            if not ws.closed:
                await ws.send_str(payload_json(payload))

    context: UtteranceContext | None = None
    start: StartMessage | None = None
    try:
        await ws.prepare(request)
        await send(
            {
                "type": "connected",
                "protocol_version": PROTOCOL_VERSION,
                "backend": backend.name,
                "model_id": backend.model_id,
                "offline_model_id": backend.offline_model_id,
                "punctuation_model_id": backend.punctuation_model_id,
            }
        )
        async for message in ws:
            if message.type == WSMsgType.BINARY:
                if context is None or start is None:
                    raise ProtocolError("start an utterance before sending audio")
                await context.feed(message.data)
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
                if context is not None:
                    raise ProtocolError("finish the active utterance before starting another")
                start = parse_start(payload)
                lease = await capacity.reserve()
                context = UtteranceContext(
                    start=start,
                    backend=backend,
                    capacity=capacity,
                    lease=lease,
                    settings=settings,
                    ws=ws,
                    send_lock=send_lock,
                )
                await send(
                    {
                        "type": "utterance_started",
                        "stream_id": start.stream_id,
                        "utterance_id": start.utterance_id,
                        "queued": context.queued,
                        "queue_position": context.queue_position,
                    }
                )
                context.start_activation()
            elif message_type == "end_utterance":
                if context is None or start is None:
                    raise ProtocolError("no active utterance")
                result = await context.finish()
                await send(
                    result_payload(
                        result,
                        stream_id=start.stream_id,
                        utterance_id=start.utterance_id,
                        backend=backend,
                    )
                )
                await context.close()
                context = None
                start = None
            elif message_type == "ping":
                await send({"type": "pong"})
            elif message_type == "close_stream":
                await ws.close(code=1000, message=b"normal closure")
            else:
                raise ProtocolError(f"unsupported message type: {message_type!r}")
    except CapacityError as exc:
        logger.info("ASR capacity rejected: %s", exc)
        if not ws.closed:
            await send(
                {
                    "type": "error",
                    "code": exc.code,
                    "message": str(exc),
                    "retryable": True,
                }
            )
            await ws.close(code=1013, message=b"capacity exceeded")
    except UtteranceLimitError as exc:
        logger.info("ASR utterance rejected: %s", exc)
        if not ws.closed:
            await send(
                {
                    "type": "error",
                    "code": "utterance_too_long",
                    "message": str(exc),
                    "retryable": False,
                }
            )
            await ws.close(code=1009, message=b"utterance too long")
    except (ProtocolError, ValueError, RuntimeError) as exc:
        logger.info("ASR stream rejected: %s", exc)
        if not ws.closed:
            error_payload = {"type": "error", "code": "bad_request", "message": str(exc)}
            await send(error_payload)
            await ws.close(code=1008, message=b"protocol error")
    except Exception:
        logger.exception("ASR stream failed")
        if not ws.closed:
            await send({"type": "error", "code": "internal_error", "message": "inference failed"})
            await ws.close(code=1011, message=b"inference error")
    finally:
        if context is not None:
            await context.close()
        await capacity.close_connection()
    return ws


def create_app(settings: Settings, backend: StreamingBackend) -> web.Application:
    app = web.Application()
    app[BACKEND_KEY] = backend
    app[SETTINGS_KEY] = settings
    app[CAPACITY_KEY] = CapacityManager(
        max_connections=settings.max_connections,
        realtime_slots=settings.realtime_slots,
        max_queued=settings.max_queued_utterances,
    )
    app.add_routes(
        [
            web.get("/healthz", health),
            web.get("/readyz", ready),
            web.get("/v1/info", info),
            web.get("/v1/stream", stream),
        ]
    )
    return app
