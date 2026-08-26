from __future__ import annotations

import asyncio

import pytest
from aiohttp import WSMsgType, WSServerHandshakeError

from eidolon_models_asr.backend import MockStreamingBackend
from eidolon_models_asr.config import Settings
from eidolon_models_asr.service import create_app


def start_payload(index: int) -> dict[str, object]:
    return {
        "type": "start",
        "stream_id": f"stream-{index}",
        "utterance_id": f"utterance-{index}",
        "sample_rate": 16000,
        "channels": 1,
        "format": "pcm_s16le",
    }


async def test_health_and_readiness(aiohttp_client) -> None:
    client = await aiohttp_client(create_app(Settings(backend="mock"), MockStreamingBackend()))
    health = await client.get("/healthz")
    assert health.status == 200
    assert (await health.json())["ok"] is True
    ready = await client.get("/readyz")
    assert ready.status == 200
    assert (await ready.json())["backend"] == "mock"
    info = await (await client.get("/v1/info")).json()
    assert info["offline_model_id"] is None
    assert info["punctuation_model_id"] is None
    assert info["capacity"]["max_connections"] == 64
    assert info["capacity"]["realtime_slots"] == 2
    assert info["capacity"]["max_queued_utterances"] == 6
    assert info["capacity"]["max_active_utterances"] == 8


async def test_streaming_contract_and_multiple_utterances(aiohttp_client) -> None:
    client = await aiohttp_client(create_app(Settings(backend="mock"), MockStreamingBackend()))
    ws = await client.ws_connect("/v1/stream")
    assert (await ws.receive_json())["type"] == "connected"

    for index in range(2):
        await ws.send_json(
            {
                "type": "start" if index == 0 else "start_utterance",
                "stream_id": "stream-1",
                "utterance_id": f"utterance-{index}",
                "sample_rate": 16000,
                "channels": 1,
                "format": "pcm_s16le",
            }
        )
        started = await ws.receive_json()
        assert started["type"] == "utterance_started"
        assert started["queued"] is False
        assert started["queue_position"] == 0
        await ws.send_bytes(b"\x00\x00" * 1600)
        interim = await ws.receive_json()
        assert interim["type"] == "transcript"
        assert interim["is_final"] is False
        await ws.send_json({"type": "end_utterance"})
        final = await ws.receive_json()
        assert final["is_final"] is True
        assert final["text"] == "测试"
        assert final["raw_text"] == "测试"
        assert final["streaming_text"] is None
        assert final["final_revised"] is False
        assert final["offline_decode_ms"] == 0.0
        assert final["punctuation_model_id"] is None
        assert final["punctuation_ms"] == 0.0
        assert final["total_inference_ms"] == final["decode_ms"]

    await ws.send_json({"type": "close_stream"})
    assert (await ws.receive()).type in {WSMsgType.CLOSE, WSMsgType.CLOSED}


async def test_audio_before_start_is_rejected(aiohttp_client) -> None:
    client = await aiohttp_client(create_app(Settings(backend="mock"), MockStreamingBackend()))
    ws = await client.ws_connect("/v1/stream")
    await ws.receive_json()
    await ws.send_bytes(b"\x00\x00")
    error = await ws.receive_json()
    assert error["type"] == "error"
    assert error["code"] == "bad_request"


async def test_queued_utterance_buffers_audio_and_runs_after_fifo_promotion(
    aiohttp_client,
) -> None:
    settings = Settings(
        backend="mock",
        max_connections=3,
        realtime_slots=1,
        max_queued_utterances=2,
        max_queue_wait_seconds=1,
    )
    client = await aiohttp_client(create_app(settings, MockStreamingBackend()))
    first = await client.ws_connect("/v1/stream")
    second = await client.ws_connect("/v1/stream")
    await first.receive_json()
    await second.receive_json()

    await first.send_json(start_payload(1))
    await second.send_json(start_payload(2))
    assert (await first.receive_json())["queued"] is False
    queued = await second.receive_json()
    assert queued["queued"] is True
    assert queued["queue_position"] == 1

    await second.send_bytes(b"\x00\x00" * 1600)
    await second.send_json({"type": "end_utterance"})

    await first.send_bytes(b"\x00\x00" * 1600)
    assert (await first.receive_json())["is_final"] is False
    await first.send_json({"type": "end_utterance"})
    assert (await first.receive_json())["is_final"] is True

    active = await second.receive_json(timeout=1)
    promoted_interim = await second.receive_json(timeout=1)
    promoted_final = await second.receive_json(timeout=1)
    assert active["type"] == "utterance_active"
    assert active["queue_wait_ms"] >= 0
    assert promoted_interim["is_final"] is False
    assert promoted_final["is_final"] is True

    await first.close()
    await second.close()


async def test_full_utterance_queue_returns_retryable_capacity_error(aiohttp_client) -> None:
    settings = Settings(
        backend="mock",
        max_connections=3,
        realtime_slots=1,
        max_queued_utterances=1,
    )
    client = await aiohttp_client(create_app(settings, MockStreamingBackend()))
    sockets = [await client.ws_connect("/v1/stream") for _ in range(3)]
    for ws in sockets:
        await ws.receive_json()
    for index, ws in enumerate(sockets):
        await ws.send_json(start_payload(index))

    assert (await sockets[0].receive_json())["queued"] is False
    assert (await sockets[1].receive_json())["queued"] is True
    error = await sockets[2].receive_json()
    assert error["code"] == "capacity_exceeded"
    assert error["retryable"] is True

    for ws in sockets:
        await ws.close()


async def test_connection_limit_returns_http_503(aiohttp_client) -> None:
    settings = Settings(
        backend="mock",
        max_connections=1,
        realtime_slots=1,
        max_queued_utterances=0,
    )
    client = await aiohttp_client(create_app(settings, MockStreamingBackend()))
    first = await client.ws_connect("/v1/stream")
    await first.receive_json()
    with pytest.raises(WSServerHandshakeError) as rejected:
        await client.ws_connect("/v1/stream")
    assert rejected.value.status == 503
    await first.close()

    replacement = await client.ws_connect("/v1/stream")
    assert (await replacement.receive_json())["type"] == "connected"
    await replacement.close()


async def test_queued_utterance_times_out_without_provider_fallback(aiohttp_client) -> None:
    settings = Settings(
        backend="mock",
        max_connections=2,
        realtime_slots=1,
        max_queued_utterances=1,
        max_queue_wait_seconds=0.02,
    )
    client = await aiohttp_client(create_app(settings, MockStreamingBackend()))
    first = await client.ws_connect("/v1/stream")
    second = await client.ws_connect("/v1/stream")
    await first.receive_json()
    await second.receive_json()
    await first.send_json(start_payload(1))
    await second.send_json(start_payload(2))
    await first.receive_json()
    assert (await second.receive_json())["queued"] is True

    error = await second.receive_json(timeout=1)
    assert error["code"] == "capacity_timeout"
    assert error["retryable"] is True
    assert (await second.receive()).type in {WSMsgType.CLOSE, WSMsgType.CLOSED}
    await first.close()


async def test_utterance_duration_is_bounded(aiohttp_client) -> None:
    settings = Settings(
        backend="mock",
        max_connections=1,
        realtime_slots=1,
        max_queued_utterances=0,
        max_utterance_seconds=0.05,
    )
    client = await aiohttp_client(create_app(settings, MockStreamingBackend()))
    ws = await client.ws_connect("/v1/stream")
    await ws.receive_json()
    await ws.send_json(start_payload(1))
    await ws.receive_json()
    await ws.send_bytes(b"\x00\x00" * 801)
    error = await ws.receive_json(timeout=1)
    assert error["code"] == "utterance_too_long"
    assert error["retryable"] is False


async def test_runtime_capacity_metrics_change_with_connections_and_queue(
    aiohttp_client,
) -> None:
    settings = Settings(
        backend="mock",
        max_connections=3,
        realtime_slots=1,
        max_queued_utterances=1,
    )
    client = await aiohttp_client(create_app(settings, MockStreamingBackend()))
    first = await client.ws_connect("/v1/stream")
    second = await client.ws_connect("/v1/stream")
    await first.receive_json()
    await second.receive_json()
    await first.send_json(start_payload(1))
    await second.send_json(start_payload(2))
    await first.receive_json()
    await second.receive_json()

    info = await (await client.get("/v1/info")).json()
    assert info["capacity"]["connections"] == 2
    assert info["capacity"]["active_utterances"] == 1
    assert info["capacity"]["queued_utterances"] == 1

    await first.close()
    await asyncio.sleep(0)
    await second.close()
