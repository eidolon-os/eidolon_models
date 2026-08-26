from __future__ import annotations

from aiohttp import WSMsgType

from eidolon_models_asr.backend import MockStreamingBackend
from eidolon_models_asr.config import Settings
from eidolon_models_asr.service import create_app


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
        assert (await ws.receive_json())["type"] == "utterance_started"
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
