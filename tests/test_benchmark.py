from __future__ import annotations

import numpy as np

from eidolon_models_asr.backend import MockStreamingBackend
from eidolon_models_asr.benchmark import benchmark_level, generate_audio_variants, percentile
from eidolon_models_asr.config import Settings
from eidolon_models_asr.service import create_app


def test_audio_variants_are_deterministic_pcm16_streams() -> None:
    source = np.arange(-1600, 1600, dtype="<i2").tobytes()
    first = generate_audio_variants(source)
    second = generate_audio_variants(source)
    assert [variant.name for variant in first] == [
        "original",
        "quiet_minus_6db",
        "noise_approx_30db",
        "quiet_with_noise",
    ]
    assert all(len(variant.pcm) == len(source) for variant in first)
    assert [variant.pcm for variant in first] == [variant.pcm for variant in second]
    assert first[0].pcm == source
    assert len({variant.pcm for variant in first}) == 4


def test_nearest_rank_percentile() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.50) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


async def test_four_concurrent_websocket_streams(aiohttp_server) -> None:
    server = await aiohttp_server(create_app(Settings(backend="mock"), MockStreamingBackend()))
    url = str(server.make_url("/v1/stream")).replace("http://", "ws://", 1)
    result = await benchmark_level(
        url=url,
        pcm=b"\x00\x00" * 3200,
        concurrency=4,
        frame_ms=100,
        realtime=False,
    )
    assert result["concurrency"] == 4
    assert len(result["streams"]) == 4
    assert {stream["variant"] for stream in result["streams"]} == {
        "original",
        "quiet_minus_6db",
        "noise_approx_30db",
        "quiet_with_noise",
    }
    assert all(stream["text"] == "测试" for stream in result["streams"])
    assert all(stream["first_interim_ms"] is not None for stream in result["streams"])
    assert result["summary"]["eot_final_ms"]["max"] >= 0
