"""Concurrent WebSocket benchmark helpers for the streaming ASR service."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from aiohttp import ClientSession


@dataclass(frozen=True)
class AudioVariant:
    name: str
    pcm: bytes


def generate_audio_variants(pcm: bytes) -> list[AudioVariant]:
    """Create deterministic streams without adding binary fixtures to the repo."""
    if len(pcm) % 2:
        raise ValueError("PCM16 payload length must be even")
    source = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    rng = np.random.default_rng(20260826)
    noise = rng.normal(0.0, 220.0, size=source.shape)

    def encode(samples: np.ndarray) -> bytes:
        return np.clip(np.rint(samples), -32768, 32767).astype("<i2").tobytes()

    return [
        AudioVariant("original", pcm),
        AudioVariant("quiet_minus_6db", encode(source * 0.5)),
        AudioVariant("noise_approx_30db", encode(source + noise)),
        AudioVariant("quiet_with_noise", encode(source * 0.65 + noise)),
    ]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty list")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return round(ordered[index], 3)


async def benchmark_stream(
    client: ClientSession,
    *,
    url: str,
    stream_index: int,
    variant: AudioVariant,
    frame_ms: int,
    realtime: bool,
) -> dict[str, Any]:
    connect_started = time.perf_counter()
    async with client.ws_connect(url) as ws:
        connected = await ws.receive_json(timeout=30)
        connected_at = time.perf_counter()
        start_sent_at = time.perf_counter()
        await ws.send_json(
            {
                "type": "start",
                "stream_id": f"bench-stream-{stream_index}",
                "utterance_id": f"bench-utterance-{stream_index}",
                "sample_rate": 16000,
                "channels": 1,
                "format": "pcm_s16le",
            }
        )
        started = await ws.receive_json(timeout=30)
        started_at = time.perf_counter()
        if connected.get("type") != "connected" or started.get("type") != "utterance_started":
            raise RuntimeError("unexpected ASR handshake response")

        first_interim_at: float | None = None
        final_at: float | None = None
        interim_count = 0

        async def receive_transcripts() -> dict[str, Any]:
            nonlocal final_at, first_interim_at, interim_count
            while True:
                value = await ws.receive_json(timeout=60)
                if value.get("type") != "transcript":
                    if value.get("type") == "error":
                        raise RuntimeError(value.get("message", "ASR service returned an error"))
                    continue
                received_at = time.perf_counter()
                if value.get("is_final"):
                    final_at = received_at
                    return value
                interim_count += 1
                if first_interim_at is None:
                    first_interim_at = received_at

        receiver = asyncio.create_task(receive_transcripts())
        first_audio_at = time.perf_counter()
        frame_bytes = frame_ms * 32
        for offset in range(0, len(variant.pcm), frame_bytes):
            chunk = variant.pcm[offset : offset + frame_bytes]
            await ws.send_bytes(chunk)
            if realtime:
                sent_audio_ms = min(offset + len(chunk), len(variant.pcm)) / 32.0
                delay = first_audio_at + sent_audio_ms / 1000.0 - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
        eot_sent_at = time.perf_counter()
        await ws.send_json({"type": "end_utterance"})
        final = await receiver
        assert final_at is not None

    audio_ms = float(final["audio_ms"])
    stream_wall_ms = (final_at - first_audio_at) * 1000.0
    return {
        "stream": stream_index,
        "variant": variant.name,
        "text": final["text"],
        "raw_text": final.get("raw_text", final["text"]),
        "interim_count": interim_count,
        "audio_ms": round(audio_ms, 3),
        "connect_ms": round((connected_at - connect_started) * 1000.0, 3),
        "start_ack_ms": round((started_at - start_sent_at) * 1000.0, 3),
        "first_interim_ms": (
            round((first_interim_at - first_audio_at) * 1000.0, 3)
            if first_interim_at is not None
            else None
        ),
        "eot_final_ms": round((final_at - eot_sent_at) * 1000.0, 3),
        "stream_wall_ms": round(stream_wall_ms, 3),
        "overhang_ms": round(stream_wall_ms - audio_ms, 3) if realtime else None,
        "decode_ms": float(final["decode_ms"]),
        "rtf": float(final["rtf"]),
        "punctuation_ms": float(final.get("punctuation_ms", 0.0)),
        "total_inference_ms": float(final.get("total_inference_ms", final["decode_ms"])),
        "total_rtf": float(final.get("total_rtf", final["rtf"])),
    }


def summarize_streams(streams: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "connect_ms",
        "start_ack_ms",
        "first_interim_ms",
        "eot_final_ms",
        "stream_wall_ms",
        "overhang_ms",
        "decode_ms",
        "rtf",
        "punctuation_ms",
        "total_inference_ms",
        "total_rtf",
    )
    summary: dict[str, Any] = {}
    for metric in metrics:
        values = [float(stream[metric]) for stream in streams if stream.get(metric) is not None]
        if values:
            summary[metric] = {
                "p50": percentile(values, 0.50),
                "p95": percentile(values, 0.95),
                "max": round(max(values), 3),
            }
    return summary


async def benchmark_level(
    *,
    url: str,
    pcm: bytes,
    concurrency: int,
    frame_ms: int = 100,
    realtime: bool = True,
) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if frame_ms < 10 or frame_ms > 1000:
        raise ValueError("frame_ms must be between 10 and 1000")
    variants = generate_audio_variants(pcm)
    level_started = time.perf_counter()
    async with ClientSession() as client:
        streams = await asyncio.gather(
            *(
                benchmark_stream(
                    client,
                    url=url,
                    stream_index=index,
                    variant=variants[index % len(variants)],
                    frame_ms=frame_ms,
                    realtime=realtime,
                )
                for index in range(concurrency)
            )
        )
    return {
        "concurrency": concurrency,
        "realtime": realtime,
        "frame_ms": frame_ms,
        "level_wall_ms": round((time.perf_counter() - level_started) * 1000.0, 3),
        "summary": summarize_streams(streams),
        "streams": streams,
    }
