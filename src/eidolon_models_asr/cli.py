"""Command-line entry point shared by every Host."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import sys
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, WSMsgType, web

from .artifacts import ArtifactError, verify_artifacts
from .backend import FunASROnnxBackend, MockStreamingBackend, StreamingBackend
from .config import Settings, detect_host_kind
from .service import create_app


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _load_backend(settings: Settings) -> StreamingBackend:
    if settings.resolved_backend == "mock":
        return MockStreamingBackend()
    verify_artifacts(settings.manifest_path, settings.model_dir)
    return FunASROnnxBackend(
        settings.model_dir,
        intra_op_threads=settings.intra_op_threads,
        chunk_size=settings.chunk_size,
    )


def _read_pcm16_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("WAV must be 16 kHz, mono, PCM16")
        return wav.readframes(wav.getnframes())


def command_doctor(settings: Settings) -> int:
    verification: dict[str, Any]
    try:
        verification = verify_artifacts(settings.manifest_path, settings.model_dir)
    except ArtifactError as exc:
        verification = {"ok": False, "error": str(exc)}
    providers: list[str] = []
    try:
        import onnxruntime

        providers = onnxruntime.get_available_providers()
    except ImportError:
        pass
    value = {
        "ok": bool(verification.get("ok")),
        "host_kind": detect_host_kind(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "requested_backend": settings.backend,
        "resolved_backend": settings.resolved_backend,
        "onnx_providers": providers,
        "model_dir": str(settings.model_dir),
        "artifacts": verification,
    }
    _print(value)
    return 0 if value["ok"] else 1


def command_verify(settings: Settings) -> int:
    _print(verify_artifacts(settings.manifest_path, settings.model_dir))
    return 0


def command_infer(settings: Settings, audio: Path) -> int:
    backend = _load_backend(settings)
    session = backend.new_session()
    pcm = _read_pcm16_wav(audio)
    events = []
    frame_bytes = 3200
    for offset in range(0, len(pcm), frame_bytes):
        events.extend(session.feed_pcm16(pcm[offset : offset + frame_bytes]))
    final = session.finish()
    _print(
        {
            "ok": True,
            "backend": backend.name,
            "model_id": backend.model_id,
            "interim_count": len(events),
            "text": final.text,
            "audio_ms": final.audio_ms,
            "decode_ms": final.decode_ms,
            "rtf": final.rtf,
        }
    )
    return 0


async def _probe(url: str, audio: Path) -> dict[str, Any]:
    pcm = _read_pcm16_wav(audio)
    transcripts: list[dict[str, Any]] = []
    async with ClientSession() as client:
        async with client.ws_connect(url) as ws:
            connected = await ws.receive_json()
            await ws.send_json(
                {
                    "type": "start",
                    "stream_id": "probe-stream",
                    "utterance_id": "probe-utterance",
                    "sample_rate": 16000,
                    "channels": 1,
                    "format": "pcm_s16le",
                }
            )
            started = await ws.receive_json()
            for offset in range(0, len(pcm), 3200):
                await ws.send_bytes(pcm[offset : offset + 3200])
                while True:
                    try:
                        message = await ws.receive(timeout=0.001)
                    except asyncio.TimeoutError:
                        break
                    if message.type == WSMsgType.TEXT:
                        transcripts.append(json.loads(message.data))
                    else:
                        break
            await ws.send_json({"type": "end_utterance"})
            while True:
                value = await ws.receive_json(timeout=30)
                transcripts.append(value)
                if value.get("type") == "transcript" and value.get("is_final"):
                    break
            return {"connected": connected, "started": started, "events": transcripts}


def command_probe(url: str, audio: Path) -> int:
    value = asyncio.run(_probe(url, audio))
    final = next(
        event
        for event in reversed(value["events"])
        if event.get("type") == "transcript" and event.get("is_final")
    )
    _print(
        {
            "ok": True,
            "url": url,
            "backend": value["connected"]["backend"],
            "interim_count": sum(
                1
                for event in value["events"]
                if event.get("type") == "transcript" and not event.get("is_final")
            ),
            "text": final["text"],
            "audio_ms": final["audio_ms"],
            "rtf": final["rtf"],
        }
    )
    return 0


def command_serve(settings: Settings) -> int:
    logging.basicConfig(
        level=os.getenv("EIDOLON_ASR_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    backend = _load_backend(settings)
    web.run_app(create_app(settings, backend), host=settings.host, port=settings.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eidolon-asr")
    parser.add_argument("--backend", help="auto, onnx-cpu, or mock")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--threads", type=int)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="show Host, runtime, and artifact readiness")
    subparsers.add_parser("verify", help="verify pinned model checksums")
    serve = subparsers.add_parser("serve", help="start the ASR HTTP/WebSocket service")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    infer = subparsers.add_parser("infer", help="run real model inference on a WAV file")
    infer.add_argument("audio", type=Path)
    probe = subparsers.add_parser("probe", help="exercise a running WebSocket service")
    probe.add_argument("audio", type=Path)
    probe.add_argument("--url", default="ws://127.0.0.1:8767/v1/stream")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    overrides: dict[str, Any] = {}
    for field, argument in (
        ("backend", args.backend),
        ("model_dir", args.model_dir),
        ("manifest_path", args.manifest),
        ("intra_op_threads", args.threads),
    ):
        if argument is not None:
            overrides[field] = argument
    if args.command == "serve":
        if args.host is not None:
            overrides["host"] = args.host
        if args.port is not None:
            overrides["port"] = args.port
    settings = replace(settings, **overrides)
    try:
        if args.command == "doctor":
            return command_doctor(settings)
        if args.command == "verify":
            return command_verify(settings)
        if args.command == "infer":
            return command_infer(settings, args.audio)
        if args.command == "probe":
            return command_probe(args.url, args.audio)
        if args.command == "serve":
            return command_serve(settings)
    except (ArtifactError, ValueError, RuntimeError) as exc:
        print(f"eidolon-asr: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
