"""Real model HTTP concurrency probe; reports admission and latency, never tunes policy."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from aiohttp import ClientSession, web

from eidolon_models_laya.config import Settings
from eidolon_models_laya.engine import load_engine
from eidolon_models_laya.service import create_app


async def run(model_dir: Path, requests: int) -> None:
    settings = Settings(backend="onnx", model_dir=model_dir, max_pending=4)
    engine, manifest = load_engine(settings)
    app = create_app(engine, settings, {"model": manifest.name, "revision": manifest.revision})
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    body = {
        "state": {"utterance": "打开客厅的灯"},
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "家居意图？",
                "criteria": ["控制", "查询", "无关"],
            }
        },
    }
    async with ClientSession() as client:

        async def one() -> tuple[int, float]:
            start = time.perf_counter()
            async with client.post(f"http://127.0.0.1:{port}/v1/systemone", json=body) as response:
                await response.read()
                return response.status, (time.perf_counter() - start) * 1000

        await one()  # warm runtime kernels
        rows = await asyncio.gather(*(one() for _ in range(requests)))
    await runner.cleanup()
    latencies = sorted(ms for status, ms in rows if status == 200)
    result = {
        "backend": "onnx",
        "revision": manifest.revision,
        "requests": requests,
        "ok": len(latencies),
        "busy": sum(status == 503 for status, _ in rows),
        "other": [status for status, _ in rows if status not in {200, 503}],
        "p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "p95_ms": round(latencies[max(0, int(0.95 * len(latencies)) - 1)], 1)
        if latencies
        else None,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=12)
    args = parser.parse_args()
    asyncio.run(run(args.model_dir, args.requests))
