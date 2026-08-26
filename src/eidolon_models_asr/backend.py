"""Streaming ASR backend abstraction and FunASR ONNX implementation."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    delta: str
    is_final: bool
    revision: int
    audio_ms: int
    decode_ms: float
    rtf: float


class StreamingSession(Protocol):
    def feed_pcm16(self, pcm: bytes) -> list[TranscriptResult]: ...

    def finish(self) -> TranscriptResult: ...


class StreamingBackend(Protocol):
    name: str
    model_id: str

    def new_session(self) -> StreamingSession: ...


def _pcm16_to_float32(pcm: bytes) -> np.ndarray:
    if len(pcm) % 2:
        raise ValueError("PCM16 payload length must be even")
    if not pcm:
        return np.empty((0,), dtype=np.float32)
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


class MockStreamingBackend:
    """Deterministic backend used by protocol and lifecycle tests."""

    name = "mock"
    model_id = "mock-streaming-asr"

    def new_session(self) -> MockStreamingSession:
        return MockStreamingSession()


class MockStreamingSession:
    def __init__(self) -> None:
        self._bytes = 0
        self._revision = 0

    def feed_pcm16(self, pcm: bytes) -> list[TranscriptResult]:
        _pcm16_to_float32(pcm)
        self._bytes += len(pcm)
        if self._bytes < 3200:
            return []
        self._revision += 1
        return [
            TranscriptResult(
                text="测试",
                delta="测试" if self._revision == 1 else "",
                is_final=False,
                revision=self._revision,
                audio_ms=self._bytes // 32,
                decode_ms=0.1,
                rtf=0.001,
            )
        ]

    def finish(self) -> TranscriptResult:
        self._revision += 1
        return TranscriptResult(
            text="测试",
            delta="" if self._bytes >= 3200 else "测试",
            is_final=True,
            revision=self._revision,
            audio_ms=self._bytes // 32,
            decode_ms=0.1,
            rtf=0.001,
        )


class FunASROnnxBackend:
    """One shared ONNX graph with isolated streaming state per utterance.

    ``funasr-onnx`` stores frontend state on the model object. We preserve a
    separate frontend per session and swap it only while holding a lock. This
    keeps model weights shared, supports interleaved streams correctly, and
    serializes CPU inference on small edge Hosts.
    """

    name = "onnx-cpu"

    def __init__(
        self,
        model_dir: Path,
        *,
        intra_op_threads: int = 4,
        chunk_size: tuple[int, int, int] = (5, 10, 5),
    ) -> None:
        from funasr_onnx.paraformer_online_bin import Paraformer
        from funasr_onnx.utils.frontend import WavFrontendOnline
        from funasr_onnx.utils.utils import read_yaml

        self.model_dir = model_dir
        self.model_id = "iic/paraformer-zh-streaming-onnx@2.0.5"
        self.chunk_size = chunk_size
        self.chunk_samples = chunk_size[1] * 960
        config = read_yaml(str(model_dir / "config.yaml"))
        self._frontend_type = WavFrontendOnline
        self._frontend_config = config["frontend_conf"]
        self._cmvn_file = str(model_dir / "am.mvn")
        self._model = Paraformer(
            model_dir=model_dir,
            quantize=True,
            device_id=-1,
            intra_op_num_threads=intra_op_threads,
            chunk_size=list(chunk_size),
        )
        self._lock = threading.Lock()

    def _new_frontend(self) -> Any:
        return self._frontend_type(cmvn_file=self._cmvn_file, **self._frontend_config)

    def new_session(self) -> FunASROnnxSession:
        return FunASROnnxSession(self)

    def infer(self, state: FunASROnnxSession, audio: np.ndarray, *, is_final: bool) -> list[str]:
        started = time.perf_counter()
        with self._lock:
            previous_frontend = self._model.frontend
            self._model.frontend = state._frontend
            try:
                raw = self._model(
                    audio,
                    param_dict={"cache": state._cache, "is_final": is_final},
                )
            finally:
                state._frontend = self._model.frontend
                self._model.frontend = previous_frontend
        state._last_decode_ms = (time.perf_counter() - started) * 1000.0
        state._total_decode_ms += state._last_decode_ms
        deltas: list[str] = []
        for item in raw or []:
            predictions = item.get("preds") if isinstance(item, dict) else None
            if isinstance(predictions, (tuple, list)) and predictions:
                text = predictions[0]
                if isinstance(text, str) and text:
                    deltas.append(text)
        return deltas


class FunASROnnxSession:
    def __init__(self, backend: FunASROnnxBackend) -> None:
        self._backend = backend
        self._frontend = backend._new_frontend()
        self._cache: dict[str, Any] = {}
        self._pending = np.empty((0,), dtype=np.float32)
        self._text = ""
        self._audio_samples = 0
        self._revision = 0
        self._last_decode_ms = 0.0
        self._total_decode_ms = 0.0
        self._finished = False

    def _result(self, delta: str, *, is_final: bool) -> TranscriptResult:
        self._revision += 1
        audio_ms = round(self._audio_samples / 16.0)
        audio_duration_ms = max(audio_ms, 1)
        return TranscriptResult(
            text=self._text,
            delta=delta,
            is_final=is_final,
            revision=self._revision,
            audio_ms=audio_ms,
            decode_ms=round(self._total_decode_ms, 3),
            rtf=round(self._total_decode_ms / audio_duration_ms, 5),
        )

    def feed_pcm16(self, pcm: bytes) -> list[TranscriptResult]:
        if self._finished:
            raise RuntimeError("utterance is already finished")
        samples = _pcm16_to_float32(pcm)
        self._audio_samples += len(samples)
        if samples.size:
            self._pending = np.concatenate((self._pending, samples))
        results: list[TranscriptResult] = []
        while self._pending.size >= self._backend.chunk_samples:
            chunk = self._pending[: self._backend.chunk_samples]
            self._pending = self._pending[self._backend.chunk_samples :]
            deltas = self._backend.infer(self, chunk, is_final=False)
            for delta in deltas:
                self._text += delta
                results.append(self._result(delta, is_final=False))
        return results

    def finish(self) -> TranscriptResult:
        if self._finished:
            raise RuntimeError("utterance is already finished")
        self._finished = True
        deltas = self._backend.infer(self, self._pending, is_final=True)
        self._pending = np.empty((0,), dtype=np.float32)
        delta = "".join(deltas)
        self._text += delta
        return self._result(delta, is_final=True)


def result_payload(
    result: TranscriptResult,
    *,
    stream_id: str,
    utterance_id: str,
    backend: StreamingBackend,
) -> dict[str, Any]:
    return {
        "type": "transcript",
        "stream_id": stream_id,
        "utterance_id": utterance_id,
        "revision": result.revision,
        "text": result.text,
        "delta": result.delta,
        "is_final": result.is_final,
        "language": "zh",
        "model_id": backend.model_id,
        "provider": backend.name,
        "audio_ms": result.audio_ms,
        "decode_ms": result.decode_ms,
        "rtf": result.rtf,
    }


def payload_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
