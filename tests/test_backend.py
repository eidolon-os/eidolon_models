from __future__ import annotations

import numpy as np

from eidolon_models_asr.backend import FunASROnnxSession


class CorrectingBackend:
    chunk_samples = 9600

    def _new_frontend(self) -> object:
        return object()

    def infer(
        self,
        state: FunASROnnxSession,
        audio: np.ndarray,
        *,
        is_final: bool,
    ) -> list[str]:
        state._last_decode_ms = 1.0
        state._total_decode_ms += 1.0
        return ["上菜"] if is_final else []

    def recognize_offline(self, audio: np.ndarray) -> tuple[str, float]:
        assert audio.size == 1600
        return "上海", 2.0

    def restore_punctuation(self, text: str) -> tuple[str, float]:
        return f"{text}。", 0.5


def test_offline_final_replaces_streaming_text_before_punctuation() -> None:
    session = FunASROnnxSession(CorrectingBackend())  # type: ignore[arg-type]
    assert session.feed_pcm16(b"\x00\x00" * 1600) == []

    final = session.finish()

    assert final.text == "上海。"
    assert final.raw_text == "上海"
    assert final.streaming_text == "上菜"
    assert final.delta == "上菜"
    assert final.final_revised is True
    assert final.offline_decode_ms == 2.0
    assert final.punctuation_ms == 0.5
    assert final.total_inference_ms == 3.5
