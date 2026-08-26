from __future__ import annotations

import wave

import pytest

from eidolon_models_asr.artifacts import verify_artifacts
from eidolon_models_asr.backend import FunASROnnxBackend
from eidolon_models_asr.config import DEFAULT_MANIFEST, DEFAULT_MODEL_DIR, PROJECT_ROOT


@pytest.mark.model
def test_real_chinese_streaming_inference() -> None:
    verify_artifacts(DEFAULT_MANIFEST, DEFAULT_MODEL_DIR)
    backend = FunASROnnxBackend(DEFAULT_MODEL_DIR, intra_op_threads=2)
    session = backend.new_session()
    audio_path = PROJECT_ROOT / "tests" / "data" / "asr_example_zh.wav"
    with wave.open(str(audio_path), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        pcm = wav.readframes(wav.getnframes())
    interims = []
    for offset in range(0, len(pcm), 3200):
        interims.extend(session.feed_pcm16(pcm[offset : offset + 3200]))
    final = session.finish()
    assert interims
    assert final.is_final is True
    assert len(final.text) >= 8
    assert any(fragment in final.text for fragment in ("语音", "识别", "模型", "体验"))
    assert final.audio_ms > 5000
    assert final.decode_ms >= interims[-1].decode_ms > 0
    assert 0 < final.rtf < 1
