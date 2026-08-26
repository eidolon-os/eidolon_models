from __future__ import annotations

import wave

import pytest

from eidolon_models_asr.artifacts import verify_artifacts
from eidolon_models_asr.backend import CTTransformerPunctuationRestorer, FunASROnnxBackend
from eidolon_models_asr.config import (
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_DIR,
    DEFAULT_PUNCTUATION_MANIFEST,
    DEFAULT_PUNCTUATION_MODEL_DIR,
    PROJECT_ROOT,
)


@pytest.mark.model
def test_real_chinese_streaming_inference() -> None:
    verify_artifacts(DEFAULT_MANIFEST, DEFAULT_MODEL_DIR)
    verify_artifacts(DEFAULT_PUNCTUATION_MANIFEST, DEFAULT_PUNCTUATION_MODEL_DIR)
    punctuation = CTTransformerPunctuationRestorer(
        DEFAULT_PUNCTUATION_MODEL_DIR,
        intra_op_threads=2,
    )
    backend = FunASROnnxBackend(
        DEFAULT_MODEL_DIR,
        intra_op_threads=2,
        punctuation=punctuation,
    )
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
    assert final.raw_text is not None
    assert len(final.text) >= 8
    assert any(fragment in final.text for fragment in ("语音", "识别", "模型", "体验"))
    assert final.audio_ms > 5000
    assert final.decode_ms >= interims[-1].decode_ms > 0
    assert 0 < final.rtf < 1
    assert final.text.endswith("。")
    assert final.raw_text == final.text[:-1]
    assert final.punctuation_ms > 0
    assert final.total_inference_ms > final.decode_ms
    assert final.total_rtf > final.rtf


@pytest.mark.model
def test_real_chinese_punctuation_quality() -> None:
    verify_artifacts(DEFAULT_PUNCTUATION_MANIFEST, DEFAULT_PUNCTUATION_MODEL_DIR)
    punctuation = CTTransformerPunctuationRestorer(
        DEFAULT_PUNCTUATION_MODEL_DIR,
        intra_op_threads=2,
    )
    cases = {
        "欢迎大家来体验达摩院推出的语音识别模型": "欢迎大家来体验达摩院推出的语音识别模型。",
        "今天天气很好我们出去散步吧然后一起吃饭": "今天天气很好，我们出去散步吧，然后一起吃饭。",
        "你好请问今天下午三点可以开会吗": "你好，请问今天下午三点可以开会吗？",
    }
    for raw, expected in cases.items():
        restored, elapsed_ms = punctuation.restore(raw)
        assert restored == expected
        assert elapsed_ms > 0
