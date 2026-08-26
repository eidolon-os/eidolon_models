from __future__ import annotations

import pytest

from eidolon_models_asr.protocol import ProtocolError, parse_start


def test_start_contract() -> None:
    start = parse_start(
        {
            "type": "start",
            "stream_id": "room-1",
            "utterance_id": "turn-1",
            "sample_rate": 16000,
            "channels": 1,
            "format": "pcm_s16le",
        }
    )
    assert start.stream_id == "room-1"
    assert start.utterance_id == "turn-1"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_rate", 48000, "16000 Hz"),
        ("channels", 2, "mono"),
        ("format", "float32", "pcm_s16le"),
    ],
)
def test_invalid_audio_contract_is_rejected(field, value, message) -> None:
    payload = {
        "type": "start",
        "stream_id": "stream",
        "utterance_id": "utterance",
        field: value,
    }
    with pytest.raises(ProtocolError, match=message):
        parse_start(payload)
