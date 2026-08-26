"""Versioned WebSocket message validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class StartMessage:
    stream_id: str
    utterance_id: str


def parse_start(value: dict[str, Any]) -> StartMessage:
    if value.get("type") not in {"start", "start_utterance"}:
        raise ProtocolError("expected start or start_utterance")
    if value.get("sample_rate", 16000) != 16000:
        raise ProtocolError("only 16000 Hz audio is supported")
    if value.get("channels", 1) != 1:
        raise ProtocolError("only mono audio is supported")
    if value.get("format", "pcm_s16le") != "pcm_s16le":
        raise ProtocolError("only pcm_s16le audio is supported")
    stream_id = str(value.get("stream_id", "")).strip()
    utterance_id = str(value.get("utterance_id", "")).strip()
    if not stream_id or not utterance_id:
        raise ProtocolError("stream_id and utterance_id are required")
    return StartMessage(stream_id=stream_id, utterance_id=utterance_id)
