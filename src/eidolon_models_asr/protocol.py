"""Versioned WebSocket message validation.

A checked mirror of ``eidolon_sdk.biz.contracts.local_asr``, which is where
this protocol is defined. Channel speaks the other end of it and imports the
contract directly — it already depends on the SDK. This service does not, and
deliberately: the SDK carries grpcio, livekit-api, sqlalchemy and cryptography,
and a local speech service on a constrained board has no business installing
any of them to learn four strings.

So this is the same arrangement the contract module prescribes for its
non-Python clients: "lightweight language-native mirrors ... cross-repository
contract tests keep those mirrors aligned with this source."
`tests/test_sdk_contract_mirror.py` is that test, and it reads the SDK's source
rather than importing it, so nothing here depends on the SDK being installed.

The SDK is the source of truth. A protocol change starts there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1

#: What a Host must declare for this service to be on it at all.
CAPABILITY = "local_asr"

#: The port role the component reserves; a Host's registry resolves the number.
PORT_ROLE = "asr_stream"

STREAM_PATH = "/v1/stream"
READY_PATH = "/readyz"
INFO_PATH = "/v1/info"

# What a client may send.
START_UTTERANCE = "start_utterance"
START_UTTERANCE_LEGACY = "start"
END_UTTERANCE = "end_utterance"
PING = "ping"
CLOSE_STREAM = "close_stream"

# What this service sends.
UTTERANCE_STARTED = "utterance_started"
TRANSCRIPT = "transcript"
PONG = "pong"
ERROR = "error"

# The one audio shape this service accepts.
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = "pcm_s16le"

# Why it refused.
ERROR_CAPACITY = "connection_capacity_exceeded"
ERROR_UTTERANCE_TOO_LONG = "utterance_too_long"
ERROR_BAD_REQUEST = "bad_request"
ERROR_INTERNAL = "internal_error"


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class StartMessage:
    stream_id: str
    utterance_id: str


def parse_start(value: dict[str, Any]) -> StartMessage:
    if value.get("type") not in {START_UTTERANCE_LEGACY, START_UTTERANCE}:
        raise ProtocolError(f"expected {START_UTTERANCE_LEGACY} or {START_UTTERANCE}")
    if value.get("sample_rate", AUDIO_SAMPLE_RATE) != AUDIO_SAMPLE_RATE:
        raise ProtocolError(f"only {AUDIO_SAMPLE_RATE} Hz audio is supported")
    if value.get("channels", AUDIO_CHANNELS) != AUDIO_CHANNELS:
        raise ProtocolError("only mono audio is supported")
    if value.get("format", AUDIO_FORMAT) != AUDIO_FORMAT:
        raise ProtocolError(f"only {AUDIO_FORMAT} audio is supported")
    stream_id = str(value.get("stream_id", "")).strip()
    utterance_id = str(value.get("utterance_id", "")).strip()
    if not stream_id or not utterance_id:
        raise ProtocolError("stream_id and utterance_id are required")
    return StartMessage(stream_id=stream_id, utterance_id=utterance_id)
