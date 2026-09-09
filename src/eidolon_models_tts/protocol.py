"""Versioned WebSocket message validation.

A checked mirror of ``eidolon_sdk.biz.contracts.local_tts``, which is where
this protocol is defined. Channel speaks the other end of it and imports the
contract directly — it already depends on the SDK. This service does not, and
deliberately: the SDK carries grpcio, livekit-api, sqlalchemy and cryptography,
and a local speech service on a constrained board has no business installing
any of them to learn a handful of strings.

`tests/test_sdk_contract_mirror.py` reads the SDK's source and holds the two
together, so nothing here depends on the SDK being installed.

The SDK is the source of truth. A protocol change starts there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1

#: What a Host must declare for this service to be on it at all.
CAPABILITY = "local_tts"

#: The port role the component reserves; a Host's registry resolves the number.
PORT_ROLE = "tts_stream"

STREAM_PATH = "/v1/stream"
READY_PATH = "/readyz"
INFO_PATH = "/v1/info"

# -- what the client sends ---------------------------------------------------

SYNTHESIZE = "synthesize"
CANCEL = "cancel"
PING = "ping"
CLOSE_STREAM = "close_stream"

# -- what this service sends -------------------------------------------------

CONNECTED = "connected"
SYNTHESIS_STARTED = "synthesis_started"
SYNTHESIS_FINISHED = "synthesis_finished"
SYNTHESIS_CANCELLED = "synthesis_cancelled"
PONG = "pong"
ERROR = "error"

PROTOCOL_VERSION_FIELD = "protocol_version"
REQUEST_ID_FIELD = "request_id"

# -- the audio this service sends -------------------------------------------

AUDIO_SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = "pcm_s16le"

# -- why this service refused ----------------------------------------------

ERROR_BUSY = "synthesis_in_progress"
ERROR_TEXT_TOO_LONG = "text_too_long"
ERROR_BAD_REQUEST = "bad_request"
ERROR_INTERNAL = "internal_error"
ERROR_ENGINE_UNAVAILABLE = "engine_unavailable"

RETRYABLE_ERROR_CODES = frozenset({ERROR_BUSY, ERROR_INTERNAL, ERROR_ENGINE_UNAVAILABLE})

MAX_TEXT_CHARACTERS = 400


class ProtocolError(ValueError):
    """A message this protocol does not contain."""


@dataclass(frozen=True)
class SynthesizeRequest:
    request_id: str
    text: str


def parse_synthesize(value: dict[str, Any]) -> SynthesizeRequest:
    if value.get("type") != SYNTHESIZE:
        raise ProtocolError(f"expected {SYNTHESIZE}")
    request_id = str(value.get(REQUEST_ID_FIELD, "")).strip()
    text = str(value.get("text", ""))
    if not request_id:
        raise ProtocolError(f"{REQUEST_ID_FIELD} is required")
    # Stripped rather than rejected for whitespace: a client that sent a
    # trailing newline meant the sentence, and the engine reads one line.
    text = text.strip()
    if not text:
        raise ProtocolError("text is required")
    if "\n" in text or "\r" in text:
        # The engine takes one utterance per line, so a newline inside one
        # request would silently become two utterances — with the second
        # attributed to this request's id.
        raise ProtocolError("text may not contain a line break")
    if len(text) > MAX_TEXT_CHARACTERS:
        raise ProtocolError(f"text exceeds {MAX_TEXT_CHARACTERS} characters")
    return SynthesizeRequest(request_id=request_id, text=text)


def parse_cancel(value: dict[str, Any]) -> str:
    if value.get("type") != CANCEL:
        raise ProtocolError(f"expected {CANCEL}")
    request_id = str(value.get(REQUEST_ID_FIELD, "")).strip()
    if not request_id:
        raise ProtocolError(f"{REQUEST_ID_FIELD} is required")
    return request_id
