"""The synthesis WebSocket surface, against a fake engine.

The real engine needs an NPU, so it is replaced here by one that answers the
same three things: readiness, audio chunks, and a per-utterance report. What is
under test is the contract — the greeting, one request at a time, cancellation,
and whether a refusal says it is worth repeating — none of which is about NPUs.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from eidolon_models_tts import protocol
from eidolon_models_tts.config import Settings
from eidolon_models_tts.engine import EngineUnavailable, SynthesisFailed, UtteranceReport
from eidolon_models_tts.service import create_app
from pathlib import Path


def _ask(request_id: str, text: str) -> dict[str, object]:
    """What a client sends. Built here rather than imported: the builders live
    in the SDK contract, which this service deliberately does not depend on —
    and in these tests the client is us."""

    return {"type": protocol.SYNTHESIZE, protocol.REQUEST_ID_FIELD: request_id, "text": text}


def _stop(request_id: str) -> dict[str, object]:
    return {"type": protocol.CANCEL, protocol.REQUEST_ID_FIELD: request_id}


class FakeEngine:
    """Says three 200 ms chunks and reports what it said."""

    def __init__(self) -> None:
        self.ready = True
        self.init_warmup_ms = 2870.0
        self.spoken: list[str] = []
        self.raise_on_next: Exception | None = None
        self.chunks = 3
        self.report_extra: dict[str, str] = {}

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def synthesize(self, text: str) -> AsyncIterator[bytes | UtteranceReport]:
        self.spoken.append(text)
        if self.raise_on_next is not None:
            error, self.raise_on_next = self.raise_on_next, None
            raise error
        chunk = b"\x01\x02" * 2400
        for _ in range(self.chunks):
            yield chunk
        yield UtteranceReport(
            {
                "status": "PASS",
                "pcm_stream_bytes": str(len(chunk) * self.chunks),
                "audio_seconds": "0.6",
                "ttft_first_pcm_ms": "2100.5",
                "steady_pcm_rtf": "0.84",
                "underrun_count": "1",
                "profile_precompute_ms": "764.0",
                **self.report_extra,
            }
        )


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
async def client(aiohttp_client, engine: FakeEngine):
    settings = Settings(model_root=Path("/nonexistent"), engine=Path("/nonexistent/engine"))
    return await aiohttp_client(create_app(settings, engine))


async def test_the_greeting_states_the_version_and_the_audio(client) -> None:
    """On the connection the client is about to use, not from a readiness
    request that could describe a different process."""

    async with client.ws_connect(protocol.STREAM_PATH) as socket:
        greeting = await socket.receive_json()

    assert greeting["type"] == protocol.CONNECTED
    assert greeting[protocol.PROTOCOL_VERSION_FIELD] == protocol.PROTOCOL_VERSION
    assert greeting["sample_rate"] == 24000
    assert greeting["format"] == "pcm_s16le"


async def test_one_request_streams_audio_then_says_what_it_said(
    client, engine: FakeEngine
) -> None:
    async with client.ws_connect(protocol.STREAM_PATH) as socket:
        await socket.receive_json()
        await socket.send_json(_ask("r-1", "你好"))

        started = await socket.receive_json()
        assert started["type"] == protocol.SYNTHESIS_STARTED
        assert started[protocol.REQUEST_ID_FIELD] == "r-1"

        audio = 0
        while True:
            message = await socket.receive()
            if message.type.name == "BINARY":
                audio += len(message.data)
                continue
            finished = json.loads(message.data)
            break

    assert engine.spoken == ["你好"]
    assert finished["type"] == protocol.SYNTHESIS_FINISHED
    assert finished[protocol.REQUEST_ID_FIELD] == "r-1"
    assert audio == finished["pcm_bytes"] == 14400
    # The engine's own quality numbers reach the client: how close the buffer
    # came to empty is the one thing it cannot observe for itself.
    assert finished["late_chunks"] == 1
    assert finished["steady_rtf"] == 0.84


async def test_text_with_a_line_break_is_refused_rather_than_split(client) -> None:
    """The engine reads one utterance per line, so a newline inside a request
    would become a second utterance carrying this request's id."""

    async with client.ws_connect(protocol.STREAM_PATH) as socket:
        await socket.receive_json()
        await socket.send_json({"type": protocol.SYNTHESIZE, "request_id": "r", "text": "a\nb"})
        refusal = await socket.receive_json()

    assert refusal["type"] == protocol.ERROR
    assert refusal["code"] == protocol.ERROR_BAD_REQUEST
    assert refusal["retryable"] is False


async def test_text_past_the_limit_is_refused_as_too_long_and_not_retryable(client) -> None:
    async with client.ws_connect(protocol.STREAM_PATH) as socket:
        await socket.receive_json()
        await socket.send_json(
            _ask("r", "字" * (protocol.MAX_TEXT_CHARACTERS + 1))
        )
        refusal = await socket.receive_json()

    assert refusal["code"] == protocol.ERROR_TEXT_TOO_LONG
    assert refusal["retryable"] is False


async def test_an_engine_that_is_loading_is_worth_waiting_for(
    client, engine: FakeEngine
) -> None:
    """Distinct from an internal error: the Host is not ready *yet*, and a
    client told otherwise would fall back to a provider for good."""

    engine.raise_on_next = EngineUnavailable("still loading")
    async with client.ws_connect(protocol.STREAM_PATH) as socket:
        await socket.receive_json()
        await socket.send_json(_ask("r-1", "你好"))
        await socket.receive_json()  # synthesis_started
        refusal = await socket.receive_json()

    assert refusal["code"] == protocol.ERROR_ENGINE_UNAVAILABLE
    assert refusal["retryable"] is True
    assert refusal[protocol.REQUEST_ID_FIELD] == "r-1"


async def test_one_failed_utterance_leaves_the_stream_usable(
    client, engine: FakeEngine
) -> None:
    """The models are still loaded, so the next sentence is not the caller's
    problem to recover from by reconnecting."""

    engine.raise_on_next = SynthesisFailed("the engine reported no reason")
    async with client.ws_connect(protocol.STREAM_PATH) as socket:
        await socket.receive_json()
        await socket.send_json(_ask("r-1", "第一句"))
        await socket.receive_json()
        refusal = await socket.receive_json()
        assert refusal["code"] == protocol.ERROR_INTERNAL

        await socket.send_json(_ask("r-2", "第二句"))
        assert (await socket.receive_json())["type"] == protocol.SYNTHESIS_STARTED

    assert engine.spoken == ["第一句", "第二句"]


async def test_a_cancel_for_a_request_already_finished_is_not_an_error(client) -> None:
    """It crosses the finish of what it meant to stop. Answering with a refusal
    would make a client retry a turn the user already interrupted."""

    async with client.ws_connect(protocol.STREAM_PATH) as socket:
        await socket.receive_json()
        await socket.send_json(_ask("r-1", "你好"))
        # Drain to the finish, so the cancel below really does arrive after it.
        while True:
            message = await socket.receive()
            if message.type.name == "BINARY":
                continue
            if json.loads(message.data)["type"] == protocol.SYNTHESIS_FINISHED:
                break
        await socket.send_json(_stop("r-1"))
        await socket.send_json({"type": protocol.PING})
        assert (await socket.receive_json())["type"] == protocol.PONG


async def test_readiness_is_a_503_while_the_engine_warms(client, engine: FakeEngine) -> None:
    """A release's gate then waits instead of calling the Host activated: the
    port answers before the NPU graphs are warm."""

    engine.ready = False
    response = await client.get(protocol.READY_PATH)
    assert response.status == 503
    assert (await response.json())["ok"] is False

    engine.ready = True
    response = await client.get(protocol.READY_PATH)
    assert response.status == 200


async def test_info_states_that_this_host_says_one_thing_at_a_time(client) -> None:
    """One NPU and one resident engine behind it. A client that queued two
    would get two utterances that both break up."""

    document = await (await client.get(protocol.INFO_PATH)).json()
    assert document["concurrent_requests"] == 1
    assert document["capability"] == "local_tts"
    assert document["max_text_characters"] == protocol.MAX_TEXT_CHARACTERS


async def test_audio_sent_to_this_stream_is_refused(client) -> None:
    """The recognition stream takes audio and this one gives it. A client that
    connected to the wrong one is told so rather than ignored."""

    async with client.ws_connect(protocol.STREAM_PATH) as socket:
        await socket.receive_json()
        await socket.send_bytes(b"\x00\x01")
        refusal = await socket.receive_json()

    assert refusal["code"] == protocol.ERROR_BAD_REQUEST


async def test_the_finish_report_says_whether_anything_was_audible(
    client, engine: FakeEngine
) -> None:
    """`underrun_count` counts chunks that missed their own deadline, which for
    a producer near rtf 1 is almost every chunk by construction — it tracks the
    audio's length, not the listener's experience. A gap is audible only when
    the buffer goes negative, and that is a different field.

    Forwarding only the first one under the name `underruns` is how this
    service reported "1-3 dropouts per utterance" for runs whose real count was
    zero. So the audible one is forwarded, and the other is named for what it
    counts.
    """

    engine.report_extra = {"minimum_buffer_after_ms": "412.5", "underrun_count": "3"}
    async with client.ws_connect(protocol.STREAM_PATH) as socket:
        await socket.receive_json()
        await socket.send_json(_ask("r-1", "你好"))
        while True:
            message = await socket.receive()
            if message.type.name == "BINARY":
                continue
            payload = json.loads(message.data)
            # Not the first text frame: that is `synthesis_started`. Breaking on
            # it is the same reading mistake this whole test is about.
            if payload["type"] == protocol.SYNTHESIS_STARTED:
                continue
            finished = payload
            break

    assert finished["minimum_buffer_ms"] == 412.5
    assert finished["late_chunks"] == 3
    # Not under a name a reader would take for audible dropouts.
    assert "underruns" not in finished
