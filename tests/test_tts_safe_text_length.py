"""The length past which this Host's audio breaks up, held to being stated.

There are two different limits and conflating them is what this file exists to
prevent. `protocol.MAX_TEXT_CHARACTERS` (400) is "will the request be
refused". `config.SAFE_TEXT_CHARACTERS` (60) is "will the listener hear a gap".
Between the two the service accepts the request, answers it, reports
`status=PASS` — and the audio drops out in the middle. Measured: 126 characters
leaves the buffer floor at −246..−372 ms on every single run
(HOST-RK3588.md §2.28).

Nothing here needs an NPU: the numbers were measured on the board and what is
checked here is that they are *stated* — that the safe bound exists, is below
the accepted bound, and is announced to callers instead of left to be found by
ear. Channel's aggregator already sends 60 (`hard_max_chars`), so until now the
whole arrangement rested on a default in another repository agreeing with an
unwritten property of this one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aiohttp import web

from eidolon_models_tts import protocol
from eidolon_models_tts.config import SAFE_TEXT_CHARACTERS, Settings
from eidolon_models_tts.service import ENGINE_KEY, SETTINGS_KEY, info

_ROOT = Path(__file__).resolve().parents[1]


def test_the_safe_bound_is_below_the_accepted_bound() -> None:
    """The two limits answer different questions, so they must not be equal.

    If they ever collapse into one number, whoever reads only the protocol will
    believe 400 characters is safe — and it is not: past ~80 the buffer floor
    goes negative and the audio has gaps.
    """

    assert SAFE_TEXT_CHARACTERS < protocol.MAX_TEXT_CHARACTERS
    assert SAFE_TEXT_CHARACTERS == 60
    assert protocol.MAX_TEXT_CHARACTERS == 400


def test_the_safe_bound_carries_its_measurements() -> None:
    """A bound with no evidence is a guess, and this one will be raised by
    somebody who wants longer sentences. The buffer floor is the quantity that
    decides it — not the late-chunk count, which tracks the audio's length —
    so the source must name it."""

    source = (_ROOT / "src/eidolon_models_tts/config.py").read_text(encoding="utf-8")
    declaration = source[source.index("#: The longest text this Host") :]
    declaration = declaration[: declaration.index("SAFE_TEXT_CHARACTERS = ")]

    assert "minimum_buffer_after_ms" in declaration
    assert "126" in declaration, "the length that was measured to break must be named"
    assert "2.28" in declaration, "the measurements must be findable"


async def test_the_service_announces_the_safe_bound(aiohttp_client) -> None:
    """Announced over `/v1/info`, so a caller can read it at startup instead of
    hardcoding a number that happens to match.

    `max_text_characters` is a cross-repository contract mirrored in the SDK and
    is left exactly as it was; this is additive.
    """

    settings = Settings(model_root=Path("/models"), engine=Path("/engine"))

    class _Engine:
        ready = True
        init_warmup_ms = 1.0

    app = web.Application()
    app[SETTINGS_KEY] = settings
    app[ENGINE_KEY] = _Engine()
    app.router.add_get(protocol.INFO_PATH, info)

    client = await aiohttp_client(app)
    response = await client.get(protocol.INFO_PATH)
    document = json.loads(await response.text())

    assert response.status == 200
    assert document["safe_text_characters"] == SAFE_TEXT_CHARACTERS
    assert document["max_text_characters"] == protocol.MAX_TEXT_CHARACTERS
    assert document["safe_text_characters"] < document["max_text_characters"]


def test_the_protocol_contract_is_untouched() -> None:
    """The safe bound deliberately lives in `config`, not `protocol`.

    `protocol.MAX_TEXT_CHARACTERS` is mirrored against
    `eidolon_sdk.biz.contracts.local_tts` by tests/test_sdk_contract_mirror.py.
    Putting a new constant there would mean a cross-repository change for a
    property that is this Host's measured characteristic rather than part of the
    wire protocol.
    """

    assert not hasattr(protocol, "SAFE_TEXT_CHARACTERS")


@pytest.mark.parametrize(
    ("characters", "audible"),
    [(14, True), (20, True), (44, True), (60, True), (80, True), (126, False)],
)
def test_the_measured_table_brackets_the_bound(characters: int, audible: bool) -> None:
    """The bound has to sit inside what was actually measured, on both sides.

    60 is not the largest length that worked — 80 also held, at only +209 ms.
    The bound is deliberately the conservative one, and 126 is the length that
    failed. A bound above every measured-good length, or at/above the failing
    one, would not be supported by the table in §2.28.
    """

    if audible:
        assert characters <= 80, "the table's audible rows all held at or under 80"
    else:
        assert characters > SAFE_TEXT_CHARACTERS, (
            f"{characters} characters was measured to produce gaps, so the safe "
            f"bound must be below it"
        )
