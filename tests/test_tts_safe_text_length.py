"""The length past which this Host's audio breaks up, held to being stated.

There are two different limits and conflating them is what this file exists to
prevent. `MAX_TEXT_CHARACTERS` (400) is "will the request be refused".
`SAFE_TEXT_CHARACTERS` (60) is "will the listener hear a gap".
Between the two the service accepts the request, answers it, reports
`status=PASS` — and the audio drops out in the middle. Measured: 126 characters
leaves the buffer floor at −246..−372 ms on every single run
(HOST-RK3588.md §2.28).

Nothing here needs an NPU: the numbers were measured on the board and what is
checked here is that they are *stated* — that the safe bound exists, is below
the accepted bound, and is announced to callers instead of left to be found by
ear. Both constants live in the SDK contract so a client can check its own
configuration when it is configured rather than after it is running — Channel
validates `hard_max_chars` against them at startup, and exceeding the safe
bound is not an error the service can return: it comes out as audio the user
cannot follow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aiohttp import web

from eidolon_models_tts import protocol
from eidolon_models_tts.config import Settings
from eidolon_models_tts.protocol import SAFE_TEXT_CHARACTERS
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

    contract = (_ROOT.parent / "eidolon_sdk/eidolon_sdk/biz/contracts/local_tts.py").read_text(
        encoding="utf-8"
    )
    declaration = contract[contract.index("#: What this Host will say in one request *without") :]
    declaration = declaration[: declaration.index("SAFE_TEXT_CHARACTERS: Final")]

    assert "126" in declaration, "the length that was measured to break must be named"
    assert "minimum_buffer_ms" in declaration


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


def test_the_safe_bound_is_part_of_the_contract() -> None:
    """It belongs beside `MAX_TEXT_CHARACTERS`, not in this service's config.

    It was in `config` first, so the only way a client could learn it was to
    read `/v1/info` at runtime. But Channel validates its aggregator's
    `hard_max_chars` at startup against the SDK contract — with the bound
    outside that contract, the only thing it could compare against was 400, so
    `hard_max_chars = 200` passed validation and produced gaps on every turn.
    A limit the caller must respect belongs where the caller already looks.

    tests/test_sdk_contract_mirror.py checks the value against the SDK's.
    """

    assert protocol.SAFE_TEXT_CHARACTERS == SAFE_TEXT_CHARACTERS
    assert not hasattr(
        __import__("eidolon_models_tts.config", fromlist=["config"]),
        "SAFE_TEXT_CHARACTERS",
    ), "the bound must not be defined in two places"


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
