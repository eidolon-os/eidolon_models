"""The argument vector handed to the engine, held to what it must contain.

This file exists because of what was missing from that vector rather than what
was wrong in it: the service never passed `--max-context`, so the ceiling on an
utterance was its text length instead of the model's context, and nothing
failed loudly — `/readyz` stayed 200 and short sentences were fine.

The flag is passed now, at the value the engine itself defaults to. Raising it
was tried and reverted: serve mode does not clear the RKLLM KV cache between
utterances, and a larger context gives that residue room to accumulate — the
fourth utterance starts reciting the voice prompt. `config.DEFAULT_MAX_CONTEXT`
carries the measurements and the condition for raising it. What is pinned here
is that the value is stated rather than inherited, and that it is not raised
without the cache fix.

An omission cannot be caught by reading the vector's own code, so the whole
vector is asserted here. Nothing needs an NPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eidolon_models_tts.config import DEFAULT_MAX_CONTEXT, Settings, load_settings
from eidolon_models_tts.engine import Engine

_MODEL = Path("/models/cosyvoice2")
_ENGINE = Path("/engine/cosyvoice2_streaming_pipeline")


def _command(**overrides: object) -> list[str]:
    settings = Settings(model_root=_MODEL, engine=_ENGINE, **overrides)  # type: ignore[arg-type]
    return Engine(settings).command()


def test_the_context_is_stated_rather_than_inherited() -> None:
    """The flag must be in the vector even though its value equals the
    engine's own default: a value nobody passes is a value nobody can find,
    and this one has a cost in both directions."""

    assert f"--max-context={DEFAULT_MAX_CONTEXT}" in _command()
    assert any(argument.startswith("--max-context=") for argument in _command())


def test_the_context_is_not_raised_without_the_cross_utterance_fix() -> None:
    """Raising this alone made output worse, not better.

    Serve mode reuses one RKLLM handle and never clears its KV cache; at 2048
    the residue accumulates and the fourth utterance recites the voice prompt,
    with steady rtf going 0.9 -> 1.5. Measured on the board, six utterances
    from a cold start: 288 gave 6/6 intact, 2048 gave 3/6.

    This assertion is here to be *deliberately* changed: when the engine's
    serve loop clears the cache per utterance, raise the value and this test
    together, and not before.
    """

    assert DEFAULT_MAX_CONTEXT == 288


def test_an_operator_can_override_the_context() -> None:
    assert "--max-context=1024" in _command(max_context=1024)


def test_the_context_comes_from_the_environment_like_every_other_knob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overridable the same way the NPU core masks are, so a Host that has
    decided differently says so in its environment rather than in this code."""

    monkeypatch.setenv("EIDOLON_TTS_MODEL_ROOT", str(_MODEL))
    monkeypatch.setenv("EIDOLON_TTS_ENGINE", str(_ENGINE))
    monkeypatch.setenv("EIDOLON_TTS_MAX_CONTEXT", "777")

    assert load_settings().max_context == 777


def test_the_context_defaults_when_the_environment_is_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIDOLON_TTS_MODEL_ROOT", str(_MODEL))
    monkeypatch.setenv("EIDOLON_TTS_ENGINE", str(_ENGINE))
    monkeypatch.delenv("EIDOLON_TTS_MAX_CONTEXT", raising=False)

    assert load_settings().max_context == DEFAULT_MAX_CONTEXT


def test_every_argument_the_engine_needs_is_present() -> None:
    """Asserted as a whole because the bug was an omission.

    The nine positionals are the engine's own order and it reads them by
    position, so a reordering is as wrong as a missing one.
    """

    command = _command()
    settings = Settings(model_root=_MODEL, engine=_ENGINE)

    assert command[:10] == [
        str(_ENGINE),
        str(settings.rkllm_model),
        str(settings.speech_head),
        str(settings.speech_embedding),
        str(settings.flow_encoder),
        str(settings.flow_estimator),
        str(settings.hift_root),
        str(settings.runtime_root),
        str(settings.flow_fixture),
        str(settings.output_root),
    ]

    options = set(command[10:])
    assert options == {
        f"--voice-profile-root={settings.voice_profile_root}",
        f"--text-frontend-root={settings.text_frontend_root}",
        "--head-core=0",
        "--encoder-core=2",
        "--flow-core=2",
        "--hift-core=2",
        f"--max-context={DEFAULT_MAX_CONTEXT}",
        "--serve",
        "--pcm-stream=-",
        "--sample",
        "--precompute-full-prompt",
    }


def test_serve_mode_and_the_pcm_channel_are_not_optional() -> None:
    """Without `--serve` the engine synthesizes once and exits; without
    `--pcm-stream=-` the audio goes to a file and the socket gets nothing."""

    command = _command()

    assert "--serve" in command
    assert "--pcm-stream=-" in command
