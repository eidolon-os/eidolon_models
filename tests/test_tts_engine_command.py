"""The argument vector handed to the engine, held to what it must contain.

This file exists because of what was missing from that vector rather than what
was wrong in it. The engine defaults `--max-context` to 288; the service never
passed the flag, so the ceiling on an utterance was its text length instead of
the model's context. 44 characters truncated mid-sentence every single time, 80
produced no audio at all, and nothing failed loudly — `/readyz` stayed 200 and
short sentences were fine. HOST-RK3588.md §2.20 had already recorded that this
argument must be passed; the hand-run benchmarks passed it and the service did
not, which is why its long-form behaviour never matched them.

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


def test_the_context_is_passed_and_is_the_model_s_own() -> None:
    """The regression this file was written for.

    2048 rather than a tuned number: the RKLLM body is
    `qwen2_body_w8a8_c2_ctx2048`, so this is what the weights hold, and the
    engine's 288 is below what a single sentence needs.
    """

    assert "--max-context=2048" in _command()
    assert DEFAULT_MAX_CONTEXT == 2048


def test_the_engine_default_is_never_what_takes_effect() -> None:
    """288 is the value that truncated every long utterance. If this vector
    ever stops carrying the flag, that default silently returns."""

    assert any(argument.startswith("--max-context=") for argument in _command())
    assert "--max-context=288" not in _command()


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
        "--max-context=2048",
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
