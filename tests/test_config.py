from __future__ import annotations

import pytest

from eidolon_models_asr.config import Settings, detect_host_kind, resolve_backend


def test_auto_backend_is_portable_cpu_baseline() -> None:
    assert resolve_backend("auto") == "onnx-cpu"
    assert Settings().resolved_backend == "onnx-cpu"
    assert Settings().offline_enabled is True
    assert Settings().punctuation_enabled is True
    assert Settings().max_connections == 64
    assert Settings().realtime_slots == 2
    assert Settings().max_queued_utterances == 6
    assert Settings().max_utterance_seconds == 60
    assert Settings().max_queue_wait_seconds == 10


def test_punctuation_can_be_disabled_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("EIDOLON_ASR_PUNCTUATION_ENABLED", "off")
    assert Settings.from_env().punctuation_enabled is False


def test_offline_can_be_disabled_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("EIDOLON_ASR_OFFLINE_ENABLED", "off")
    assert Settings.from_env().offline_enabled is False


def test_invalid_punctuation_environment_value_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("EIDOLON_ASR_PUNCTUATION_ENABLED", "sometimes")
    with pytest.raises(ValueError, match="EIDOLON_ASR_PUNCTUATION_ENABLED"):
        Settings.from_env()


def test_invalid_offline_environment_value_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("EIDOLON_ASR_OFFLINE_ENABLED", "sometimes")
    with pytest.raises(ValueError, match="EIDOLON_ASR_OFFLINE_ENABLED"):
        Settings.from_env()


def test_capacity_can_be_configured_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("EIDOLON_ASR_MAX_CONNECTIONS", "12")
    monkeypatch.setenv("EIDOLON_ASR_REALTIME_SLOTS", "3")
    monkeypatch.setenv("EIDOLON_ASR_MAX_QUEUED_UTTERANCES", "4")
    monkeypatch.setenv("EIDOLON_ASR_MAX_UTTERANCE_SECONDS", "45")
    monkeypatch.setenv("EIDOLON_ASR_MAX_QUEUE_WAIT_SECONDS", "7.5")
    settings = Settings.from_env()
    assert settings.max_connections == 12
    assert settings.realtime_slots == 3
    assert settings.max_queued_utterances == 4
    assert settings.max_utterance_seconds == 45
    assert settings.max_queue_wait_seconds == 7.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_connections": 0},
        {"realtime_slots": 0},
        {"max_queued_utterances": -1},
        {"max_utterance_seconds": 0},
        {"max_queue_wait_seconds": 0},
        {"max_connections": 2, "realtime_slots": 2, "max_queued_utterances": 1},
    ],
)
def test_invalid_capacity_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_rknn_cannot_be_selected_without_artifact() -> None:
    with pytest.raises(ValueError, match="not available"):
        resolve_backend("rknn")


def test_host_kind_is_diagnostic_string() -> None:
    assert detect_host_kind()
