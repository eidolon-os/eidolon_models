from __future__ import annotations

import pytest

from eidolon_models_asr.config import Settings, detect_host_kind, resolve_backend


def test_auto_backend_is_portable_cpu_baseline() -> None:
    assert resolve_backend("auto") == "onnx-cpu"
    assert Settings().resolved_backend == "onnx-cpu"
    assert Settings().punctuation_enabled is True


def test_punctuation_can_be_disabled_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("EIDOLON_ASR_PUNCTUATION_ENABLED", "off")
    assert Settings.from_env().punctuation_enabled is False


def test_invalid_punctuation_environment_value_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("EIDOLON_ASR_PUNCTUATION_ENABLED", "sometimes")
    with pytest.raises(ValueError, match="EIDOLON_ASR_PUNCTUATION_ENABLED"):
        Settings.from_env()


def test_rknn_cannot_be_selected_without_artifact() -> None:
    with pytest.raises(ValueError, match="not available"):
        resolve_backend("rknn")


def test_host_kind_is_diagnostic_string() -> None:
    assert detect_host_kind()
