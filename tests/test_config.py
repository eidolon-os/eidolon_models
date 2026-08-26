from __future__ import annotations

import pytest

from eidolon_models_asr.config import Settings, detect_host_kind, resolve_backend


def test_auto_backend_is_portable_cpu_baseline() -> None:
    assert resolve_backend("auto") == "onnx-cpu"
    assert Settings().resolved_backend == "onnx-cpu"


def test_rknn_cannot_be_selected_without_artifact() -> None:
    with pytest.raises(ValueError, match="not available"):
        resolve_backend("rknn")


def test_host_kind_is_diagnostic_string() -> None:
    assert detect_host_kind()
