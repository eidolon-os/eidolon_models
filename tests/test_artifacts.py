from __future__ import annotations

import json

import pytest

from eidolon_models_asr.artifacts import ArtifactError, verify_artifacts
from eidolon_models_asr.config import (
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_DIR,
    DEFAULT_OFFLINE_MANIFEST,
    DEFAULT_OFFLINE_MODEL_DIR,
    DEFAULT_PUNCTUATION_MANIFEST,
    DEFAULT_PUNCTUATION_MODEL_DIR,
)


def test_pinned_model_artifacts_are_complete() -> None:
    result = verify_artifacts(DEFAULT_MANIFEST, DEFAULT_MODEL_DIR)
    assert result["ok"] is True
    assert "model_quant.onnx" in result["checked_files"]
    assert "decoder_quant.onnx" in result["checked_files"]

    offline = verify_artifacts(DEFAULT_OFFLINE_MANIFEST, DEFAULT_OFFLINE_MODEL_DIR)
    assert offline["ok"] is True
    assert offline["checked_files"] == [
        "am.mvn",
        "config.yaml",
        "configuration.json",
        "model_quant.onnx",
        "tokens.json",
    ]

    punctuation = verify_artifacts(
        DEFAULT_PUNCTUATION_MANIFEST,
        DEFAULT_PUNCTUATION_MODEL_DIR,
    )
    assert punctuation["ok"] is True
    assert punctuation["checked_files"] == [
        "config.yaml",
        "configuration.json",
        "model_quant.onnx",
        "tokens.json",
    ]


def test_checksum_mismatch_is_rejected(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weight.onnx").write_bytes(b"changed")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "test",
                "source": {"revision": "fixed"},
                "files": {"weight.onnx": "0" * 64},
            }
        )
    )
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        verify_artifacts(manifest, model_dir)
