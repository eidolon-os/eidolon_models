from __future__ import annotations

import json

import pytest

from eidolon_models_asr.artifacts import ArtifactError, verify_artifacts
from eidolon_models_asr.config import DEFAULT_MANIFEST, DEFAULT_MODEL_DIR


def test_pinned_model_artifacts_are_complete() -> None:
    result = verify_artifacts(DEFAULT_MANIFEST, DEFAULT_MODEL_DIR)
    assert result["ok"] is True
    assert "model_quant.onnx" in result["checked_files"]
    assert "decoder_quant.onnx" in result["checked_files"]


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
