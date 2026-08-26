"""Offline model artifact verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ArtifactError(RuntimeError):
    """A required model artifact is missing or has changed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactError(f"manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid manifest JSON: {path}: {exc}") from exc
    if value.get("schema_version") != 1 or not isinstance(value.get("files"), dict):
        raise ArtifactError(f"unsupported manifest schema: {path}")
    return value


def verify_artifacts(manifest_path: Path, model_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    checked: list[str] = []
    for relative, expected in manifest["files"].items():
        candidate = model_dir / relative
        if not candidate.is_file():
            raise ArtifactError(f"required model file not found: {candidate}")
        actual = sha256_file(candidate)
        if actual != expected:
            raise ArtifactError(
                f"checksum mismatch for {candidate}: expected {expected}, got {actual}"
            )
        checked.append(relative)
    return {
        "ok": True,
        "model_id": manifest["model_id"],
        "revision": manifest["source"]["revision"],
        "checked_files": checked,
    }
