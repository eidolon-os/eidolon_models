"""Typed configuration and Host-neutral backend selection."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "asr" / "paraformer-zh-streaming" / "2.0.5" / "model"
DEFAULT_MANIFEST = DEFAULT_MODEL_DIR.parent / "manifest.json"
DEFAULT_PUNCTUATION_MODEL_DIR = (
    PROJECT_ROOT / "asr" / "punc-ct-transformer-zh-cn" / "2024-09-25" / "model"
)
DEFAULT_PUNCTUATION_MANIFEST = DEFAULT_PUNCTUATION_MODEL_DIR.parent / "manifest.json"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of: 1, 0, true, false, yes, no, on, off")


def detect_host_kind() -> str:
    """Return a diagnostic Host kind without changing the service contract."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "macos-arm64"
    if system == "linux" and machine in {"arm64", "aarch64"}:
        model_path = Path("/proc/device-tree/model")
        try:
            model = model_path.read_text(errors="ignore").rstrip("\x00").lower()
        except OSError:
            model = ""
        if "raspberry pi 5" in model:
            return "raspberry-pi-5"
        if "rk3588" in model:
            return "rk3588"
        return "linux-arm64"
    return f"{system}-{machine}"


def resolve_backend(requested: str) -> str:
    """Resolve a backend name.

    Mac and Raspberry Pi deliberately use the same ONNX CPU backend. RKNN is
    reserved until an audited RKNN artifact exists; auto selection must never
    silently claim NPU acceleration.
    """
    normalized = requested.strip().lower()
    if normalized in {"", "auto"}:
        return "onnx-cpu"
    if normalized in {"onnx", "onnx-cpu", "mock"}:
        return "onnx-cpu" if normalized == "onnx" else normalized
    if normalized == "rknn":
        raise ValueError("rknn backend is not available until an RKNN artifact is shipped")
    raise ValueError(f"unsupported ASR backend: {requested!r}")


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8767
    backend: str = "auto"
    model_dir: Path = DEFAULT_MODEL_DIR
    manifest_path: Path = DEFAULT_MANIFEST
    punctuation_enabled: bool = True
    punctuation_model_dir: Path = DEFAULT_PUNCTUATION_MODEL_DIR
    punctuation_manifest_path: Path = DEFAULT_PUNCTUATION_MANIFEST
    intra_op_threads: int = max(1, min(4, os.cpu_count() or 1))
    chunk_size: tuple[int, int, int] = (5, 10, 5)
    max_binary_message_bytes: int = 1024 * 1024

    @property
    def resolved_backend(self) -> str:
        return resolve_backend(self.backend)

    @classmethod
    def from_env(cls) -> Settings:
        model_dir = Path(os.getenv("EIDOLON_ASR_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
        manifest = Path(os.getenv("EIDOLON_ASR_MANIFEST", str(model_dir.parent / "manifest.json")))
        punctuation_model_dir = Path(
            os.getenv("EIDOLON_ASR_PUNCTUATION_MODEL_DIR", str(DEFAULT_PUNCTUATION_MODEL_DIR))
        )
        punctuation_manifest = Path(
            os.getenv(
                "EIDOLON_ASR_PUNCTUATION_MANIFEST",
                str(punctuation_model_dir.parent / "manifest.json"),
            )
        )
        return cls(
            host=os.getenv("EIDOLON_ASR_HOST", "127.0.0.1"),
            port=int(os.getenv("EIDOLON_ASR_PORT", "8767")),
            backend=os.getenv("EIDOLON_ASR_BACKEND", "auto"),
            model_dir=model_dir,
            manifest_path=manifest,
            punctuation_enabled=_env_bool("EIDOLON_ASR_PUNCTUATION_ENABLED", True),
            punctuation_model_dir=punctuation_model_dir,
            punctuation_manifest_path=punctuation_manifest,
            intra_op_threads=int(
                os.getenv("EIDOLON_ASR_THREADS", str(max(1, min(4, os.cpu_count() or 1))))
            ),
        )
