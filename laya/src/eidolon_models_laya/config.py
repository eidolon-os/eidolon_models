"""Settings, read from ``EIDOLON_LAYA_*`` environment variables.

The launcher (`scripts/eidolon-laya`) sources ``laya/.env`` when it exists, and
systemd passes ``EnvironmentFile=``; nothing here reads files itself.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

BACKENDS = ("torch", "onnx", "rknn")
DEVICES = ("auto", "cpu", "mps", "cuda")

#: The laya project directory (the one holding ``models/``). The launcher exports
#: it; the fallback is right for a source checkout, which is how uv installs it.
HOME_ENV = "EIDOLON_LAYA_HOME"
DEFAULT_MODEL = Path("models") / "laya-multilingual" / "1c5edc17"
DEFAULT_PORT = 8771  # after ASR 8768, LLM 8769, TTS 8770


def laya_home(env: Mapping[str, str] = os.environ) -> Path:
    configured = env.get(HOME_ENV, "").strip()
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[2]


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0/true/false/yes/no/on/off, got {value!r}")


def _int(env: Mapping[str, str], name: str, default: int | None, minimum: int = 0) -> int | None:
    value = env.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _choice(env: Mapping[str, str], name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = (env.get(name) or default).strip().lower()
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}, got {value!r}")
    return value


def is_loopback(host: str) -> bool:
    if host in {"localhost", ""}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class Settings:
    backend: str = "torch"
    model_dir: Path = DEFAULT_MODEL
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    api_key: str | None = None
    allow_no_auth: bool = False
    #: torch only; ``auto`` picks cuda, then mps, then cpu.
    device: str = "auto"
    #: intra-op threads; 0 leaves the runtime's own default. On a host with one
    #: physical core and hyperthreads, 1 measured faster than 2.
    threads: int = 0
    #: None keeps the checkpoint's trained length (1024). Raising it works up to
    #: 8192 but costs quadratically on CPU; see README.
    max_len: int | None = None
    #: None keeps the checkpoint's own budget for instruction + options (256 for
    #: laya-multilingual). Many options share it, so a long device list needs more.
    head_max_len: int | None = None
    max_pending: int = 4
    max_questions: int = 32
    max_body_bytes: int = 256 * 1024

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        home = laya_home(env)
        model_dir = Path(env.get("EIDOLON_LAYA_MODEL_DIR", "").strip() or home / DEFAULT_MODEL)
        if not model_dir.is_absolute():
            model_dir = home / model_dir
        return cls(
            backend=_choice(env, "EIDOLON_LAYA_BACKEND", "torch", BACKENDS),
            model_dir=model_dir,
            host=(env.get("EIDOLON_LAYA_HOST") or "127.0.0.1").strip(),
            port=_int(env, "EIDOLON_LAYA_PORT", DEFAULT_PORT, minimum=1),
            api_key=(env.get("EIDOLON_LAYA_API_KEY") or "").strip() or None,
            allow_no_auth=_bool(env, "EIDOLON_LAYA_ALLOW_NO_AUTH", False),
            device=_choice(env, "EIDOLON_LAYA_DEVICE", "auto", DEVICES),
            threads=_int(env, "EIDOLON_LAYA_THREADS", 0),
            max_len=_int(env, "EIDOLON_LAYA_MAX_LEN", None, minimum=64),
            head_max_len=_int(env, "EIDOLON_LAYA_HEAD_MAX_LEN", None, minimum=32),
            max_pending=_int(env, "EIDOLON_LAYA_MAX_PENDING", 4, minimum=1),
            max_questions=_int(env, "EIDOLON_LAYA_MAX_QUESTIONS", 32, minimum=1),
            max_body_bytes=_int(env, "EIDOLON_LAYA_MAX_BODY_BYTES", 256 * 1024, minimum=1024),
        )

    def with_overrides(self, **overrides: object) -> Settings:
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    def validate_exposure(self) -> None:
        """Refuse to listen beyond loopback without a key, unless told to on purpose."""
        if self.api_key or self.allow_no_auth or is_loopback(self.host):
            return
        raise ValueError(
            f"refusing to listen on {self.host} without EIDOLON_LAYA_API_KEY; "
            "set a key, or EIDOLON_LAYA_ALLOW_NO_AUTH=1 if the network is already trusted"
        )
