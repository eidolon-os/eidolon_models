"""Settings for one Laya service, read from the services file.

Every Laya service a Host may run is described in one file, ``deploy/services.toml``: a table per
service, and under it a table per backend. A process reads only its own service. What differs
between Hosts is not written there: which services run is the Host's capability declaration, and the
backend follows from ``rknpu2`` in that same declaration — ``EIDOLON_HOST_CAPABILITIES``, which Ops
writes for every Host and is the one variable read here. A source checkout without it runs torch.

Command-line options override single values for a one-off run (a scratch port, another model).
"""

from __future__ import annotations

import ipaddress
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

BACKENDS = ("torch", "onnx", "rknn")
DEVICES = ("auto", "cpu", "mps", "cuda")

DEFAULT_MODEL = Path("models") / "laya-multilingual" / "1c5edc17"
DEFAULT_PORT = 8771  # after ASR 8768, LLM 8769, TTS 8770
#: smart_home and participation are the product's; systemone is the bare choice API over a base
#: model that a standalone server (deploy/ecs) exposes.
SERVICES = ("smart_home", "participation", "systemone")
HOST_CAPABILITIES_ENV = "EIDOLON_HOST_CAPABILITIES"


def laya_home() -> Path:
    """The laya project directory, holding ``deploy/`` and ``models/``, of a source checkout."""

    return Path(__file__).resolve().parents[2]


def default_services_file() -> Path:
    return laya_home() / "deploy" / "services.toml"


def host_capabilities(env: Mapping[str, str] = os.environ) -> frozenset[str]:
    return frozenset(
        item.strip() for item in env.get(HOST_CAPABILITIES_ENV, "").split(",") if item.strip()
    )


def backend_for(capabilities: frozenset[str]) -> str:
    """The NPU where the Host declares one; otherwise torch, whose weights only a checkout has.

    There is no CPU path for a released Host: ONNX on the RK3588's and the Pi 5's A76 cores
    measured 2.4–3.0 s per request (evals/opi5max-npu-20260929), past every caller's budget.
    """

    return "rknn" if "rknpu2" in capabilities else "torch"


def is_loopback(host: str) -> bool:
    if host in {"localhost", ""}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class Settings:
    #: Which table of ``deploy/services.toml`` this is; empty when built by hand (tools, tests).
    service: str = ""
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
    #: with ``options.ask_if``: start every question at once and answer as soon as the earlier
    #: answers settle what is needed (an utterance judged 无关 returns after the intent question).
    #: Only backends that run questions in parallel (``rknn``) can; the others stage them.
    speculative: bool = True
    #: rknn only: which NPU core loads which sequence buckets, as in
    #: ``"1:128,256,384,512|2:128,256"``. None = the backend's default placement.
    rknn_placement: str | None = None
    #: rknn only: the RKNN runtime library. None = ``librknnrt.so`` in the model directory (the
    #: pinned artifact carries it), falling back to the system's.
    rknn_library: Path | None = None
    max_pending: int = 4
    max_questions: int = 32
    max_body_bytes: int = 256 * 1024
    #: A task-qualified profile in the pinned model root is required to expose participation v2.
    enable_participation: bool = False

    @classmethod
    def for_service(
        cls,
        service: str,
        *,
        path: Path | None = None,
        capabilities: frozenset[str] | None = None,
        backend: str | None = None,
    ) -> Settings:
        """``service``'s table in the services file, for this Host's backend."""

        path = (path or default_services_file()).resolve()
        document = load_services(path)
        if service not in document["services"]:
            raise ValueError(
                f"{path}: no service {service!r}; it describes {', '.join(document['services'])}"
            )
        chosen = backend or backend_for(
            host_capabilities() if capabilities is None else capabilities
        )
        table = document["services"][service]
        if chosen not in table["backends"]:
            raise ValueError(
                f"{path}: service {service!r} has no {chosen!r} backend; "
                f"it describes {', '.join(table['backends']) or 'none'}"
            )
        values = {**document["defaults"], **table["common"], **table["backends"][chosen]}
        model_dir = Path(values.pop("model_dir"))
        api_key_file = values.pop("api_key_file", None)
        api_key = None
        if api_key_file is not None:
            api_key = Path(api_key_file).read_text(encoding="utf-8").strip() or None
            if api_key is None:
                raise ValueError(f"{path}: {service}.api_key_file {api_key_file} is empty")
        library = values.pop("rknn_library", None)
        return cls(
            service=service,
            backend=chosen,
            # A relative model directory is one of this checkout's (models/ in the laya project).
            model_dir=model_dir if model_dir.is_absolute() else laya_home() / model_dir,
            api_key=api_key,
            rknn_library=Path(library) if library else None,
            **{_FIELD_OF.get(key, key): value for key, value in values.items()},
        )

    def with_overrides(self, **overrides: object) -> Settings:
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    def validate_exposure(self) -> None:
        """Refuse to listen beyond loopback without a key, unless told to on purpose."""
        if self.api_key or self.allow_no_auth or is_loopback(self.host):
            return
        raise ValueError(
            f"refusing to listen on {self.host} without a key; set api_key_file for the service, "
            "or allow_no_auth = true if the network is already trusted"
        )

    def require_model(self) -> None:
        """Fail on a missing pinned model before loading anything, saying why it is missing."""
        if (self.model_dir / "manifest.json").is_file():
            return
        hint = (
            "the Host's pinned artifact is not installed"
            if self.backend == "rknn"
            else "torch weights live in a source checkout (git lfs pull); "
            "a released Host without rknpu2 does not run Laya"
        )
        raise FileNotFoundError(f"{self.service or 'laya'}: no model at {self.model_dir}: {hint}")


#: Keys of the services file that name a Settings field differently.
_FIELD_OF = {"placement": "rknn_placement", "participation": "enable_participation"}
_VALUE_KEYS = {
    "model_dir": (str,),
    "host": (str,),
    "port": (int,),
    "device": (str,),
    "threads": (int,),
    "max_len": (int,),
    "head_max_len": (int,),
    "speculative": (bool,),
    "placement": (str,),
    "rknn_library": (str,),
    "max_pending": (int,),
    "max_questions": (int,),
    "max_body_bytes": (int,),
    "participation": (bool,),
    "api_key_file": (str,),
    "allow_no_auth": (bool,),
}


def load_services(path: Path) -> dict:
    """Read and check the services file: every key one this module knows, every value its type."""

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"services file is unreadable: {path}: {exc}") from exc
    if raw.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")
    unknown = set(raw) - {"schema_version", "defaults", *SERVICES}
    if unknown:
        raise ValueError(
            f"{path}: unknown tables {', '.join(sorted(unknown))}; "
            f"services are {', '.join(SERVICES)}"
        )

    def values(table: object, label: str) -> dict:
        if not isinstance(table, dict):
            raise ValueError(f"{path}: {label} must be a table")
        out = {}
        for key, value in table.items():
            if key in BACKENDS:
                continue
            if key not in _VALUE_KEYS:
                raise ValueError(f"{path}: {label}.{key} is not a setting")
            if not isinstance(value, _VALUE_KEYS[key]) or (
                isinstance(value, bool) and bool not in _VALUE_KEYS[key]
            ):
                raise ValueError(f"{path}: {label}.{key} has the wrong type")
            if key == "device" and value not in DEVICES:
                raise ValueError(f"{path}: {label}.device must be one of {', '.join(DEVICES)}")
            out[key] = value
        return out

    defaults = values(raw.get("defaults", {}), "defaults")
    services = {}
    for name in SERVICES:
        if name not in raw:
            continue
        table = raw[name]
        common = values(table, name)
        backends = {b: values(table[b], f"{name}.{b}") for b in BACKENDS if b in table}
        for backend, chosen in backends.items():
            if "model_dir" not in {**common, **chosen}:
                raise ValueError(f"{path}: {name}.{backend} names no model_dir")
        services[name] = {"common": common, "backends": backends}
    return {"defaults": defaults, "services": services}
