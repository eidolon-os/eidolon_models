"""Which cores this Host gives a service, and how each runtime is told.

One board, one allocation, three consumers that cannot share code: two Python
services (ASR, TTS) pin themselves through `sched_setaffinity`, and the chat
model's launcher is a POSIX shell script handing `llama-server` its own
`--cpu-mask` flags. So the *decision* lives in
`deploy/cpu-allocation.env` and the *syntax* — a taskset-style CPU list —
lives here, with `hex_mask` as the bridge to the one consumer that speaks hex.
`tests/test_cpu_allocation.py` holds the shell's copy of the parser against
this one.
"""

from __future__ import annotations

import os

__all__ = ["apply_cpu_affinity", "effective_cpu_affinity", "hex_mask", "parse_cpu_list"]


def parse_cpu_list(spec: str) -> frozenset[int]:
    """Parse a taskset-style CPU list: ``4``, ``4,5``, ``0-3``, ``0-3,7``."""
    cpus: set[int] = set()
    for part in spec.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if "-" in chunk.lstrip("-"):
            low, _, high = chunk.partition("-")
            start, stop = int(low), int(high)
            if start > stop:
                raise ValueError(f"invalid CPU range {chunk!r}: start is above end")
            cpus.update(range(start, stop + 1))
        else:
            cpus.add(int(chunk))
    if not cpus:
        raise ValueError(f"no CPUs in {spec!r}")
    if any(cpu < 0 for cpu in cpus):
        raise ValueError(f"negative CPU index in {spec!r}")
    return frozenset(cpus)


def hex_mask(cpus: frozenset[int] | set[int]) -> str:
    """Render a CPU set as llama.cpp's ``--cpu-mask``: lowercase hex, no ``0x``.

    The only reason this exists rather than passing ``--cpu-range lo-hi``: a
    range cannot say ``0-3,7``, and the allocation file's syntax can.
    """
    if not cpus:
        raise ValueError("no CPUs to render")
    value = 0
    for cpu in cpus:
        value |= 1 << cpu
    return f"{value:x}"


def apply_cpu_affinity(env_name: str, spec: str | None = None) -> frozenset[int] | None:
    """Pin this process before anything derives a thread count from it.

    Call this once at startup, ahead of reading settings: a pool sized from
    ``os.process_cpu_count()`` follows the cores without a second knob to keep
    in sync. A service that spawns a worker pins itself for the same reason —
    the child inherits the mask, so there is nothing to pass down.

    Returns the mask applied, or ``None`` when the variable is unset -- in
    which case whatever the caller inherited (taskset, systemd ``CPUAffinity``,
    or nothing) is left untouched. Asking to pin on a platform that cannot is
    an error rather than a silent no-op, because a pin that quietly does
    nothing is what oversubscribes the pool.
    """
    raw = os.getenv(env_name) if spec is None else spec
    if raw is None or not raw.strip():
        return None
    try:
        cpus = parse_cpu_list(raw)
    except ValueError as exc:
        raise ValueError(f"{env_name}: {exc}") from exc
    if not hasattr(os, "sched_setaffinity"):
        raise ValueError(
            f"{env_name} is set to {raw!r} but this platform has no "
            "CPU affinity support (Linux only); unset it or run on the target host"
        )
    try:
        os.sched_setaffinity(0, cpus)
    except OSError as exc:
        total = os.cpu_count() or 0
        raise ValueError(
            f"{env_name}={raw!r} could not be applied ({exc}); "
            f"this host reports {total} CPUs, so valid indices are 0-{max(total - 1, 0)}"
        ) from exc
    return cpus


def effective_cpu_affinity() -> list[int] | None:
    """Return the CPUs this process may run on, or None where unsupported."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    return sorted(os.sched_getaffinity(0))
