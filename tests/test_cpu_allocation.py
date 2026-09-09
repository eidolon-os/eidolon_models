"""The core allocation, and the two parsers that have to agree about it.

`deploy/cpu-allocation.env` is the board's single tuning point. Three runtimes
read it: two Python services pin themselves, and the chat model's launcher is a
POSIX shell script that hands `llama-server` a hex mask. The launcher cannot
import Python -- it is deliberately dependency-free, because it starts before
any venv is on the path -- so the taskset-list syntax genuinely exists twice.
These tests hold the second copy against the first, and hold the file against
what the services and the launcher actually read out of it.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from eidolon_models_host.cpu import hex_mask, parse_cpu_list

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOCATION = REPO_ROOT / "deploy" / "cpu-allocation.env"
LAUNCHER = REPO_ROOT / "scripts" / "eidolon-llm"

# Every list the allocation file could plausibly hold, plus the shapes the
# syntax allows that it does not currently use -- a single core, a bare pair,
# and the discontiguous set that is the whole reason the launcher speaks hex
# instead of `--cpu-range lo-hi`.
SPECS = ["0", "4", "0-3", "4-6", "4-7", "0-7", "4,5", "6,7", "0-3,7", "0,2,4,6"]


def _shell_cpu_mask(spec: str) -> tuple[str, int]:
    """Run the launcher's own `cpu_mask` and return (hex mask, thread count)."""
    body = LAUNCHER.read_text(encoding="utf-8")
    match = re.search(r"^cpu_mask\(\) \{.*?^\}", body, re.S | re.M)
    assert match, "scripts/eidolon-llm no longer defines cpu_mask()"
    result = subprocess.run(
        ["sh", "-c", f'{match.group(0)}\ncpu_mask "$1"', "sh", spec],
        capture_output=True,
        text=True,
        check=True,
    )
    mask, _, count = result.stdout.strip().partition(" ")
    return mask, int(count)


@pytest.mark.parametrize("spec", SPECS)
def test_launcher_mask_matches_the_python_parser(spec: str) -> None:
    cpus = parse_cpu_list(spec)
    assert _shell_cpu_mask(spec) == (hex_mask(cpus), len(cpus))


@pytest.mark.parametrize("spec", ["3-1", "", "   ", ","])
def test_launcher_refuses_what_the_python_parser_refuses(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_cpu_list(spec)
    with pytest.raises(subprocess.CalledProcessError):
        _shell_cpu_mask(spec)


def test_hex_mask_is_what_llama_cpp_reads() -> None:
    # Checked against the binary on the board: `--cpu-mask f --threads 4` put
    # the decode threads on cores 0-3 at 5.5 tok/s, the same as `taskset -c
    # 0-3`. Bit n is core n; no `0x`, lowercase, no padding.
    assert hex_mask(parse_cpu_list("0-3")) == "f"
    assert hex_mask(parse_cpu_list("4-6")) == "70"
    assert hex_mask(parse_cpu_list("0-7")) == "ff"
    assert hex_mask(parse_cpu_list("0-3,7")) == "8f"


def _allocation() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in ALLOCATION.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        values[name] = value
    return values


def test_every_assignment_is_a_variable_a_consumer_reads() -> None:
    """A line nobody reads is the failure this file had for a release.

    `EIDOLON_TTS_CPU_AFFINITY` sat here commented out while HOST-RK3588.md
    2.24 measured the board's headroom with TTS pinned -- so the shipped
    arrangement was one nothing had measured. Assignments and readers are
    checked against each other in both directions.
    """
    from eidolon_models_asr.config import CPU_AFFINITY_ENV as ASR_ENV
    from eidolon_models_tts.config import CPU_AFFINITY_ENV as TTS_ENV

    launcher = LAUNCHER.read_text(encoding="utf-8")
    # Only the core variables: the launcher also reads its port, host and
    # weights, and those are the unit's business, not this file's.
    readers = {ASR_ENV, TTS_ENV} | {
        name
        for name in re.findall(r"\$\{(EIDOLON_[A-Z_]+):-\}", launcher)
        if "CPU_AFFINITY" in name or name.endswith("_THREADS")
    }
    assert set(_allocation()) <= readers, "an assignment here is read by nobody"
    # `EIDOLON_LLM_THREADS` is deliberately commented out -- the count is
    # derived from the mask, and the variable is only an override for probing.
    assert readers <= set(_allocation()) | {"EIDOLON_LLM_THREADS"}


@pytest.mark.parametrize(
    "name",
    [
        "EIDOLON_ASR_CPU_AFFINITY",
        "EIDOLON_LLM_CPU_AFFINITY",
        "EIDOLON_LLM_PREFILL_CPU_AFFINITY",
        "EIDOLON_TTS_CPU_AFFINITY",
    ],
)
def test_allocated_values_parse(name: str) -> None:
    value = _allocation().get(name)
    assert value is not None, f"{name} is not assigned in deploy/cpu-allocation.env"
    if not value.strip():  # deliberately unpinned, as ASR is
        return
    cpus = parse_cpu_list(value)
    assert max(cpus) <= 7, f"{name}={value} names a core RK3588 does not have"


def test_the_two_clusters_are_not_shared_while_tts_is_speaking() -> None:
    """Decode and synthesis must not overlap; prefill and synthesis may.

    Measured, HOST-RK3588.md 2.29: sharing a cluster with sustained decode
    takes synthesis from rtf 0.94 to 1.10 and drains the playback buffer to
    -620 ms, which is audible. Prefill is allowed to overlap because it runs
    before there is any text to speak, and the one case where it does overlap
    -- a barge-in over the previous reply -- still held +272 ms.
    """
    allocation = _allocation()
    decode = parse_cpu_list(allocation["EIDOLON_LLM_CPU_AFFINITY"])
    synthesis = parse_cpu_list(allocation["EIDOLON_TTS_CPU_AFFINITY"])
    assert not (decode & synthesis), (
        f"decode {sorted(decode)} and synthesis {sorted(synthesis)} share a core"
    )


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="Linux only")
def test_tts_pins_itself_so_the_engine_inherits() -> None:
    from eidolon_models_tts.config import apply_cpu_affinity

    available = sorted(os.sched_getaffinity(0))
    try:
        assert apply_cpu_affinity(str(available[0])) == {available[0]}
        assert sorted(os.sched_getaffinity(0)) == [available[0]]
    finally:
        os.sched_setaffinity(0, available)
