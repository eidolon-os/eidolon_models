"""How this Host starts its chat model, held to what the board measured.

The launcher is shell, so what is checked here is the argument vector it builds
— which is the whole of the configuration, and every part of it was measured on
RK3588 rather than chosen.
"""

from __future__ import annotations

from pathlib import Path

LAUNCHER = (Path(__file__).resolve().parents[1] / "scripts/eidolon-llm").read_text(encoding="utf-8")


def test_reasoning_is_off_and_not_merely_budgeted() -> None:
    """`--reasoning-budget 0` lets the template start thinking and then cuts it
    off, so every reply came back beginning with a stray `</think>` and two
    newlines — measured on the board, 4 out of 4.

    Downstream that is not cosmetic: the local TTS refuses text containing a
    line break, because its engine reads one utterance per line. A Host running
    both local models could not speak its own replies, and anything that did
    would say the tag out loud.
    """

    assert "--reasoning off" in LAUNCHER
    assert "--reasoning-budget" not in LAUNCHER.split("# Qwen3 emits reasoning")[0]


def test_the_template_is_applied_at_all() -> None:
    """Without `--jinja` the model's own chat template is not used, and the
    reasoning flags have nothing to act on."""

    assert "--jinja" in LAUNCHER


def test_the_model_and_port_are_required_rather_than_defaulted() -> None:
    """A launcher that guessed either would start a server against the wrong
    weights, or on a port the Host's registry did not assign."""

    assert "EIDOLON_LLM_MODEL and EIDOLON_LLM_PORT are required" in LAUNCHER


def test_absent_weights_and_absent_server_each_say_which() -> None:
    """Two different operator actions, so two different messages. The service
    exits 2 either way, and the journal's first line is what distinguishes
    them."""

    assert "model weights are absent" in LAUNCHER
    assert "is absent; it is a pinned foundation artifact" in LAUNCHER


def test_prefill_and_decode_are_given_their_own_cores() -> None:
    """Two masks, because the two phases do not run at the same moment.

    Decode runs while the local TTS speaks the sentence before it, and
    synthesis has 15% of realtime to spare, so it must stay off the cluster
    synthesis was given. Prefill runs before there is any text to speak, so it
    may borrow all eight: 20.2 -> 86.2 tok/s, a production first token at 3.2 s
    instead of 13.8 (HOST-RK3588.md 2.29).

    `taskset` cannot express that -- it confines the whole process, which is
    exactly why prefill used to be stuck on four little cores -- so the flags
    are llama.cpp's own.
    """

    # The word still appears, naming the syntax the allocation file uses;
    # what must be gone is the invocation.
    assert "taskset -c" not in LAUNCHER
    assert "--cpu-mask" in LAUNCHER
    assert "--cpu-mask-batch" in LAUNCHER
    assert "EIDOLON_LLM_CPU_AFFINITY" in LAUNCHER
    assert "EIDOLON_LLM_PREFILL_CPU_AFFINITY" in LAUNCHER


def test_a_mask_is_never_passed_without_a_thread_count() -> None:
    """`--cpu-mask` alone leaves llama.cpp sizing its pool from the machine's
    core count, so four cores got eight threads and decode collapsed to 1.36
    tok/s -- slower than either arrangement the mask exists to choose between.
    Measured on the board. The count is derived from the mask for that reason,
    and the two are passed together or not at all."""

    assert "--cpu-mask --threads" in LAUNCHER
    assert "--cpu-mask-batch --threads-batch" in LAUNCHER


def test_the_decode_mask_is_strict() -> None:
    """Without it the kernel may migrate a decode thread onto a core this Host
    gave to synthesis, which is the one thing the mask is there to prevent."""

    assert "--cpu-strict 1" in LAUNCHER


PROBE = (Path(__file__).resolve().parents[1] / "scripts/llm-reasoning-probe").read_text(
    encoding="utf-8"
)


def test_the_flag_has_a_live_probe_behind_it_and_not_only_this_file() -> None:
    """Everything above reads the launcher as text, which is the right check
    for a flag and no check at all on the model. `--reasoning off` was first
    accepted on a hand-run of one path — non-streaming chat — while the product
    path is the Agent, through LiteLLM, streaming. The script is what closes
    that; this test is what keeps it committed."""

    assert PROBE
    assert "--reasoning off" in PROBE


def test_the_probe_covers_the_paths_a_turn_can_actually_take() -> None:
    """Named individually rather than counted, so that dropping one is a
    deletion someone has to make on purpose. Non-streaming was the only path
    checked the first time, and that is the whole reason this exists."""

    for leg in (
        "http-stream",
        "http-plain",
        "bare-no-system",
        "http-completions",
        "litellm-stream",
        "litellm-multiturn",
    ):
        assert f'"{leg}"' in PROBE, leg


def test_the_probe_can_be_pointed_at_the_failure_it_is_for() -> None:
    """A green check that cannot go red proves nothing. `--expect-leak` runs
    the probe against a Host started the old way — `--reasoning-budget 0` — and
    passes only if something leaks. On the Mac build of the pinned llama.cpp it
    does: 17 of 32 bare replies, against 0 of 32 with the flag as it now is."""

    assert "--expect-leak" in PROBE
