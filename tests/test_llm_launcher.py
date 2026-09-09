"""How this Host starts its chat model, held to what the board measured.

The launcher is shell, so what is checked here is the argument vector it builds
— which is the whole of the configuration, and every part of it was measured on
RK3588 rather than chosen.
"""

from __future__ import annotations

from pathlib import Path

LAUNCHER = (Path(__file__).resolve().parents[1] / "scripts/eidolon-llm").read_text(
    encoding="utf-8"
)


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


def test_the_core_allocation_is_applied_when_this_host_states_one() -> None:
    """The A55 pinning is the arrangement HOST-RK3588.md 2.22-2.23 measured:
    the LLM keeps the little cores so TTS can keep the NPU. A launcher that
    ignored it would run on all eight and take the NPU's air."""

    assert "taskset -c" in LAUNCHER
    assert "EIDOLON_LLM_CPU_AFFINITY" in LAUNCHER
