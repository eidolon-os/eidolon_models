"""The verdict `scripts/llm-reasoning-probe` reaches, checked without a model.

The probe itself needs a Host that is serving and, for the product path, the
Agent's environment; neither is here. What is here is the part that decides —
given a reply, is this something the Host could say — and that part can be held
to the replies the board actually produced.

The division these tests pin is the probe's finding: a `</think>` at the head
of a reply is the launcher's business, and a line break in the middle of one is
not. Collapsing the two is how a Markdown answer gets read as a reasoning leak,
or a reasoning leak as a formatting quirk.
"""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

PROBE_PATH = Path(__file__).resolve().parents[1] / "scripts/llm-reasoning-probe"


def _load():
    # No `.py` on it: it is a command, and the tests read the command rather
    # than a copy of it.
    loader = SourceFileLoader("llm_reasoning_probe", str(PROBE_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: `@dataclass` resolves the defining module out
    # of `sys.modules`, and a module that is not there yet has no dict to read.
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


probe = _load()

#: What the board returned, 4 times out of 4, when the launcher passed
#: `--reasoning-budget 0`: the template started thinking, the budget cut it
#: off, and the closing tag stayed at the head of the content.
BOARD_LEAK = "</think>\n\n欢迎体验达摩院的语音服务。"


def test_the_board_s_own_leak_is_caught_as_a_reasoning_fault() -> None:
    faults = probe.reasoning_faults(probe.Reply("_", BOARD_LEAK, first_delta="</think>\n\n"))
    assert any("</think>" in fault for fault in faults)


def test_the_leak_is_caught_at_the_head_of_the_stream_too() -> None:
    """The tag arrives in the first delta. A leg that judged only the joined
    text would notice the same thing twice and the head not at all — which is
    the shape of the gap this probe was written to close."""

    faults = probe.reasoning_faults(probe.Reply("_", BOARD_LEAK, first_delta="</think>\n\n"))
    assert any("first delta" in fault for fault in faults)


def test_an_empty_reply_reads_as_a_model_that_thought_the_budget_away() -> None:
    """Reasoning left on returns no error — it returns nothing, having spent
    the completion inside its own head."""

    faults = probe.reasoning_faults(probe.Reply("_", "", finish_reason="length"))
    assert any("empty" in fault for fault in faults)


def test_reasoning_arriving_on_its_own_channel_is_a_fault_and_not_a_relief() -> None:
    """`reasoning_content` filled means the content field is the remainder of a
    reply rather than the whole of one. Clean-looking, and half a sentence."""

    faults = probe.reasoning_faults(
        probe.Reply("_", "答案是 1161。", reasoning="Let me work this out...")
    )
    assert any("remainder" in fault for fault in faults)


def test_one_spoken_line_is_faultless_on_both_halves() -> None:
    reply = probe.Reply("_", "27 乘以 43 等于 1161。", finish_reason="stop", first_delta="27")
    assert probe.reasoning_faults(reply) == []
    assert probe.speech_faults(reply) == []


def test_a_line_break_is_a_speech_fault_and_not_a_reasoning_one() -> None:
    """Asked a bare question this model answers in Markdown, thinking disabled
    or not. The launcher's flag has no say in it, so neither does the half of
    the verdict the flag answers for."""

    reply = probe.Reply("_", "我们来计算：\n\n27 × 43 = 1161。", finish_reason="stop")
    assert probe.reasoning_faults(reply) == []
    assert any("line break" in fault for fault in probe.speech_faults(reply))


def test_the_speech_half_is_the_local_tts_s_own_rule_rather_than_a_copy() -> None:
    """The check and the thing checked are the same function, so they cannot
    drift: the fault text is the gate's own refusal, quoted."""

    from eidolon_models_tts.protocol import ProtocolError, parse_synthesize

    text = "第一句。\n第二句。"
    with pytest.raises(ProtocolError) as refusal:
        parse_synthesize({"type": "synthesize", "request_id": "r", "text": text})
    assert str(refusal.value) in " ".join(probe.speech_faults(probe.Reply("_", text)))


def test_the_gate_s_length_cap_is_left_to_channel() -> None:
    """Channel's aggregator cuts a reply into sentences before the gate sees
    it, so a long reply is its business. Counting it here would fail a Host for
    answering at length on one line, which is a thing it may do."""

    long_line = "一" * (probe.MAX_TEXT_CHARACTERS + 50) + "。"
    assert probe.speech_faults(probe.Reply("_", long_line, finish_reason="stop")) == []


def test_a_stump_is_a_speech_fault() -> None:
    reply = probe.Reply("_", "27 乘以 43 等于", finish_reason="length")
    assert any("cut off" in fault for fault in probe.speech_faults(reply))


def test_the_contrast_leg_reports_line_breaks_without_failing_on_them() -> None:
    """`bare-no-system` exists to show what the flag does not cover. Holding it
    to the speakable half would file the finding as a failure."""

    leg = probe.Leg("bare-no-system", "_", speakable_required=False)
    probe.judge(leg, [probe.Reply("_", "第一行\n第二行。", finish_reason="stop")])
    assert leg.faults == []
    assert any("line break" in note for note in leg.notes)
    assert leg.leaked is False


def test_a_leak_on_the_contrast_leg_still_fails_it() -> None:
    """Only the speakable half is relaxed there. The reasoning half is the
    whole reason that leg is the sensitive one: with `--reasoning-budget 0`,
    17 of 32 bare replies leaked, while the same prompts under a system message
    asking for one spoken line leaked none at all — the instruction masks the
    fault, so the leg that sends no instruction is the one that can see it."""

    leg = probe.Leg("bare-no-system", "_", speakable_required=False)
    probe.judge(leg, [probe.Reply("_", BOARD_LEAK, finish_reason="stop")])
    assert leg.leaked is True
    assert leg.status == "fail"


def test_reasoning_that_never_enters_the_text_is_still_caught() -> None:
    """The failure mode the tag-hunt would miss entirely.

    Left to think, llama.cpp parses the block out into `reasoning_content`, so
    the text carries no tag and reads as a clean answer — while being the
    remainder of one, or empty. Measured against a Host started with no
    reasoning flag at all: `reasoning_content` came back 449 characters long
    and every leg went red, the two driving the Agent's provider included."""

    reply = probe.Reply("_", "小明现在有 15 个苹果。", reasoning="嗯，让我仔细想想这个问题。")
    assert probe.speech_faults(reply) == []
    assert any("remainder" in fault for fault in probe.reasoning_faults(reply))
