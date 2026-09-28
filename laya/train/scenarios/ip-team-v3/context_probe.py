"""Token budget of text-only Ensemble decisions under the actual Laya tokenizer.

These are designed stress cases, not claims about measured user traffic. The
short utterance sizes also cover the synthetic Agent reply examples in the
repository; medium and long cases test default and extended discussion.
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from eidolon_models_laya.vendor.laya.common import build_sequence, serialize_state
from eidolon_laya_train.model import to_internal
from episodes import _options

ROOT = Path(__file__).resolve().parents[3]
TOKENIZER = ROOT / "train/runs/r14/checkpoint/tokenizer"
PHRASES = (
    "我们先讨论这件事，考虑用户刚提出的目标和已经公开的意见。",
    "这个建议有吸引力，不过时间和体力都需要留出余地。",
    "我听到前一位说的重点了，可以把方案改得更简单一些。",
    "如果信息还不够明确，应当先问清楚关键条件再继续。",
)


def stretch(length: int, index: int) -> str:
    phrase = PHRASES[index % len(PHRASES)]
    return (phrase * (length // len(phrase) + 1))[:length]


def case(name: str, members: int, public_messages: int, reply_chars: int, profile_chars: int):
    names = [f"成员{i}" for i in range(members)]
    profiles = {f"M{i}": f"{names[i]}；{stretch(profile_chars, i)}" for i in range(members)}
    original = "为明天半天的轻松活动出主意，听过大家的建议后再决定。"
    history = [{"author_kind": "user", "author_id": "user", "author_name": "用户", "text": original}]
    for i in range(1, public_messages):
        author = (i - 1) % members
        history.append({"author_kind": "companion", "author_id": f"M{author}", "author_name": names[author],
                        "text": f"第{i}次公开发言：{stretch(reply_chars, i)}"})
    latest = history[-1]
    controls = {"allowed_actions": ["respond", "clarify", "wait", "finish"],
                "remaining_replies": 8}
    verbose = {"user_request": original, "scene_goal": "给出一个具体但不冗长的建议",
               "trigger": latest, "recent_public_messages": history,
               "candidates": profiles, **controls}
    # The pinned request and trigger occur once; prior messages remain in
    # chronological order. This is an experiment projection, not a runtime edit.
    compact = {"user_request": original, "scene_goal": "给出一个具体但不冗长的建议",
               "candidates": profiles, **controls,
               "trigger": latest, "prior_public_messages": history[1:-1]}
    moves = _options(profiles, controls["allowed_actions"])
    question = {"type": "choice", "instructions": "选出下一步动作及至多一位成员", "criteria": moves}
    return name, verbose, compact, question


def measure(tok, state: dict, question: dict, window: int, head_cap: int) -> dict:
    internal = to_internal(question)
    all_state = tok(serialize_state(state), add_special_tokens=False)["input_ids"]
    empty, _ = build_sequence(tok, "", internal, window, head_cap)
    head_len = len(empty)  # includes the final separator reserved after state
    room = max(0, window - head_len)
    ids, actual_markers = build_sequence(tok, state, internal, window, head_cap)
    used = min(len(all_state), room)
    return {"window": window, "head_cap": head_cap, "head_tokens": head_len,
            "state_tokens": len(all_state), "state_room": room, "retained_state_tokens": used,
            "state_complete": len(all_state) <= room, "options_complete": len(actual_markers) == len(question["criteria"]),
            "input_tokens": len(ids), "retained_fraction": round(used / len(all_state), 3)}


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    cases = [case("short", 2, 2, 80, 60),
             case("normal", 4, 4, 120, 100),
             case("default_budget", 4, 9, 160, 120),
             case("long_discussion", 4, 13, 220, 160),
             case("many_members", 6, 9, 160, 120),
             case("wire_max_history", 4, 64, 160, 120)]
    results = []
    for name, verbose, compact, question in cases:
        for projection, state in (("verbose_v2", verbose), ("compact_trial", compact)):
            for window in (768, 1024, 1536, 2048):
                results.append({"case": name, "projection": projection,
                                **measure(tok, state, question, window, 256)})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
