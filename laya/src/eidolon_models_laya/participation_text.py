"""Experimental text-only v5 projection; not registered by the release loader.

Shared by offline training and SDK shadow replay. Full character descriptions live
only in state. This classifier cannot generate a clarification task, so uncertainty
is abstention; no fixed clarification dialogue is synthesized.
"""
from __future__ import annotations

import json

from eidolon_sdk.biz.participation import DecisionRequest

from .participation import ContextTooLong, ModelChoice
from .sequence import build_sequence, render_options, to_internal

INSTRUCTIONS = (
    "选择一个合适的下一步。依据用户意图、角色关系和已公开内容选择回应者，不机械轮流。"
    "暂停等用户选wait；已完成选finish；无法可靠选择或需要消歧选abstain。"
    "安静陪伴可以文字回应；只提及名字不等于点名。"
)


def project_request(request: DecisionRequest, *, include_names: bool = False) -> tuple[dict, dict, dict[str, str]]:
    if not 1 <= len(request.candidates) <= 6 or request.context.summary or request.context.pending_requirements:
        raise ContextTooLong("v5 supports 1–6 candidates and complete unsummarized text")
    by_id = {c.companion_id: f"M{i}" for i, c in enumerate(request.candidates)}
    slots = {v: k for k, v in by_id.items()}
    history = request.context.recent_messages
    if not history or history[-1] != request.trigger:
        raise ContextTooLong("trigger must equal latest public message")
    if len({m.message_id for m in history}) != len(history):
        raise ContextTooLong("duplicate public message IDs")
    if any(m.message_id == request.user_request.message_id and m != request.user_request for m in history):
        raise ContextTooLong("pinned user message differs from public message with same ID")

    def public(message):
        if message.author_kind == "user":
            author = "user"
        elif message.author_kind == "companion" and message.author_id in by_id:
            author = by_id[message.author_id]
        else:
            raise ContextTooLong("unsupported public author")
        return {"author": author, "text": message.text}

    state = {
        "user_request": request.user_request.text,
        "scene_goal": request.scene_goal,
        "candidates": {by_id[c.companion_id]: {"name": c.display_name, "character": c.description}
                       for c in request.candidates},
        "allowed_actions": list(request.constraints.allowed_actions),
        "remaining_replies": request.constraints.remaining_replies,
        "prior_public_messages": [public(m) for m in history[:-1]
                                  if m.message_id != request.user_request.message_id],
        "trigger": ({"same_as_user_request": True}
                    if request.trigger.message_id == request.user_request.message_id else public(request.trigger)),
    }
    choices = {}
    if "respond" in request.constraints.allowed_actions:
        choices.update({f"respond:{slot}": f"由{slot}回应" for slot in slots})
        if include_names:
            choices.update({f"respond:{by_id[c.companion_id]}": f"由{c.display_name}（{by_id[c.companion_id]}）回应"
                            for c in request.candidates})
    for action, meaning in (("wait", "暂停等用户"), ("finish", "本轮已完成")):
        if action in request.constraints.allowed_actions:
            choices[action] = meaning
    choices["abstain"] = "无法可靠选择或需要澄清"
    return state, {"type": "choice", "instructions": INSTRUCTIONS, "criteria": choices}, slots


def check_capacity(tok, state, question, max_len=2048, head_max_len=256):
    internal = to_internal(question)
    if tok.mask_token in json.dumps([state, question], ensure_ascii=False):
        raise ContextTooLong("model control token in input")
    sizes = [len(tok.encode(" " + s)) for s in render_options(internal)]
    instruction_size = len(tok.encode("choice question: " + question["instructions"]))
    if max(sizes) > 48 or head_max_len - sum(s + 1 for s in sizes) < max(16, instruction_size):
        raise ContextTooLong("question or option would be truncated")
    ids, markers, truncated = build_sequence(tok, state, internal, max_len, head_max_len)
    if truncated or len(markers) != len(sizes):
        raise ContextTooLong("public context would be truncated")
    return len(ids)


class CompactTextPredictor:
    def __init__(self, engine, *, include_names: bool = False):
        self.engine = engine
        self.include_names = include_names

    def predict(self, request):
        state, question, slots = project_request(request, include_names=self.include_names)
        check_capacity(self.engine.tokenizer, state, question, self.engine.max_len, self.engine.head_max_len)
        prediction = self.engine.predict(state, {"move": question}, truncate_left=False)
        answer = prediction.answers.get("move", {})
        choice, confidence = answer.get("choice"), answer.get("answer_confidence")
        if prediction.truncated or choice not in question["criteria"] or type(confidence) not in (int, float):
            return ModelChoice("abstain")
        if choice.startswith("respond:"):
            return ModelChoice("respond", slots[choice.split(":")[1]], confidence=float(confidence))
        return ModelChoice(choice, confidence=float(confidence))
