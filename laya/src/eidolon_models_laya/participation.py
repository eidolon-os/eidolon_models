"""IP-team participation v2 boundary for a task-qualified model.

This module deliberately does not map Laya's generic question/choice output to
the business contract. A trained model must supply a predictor with a frozen
input format, calibration policy and version, pinned by its manifest
(``task_profiles``); without one the HTTP endpoint is unavailable.
Two formats: ``ip-team-v3`` (one "move" question, clarify abstains) and
``participation-laya-v1`` (action / speaker / clarify_about, the participation release).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from eidolon_sdk.biz.participation import (
    DecisionRequest,
    DecisionResult,
    Proposal,
    Snapshot,
    validate_proposal,
)

from .engine import DecisionEngine
from .sequence import build_sequence, render_options, to_internal

MOVE_INSTRUCTIONS = (
    "根据当前用户请求、真实已公开发言、角色资料与剩余预算，决定唯一下一步。"
    "每步最多一位成员；用户要求安静或已获得足够答案时不要继续发言。"
)


class ContextTooLong(ValueError):
    """The complete SDK snapshot cannot fit the trained model's input budget."""


@dataclass(frozen=True)
class ModelChoice:
    """One model prediction, before SDK and current-authority validation."""

    action: str
    companion_id: str = ""
    instruction: str = ""
    confidence: float = 0.0
    truncated: bool = False


class ParticipationPredictor(Protocol):
    """Task-qualified inference; never truncate the supplied snapshot."""

    def predict(self, request: DecisionRequest) -> ModelChoice: ...


class ParticipationAdapter:
    def __init__(
        self,
        predictor: ParticipationPredictor,
        *,
        model_version: str,
        policy_version: str,
        min_confidence: float,
    ) -> None:
        if not model_version or not policy_version:
            raise ValueError("model and policy versions are required")
        if (
            type(min_confidence) not in (int, float)
            or not math.isfinite(min_confidence)
            or not 0 < min_confidence <= 1
        ):
            raise ValueError("min_confidence must be in (0, 1]")
        self.predictor = predictor
        self.model_version = model_version
        self.policy_version = policy_version
        self.min_confidence = min_confidence

    def decide(self, request: DecisionRequest) -> DecisionResult:
        snapshot = {key: getattr(request, key) for key in Snapshot.model_fields}

        def result(proposal: Proposal | None = None) -> DecisionResult:
            return DecisionResult(
                **snapshot,
                status="decided" if proposal is not None else "abstained",
                proposal=proposal,
                policy_version=self.policy_version,
                model_version=self.model_version,
            )

        try:
            choice = self.predictor.predict(request)
        except ContextTooLong:
            return result()
        if (
            choice.truncated
            or not math.isfinite(choice.confidence)
            or choice.confidence < self.min_confidence
        ):
            return result()
        if choice.action == "abstain":
            return result()
        participants = (choice.companion_id,) if choice.companion_id else ()
        try:
            proposal = Proposal(
                action=choice.action,
                participants=participants,
                instruction=choice.instruction,
            )
            decided = result(proposal)
            validate_proposal(request, decided)
        except ValueError:
            return result()
        return decided


class LayaMovePredictor:
    """The frozen ip-team-v3 classification input, with no silent token loss.

    The current head selects an action and a candidate slot. It cannot produce
    a concrete clarification question, so that choice always abstains.
    """

    def __init__(self, engine: DecisionEngine, *, max_candidates: int = 6) -> None:
        if type(max_candidates) is not int or not 1 <= max_candidates <= 6:
            raise ValueError("ip-team-v3 supports at most six candidates")
        self.engine = engine
        self.max_candidates = max_candidates

    def predict(self, request: DecisionRequest) -> ModelChoice:
        if (
            not request.candidates
            or len(request.candidates) > self.max_candidates
            or request.context.summary
            or request.context.pending_requirements
        ):
            raise ContextTooLong("snapshot has unsupported or over-capacity context")

        members = {}
        by_id = {}
        for index, candidate in enumerate(request.candidates):
            slot = f"M{index}"
            by_id[candidate.companion_id] = (slot, candidate.display_name)
            description = candidate.description
            if candidate.display_name and not description.startswith(candidate.display_name):
                description = f"{candidate.display_name}；{description}"
            members[slot] = description

        def public(message):
            if message.author_kind == "user":
                author_id, author_name = "user", "用户"
            elif message.author_kind == "companion" and message.author_id in by_id:
                author_id, author_name = by_id[message.author_id]
            else:
                raise ContextTooLong("public history contains an unsupported author")
            return {
                "author_kind": message.author_kind,
                "author_id": author_id,
                "author_name": author_name,
                "text": message.text,
            }

        history = request.context.recent_messages
        if not history or history[-1] != request.trigger:
            raise ContextTooLong("trigger is not the latest public message")
        prior = [
            public(message) for message in history[:-1]
            if message.message_id != request.user_request.message_id
        ]
        trigger = (
            {"author_kind": "user", "same_as_user_request": True}
            if request.trigger.message_id == request.user_request.message_id
            else public(request.trigger)
        )
        options = {}
        for slot, profile in members.items():
            if "respond" in request.constraints.allowed_actions:
                options[f"respond:{slot}"] = f"让 {profile} 回应或继续讨论"
            if "clarify" in request.constraints.allowed_actions:
                options[f"clarify:{slot}"] = f"让 {profile} 询问必要的澄清问题"
        if "wait" in request.constraints.allowed_actions:
            options["wait"] = "等待用户再次输入，本轮保持开放"
        if "finish" in request.constraints.allowed_actions:
            options["finish"] = "当前讨论已经完成，结束本轮发言"
        options["abstain"] = "无法可靠提出合法且有帮助的下一步"
        state = {
            "user_request": request.user_request.text,
            "scene_goal": request.scene_goal,
            "candidates": members,
            "allowed_actions": list(request.constraints.allowed_actions),
            "remaining_replies": request.constraints.remaining_replies,
            "trigger": trigger,
            "prior_public_messages": prior,
        }
        question = {"type": "choice", "instructions": MOVE_INSTRUCTIONS, "criteria": options}
        self._check_input_budget(state, question)
        prediction = self.engine.predict(state, {"move": question}, truncate_left=False)
        if prediction.truncated:
            raise ContextTooLong("model truncated its input")
        answer = prediction.answers.get("move", {})
        choice = answer.get("choice")
        confidence = answer.get("answer_confidence")
        if choice not in options or type(confidence) not in (int, float):
            return ModelChoice("abstain")
        if choice in {"abstain"} or choice.startswith("clarify:"):
            return ModelChoice("abstain", confidence=float(confidence))
        if choice in {"wait", "finish"}:
            return ModelChoice(choice, confidence=float(confidence))
        action, slot = choice.split(":", 1)
        candidate_id = request.candidates[int(slot[1:])].companion_id
        return ModelChoice(action, candidate_id, confidence=float(confidence))

    def _check_input_budget(self, state: dict, question: dict) -> None:
        internal = to_internal(question)
        tokenizer = self.engine.tokenizer
        options = render_options(internal)
        if tokenizer.mask_token in json.dumps(state, ensure_ascii=False) or any(
            tokenizer.mask_token in option for option in options
        ):
            raise ContextTooLong("model control token occurs in public input")
        option_lengths = [len(tokenizer.encode(" " + option)) for option in options]
        if any(length > 48 for length in option_lengths):
            raise ContextTooLong("candidate option would be truncated")
        head_budget = self.engine.head_max_len - sum(length + 1 for length in option_lengths)
        instruction_length = len(tokenizer.encode("choice question: " + MOVE_INSTRUCTIONS))
        if head_budget < 16 or instruction_length > head_budget:
            raise ContextTooLong("question head would be truncated")
        _, markers, truncated = build_sequence(
            tokenizer, state, internal, self.engine.max_len, self.engine.head_max_len,
            truncate_left=False,
        )
        if truncated or len(markers) != len(options):
            raise ContextTooLong("public context or model options would be truncated")


ACTIONS = ("respond", "clarify", "wait", "finish")
# speaker is asked only when the action speaks; clarify_about only for clarify (as trained: a question
# is labelled only where its answer matters, so it is never seen in the other states)
ASK_IF = {"speaker": {"action": ["respond", "clarify"]}, "clarify_about": {"action": ["clarify"]}}


class LayaParticipationPredictor:
    """participation-laya-v1: the state and three questions of the participation release
    (laya/train/scenarios/participation): action, then speaker, then clarify_about.

    The state is built exactly as the training snapshots were; the questions come frozen from the
    model's pinned profile. A clarify carries the profile's bounded task for its reason.
    """

    HISTORY = 16  # public entries the model was trained on (episodes.py HISTORY)

    def __init__(
        self,
        engine: DecisionEngine,
        *,
        questions: dict,
        clarify_instructions: dict[str, str],
        max_candidates: int = 6,
    ) -> None:
        if type(max_candidates) is not int or not 1 <= max_candidates <= 6:
            raise ValueError("participation-laya-v1 supports at most six candidates")
        action = questions.get("action") or {}
        about = questions.get("clarify_about") or {}
        speaker = questions.get("speaker") or {}
        if (
            set(questions) != {"action", "speaker", "clarify_about"}
            or action.get("type") != "choice"
            or tuple(action.get("criteria") or {}) != ACTIONS
            or speaker.get("type") != "choice"
            or "criteria" in speaker
            or about.get("type") != "choice"
            or set(about.get("criteria") or {}) != set(clarify_instructions)
            or not all(isinstance(v, str) and v.strip() for v in clarify_instructions.values())
        ):
            raise ValueError("participation-laya-v1 profile questions do not match the trained task")
        self.engine = engine
        self.questions = questions
        self.clarify_instructions = dict(clarify_instructions)
        self.max_candidates = max_candidates

    def state(self, request: DecisionRequest) -> tuple[dict, dict, list[str]]:
        """(state, questions, slot -> companion id) for one request; raises ContextTooLong."""
        if (
            not request.candidates
            or len(request.candidates) > self.max_candidates
            or request.context.summary
            or request.context.pending_requirements
            or not all(c.display_name.strip() for c in request.candidates)
        ):
            raise ContextTooLong("snapshot has unsupported or over-capacity context")
        slots = {c.companion_id: f"M{i}" for i, c in enumerate(request.candidates)}
        names = {c.companion_id: c.display_name for c in request.candidates}
        history = request.context.recent_messages
        if not history or history[-1] != request.trigger:
            raise ContextTooLong("trigger is not the latest public message")
        public = []
        for message in history:
            if message.author_kind == "user":
                speaker = "用户"
            elif message.author_kind == "companion" and message.author_id in slots:
                speaker = f"{slots[message.author_id]} {names[message.author_id]}"
            else:
                raise ContextTooLong("public history contains an unsupported author")
            public.append({"说话": speaker, "内容": message.text})
        state = {
            "场景目标": request.scene_goal,
            "候选": [
                {"编号": slots[c.companion_id], "名字": c.display_name, "角色": c.description}
                for c in request.candidates
            ],
            "本轮用户请求": request.user_request.text,
            "公开记录": public[-self.HISTORY:],
            "最新一条": f"{public[-1]['说话']}：{public[-1]['内容']}",
        }
        questions = {
            "action": self.questions["action"],
            "speaker": self.questions["speaker"] | {"criteria": {
                slots[c.companion_id]: f"{c.display_name}：{c.description}" for c in request.candidates
            }},
            "clarify_about": self.questions["clarify_about"],
        }
        return state, questions, [c.companion_id for c in request.candidates]

    def predict(self, request: DecisionRequest) -> ModelChoice:
        state, questions, ids = self.state(request)
        self._check_input_budget(state, questions)
        prediction = self.engine.predict(state, questions, truncate_left=False, ask_if=ASK_IF)
        if prediction.truncated:
            raise ContextTooLong("model truncated its input")
        answers = prediction.answers
        action = answers.get("action", {}).get("choice")
        confidence = answers.get("action", {}).get("answer_confidence")
        if action not in ACTIONS or type(confidence) not in (int, float):
            return ModelChoice("abstain")
        if action in {"wait", "finish"}:
            return ModelChoice(action, confidence=float(confidence))
        slot = answers.get("speaker", {}).get("choice")
        if not isinstance(slot, str) or not slot.startswith("M") or not slot[1:].isdigit() \
                or int(slot[1:]) >= len(ids):
            return ModelChoice("abstain", confidence=float(confidence))
        if action == "respond":
            return ModelChoice("respond", ids[int(slot[1:])], confidence=float(confidence))
        instruction = self.clarify_instructions.get(answers.get("clarify_about", {}).get("choice"))
        if not instruction:
            return ModelChoice("abstain", confidence=float(confidence))
        return ModelChoice("clarify", ids[int(slot[1:])], instruction, float(confidence))

    def _check_input_budget(self, state: dict, questions: dict) -> None:
        tokenizer = self.engine.tokenizer
        if tokenizer.mask_token in json.dumps(state, ensure_ascii=False) or any(
            tokenizer.mask_token in json.dumps(q, ensure_ascii=False) for q in questions.values()
        ):
            raise ContextTooLong("model control token occurs in public input")
        for question in questions.values():
            internal = to_internal(question)
            _, markers, truncated = build_sequence(
                tokenizer, state, internal, self.engine.max_len, self.engine.head_max_len,
                truncate_left=False,
            )
            if truncated or len(markers) != len(render_options(internal)):
                raise ContextTooLong("public context or model options would be truncated")


def participation_profile_digest(manifest: dict) -> str:
    profiles = manifest.get("task_profiles")
    pin = profiles.get("ip_team.participation") if isinstance(profiles, dict) else None
    if (
        not isinstance(pin, dict)
        or set(pin) != {"path", "sha256"}
        or pin["path"] != "participation.json"
        or not isinstance(pin["sha256"], str)
    ):
        raise ValueError("model manifest does not pin an IP-team participation profile")
    return pin["sha256"]


def read_participation_profile(
    profile_path: Path, *, model_dir: Path, model_revision: str, expected_sha256: str,
) -> dict:
    """Read only the profile pinned by the model manifest and Ops artifact."""
    if profile_path.resolve() != (model_dir.resolve() / "participation.json"):
        raise ValueError("participation profile must be model_dir/participation.json")
    if profile_path.is_symlink() or not profile_path.is_file():
        raise ValueError("participation profile must be a regular pinned file")
    raw = profile_path.read_bytes()
    if len(expected_sha256) != 64 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("participation profile digest mismatch")
    profile = json.loads(raw)
    base = {
        "schema_version", "task", "state_format", "model_revision", "policy_version",
        "min_confidence", "max_candidates",
    }
    # v1: the ip-team-v3 single "move" question; v2: participation-laya-v1 with its frozen questions
    formats = {1: ("ip-team-v3", base), 2: ("participation-laya-v1", base | {"questions", "clarify_instructions"})}
    version = profile.get("schema_version") if isinstance(profile, dict) else None
    if type(version) is not int or version not in formats or set(profile) != formats[version][1]:
        raise ValueError("invalid participation profile fields")
    if (
        profile["task"] != "ip_team.participation"
        or profile["state_format"] != formats[version][0]
        or profile["model_revision"] != model_revision
    ):
        raise ValueError("participation profile task or model revision mismatch")
    if (
        not isinstance(profile["policy_version"], str)
        or not profile["policy_version"].strip()
        or len(profile["policy_version"]) > 512
    ):
        raise ValueError("invalid participation policy version")
    return profile


def load_participation_adapter(
    profile_path: Path,
    engine: DecisionEngine,
    *,
    model_dir: Path,
    model_revision: str,
    expected_sha256: str,
) -> ParticipationAdapter:
    """Opt in only with a pinned model-local task profile matching its revision."""

    profile = read_participation_profile(
        profile_path, model_dir=model_dir, model_revision=model_revision,
        expected_sha256=expected_sha256,
    )
    if profile["state_format"] == "participation-laya-v1":
        predictor = LayaParticipationPredictor(
            engine,
            questions=profile["questions"],
            clarify_instructions=profile["clarify_instructions"],
            max_candidates=profile["max_candidates"],
        )
    else:
        predictor = LayaMovePredictor(engine, max_candidates=profile["max_candidates"])
    return ParticipationAdapter(
        predictor,
        model_version=model_revision,
        policy_version=profile["policy_version"],
        min_confidence=profile["min_confidence"],
    )
