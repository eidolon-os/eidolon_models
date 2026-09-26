"""Validate → build sequences → forward → decode. Backend-agnostic.

All questions go through one forward pass, unless ``ask_if`` makes some depend on another's answer
(``{"device": {"intent": ["控制", "查询"]}}``): then they run in stages and the rest are skipped —
e.g. an utterance judged 无关 costs only the intent question.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .artifacts import Manifest, verify_onnx, verify_torch
from .backends import Backend
from .config import Settings
from .sequence import (
    QTYPES,
    ModelConfig,
    Tokenizer,
    build_sequence,
    check_question,
    collate,
    decode_answers,
    render_options,
    to_internal,
)


@dataclass
class Prediction:
    answers: dict[str, dict]
    input_tokens: int
    truncated: list[str]
    forward_ms: float
    total_ms: float
    skipped: dict[str, str] = field(default_factory=dict)  # question id -> why it was not asked

    def as_response(self, model: str, backend: str) -> dict:
        extra = {"skipped": self.skipped} if self.skipped else {}
        return extra | {
            "model": model,
            "answers": self.answers,
            "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
            "truncated": self.truncated,
            "backend": backend,
            "timing_ms": {"forward": round(self.forward_ms, 1), "total": round(self.total_ms, 1)},
        }


class DecisionEngine:
    def __init__(
        self,
        backend: Backend,
        tokenizer: Tokenizer,
        cfg: ModelConfig,
        *,
        max_len: int | None = None,
        head_max_len: int | None = None,
        name: str = "laya",
    ):
        self.backend = backend
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.max_len = max_len or cfg.max_len
        # Budget for instruction + options; raise it for questions with many options.
        self.head_max_len = head_max_len or cfg.head_max_len
        self.name = name

    def predict(
        self,
        state: Any,
        questions: dict[str, dict],
        *,
        truncate_left: bool = False,
        ask_if: dict[str, dict[str, list[str]]] | None = None,
    ) -> Prediction:
        t0 = time.perf_counter()
        if not isinstance(questions, dict):
            raise ValueError("'questions' must be an object mapping question id -> definition")
        if not questions:
            return Prediction({}, 0, [], 0.0, 0.0)
        for qid, qdef in questions.items():
            check_question(qid, qdef)
        ask_if = check_ask_if(questions, ask_if or {})
        pending, answers, skipped = dict(questions), {}, {}
        tokens, truncated, forward_ms = 0, [], 0.0
        while pending:
            ready = [q for q in pending if all(d in answers or d in skipped for d in ask_if.get(q, {}))]
            run = {}
            for qid in ready:
                why = next((f"{d}={answers[d]['choice']}" if d in answers else f"{d} skipped"
                            for d, allowed in ask_if.get(qid, {}).items()
                            if d in skipped or answers[d]["choice"] not in allowed), None)
                if why:
                    skipped[qid] = why
                else:
                    run[qid] = pending[qid]
                del pending[qid]
            if run:
                a, n, cut, ms = self._forward(state, run, truncate_left)
                answers |= a
                tokens, truncated, forward_ms = tokens + n, truncated + cut, forward_ms + ms
        return Prediction(
            {q: answers[q] for q in questions if q in answers},
            tokens,
            truncated,
            forward_ms,
            (time.perf_counter() - t0) * 1000,
            {q: skipped[q] for q in questions if q in skipped},
        )

    def _forward(self, state: Any, questions: dict[str, dict], truncate_left: bool):
        ids, internals, items, truncated = [], [], [], []
        for qid, qdef in questions.items():
            q = to_internal(qdef)
            seq, markers, cut = build_sequence(
                self.tokenizer, state, q, self.max_len, self.head_max_len, truncate_left
            )
            if len(markers) != len(render_options(q)):
                raise ValueError(
                    f"question {qid!r} options exceed head_max_len={self.head_max_len}"
                )
            ids.append(qid)
            internals.append(q)
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
            if cut:
                truncated.append(qid)
        batch = collate(items, self.tokenizer.pad_token_id)
        t1 = time.perf_counter()
        logits, act = self.backend.forward(batch)
        ms = (time.perf_counter() - t1) * 1000
        answers = decode_answers(ids, internals, items, logits, act, self.cfg)
        return answers, int(batch["attention_mask"].sum()), truncated, ms

    def describe(self) -> dict:
        return {
            **self.backend.describe(),
            "max_len": self.max_len,
            "head_max_len": self.head_max_len,
        }


def check_ask_if(questions: dict[str, dict], ask_if: Any) -> dict[str, dict[str, list[str]]]:
    """``{qid: {depends_on_qid: [answers that make qid worth asking]}}``, checked against the questions."""
    if not isinstance(ask_if, dict):
        raise ValueError("'ask_if' must map question id -> {question id: [answers]}")
    for qid, conds in ask_if.items():
        if qid not in questions:
            raise ValueError(f"ask_if: {qid!r} is not one of the questions")
        if not isinstance(conds, dict) or not conds:
            raise ValueError(f"ask_if[{qid!r}] must map a question id to the answers that make it worth asking")
        for dep, allowed in conds.items():
            if dep == qid or dep not in questions:
                raise ValueError(f"ask_if[{qid!r}]: {dep!r} is not another of the questions")
            if questions[dep].get("type") != "choice":
                raise ValueError(f"ask_if[{qid!r}]: {dep!r} must be a choice question")
            options = set(to_internal(questions[dep])["crit"])
            if not isinstance(allowed, list) or not allowed or not set(allowed) <= options:
                raise ValueError(f"ask_if[{qid!r}][{dep!r}] must be a non-empty list of {dep!r}'s options {sorted(options)}")
    seen: set[str] = set()

    def visit(q: str, path: tuple[str, ...]) -> None:
        if q in path:
            raise ValueError(f"ask_if has a cycle: {' -> '.join(path + (q,))}")
        if q not in seen:
            seen.add(q)
            for d in ask_if.get(q, {}):
                visit(d, path + (q,))

    for q in ask_if:
        visit(q, ())
    return ask_if


def load_engine(settings: Settings, log=print) -> tuple[DecisionEngine, Manifest]:
    """Build the engine the settings ask for, refusing incomplete model files."""
    manifest = Manifest.load(settings.model_dir)
    problems = verify_torch(manifest, checksums=False)  # tokenizer + config needed by both
    if settings.backend == "onnx":
        problems += verify_onnx(manifest, checksums=False)
    if problems:
        raise FileNotFoundError("model files incomplete:\n  " + "\n  ".join(problems))
    t0 = time.perf_counter()
    if settings.backend == "torch":
        from .backends import TorchBackend

        backend = TorchBackend(manifest.torch_dir, device=settings.device, threads=settings.threads)
    else:
        from .backends import OnnxBackend

        backend = OnnxBackend(manifest.onnx_path, threads=settings.threads)
    engine = DecisionEngine(
        backend,
        Tokenizer(manifest.tokenizer_dir),
        ModelConfig.from_dict(manifest.model_config()),
        max_len=settings.max_len,
        head_max_len=settings.head_max_len,
        name=manifest.name,
    )
    log(
        f"loaded {manifest.name}@{manifest.revision[:8]} backend={backend.name} "
        f"in {time.perf_counter() - t0:.1f}s ({engine.describe()})"
    )
    return engine, manifest
