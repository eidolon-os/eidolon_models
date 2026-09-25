"""Validate → build sequences → one forward pass → decode. Backend-agnostic."""

from __future__ import annotations

import time
from dataclasses import dataclass
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

    def as_response(self, model: str, backend: str) -> dict:
        return {
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
        self, state: Any, questions: dict[str, dict], *, truncate_left: bool = False
    ) -> Prediction:
        t0 = time.perf_counter()
        if not isinstance(questions, dict):
            raise ValueError("'questions' must be an object mapping question id -> definition")
        if not questions:
            return Prediction({}, 0, [], 0.0, 0.0)
        ids, internals, items, truncated = [], [], [], []
        for qid, qdef in questions.items():
            check_question(qid, qdef)
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
        t2 = time.perf_counter()
        answers = decode_answers(ids, internals, items, logits, act, self.cfg)
        return Prediction(
            answers,
            int(batch["attention_mask"].sum()),
            truncated,
            (t2 - t1) * 1000,
            (time.perf_counter() - t0) * 1000,
        )

    def describe(self) -> dict:
        return {
            **self.backend.describe(),
            "max_len": self.max_len,
            "head_max_len": self.head_max_len,
        }


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
