"""Torch-free sequence building and answer decoding, shared by both backends.

Ported from laya 0.3.20 (``laya/common.py`` and ``Agent.system_one`` in
``laya/agent.py``), Copyright Convai Innovations, Apache License 2.0. Changes:
the tokenizer is ``tokenizers`` instead of ``transformers``, batches are numpy
instead of torch, and ``build_sequence`` also reports whether the state was
truncated. Everything that decides the model's input or the reported numbers is
kept byte-for-byte — ``tests/test_parity.py`` checks it against upstream — so the
ONNX backend answers exactly what the PyTorch one does.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
INPUT_NAMES = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")

# A fitted temperature below 1 sharpens instead of softening; laya refuses the
# extreme buckets some checkpoints ship (see laya/common.py).
TEMP_MIN = 0.5
TEMP_MAX = 5.0


class Tokenizer:
    """The checkpoint's ``tokenizer.json`` via the Rust ``tokenizers`` library."""

    def __init__(self, tokenizer_dir: Path):
        from tokenizers import Tokenizer as _RustTokenizer

        self._tok = _RustTokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
        self._tok.no_truncation()
        self._tok.no_padding()
        cfg = json.loads((tokenizer_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.mask_token = cfg["mask_token"]
        self.mask_token_id = self._id(cfg["mask_token"])
        self.cls_token_id = self._id(cfg["cls_token"])
        self.sep_token_id = self._id(cfg["sep_token"])
        self.pad_token_id = self._id(cfg["pad_token"])

    def _id(self, token: str) -> int:
        token_id = self._tok.token_to_id(token)
        if token_id is None:
            raise ValueError(f"special token {token!r} is not in the vocabulary")
        return token_id

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids


@dataclass(frozen=True)
class ModelConfig:
    """The parts of ``rl_agent_config.json`` inference reads."""

    max_len: int
    head_max_len: int
    temperature: tuple[float, float, float]
    temperature_by_options: dict[str, float]

    @classmethod
    def from_dict(cls, cfg: dict) -> ModelConfig:
        raw = cfg.get("temperature", [1.0, 1.0, 1.0])
        return cls(
            max_len=int(cfg.get("max_len", 512)),
            head_max_len=int(cfg.get("head_max_len", 192)),
            temperature=tuple(clamp_temperature(t) for t in raw),
            temperature_by_options={
                k: clamp_temperature(v) for k, v in cfg.get("temperature_by_options", {}).items()
            },
        )


def clamp_temperature(t: Any, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):
        return 1.0
    return min(hi, max(lo, t))


def serialize_state(state: str | dict | list) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: dict) -> list[str]:
    """Option texts in label-index order. Noul is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [
            k if v is None or v == "" else "%s: %s" % (k, render_criterion(v))
            for k, v in crit.items()
        ]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        "false: "
        + (
            render_criterion(false_crit)
            if false_crit not in (None, "")
            else "no, the statement does not hold"
        ),
        "true: "
        + (
            render_criterion(true_crit)
            if true_crit not in (None, "")
            else "yes, the statement holds"
        ),
    ]


def check_question(qid: str, qdef: Any) -> None:
    """Reject a question that cannot be answered, naming it and what to fix."""
    if not isinstance(qdef, dict):
        raise ValueError(
            "question %r: definition must be a dict, got %s" % (qid, type(qdef).__name__)
        )
    t = qdef.get("type")
    if t not in QTYPES:
        raise ValueError("question %r: unknown type %r; use one of %s" % (qid, t, sorted(QTYPES)))
    if "instructions" not in qdef:
        raise ValueError(
            "question %r: no 'instructions'; add the text the model should answer" % (qid,)
        )
    crit = qdef.get("criteria")
    if t == "choice":
        if not isinstance(crit, (dict, list)):
            raise ValueError(
                "question %r: a choice question takes 'criteria' as a dict of "
                "label -> description, or a list of labels" % (qid,)
            )
        if not crit:
            raise ValueError("question %r: a choice question needs at least one criterion" % (qid,))
    elif t == "score":
        if not isinstance(crit, list):
            raise ValueError(
                "question %r: a score question takes 'criteria' as a list of level "
                "descriptions, index 0 first" % (qid,)
            )
        if not crit:
            raise ValueError("question %r: a score question needs at least one level" % (qid,))
    elif crit is not None and not isinstance(crit, dict):
        raise ValueError(
            "question %r: a noul question takes 'criteria' as a dict with optional "
            "'true'/'false' descriptions, or omits it" % (qid,)
        )


def to_internal(qdef: dict) -> dict:
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    elif t == "noul" and isinstance(crit, dict):
        crit = {str(k).lower(): v for k, v in crit.items()}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


def build_sequence(
    tok: Tokenizer,
    state: Any,
    q: dict,
    max_len: int,
    head_max_len: int,
    truncate_left: bool = False,
) -> tuple[list[int], list[int], bool]:
    """``[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]``.

    Returns ``(ids, marker_positions, state_truncated)``. By default the *end*
    of an over-long state is dropped (laya's behaviour); pass
    ``truncate_left=True`` to keep the most recent end of a chat history.
    """
    mask_tok = tok.mask_token
    opts = render_options(q)
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok.encode("%s question: %s" % (q["t"], ins))
    opt_ids = [[tok.mask_token_id] + tok.encode(" " + o.replace(mask_tok, " "))[:48] for o in opts]
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok.encode(serialize_state(state).replace(mask_tok, " "))
    truncated = len(st) > room
    if room == 0:
        st = []
    else:
        st = st[-room:] if truncate_left else st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len], truncated


def collate(items: list[dict], pad_id: int) -> dict[str, np.ndarray]:
    n, length = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    batch = {
        "input_ids": np.full((n, length), pad_id, dtype=np.int64),
        "attention_mask": np.zeros((n, length), dtype=np.int64),
        "marker_pos": np.zeros((n, kmax), dtype=np.int64),
        "marker_mask": np.zeros((n, kmax), dtype=bool),
        "qtype": np.array([it["qtype"] for it in items], dtype=np.int64),
    }
    for i, it in enumerate(items):
        batch["input_ids"][i, : len(it["ids"])] = it["ids"]
        batch["attention_mask"][i, : len(it["ids"])] = 1
        k = len(it["markers"])
        batch["marker_pos"][i, :k] = it["markers"]
        batch["marker_mask"][i, :k] = True
    return batch


def answer_confidence(p: np.ndarray, k: int) -> float:
    """Probability mass on the reported answer: max(p). laya 0.3.20 reports it next to
    ``confidence`` because it is the quantity temperature scaling fits (and ECE measures);
    ``confidence`` below is entropy-based and not comparable against the same threshold."""
    if k < 1:
        return 1.0
    return float(np.clip(np.max(p[:k]), 0.0, 1.0))


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k)."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def _softmax32(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def decode_answers(
    ids: list[str],
    internals: list[dict],
    items: list[dict],
    logits: np.ndarray,
    act_logits: np.ndarray,
    cfg: ModelConfig,
) -> dict[str, dict]:
    """Logits -> laya's answer objects, in float32 exactly as upstream computes them."""
    logits = logits.astype(np.float32)
    act = _softmax32(act_logits)
    answers = {}
    for r, qid in enumerate(ids):
        q = internals[r]
        k = len(items[r]["markers"])
        qt = QTYPES[q["t"]]
        t_scale = cfg.temperature_by_options.get(temp_bucket(qt, k), cfg.temperature[qt])
        z = logits[r, :k] / t_scale
        p = np.exp(z - z.max())
        p = p / p.sum()
        conf = round(confidence_from_probs(p, k), 4)
        ans_conf = round(answer_confidence(p, k), 4)
        ext = {"act_probability": round(float(act[r, 0]), 4)}
        if q["t"] == "choice":
            keys = list(q["crit"].keys())
            answers[qid] = {
                "type": "choice",
                "choice": keys[int(p.argmax())],
                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p, strict=False)},
                "confidence": conf,
                "answer_confidence": ans_conf,
                "action": ext,
            }
        elif q["t"] == "score":
            answers[qid] = {
                "type": "score",
                "score": round(float((np.arange(k) * p).sum()), 4),
                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                "confidence": conf,
                "answer_confidence": ans_conf,
                "action": ext,
            }
        else:
            p1 = float(p[1])
            answers[qid] = {
                "type": "noul",
                "noul": round(p1, 4),
                "confidence": round(max(p1, 1.0 - p1), 4),
                "answer_confidence": ans_conf,
                "action": ext,
            }
    return answers
