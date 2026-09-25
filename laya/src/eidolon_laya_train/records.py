"""The one record format every stage reads and writes.

A record is a Jev-shaped request (``state`` + ``questions``) plus ``labels``, so the same
line trains a model and evaluates a service. Labels carry a gold answer, a soft target
distribution, or both; ``target_vector`` turns either into the per-option distribution the
trainer consumes, in the option order the question defines.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

QTYPES = ("choice", "score", "noul")
SPLITS = ("train", "val", "calib", "eval")


@dataclass
class Record:
    id: str
    scenario: str
    source: str
    state: Any
    questions: dict[str, dict]
    labels: dict[str, dict] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    split: str | None = None
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        if d["split"] is None:
            del d["split"]
        if not d["meta"]:
            del d["meta"]
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> Record:
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)

    def request(self) -> dict:
        """The Jev protocol body for this record."""
        return {"state": self.state, "questions": self.questions}


def read_jsonl(path: str | Path) -> Iterator[Record]:
    with Path(path).open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Record.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"{path}:{n}: bad record: {e}") from e


def write_jsonl(path: str | Path, records: Iterable[Record]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.to_json() + "\n")
            n += 1
    return n


# ------------------------------------------------------------------ questions and targets


def option_names(q: dict) -> list[str]:
    """Option labels in the order the model scores them (laya's render_options order)."""
    t = q["type"]
    if t == "choice":
        return list(q["criteria"])
    if t == "score":
        crit = q["criteria"]
        return [str(i) for i in range(len(crit))]
    if t == "noul":
        return ["false", "true"]
    raise ValueError(f"unknown question type {t!r}")


def _as_names(label, names: list[str]) -> list[str]:
    if isinstance(label, bool):
        return ["true" if label else "false"]
    if isinstance(label, list | tuple):
        out = []
        for x in label:
            out.extend(_as_names(x, names))
        return out
    if isinstance(label, int | float) and not isinstance(label, bool):
        return [str(int(label))]
    return [str(label)]


def target_vector(q: dict, label: dict) -> list[float]:
    """Per-option target distribution: ``target`` if present, else one-hot over ``gold``.

    Several acceptable gold answers share the mass equally. Anything named in the label
    that is not an option is an error: a label that drifted from its question is the
    kind of silent bug that trains a wrong model.
    """
    names = option_names(q)
    if "target" in label and label["target"] is not None:
        t = label["target"]
        vec = [float(t.get(n, 0.0)) for n in names]
        unknown = set(t) - set(names)
        if unknown:
            raise ValueError(f"target names {sorted(unknown)} are not options of the question")
        s = sum(vec)
        if s <= 0:
            raise ValueError("target distribution sums to zero")
        return [v / s for v in vec]
    if "gold" in label and label["gold"] is not None:
        golds = _as_names(label["gold"], names)
        unknown = set(golds) - set(names)
        if unknown:
            raise ValueError(f"gold {sorted(unknown)} is not an option of the question")
        w = 1.0 / len(golds)
        return [w if n in golds else 0.0 for n in names]
    raise ValueError("label has neither target nor gold")


def gold_names(q: dict, label: dict) -> set[str]:
    """The acceptable answers for scoring; from ``gold``, else the argmax of ``target``."""
    names = option_names(q)
    if "gold" in label and label["gold"] is not None:
        return set(_as_names(label["gold"], names))
    vec = target_vector(q, label)
    return {names[max(range(len(vec)), key=vec.__getitem__)]}


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_id(scenario: str, *parts: Any) -> str:
    """Deterministic id from the record's identifying content, prefixed by scenario."""
    return f"{scenario}/{stable_hash(json.dumps(parts, ensure_ascii=False, sort_keys=True))[:12]}"
