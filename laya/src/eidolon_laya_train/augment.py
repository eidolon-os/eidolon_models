"""``augment``: derived records. Each transform takes a record and returns zero or more new
records with ``derived_from`` in meta; the originals pass through untouched.

Transforms are chosen by name in the config::

    transforms:
      - name: exit_injection        # drop the gold option from a dynamic criteria list;
        question: device            # the answer becomes the "none" exit
        exit: 没有对应的设备
        rate: 0.3
      - name: option_subset          # keep the gold plus k random others (+ exits)
        question: device
        keep: 6
        rate: 0.3
      - name: char_confusion         # ASR-style substitutions from a table
        field: utterance
        table: {灯: [登, 等], 空调: [空掉]}
        rate: 0.3
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator
from copy import deepcopy

from .records import Record, gold_names
from .scenario import Scenario

Transform = Callable[[Record, Scenario, dict, random.Random], list[Record]]


def _derive(r: Record, name: str, suffix: str) -> Record:
    d = deepcopy(r)
    d.id = f"{r.id}~{suffix}"
    d.source = f"{r.source}+{name}"
    d.tags = list(r.tags) + [f"aug:{name}"]
    d.meta = dict(r.meta, derived_from=r.id)
    return d


def exit_injection(r: Record, scenario: Scenario, cfg: dict, rng: random.Random) -> list[Record]:
    qid, exit_name = cfg["question"], cfg["exit"]
    q = r.questions.get(qid)
    lab = r.labels.get(qid)
    if not q or not lab or q["type"] != "choice":
        return []
    spec = scenario.questions.get(qid)
    exits = set(spec.exits) if spec else set()
    golds = gold_names(q, lab)
    if golds & exits or exit_name not in q["criteria"]:
        return []  # already an exit answer, or this question has no such exit
    d = _derive(r, "exit_injection", "noexit")
    crit = {k: v for k, v in q["criteria"].items() if k not in golds}
    d.questions[qid] = dict(q, criteria=crit)
    d.labels[qid] = {"gold": exit_name}
    d.tags.append("exit:none")
    return [d]


def option_subset(r: Record, scenario: Scenario, cfg: dict, rng: random.Random) -> list[Record]:
    qid, keep = cfg["question"], int(cfg.get("keep", 6))
    q = r.questions.get(qid)
    lab = r.labels.get(qid)
    if not q or not lab or q["type"] != "choice":
        return []
    spec = scenario.questions.get(qid)
    exits = list(spec.exits) if spec else []
    golds = gold_names(q, lab)
    pool = [k for k in q["criteria"] if k not in golds and k not in exits]
    if len(pool) <= keep:
        return []
    others = rng.sample(pool, keep)
    kept = [k for k in q["criteria"] if k in golds or k in others or k in exits]
    d = _derive(r, "option_subset", f"sub{keep}")
    d.questions[qid] = dict(q, criteria={k: q["criteria"][k] for k in kept})
    if "target" in lab:
        t = {k: v for k, v in lab["target"].items() if k in kept}
        s = sum(t.values()) or 1.0
        d.labels[qid] = dict(lab, target={k: v / s for k, v in t.items()})
    return [d]


def char_confusion(r: Record, scenario: Scenario, cfg: dict, rng: random.Random) -> list[Record]:
    field = cfg.get("field", "utterance")
    table: dict[str, list[str]] = cfg["table"]
    if not isinstance(r.state, dict) or field not in r.state:
        return []
    text = str(r.state[field])
    hits = [k for k in table if k in text]
    if not hits:
        return []
    k = rng.choice(hits)
    new = text.replace(k, rng.choice(table[k]), 1)
    if new == text:
        return []
    d = _derive(r, "char_confusion", "asr")
    d.state = dict(r.state, **{field: new})
    d.tags.append("asr-noise")
    return [d]


TRANSFORMS: dict[str, Transform] = {
    "exit_injection": exit_injection,
    "option_subset": option_subset,
    "char_confusion": char_confusion,
}


def augment_records(
    records: Iterable[Record],
    scenario: Scenario,
    transforms: list[dict],
    seed: int = 7,
    stats: dict | None = None,
) -> Iterator[Record]:
    rng = random.Random(seed)
    stats = stats if stats is not None else {}
    for r in records:
        yield r
        for cfg in transforms:
            fn = TRANSFORMS.get(cfg["name"])
            if fn is None:
                raise ValueError(f"unknown transform {cfg['name']!r}")
            if rng.random() > float(cfg.get("rate", 1.0)):
                continue
            for d in fn(r, scenario, cfg, rng):
                stats[cfg["name"]] = stats.get(cfg["name"], 0) + 1
                yield d
