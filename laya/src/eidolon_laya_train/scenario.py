"""A scenario describes the questions, not the data.

``scenario.yaml``::

    name: smart-home
    questions:
      intent:
        type: choice
        instructions: "`utterance` 是在让智能家居做什么？"
        criteria: {控制: "...", 查询: "...", 无关: "..."}
      device:
        type: choice
        instructions: "`utterance` 说的是家里的哪台设备？"
        criteria_from: devices            # filled per record by the generator
        exits: {多个设备或整屋: "...", 没有对应的设备: "..."}
      action:
        type: choice
        ...
    eval_sets:                            # locked: assemble never lets these ids into train
      - ../../evals/smart-home/scenarios   # (resolved relative to scenario.yaml)

``criteria_from`` names a slot the generator supplies (a ``{name: description}`` map);
``exits`` are appended after it, so every dynamic option list ends with the same exits in
the same order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class QuestionSpec:
    id: str
    type: str
    instructions: str
    criteria: dict | list | None = None
    criteria_from: str | None = None
    exits: dict[str, str] = field(default_factory=dict)

    def materialize(self, dynamic: dict[str, dict] | None = None) -> dict:
        """The concrete Jev question for one record."""
        q: dict = {"type": self.type, "instructions": self.instructions}
        if self.type == "noul":
            if self.criteria:
                q["criteria"] = self.criteria
            return q
        if self.criteria_from:
            if not dynamic or self.criteria_from not in dynamic:
                raise ValueError(
                    f"question {self.id!r} needs dynamic criteria {self.criteria_from!r}"
                )
            crit = dict(dynamic[self.criteria_from])
        else:
            crit = (
                dict(self.criteria)
                if isinstance(self.criteria, dict)
                else list(self.criteria or [])
            )
        if self.exits:
            if not isinstance(crit, dict):
                raise ValueError(f"question {self.id!r}: exits need choice criteria")
            crit.update(self.exits)
        q["criteria"] = crit
        return q


@dataclass
class Scenario:
    name: str
    questions: dict[str, QuestionSpec]
    eval_sets: list[Path]
    root: Path
    raw: dict

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        path = Path(path)
        if path.is_dir():
            path = path / "scenario.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        qs = {}
        for qid, spec in raw["questions"].items():
            qs[qid] = QuestionSpec(
                id=qid,
                type=spec["type"],
                instructions=spec["instructions"],
                criteria=spec.get("criteria"),
                criteria_from=spec.get("criteria_from"),
                exits=dict(spec.get("exits") or {}),
            )
        eval_sets = [(path.parent / p).resolve() for p in raw.get("eval_sets", [])]
        return cls(name=raw["name"], questions=qs, eval_sets=eval_sets, root=path.parent, raw=raw)

    def build_questions(
        self, dynamic: dict[str, dict] | None = None, only: list[str] | None = None
    ) -> dict:
        ids = only or list(self.questions)
        return {qid: self.questions[qid].materialize(dynamic) for qid in ids}

    @property
    def exit_names(self) -> dict[str, list[str]]:
        return {qid: list(q.exits) for qid, q in self.questions.items() if q.exits}
