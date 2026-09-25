"""``label``: soft targets from a teacher behind any ``/v1/systemone`` endpoint.

For each question the teacher's distribution becomes ``target``; when the record already has
``gold``, the target is ``alpha * onehot(gold) + (1 - alpha) * teacher`` so a human label
anchors the answer while the teacher shapes the rest of the mass. Records the teacher fails on
keep their gold-only label and are counted, never dropped silently.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor

from .records import Record, option_names, target_vector


class Teacher:
    def __init__(self, url: str, timeout: float = 120.0, api_key: str | None = None):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def answers(self, body: dict) -> dict:
        req = urllib.request.Request(
            self.url + "/v1/systemone",
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers=self.headers,
            method="POST",
        )
        with self.opener.open(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["answers"]


def teacher_distribution(q: dict, answer: dict) -> dict[str, float]:
    """A Jev answer → ``{option: p}`` over the question's option names."""
    names = option_names(q)
    t = q["type"]
    if t == "noul":
        p = float(answer["noul"])
        return {"false": 1.0 - p, "true": p}
    probs = answer.get("probabilities") or {}
    out = {n: float(probs.get(n, 0.0)) for n in names}
    if sum(out.values()) <= 0:
        raise ValueError(f"teacher returned no probability mass for {t} question")
    return out


def blend(q: dict, label: dict, teacher: dict[str, float], alpha: float) -> dict[str, float]:
    names = option_names(q)
    tv = [teacher.get(n, 0.0) for n in names]
    s = sum(tv) or 1.0
    tv = [x / s for x in tv]
    if label.get("gold") is not None and alpha > 0:
        gv = target_vector(q, {"gold": label["gold"]})
        tv = [alpha * g + (1 - alpha) * t for g, t in zip(gv, tv, strict=True)]
    return {n: round(v, 6) for n, v in zip(names, tv, strict=True)}


def label_records(
    records: Iterable[Record],
    teacher: Teacher,
    *,
    alpha: float = 0.7,
    workers: int = 2,
    teacher_name: str = "teacher",
    stats: dict | None = None,
) -> Iterator[Record]:
    stats = stats if stats is not None else {}
    stats.setdefault("labelled", 0)
    stats.setdefault("failed", 0)

    def one(r: Record) -> Record:
        try:
            ans = teacher.answers(r.request())
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
            stats["failed"] += 1
            r.meta.setdefault("label_errors", []).append(f"{teacher_name}: {e}")
            return r
        for qid, q in r.questions.items():
            if qid not in ans:
                continue
            lab = r.labels.setdefault(qid, {})
            dist = teacher_distribution(q, ans[qid])
            lab["teacher"] = {n: round(p, 6) for n, p in dist.items()}
            lab["target"] = blend(q, lab, dist, alpha)
        r.meta["teacher"] = teacher_name
        stats["labelled"] += 1
        return r

    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(one, records)
