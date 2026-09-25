"""``label``: soft targets from a teacher.

Two kinds of teacher:

- ``systemone``: a decision model behind ``/v1/systemone`` (decider, laya, Jev). One call returns
  the whole distribution; cheap, but only as reliable as that model's judgement.
- ``llm``: a chat model (DeepSeek / Qwen / GPT class) asked the same question ``samples`` times
  at temperature ``sample_temperature``; the vote counts are the distribution. This is how the
  typed-decisions dataset was labelled (3 samples at 0.7). Slower and pricier, but the judgement
  is a frontier model's — prefer it as the teacher and keep a decision model as a cheap second
  opinion (``disagreements`` in the stats).

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

from .generators import ChatClient
from .records import Record, gold_names, option_names, target_vector

LLM_PROMPT = """你是一个严格的判断器。根据下面的 state 回答一道题，只能从给定选项里选一个。

state:
{state}

题目：{instructions}
选项（只回答键名，不要解释）：
{options}

只输出一个 JSON：{{"answer": "<选项键名>"}}"""


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


class LLMTeacher:
    """Vote-count distributions from a chat model. ``answers(body)`` mirrors ``Teacher.answers``
    so ``label_records`` does not care which kind it got."""

    def __init__(self, config: dict, samples: int = 5, temperature: float = 0.7):
        self.client = ChatClient(config)
        self.samples = samples
        self.temperature = temperature

    def _ask(self, state, q: dict) -> str | None:
        names = option_names(q)
        if q["type"] == "noul":
            opts = "true: 成立\nfalse: 不成立"
        elif q["type"] == "score":
            opts = "\n".join(f"{i}: {c}" for i, c in enumerate(q["criteria"]))
        else:
            opts = "\n".join(f"{k}: {v}" if v else k for k, v in q["criteria"].items())
        prompt = LLM_PROMPT.format(
            state=json.dumps(state, ensure_ascii=False),
            instructions=q["instructions"],
            options=opts,
        )
        text = self.client.complete(prompt, temperature=self.temperature)
        start, end = text.find("{"), text.rfind("}")
        try:
            ans = json.loads(text[start : end + 1])["answer"]
        except (ValueError, KeyError, TypeError):
            return None
        ans = str(ans).strip().lower() if q["type"] == "noul" else str(ans).strip()
        return ans if ans in names else None

    def answers(self, body: dict) -> dict:
        out = {}
        for qid, q in body["questions"].items():
            names = option_names(q)
            votes = {n: 0 for n in names}
            valid = 0
            for _ in range(self.samples):
                a = self._ask(body["state"], q)
                if a is not None:
                    votes[a] += 1
                    valid += 1
            if valid == 0:
                raise ValueError(f"llm teacher gave no valid answer for {qid}")
            probs = {n: c / valid for n, c in votes.items()}
            if q["type"] == "noul":
                out[qid] = {"type": "noul", "noul": probs["true"]}
            else:
                out[qid] = {"type": q["type"], "probabilities": probs, "samples": valid}
        return out


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
    teacher,
    *,
    alpha: float = 0.7,
    workers: int = 2,
    teacher_name: str = "teacher",
    stats: dict | None = None,
) -> Iterator[Record]:
    stats = stats if stats is not None else {}
    stats.setdefault("labelled", 0)
    stats.setdefault("failed", 0)
    stats.setdefault("disagreements", 0)  # teacher argmax not among the gold answers

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
            if lab.get("gold") is not None:
                top = max(dist, key=dist.get)
                if top not in gold_names(q, {"gold": lab["gold"]}):
                    stats["disagreements"] += 1
                    r.tags.append(f"disagree:{qid}")
        r.meta["teacher"] = teacher_name
        stats["labelled"] += 1
        return r

    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(one, records)
