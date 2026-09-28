"""``gen``: turn a scenario into cases. Three kinds of generator, all yielding Records.

- ``template``: a Python function in the scenario directory, ``generate(scenario, config, rng)``.
- ``import``: an adapter that reads an existing dataset (our evals, record-shaped jsonl, ...).
- ``llm``: an OpenAI-compatible chat endpoint fills a prompt template and returns JSON cases.

A generator decides ``state``, the dynamic criteria (e.g. the device list) and any gold it
knows; the scenario decides the questions. Records carry ``source`` so assemble can weight
them and eval can slice by origin.
"""

from __future__ import annotations

import importlib
import json
import os
import random
import sys
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path

from .records import Record, record_id
from .scenario import Scenario


def generate(scenario: Scenario, config: dict, seed: int = 7) -> Iterator[Record]:
    rng = random.Random(seed)
    kind = config["kind"]
    if kind == "template":
        yield from _template(scenario, config, rng)
    elif kind == "import":
        yield from _import(scenario, config)
    elif kind == "llm":
        yield from _llm(scenario, config, rng)
    else:
        raise ValueError(f"unknown generator kind {kind!r}")


# ------------------------------------------------------------------ template


def _template(scenario: Scenario, config: dict, rng: random.Random) -> Iterable[Record]:
    module_name, _, func_name = config["module"].partition(":")
    root = str(scenario.root)
    if root not in sys.path:
        sys.path.insert(0, root)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name or "generate")
    for r in fn(scenario, config, rng):
        if not isinstance(r, Record):
            raise TypeError(f"{config['module']} yielded {type(r).__name__}, not Record")
        yield r


# ------------------------------------------------------------------ import adapters


def _import(scenario: Scenario, config: dict) -> Iterable[Record]:
    adapter = config["adapter"]
    if adapter == "records":
        yield from _import_records(scenario, config)
    elif adapter == "evals_cases":
        yield from _import_evals_cases(scenario, config)
    elif adapter == "authored_cases":
        yield from _import_authored_cases(scenario, config)
    elif adapter in ("zhihao_smarthome", "scenic", "massive_zh"):
        from . import external

        yield from getattr(external, f"import_{adapter}")(scenario, config)
    else:
        raise ValueError(f"unknown import adapter {adapter!r}")


def _import_records(scenario: Scenario, config: dict) -> Iterable[Record]:
    """Already record-shaped jsonl (another run's output, a hand-written set)."""
    from .records import read_jsonl

    for r in read_jsonl(_resolve(scenario, config["path"])):
        if config.get("scenario_filter") and r.scenario != config["scenario_filter"]:
            continue
        yield r


def _import_evals_cases(scenario: Scenario, config: dict) -> Iterable[Record]:
    """``laya/evals/<name>/scenarios/*/cases.jsonl`` + ``homes/*.json`` (our evaluation format).

    Each case is ``{id, home, text, gold: {intent, device?, action?}}``; the device criteria
    come from the home. ``config.dynamic`` maps the scenario's ``criteria_from`` slot to the
    home's device map, e.g. ``{devices: home_devices}``.
    """
    root = _resolve(scenario, config["path"])
    homes = _load_homes(root / "homes")
    slot = config.get("device_slot", "devices")
    # ``split``: keep only the ids a split file assigns to this split (dev / test of a locked set)
    keep_ids = None
    if config.get("split"):
        split = json.loads(
            (root / config.get("split_file", "split.json")).read_text(encoding="utf-8")
        )
        keep_ids = set(split[config["split"]])
    files = sorted((root / "scenarios").glob("*/cases.jsonl"))
    if not files:
        raise FileNotFoundError(f"no scenarios/*/cases.jsonl under {root}")
    for f in files:
        scn_tag = f.parent.name
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            if keep_ids is not None and c["id"] not in keep_ids:
                continue
            devices = homes[c["home"]]
            questions = scenario.build_questions({slot: devices})
            labels = {}
            for qid, gold in c["gold"].items():
                if qid in questions and gold is not None:
                    labels[qid] = {"gold": gold}
            yield Record(
                id=f"{scenario.name}/{c['id']}",
                scenario=scenario.name,
                source=f"import:{config.get('name', root.name)}",
                state={"utterance": c["text"]},
                questions=questions,
                labels=labels,
                tags=[scn_tag, f"home:{c['home']}"],
                meta={"home": c["home"]},
            )


def _import_authored_cases(scenario: Scenario, config: dict) -> Iterable[Record]:
    """``train/data/authored/<scenario>/*.jsonl`` in the authoring format (see train/data/README.md):
    ``{home, text, intent, device?, action?, tags}``. Every gold is checked against the home's
    options and the line is named on failure, so a typo in a device name cannot train."""
    from .records import target_vector

    root = _resolve(scenario, config["path"])
    # ``homes``: one or more directories, each holding ``*.json`` homes directly or under ``homes/``
    # (the eval homes plus training-only layouts, so the model sees many device lists).
    home_dirs = config.get("homes", "../../../evals/smart-home")
    homes: dict[str, dict[str, str]] = {}
    for d in [home_dirs] if isinstance(home_dirs, str) else home_dirs:
        d = _resolve(scenario, d)
        loaded = _load_homes(d / "homes" if (d / "homes").is_dir() else d)
        clash = set(loaded) & set(homes)
        if clash:
            raise ValueError(f"home names defined twice: {sorted(clash)}")
        homes.update(loaded)
    slot = config.get("device_slot", "devices")
    files = sorted(root.glob("*.jsonl")) if root.is_dir() else [root]
    if not files:
        raise FileNotFoundError(f"no *.jsonl under {root}")
    seen: dict[str, str] = {}
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            where = f"{f.name}:{n}"
            try:
                c = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{where}: bad json: {e}") from e
            for key in ("home", "text", "intent"):
                if key not in c:
                    raise ValueError(f"{where}: missing {key!r}")
            if c["home"] not in homes:
                raise ValueError(f"{where}: unknown home {c['home']!r}")
            key = f"{c['home']}|{c['text'].strip()}"
            if key in seen:  # the same sentence in two files: keep the first, say so
                print(f"authored: {where} duplicates {seen[key]}, skipped", file=sys.stderr)
                continue
            seen[key] = where
            questions = scenario.build_questions({slot: homes[c["home"]]})
            labels = {"intent": {"gold": c["intent"]}}
            if c.get("device") is not None:
                labels["device"] = {"gold": c["device"]}
            if c.get("action") is not None:
                labels["action"] = {"gold": c["action"]}
            if c["intent"] == "控制" and "device" not in labels:
                raise ValueError(f"{where}: 控制 needs a device (or an exit)")
            if c["intent"] == "无关" and ("device" in labels or "action" in labels):
                raise ValueError(f"{where}: 无关 must not carry device/action")
            for qid, lab in labels.items():
                try:
                    target_vector(questions[qid], lab)
                except ValueError as e:
                    raise ValueError(f"{where}: {qid}: {e}") from e
            tags = list(c.get("tags") or [])
            yield Record(
                id=record_id(scenario.name, "authored", c["home"], c["text"]),
                scenario=scenario.name,
                source=f"authored:{f.stem}",
                state={"utterance": c["text"]},
                questions=questions,
                labels=labels,
                tags=tags + [f"home:{c['home']}"],
                meta={"home": c["home"], "file": where},
            )


def _load_homes(homes_dir: Path) -> dict[str, dict[str, str]]:
    raw = {p.stem: json.loads(p.read_text("utf-8")) for p in homes_dir.glob("*.json")}
    out = {}
    for name, home in raw.items():
        devices = list(home.get("devices", []))
        for inc in home.get("include", []):
            devices.extend(raw[inc]["devices"])
        out[name] = {d["name"]: f"{d['room']}·{d['type']}" for d in devices}
    return out


def _resolve(scenario: Scenario, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (scenario.root / path).resolve()


# ------------------------------------------------------------------ llm


def _llm(scenario: Scenario, config: dict, rng: random.Random) -> Iterable[Record]:
    """Ask a chat model for cases. The prompt template gets ``{questions}`` (the scenario's
    questions as JSON), ``{context}`` (one item from ``config.contexts``) and ``{n}``; the
    model must answer with a JSON array of ``{state, dynamic?, labels?, tags?}``.

    ``contexts`` is a list (or a ``contexts_file`` jsonl) of dicts the template can cite —
    a home, a persona, a scene. Each call samples one. Dynamic criteria come back from the
    model only when the context does not fix them; otherwise ``context.dynamic`` wins.
    """
    prompt_tpl = (scenario.root / config["prompt_file"]).read_text(encoding="utf-8")
    contexts = list(config.get("contexts", []))
    if config.get("contexts_file"):
        for line in _resolve(scenario, config["contexts_file"]).read_text("utf-8").splitlines():
            if line.strip():
                contexts.append(json.loads(line))
    if not contexts:
        contexts = [{}]
    n_calls, per_call = int(config.get("calls", 1)), int(config.get("per_call", 10))
    client = ChatClient(config)
    only = config.get("questions")
    for call in range(n_calls):
        ctx = rng.choice(contexts)
        dynamic = ctx.get("dynamic")
        questions = (
            scenario.build_questions(dynamic, only=only)
            if (dynamic or not _needs_dynamic(scenario, only))
            else None
        )
        prompt = prompt_tpl.format(
            questions=json.dumps(
                questions
                or {q: scenario.raw["questions"][q] for q in (only or scenario.questions)},
                ensure_ascii=False,
                indent=1,
            ),
            context=json.dumps(ctx.get("text", ctx), ensure_ascii=False),
            n=per_call,
        )
        text = client.complete(prompt, temperature=float(config.get("temperature", 0.9)))
        items = _parse_json_array(text)
        for i, item in enumerate(items):
            dyn = dynamic or item.get("dynamic")
            qs = scenario.build_questions(dyn, only=only)
            labels = {}
            for qid, lab in (item.get("labels") or {}).items():
                if qid in qs:
                    labels[qid] = lab if isinstance(lab, dict) else {"gold": lab}
            yield Record(
                id=record_id(scenario.name, "llm", client.model, item["state"], sorted(qs)),
                scenario=scenario.name,
                source=f"llm:{client.model}",
                state=item["state"],
                questions=qs,
                labels=labels,
                tags=list(item.get("tags", [])) + [f"llm-call:{call}"],
                meta={"context": ctx.get("name"), "index": i},
            )


def _needs_dynamic(scenario: Scenario, only: list[str] | None) -> bool:
    return any(q.criteria_from for qid, q in scenario.questions.items() if not only or qid in only)


def _parse_json_array(text: str) -> list[dict]:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    start, end = s.find("["), s.rfind("]")
    if start < 0 or end < 0:
        raise ValueError(f"model did not return a JSON array: {text[:200]!r}")
    data = json.loads(s[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("model returned JSON that is not an array")
    return [d for d in data if isinstance(d, dict) and "state" in d]


class ChatClient:
    """Minimal OpenAI-compatible chat client (DashScope, OpenAI, local llama-server all fit)."""

    def __init__(self, config: dict):
        self.base_url = (
            config.get("base_url") or os.environ.get("EIDOLON_TRAIN_LLM_BASE_URL") or ""
        ).rstrip("/")
        self.model = config.get("model") or os.environ.get("EIDOLON_TRAIN_LLM_MODEL") or ""
        key_env = config.get("api_key_env", "EIDOLON_TRAIN_LLM_API_KEY")
        self.api_key = os.environ.get(key_env, "")
        self.timeout = float(config.get("timeout", 120))
        self.request_options = {
            name: config[name]
            for name in ("max_tokens", "thinking", "reasoning_effort", "response_format")
            if name in config
        }
        if not self.base_url or not self.model:
            raise ValueError(
                "llm generator needs base_url and model (or EIDOLON_TRAIN_LLM_BASE_URL / _MODEL)"
            )

    def complete(self, prompt: str, temperature: float = 0.9) -> str:
        return self.complete_with_meta(prompt, temperature=temperature)["text"]

    def complete_with_meta(self, prompt: str, temperature: float = 0.9) -> dict:
        """Return the same completion plus provider usage for bounded data generation."""
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            **self.request_options,
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("chat completion returned empty text content")
        return {
            "text": content,
            "id": data.get("id"),
            "model": data.get("model", self.model),
            "usage": data.get("usage") or {},
        }
