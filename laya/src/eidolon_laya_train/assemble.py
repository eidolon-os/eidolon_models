"""``assemble``: mix sources into one dataset with train / val / calib splits.

Config::

    sources:
      - path: runs/a/augmented.jsonl
        weight: 1.0             # >1 repeats, <1 subsamples (deterministic by id hash)
      - path: train/data/chinese-laya/train.jsonl
        weight: 0.3
        scenario: replay        # optional: override the scenario tag for balancing
    locked_eval:                # ids (and identical states) that must never reach train
      - ../evals/smart-home
    split: {val: 0.1, calib: 0.1}
    balance: {by: scenario, max_ratio: 3.0}   # cap any group at max_ratio × the smallest
    tag_weights: {implicit-intent: 1.5}        # multiply the weight of records whose slice (first
                                               # tag) matches; also allowed per source
    seed: 7

Splits are by the hash of a connected group: an original and its augmentations, plus
records with identical states, always stay together. Re-assembling normally keeps old
groups in their split; adding a new duplicate that joins two groups can move one group.
Locked eval sets are excluded both by id and by the hash of the serialized state, so a
generated case that happens to repeat an eval utterance is excluded too.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .records import Record, read_jsonl, stable_hash, write_jsonl


def stable_file_hash(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _state_key(r: Record) -> str:
    return stable_hash(json.dumps(r.state, ensure_ascii=False, sort_keys=True))


def locked_keys(paths: list[Path]) -> tuple[set[str], set[str]]:
    """Ids and state hashes of every locked eval record. Accepts record jsonl files, or an
    evals directory (``scenarios/*/cases.jsonl`` with ``text``)."""
    ids, states = set(), set()
    for p in paths:
        if p.is_dir():
            for f in p.glob("scenarios/*/cases.jsonl"):
                for line in f.read_text("utf-8").splitlines():
                    if line.strip():
                        c = json.loads(line)
                        ids.add(c["id"])
                        states.add(
                            stable_hash(
                                json.dumps(
                                    {"utterance": c["text"]}, ensure_ascii=False, sort_keys=True
                                )
                            )
                        )
        else:
            for r in read_jsonl(p):
                ids.add(r.id)
                ids.add(r.id.split("/", 1)[-1])
                states.add(_state_key(r))
    return ids, states


def _split_of(group_id: str, fractions: dict[str, float]) -> str:
    x = int(stable_hash("split:" + group_id)[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, frac in fractions.items():
        acc += frac
        if x < acc:
            return name
    return "train"


def _split_groups(records: list[Record]) -> dict[str, str]:
    """Join augmentation families and identical states before assigning splits.

    ``derived_from`` points to the immediate parent, which can itself be augmented.
    The id prefix before the first ``~`` is the stable root even if intermediate
    records were removed by source weighting.
    """
    parent = {r.id: r.id for r in records}

    def root(rid: str) -> str:
        while parent[rid] != rid:
            parent[rid] = parent[parent[rid]]
            rid = parent[rid]
        return rid

    def join(a: str, b: str) -> None:
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    by_family: dict[str, str] = {}
    by_state: dict[str, str] = {}
    for r in records:
        family = str(r.meta.get("derived_from") or r.id).split("~", 1)[0]
        for index, key in ((by_family, family), (by_state, _state_key(r))):
            if key in index:
                join(r.id, index[key])
            else:
                index[key] = r.id
    return {r.id: root(r.id) for r in records}


def assemble(config: dict, out_dir: Path, base: Path) -> dict:
    rng = random.Random(int(config.get("seed", 7)))
    fractions = {
        k: float(v) for k, v in (config.get("split") or {"val": 0.1, "calib": 0.1}).items()
    }
    lock_ids, lock_states = locked_keys(
        [(base / p).resolve() for p in config.get("locked_eval", [])]
    )

    kept: dict[str, Record] = {}
    per_source: Counter = Counter()
    dropped = Counter()
    for src in config["sources"]:
        path = (base / src["path"]).resolve()
        weight = float(src.get("weight", 1.0))
        tag_weights = {**(config.get("tag_weights") or {}), **(src.get("tag_weights") or {})}
        for r in read_jsonl(path):
            if src.get("scenario"):
                r.scenario = src["scenario"]
            if not r.labels:
                dropped["unlabelled"] += 1
                continue
            bare = r.id.split("/", 1)[-1]
            if r.id in lock_ids or bare in lock_ids or _state_key(r) in lock_states:
                dropped["locked"] += 1
                continue
            if r.id in kept:
                dropped["duplicate_id"] += 1
                continue
            # deterministic subsample / repeat
            u = int(stable_hash("w:" + r.id)[:8], 16) / 0xFFFFFFFF
            # per-slice multiplier on the record's first tag (its slice), so a growing slice
            # elsewhere does not dilute the ones the gate cares about
            w = weight * float(tag_weights.get(r.tags[0] if r.tags else "", 1.0))
            copies = int(w) + (1 if u < w - int(w) else 0)
            if copies <= 0:
                dropped["weighted_out"] += 1
                continue
            r.meta["copies"] = copies
            r.meta["source_file"] = str(path)
            kept[r.id] = r
            per_source[str(path)] += 1

    # balance groups (before splitting, on unique records)
    bal = config.get("balance")
    if bal:
        by = bal.get("by", "scenario")
        max_ratio = float(bal.get("max_ratio", 3.0))
        groups: dict[str, list[Record]] = defaultdict(list)
        for r in kept.values():
            groups[getattr(r, by) if by in ("scenario", "source") else r.meta.get(by, "")].append(r)
        smallest = min(len(g) for g in groups.values())
        cap = int(max_ratio * smallest)
        for g, rs in groups.items():
            if len(rs) > cap:
                rs_sorted = sorted(rs, key=lambda r: r.id)
                rng.shuffle(rs_sorted)
                for r in rs_sorted[cap:]:
                    del kept[r.id]
                    dropped[f"balanced_out:{g}"] += 1

    group_ids = _split_groups(list(kept.values()))
    splits: dict[str, list[Record]] = {"train": [], "val": [], "calib": []}
    for r in kept.values():
        s = _split_of(group_ids[r.id], fractions)
        r.split = s
        if s == "train":
            splits["train"].extend([r] * r.meta.get("copies", 1))
        else:
            splits.setdefault(s, []).append(r)
    rng.shuffle(splits["train"])

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rs in splits.items():
        counts[name] = write_jsonl(out_dir / f"{name}.jsonl", rs)
    manifest = {
        "config": config,
        "counts": counts,
        "unique": len(kept),
        "split_groups": len(set(group_ids.values())),
        "per_source": dict(per_source),
        "source_sha256": {
            str((base / src["path"]).resolve()): stable_file_hash((base / src["path"]).resolve())
            for src in config["sources"]
        },
        "dropped": dict(dropped),
        "by_scenario": dict(Counter(r.scenario for r in kept.values())),
        "by_qtype": dict(Counter(q["type"] for r in kept.values() for q in r.questions.values())),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
