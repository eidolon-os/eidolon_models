"""Prepare paired IP-only and joint pilot datasets with family-stratified splits.

This local experiment stage leaves the generic assembler and historical home
splits untouched. Home replay enters train only; both pilots use identical IP
validation and calibration records.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from eidolon_laya_train.assemble import stable_file_hash
from eidolon_laya_train.records import Record, read_jsonl, stable_hash, write_jsonl


RESERVED_TRAIN = {
    "conditional_cost_or_visual",
    "conditional_fact_or_story",
    "answer_named",
    "answer_role",
}


def _state_hash(record: Record) -> str:
    return stable_hash(json.dumps(record.state, ensure_ascii=False, sort_keys=True))


def prepare(ip_source: Path, home_source: Path, eval_sets: list[Path], ip_out: Path, joint_out: Path) -> dict:
    ip = list(read_jsonl(ip_source))
    by_family: dict[str, list[Record]] = defaultdict(list)
    for record in ip:
        if record.scenario != "ip-team" or record.meta.get("synthetic") is not True:
            raise ValueError(f"invalid IP source record {record.id}")
        by_family[record.meta["family"]].append(record)
    if len({r.id for r in ip}) != len(ip):
        raise ValueError("duplicate IP record ids")
    if not RESERVED_TRAIN <= by_family.keys():
        raise ValueError("reserved train family missing")

    by_action: dict[str, list[str]] = defaultdict(list)
    for family, family_records in by_family.items():
        actions = {r.labels["action"]["gold"] for r in family_records}
        if len(actions) != 1:
            raise ValueError(f"family {family} changes action; split rule needs a primary class")
        by_action[next(iter(actions))].append(family)

    split_of = {family: "train" for family in by_family}
    for action, families in sorted(by_action.items()):
        available = sorted(
            (family for family in families if family not in RESERVED_TRAIN),
            key=lambda family: stable_hash(f"ip-pilot-split-v1:{family}"),
        )
        take = 2 if action == "respond" else 1
        if len(available) < take * 2 + 1:
            raise ValueError(f"not enough {action} families for stratified split")
        for family in available[:take]:
            split_of[family] = "val"
        for family in available[take : take * 2]:
            split_of[family] = "calib"

    locked = list(r for path in eval_sets for r in read_jsonl(path))
    locked_ids = {r.id for r in locked}
    locked_states = {_state_hash(r) for r in locked}
    if any(r.id in locked_ids or _state_hash(r) in locked_states for r in ip):
        raise ValueError("IP source overlaps locked comparison/holdout")

    ip_parts: dict[str, list[Record]] = {"train": [], "val": [], "calib": []}
    state_parts: dict[str, set[str]] = defaultdict(set)
    for family, family_records in by_family.items():
        split = split_of[family]
        for r in family_records:
            r.split = split
            state_parts[_state_hash(r)].add(split)
            ip_parts[split].append(r)
    if any(len(splits) > 1 for splits in state_parts.values()):
        raise ValueError("identical IP state crosses train/val/calib")

    # r17's source is already its training split; use unique IDs and a fixed
    # 2.5% hash sample. Keeping it out of val/calib makes the comparison fair.
    home_by_id = {r.id: r for r in read_jsonl(home_source)}
    home = [
        r for rid, r in sorted(home_by_id.items())
        if int(stable_hash(f"ip-pilot-home-replay-v1:{rid}")[:8], 16) / 0xFFFFFFFF < 0.025
    ]
    if any(r.scenario != "smart-home" or r.split != "train" for r in home):
        raise ValueError("home replay must come only from r17 train")
    if any(r.id in locked_ids or _state_hash(r) in locked_states for r in home):
        raise ValueError("home replay overlaps locked IP set")

    manifest = {
        "ip_source": str(ip_source.resolve()),
        "ip_source_sha256": stable_file_hash(ip_source),
        "home_source": str(home_source.resolve()),
        "home_source_sha256": stable_file_hash(home_source),
        "locked_eval": {str(p.resolve()): stable_file_hash(p) for p in eval_sets},
        "split_rule": "family-stratified-v1; respond 2/2 val/calib, each other action 1/1; reserved four train families",
        "family_split": split_of,
        "ip_count": {k: len(v) for k, v in ip_parts.items()},
        "ip_action_by_split": {
            k: dict(Counter(r.labels["action"]["gold"] for r in v)) for k, v in ip_parts.items()
        },
        "joint_home_replay_unique": len(home),
        "joint_home_fraction": 0.025,
    }
    for out, train in ((ip_out, ip_parts["train"]), (joint_out, ip_parts["train"] + home)):
        dataset = out / "dataset"
        dataset.mkdir(parents=True, exist_ok=True)
        for part in ("train", "val", "calib"):
            rows = train if part == "train" else ip_parts[part]
            write_jsonl(dataset / f"{part}.jsonl", rows)
        (dataset / "manifest.json").write_text(
            json.dumps({**manifest, "mode": "ip-only" if out == ip_out else "joint"}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip-source", type=Path, required=True)
    parser.add_argument("--home-source", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, action="append", required=True)
    parser.add_argument("--ip-out", type=Path, required=True)
    parser.add_argument("--joint-out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.ip_source, args.home_source, args.eval_set, args.ip_out, args.joint_out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
