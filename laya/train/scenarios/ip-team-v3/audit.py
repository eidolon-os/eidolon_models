"""Reject split leakage, illegal text decisions and silent Laya truncation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from eidolon_models_laya.vendor.laya.common import build_sequence, serialize_state
from eidolon_laya_train.model import record_items, to_internal
from eidolon_laya_train.records import read_jsonl, target_vector


def percentile(xs: list[int], p: float) -> int:
    return sorted(xs)[min(len(xs) - 1, int((len(xs) - 1) * p))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--comparison", required=True, type=Path)
    ap.add_argument("--holdout", required=True, type=Path)
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--head-max-len", type=int, default=256)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    paths = {split: args.dataset / f"{split}.jsonl" for split in ("train", "val", "calib")}
    paths.update(comparison=args.comparison, holdout=args.holdout)
    families, ids, pair_groups, out = {}, set(), defaultdict(list), {}
    for split, path in paths.items():
        records = list(read_jsonl(path))
        if not records:
            raise ValueError(f"{split}: empty")
        fs = {r.meta["family"] for r in records}
        for other, prior in families.items():
            if fs & prior:
                raise ValueError(f"family leakage {split}/{other}: {fs & prior}")
        families[split] = fs
        actions, modes, candidate_counts, context_counts, lengths = Counter(), Counter(), Counter(), Counter(), []
        for r in records:
            if r.id in ids:
                raise ValueError(f"duplicate record id {r.id}")
            ids.add(r.id)
            if list(r.questions) != ["move"] or list(r.labels) != ["move"]:
                raise ValueError(f"{r.id}: wrong question set")
            q = r.questions["move"]
            target_vector(q, r.labels["move"])
            action = r.labels["move"]["gold"].split(":", 1)[0]
            allowed = r.state["allowed_actions"]
            if not allowed or len(set(allowed)) != len(allowed) or action not in allowed:
                raise ValueError(f"{r.id}: SDK-invalid allowed actions/gold")
            if action in ("respond", "clarify") and r.labels["move"]["gold"].split(":", 1)[1] not in r.state["candidates"]:
                raise ValueError(f"{r.id}: candidate missing")
            if len(r.state["candidates"]) > 6:
                raise ValueError(f"{r.id}: candidate limit")
            state_tokens = tok(serialize_state(r.state), add_special_tokens=False)["input_ids"]
            head, _ = build_sequence(tok, "", to_internal(q), args.max_len, args.head_max_len)
            total = len(head) + len(state_tokens)
            if total > args.max_len:
                raise ValueError(f"{r.id}: {total} tokens exceed {args.max_len}, must not truncate")
            if len(record_items(r, tok, {"max_len": args.max_len, "head_max_len": args.head_max_len})) != 1:
                raise ValueError(f"{r.id}: lost a choice option")
            lengths.append(total)
            actions[action] += 1
            modes[r.meta["page_mode"]] += 1
            candidate_counts[len(r.state["candidates"])] += 1
            context_counts[
                len(r.state["prior_public_messages"]) + 1
                + (0 if r.state["trigger"].get("same_as_user_request") else 1)
            ] += 1
            if r.meta.get("pair_id"):
                variant = r.id.split("~v", 1)[1].split("-", 1)[0]
                pair_groups[(split, r.meta["pair_id"], variant)].append(r)
        out[split] = {
            "records": len(records), "families": len(fs), "actions": dict(actions),
            "modes": dict(modes), "candidates": dict(candidate_counts),
            "public_context_messages": dict(context_counts),
            "input_tokens": {"min": min(lengths), "median": percentile(lengths, .5),
                             "p95": percentile(lengths, .95), "max": max(lengths)},
        }
    for key, group in pair_groups.items():
        if len(group) != 2:
            raise ValueError(f"{key}: expected two counterfactual branches")
        x, y = group
        if x.state["candidates"] != y.state["candidates"] or x.state["user_request"] != y.state["user_request"]:
            raise ValueError(f"{key}: candidate/request drift")
        if x.state["trigger"] == y.state["trigger"] and x.state["prior_public_messages"] == y.state["prior_public_messages"]:
            raise ValueError(f"{key}: branches are identical")
    result = json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
