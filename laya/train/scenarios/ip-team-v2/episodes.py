"""Expand authored Ensemble episodes into Laya one-step records.

An episode is the split unit. Every state contains the original user request,
the actual completed public conversation and an opaque, shuffled candidate
list. The model never sees a future scripted turn or a device/permission grant.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from eidolon_laya_train.records import Record, target_vector

NAMES = {
    "train": ("阿岚", "小禾", "明川", "若竹", "云舟", "星野", "阿乔", "闻溪", "小岳", "林序", "青禾", "知遥"),
    "val": ("初晴", "小满", "秋山", "知夏", "路遥", "阿榕", "云鹤", "雨声"),
    "calib": ("沐言", "知远", "小檀", "景明", "苏荷", "望舒", "阿澄", "凌川"),
    "comparison": ("清川", "小鹿", "南星", "景初", "知鱼", "安禾", "向晚", "小石"),
    "holdout": ("一帆", "半夏", "予安", "林深", "晏宁", "远山", "小澈", "如月"),
}
ALL_ACTIONS = ("respond", "clarify", "wait", "finish")
TASKS = ("answer", "propose", "build", "challenge", "compare", "synthesize", "comfort", "clarify", "none")


def _render(value: str, names: dict[str, str]) -> str:
    return value.format_map(names)


def _moves(candidates: dict[str, str], allowed: list[str]) -> dict[str, str]:
    options = {}
    for key, profile in candidates.items():
        if "respond" in allowed:
            options[f"respond:{key}"] = f"让这一位成员回应或继续讨论：{profile}"
        if "clarify" in allowed:
            options[f"clarify:{key}"] = f"让这一位成员只问一个必要的澄清问题：{profile}"
    for action in ("wait", "finish"):
        if action in allowed:
            options[action] = {
                "wait": "当前轮次暂不接话，等待用户再次输入；不关闭本轮",
                "finish": "本轮讨论已经完成；团队仍保持，允许用户再输入",
            }[action]
    options["abstain"] = "无法可靠提出合法且有帮助的下一步，不授权任何人发言"
    return options


def _make_record(scenario, episode: dict, step: dict, state: dict, keys: dict[str, str], ordinal: int, variant: int, split: str) -> Record:
    action = step["action"]
    target = step.get("speaker")
    task = step["task"]
    if action not in (*ALL_ACTIONS, "abstain") or task not in TASKS:
        raise ValueError(f"{episode['id']}: unknown action/task")
    if action in ("respond", "clarify") and target not in keys:
        raise ValueError(f"{episode['id']}: speaker not in candidates")
    move = f"{action}:{keys[target]}" if action in ("respond", "clarify") else action
    if action in ("wait", "finish", "abstain") and (target is not None or task != "none"):
        raise ValueError(f"{episode['id']}: silent move has a speaker/task")
    if action == "clarify" and (task != "clarify" or not step.get("brief")):
        raise ValueError(f"{episode['id']}: clarify needs a bounded question task")
    if action == "respond" and (task in ("none", "clarify") or not step.get("brief")):
        raise ValueError(f"{episode['id']}: respond needs a bounded task")
    allowed = state["allowed_actions"]
    if action != "abstain" and action not in allowed:
        raise ValueError(f"{episode['id']}: gold action forbidden")
    members = state["candidates"]
    questions = scenario.build_questions({"moves": _moves(members, allowed)})
    labels = {"move": {"gold": move}, "task": {"gold": task}}
    for qid, label in labels.items():
        target_vector(questions[qid], label)
    root = f"ip-team-v2/{episode['id']}"
    return Record(
        id=f"{root}~v{variant:02d}-s{ordinal:02d}",
        scenario="ip-team-v2",
        source="authored:ensemble-episodes",
        state=state,
        questions=questions,
        labels=labels,
        tags=[episode["tag"], f"family:{episode['id']}", f"split-source:{split}"],
        split="eval" if split in ("comparison", "holdout") else split,
        meta={
            "family": episode["id"], "derived_from": root, "synthetic": True,
            "brief": _render(step.get("brief", ""), state["_names"]),
            "page_mode": episode["mode"],
        },
    )


def generate(scenario, config, rng):
    path = Path(config["path"])
    if not path.is_absolute():
        path = (scenario.root / path).resolve()
    episodes = yaml.safe_load(path.read_text(encoding="utf-8"))
    split = config["split_name"]
    variants = int(config.get("variants", 3))
    if split not in NAMES or variants < 1:
        raise ValueError("invalid split or variant count")
    seen = set()
    for episode in episodes:
        eid = episode["id"]
        if eid in seen:
            raise ValueError(f"duplicate episode {eid}")
        seen.add(eid)
        slots = list(episode["cast"])
        if not 1 <= len(slots) <= 6:
            raise ValueError(f"{eid}: candidate count outside pilot capacity")
        for variant in range(variants):
            chosen = rng.sample(NAMES[split], len(slots))
            names = dict(zip(slots, chosen, strict=True))
            candidates = [(slot, f"{names[slot]}；{_render(episode['cast'][slot], names)}") for slot in slots]
            rng.shuffle(candidates)
            keys = {slot: f"M{i}" for i, (slot, _) in enumerate(candidates)}
            members = {f"M{i}": profile for i, (_, profile) in enumerate(candidates)}
            history = []
            for item in episode.get("prelude", []):
                author = item["author"]
                history.append({
                    "author_kind": "user" if author == "user" else "companion",
                    "author": "用户" if author == "user" else names[author],
                    "text": _render(item["text"], names),
                })
            user = _render(episode["user"], names)
            trigger = {"author_kind": "user", "author": "用户", "text": user}
            history.append(trigger)
            budget = int(episode.get("budget", 5))
            for ordinal, step in enumerate(episode["steps"]):
                if "heard" in step:
                    heard = step["heard"]
                    slot = heard["speaker"]
                    if slot not in names:
                        raise ValueError(f"{eid}: unknown heard speaker")
                    trigger = {"author_kind": "companion", "author": names[slot], "text": _render(heard["text"], names)}
                    history.append(trigger)
                    budget = max(1, budget - 1)
                if "new_user" in step:
                    user = _render(step["new_user"], names)
                    trigger = {"author_kind": "user", "author": "用户", "text": user}
                    history.append(trigger)
                    budget = int(episode.get("budget", 5))
                allowed = step.get("allowed_actions", list(ALL_ACTIONS))
                if set(allowed) - set(ALL_ACTIONS):
                    raise ValueError(f"{eid}: invalid allowed actions")
                state = {
                    "user_request": user,
                    "scene_goal": _render(episode.get("goal", ""), names),
                    "trigger": dict(trigger),
                    "recent_public_messages": list(history),
                    "candidates": members,
                    "allowed_actions": allowed,
                    "remaining_replies": budget,
                    "_names": names,
                }
                record = _make_record(scenario, episode, step, state, keys, ordinal, variant, split)
                del record.state["_names"]
                yield record
