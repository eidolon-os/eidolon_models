"""Expand reviewed IP Team decision families into record-shaped Laya examples.

The family is the split unit. Names, candidate order, topics, and opaque member
keys vary within a family, so a paraphrase of one decision cannot cross a split.
This is synthetic development data, never a substitute for real PTT/ASR evidence.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from eidolon_laya_train.records import Record, target_vector

ROLE_DESCRIPTION = {
    "creative": "负责创意构思、表达和画面",
    "fact": "负责查证事实、来源和历史细节",
    "budget": "负责预算、成本和资源取舍",
    "tech": "负责技术可行性和实现步骤",
    "host": "负责主持讨论以及向用户澄清歧义",
    "story": "负责叙事结构和人物动机",
}

NAMES = {
    "train": ("阿岚", "小禾", "明川", "若竹", "云舟", "星野", "阿乔", "闻溪", "小岳", "林序", "青禾", "知遥"),
    "comparison": ("初晴", "小满", "秋山", "知夏", "路遥", "阿榕", "云鹤", "雨声"),
    "holdout": ("沐言", "知远", "小檀", "景明", "苏荷", "望舒", "阿澄", "凌川"),
}

TOPICS = {
    "train": ("登山路线", "社区阅读活动", "海边短片", "校园展览", "旧城改造", "科普播客", "公益市集"),
    "comparison": ("博物馆夜游", "社区花园", "儿童戏剧", "湿地步道"),
    "holdout": ("天文展厅", "街区音乐节", "河岸骑行", "城市寻宝"),
}


def _render(value: str, bindings: dict[str, str]) -> str:
    return value.format_map(bindings)


def _record(scenario, family: dict, alternative: dict, variant: int, split_name: str, rng) -> Record:
    roles = family.get("roles", ["creative", "fact", "host"])
    if family.get("no_candidates"):
        roles = []
    if len(set(roles)) != len(roles) or any(role not in ROLE_DESCRIPTION for role in roles):
        raise ValueError(f"{family['id']}: invalid or duplicate roles")
    names = rng.sample(NAMES[split_name], len(roles))
    if family.get("duplicate_names"):
        if len(names) < 2:
            raise ValueError(f"{family['id']}: duplicate_names needs two candidates")
        names[1] = names[0]
    by_role = dict(zip(roles, names, strict=True))
    bindings = {
        "a": names[0] if names else "",
        "b": names[1] if len(names) > 1 else "",
        "c": names[2] if len(names) > 2 else "",
        "topic": rng.choice(TOPICS[split_name]),
        **{f"{role}_name": name for role, name in by_role.items()},
    }
    # The model sees positional keys, not production Companion IDs. The adapter
    # maps a chosen option back to an ID after validating the current snapshot.
    candidates = [(role, name) for role, name in zip(roles, names, strict=True)]
    rng.shuffle(candidates)
    members = {}
    role_to_key = {}
    name_to_key = {}
    for index, (role, name) in enumerate(candidates):
        key = f"M{index}"
        members[key] = f"{name}；{ROLE_DESCRIPTION[role]}"
        role_to_key[role] = key
        name_to_key.setdefault(name, []).append(key)
    user = _render(str(alternative.get("user", family["user"])), bindings)
    kind = alternative.get("trigger_kind", family.get("trigger_kind", "user"))
    trigger_tpl = alternative.get("trigger", family.get("trigger", ""))
    trigger = _render(str(trigger_tpl), bindings) if kind == "companion" else user
    author_role = alternative.get("trigger_author", family.get("trigger_author", roles[0] if roles else ""))
    public = []
    for entry in family.get("history", []):
        history_author = entry.get("author", "user")
        if history_author != "user" and history_author not in by_role:
            raise ValueError(f"{family['id']}: history author is not a candidate")
        public.append({
            "author": "用户" if history_author == "user" else by_role[history_author],
            "text": _render(str(entry["text"]), bindings),
            "played": True,
        })
    if kind == "companion":
        if author_role not in by_role:
            raise ValueError(f"{family['id']}: trigger author is not a candidate")
        public.append({"author": by_role[author_role], "text": trigger, "played": True})
    if family.get("long_history"):
        # Short, completed public messages only. The initiating request stays in
        # its own field even when these entries fill the recent-history window.
        public = [
            {"author": names[i % len(names)], "text": f"关于{bindings['topic']}的第{i + 1}点讨论已经讲完。", "played": True}
            for i in range(10)
        ] + public
    allowed = alternative.get("allowed_actions", family.get("allowed_actions", ["respond", "clarify", "wait", "finish"]))
    state = {
        "user_request": user,
        "trigger": {"author_kind": kind, "author": by_role.get(author_role, "用户") if kind == "companion" else "用户", "text": trigger},
        "scene_goal": _render(str(family.get("scene_goal", "")), bindings),
        "recent_public_messages": public,
        "candidates": members,
        "allowed_actions": allowed,
        "remaining_replies": 1 + variant % 4,
    }
    action = alternative.get("action", family["action"])
    target = alternative.get("target", family.get("target"))
    speaker = "NONE"
    if action in ("respond", "clarify"):
        if target in role_to_key:
            speaker = role_to_key[target]
        elif target in ("a", "b", "c"):
            name = bindings[target]
            keys = name_to_key.get(name, [])
            if len(keys) != 1:
                raise ValueError(f"{family['id']}: {target} is not uniquely resolvable")
            speaker = keys[0]
        else:
            raise ValueError(f"{family['id']}: no valid target for {action}")
    if action not in scenario.questions["action"].criteria:
        raise ValueError(f"{family['id']}: invalid action {action}")
    if action != "abstain" and action not in allowed:
        raise ValueError(f"{family['id']}: gold action forbidden by constraints")
    questions = scenario.build_questions({"members": members})
    labels = {"action": {"gold": action}, "speaker": {"gold": speaker}}
    for qid, label in labels.items():
        target_vector(questions[qid], label)
    root = f"ip-team/{family['id']}"
    return Record(
        id=f"{root}~v{variant:02d}-{alternative.get('id', 'main')}",
        scenario="ip-team",
        source="authored:ip-team-families",
        state=state,
        questions=questions,
        labels=labels,
        tags=[family["tag"], f"family:{family['id']}", f"split-source:{split_name}"],
        meta={"family": family["id"], "derived_from": root, "synthetic": True},
    )


def generate(scenario, config, rng):
    source = Path(config["path"])
    if not source.is_absolute():
        source = (scenario.root / source).resolve()
    families = yaml.safe_load(source.read_text(encoding="utf-8"))
    split_name = config.get("split_name", "train")
    variants = int(config.get("variants", 20))
    if split_name not in NAMES or variants < 1:
        raise ValueError("invalid split_name or variants")
    ids = set()
    for family in families:
        if family["id"] in ids:
            raise ValueError(f"duplicate family {family['id']}")
        ids.add(family["id"])
        for variant in range(variants):
            for alternative in family.get("alternatives", [{}]):
                yield _record(scenario, family, alternative, variant, split_name, rng)
