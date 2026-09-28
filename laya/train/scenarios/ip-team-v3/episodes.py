"""Expand original text-only Ensemble episodes into one legal next-move choice.

The pinned user request and newest trigger are represented once. The full
public text supplied to this generator is retained; a separate tokenizer
preflight rejects samples that do not fit the model window.
"""

from __future__ import annotations

import hashlib
import random
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
ACTIONS = ("respond", "clarify", "wait", "finish")
BACKGROUND = (
    "刚才我们讨论过{topic}。我原本想把所有材料一次准备齐，但还不知道哪些已经有了，哪些需要另外借用。先把想法写下来，等看过现有条件后再确定具体做法。",
    "我的建议是从一件容易完成的小事开始，不要一上来排满整个周末。可以先列出两三种做法，再看大家的时间是否合得上；如果要花钱，也先估个大致范围。",
    "这个办法有趣，不过场地和时间还得再确认。上次我们也遇到过计划太满的问题，最后只做了最简单的一部分，反而更轻松。让参与的人有休息和改变主意的余地会好些。",
    "我把关于{topic}的意见归一下：有可先试的小步骤，也有需要核实的限制。现在还没有确定日期和负责的人，所以只是几个备选。等信息齐一点，再看哪种方式最省力。",
    "如果使用家里已有的工具，成本会低不少。收拾旧物时也可以看看哪些还能继续用，不必因为一个想法就买一整套新东西。只是这样可能会多花一点整理时间。",
    "时间安排上，我希望先把准备和执行分开：周五晚上检查材料，周末只做实际活动。这样不用一边寻找东西一边赶进度，也更容易让临时来帮忙的人看懂该做什么。",
    "大家的思路已经比较清楚了。我还想知道参与人数大概有多少；人数不同，场地和材料的需求就不一样。先把这个问题记在这里，等有消息再继续往下安排。",
    "我觉得还有一个容易忽略的地方：结束后谁负责收拾。活动本身也许只要一个小时，但清理和归还借来的东西同样要时间。若不提前想到，临走时会有点手忙脚乱。",
    "对，收尾可以简单一些，比如先把用过的东西按原位放回，再确认桌面和地面没有遗漏。只要流程清楚，下一次再做类似的事就不必重新从头摸索。",
    "我听下来，大部分人都希望事情轻松些，不追求一次做得特别完整。把准备工作压缩到必要的几项，真正开始后再根据情况微调，这样更符合最初的想法。",
    "关于{topic}，现阶段能说清的就是先小规模尝试、确认时间与材料，结束后再回顾。其他细节等条件明确再定。先把这些想法留着，后面有新的需求时再继续聊。",
)


def _render(value: str, names: dict[str, str]) -> str:
    return value.format_map(names)


def _options(members: dict[str, str], allowed: list[str]) -> dict[str, str]:
    options = {}
    for key, profile in members.items():
        if "respond" in allowed:
            options[f"respond:{key}"] = f"让 {profile} 回应或继续讨论"
        if "clarify" in allowed:
            options[f"clarify:{key}"] = f"让 {profile} 询问必要的澄清问题"
    if "wait" in allowed:
        options["wait"] = "等待用户再次输入，本轮保持开放"
    if "finish" in allowed:
        options["finish"] = "当前讨论已经完成，结束本轮发言"
    options["abstain"] = "无法可靠提出合法且有帮助的下一步"
    return options


def _local_rng(config: dict, split: str, episode: dict, variant: int, fallback: random.Random) -> random.Random:
    pair = episode.get("pair_id")
    if not pair:
        return fallback
    digest = hashlib.sha256(f"{config.get('pair_seed', 0)}:{split}:{pair}:{variant}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _background(topic: str, slots: list[str], names: dict[str, str], keys: dict[str, str]) -> list[dict]:
    out = []
    for i, template in enumerate(BACKGROUND):
        speaker = None if i == 0 else slots[(i - 1) % len(slots)]
        out.append({"author_kind": "user" if speaker is None else "companion",
                    "author_id": "user" if speaker is None else keys[speaker],
                    "author_name": "用户" if speaker is None else names[speaker],
                    "text": template.format(topic=topic)})
    return out


def generate(scenario, config, rng):
    path = Path(config["path"])
    if not path.is_absolute():
        path = (scenario.root / path).resolve()
    episodes = yaml.safe_load(path.read_text(encoding="utf-8"))
    split = config["split_name"]
    variants = int(config.get("variants", 3))
    skip = set(config.get("skip_ids", []))
    select_split = config.get("select_split")
    if split not in NAMES or variants < 1:
        raise ValueError("invalid split or variants")
    seen = set()
    for episode in episodes:
        eid = episode["id"]
        if eid in seen:
            raise ValueError(f"duplicate episode {eid}")
        seen.add(eid)
        if eid in skip or (select_split and episode.get("split") != select_split):
            continue
        slots = list(episode["cast"])
        if not 1 <= len(slots) <= 6:
            raise ValueError(f"{eid}: invalid candidate count")
        budget_initial = int(episode.get("budget", 8))
        if not 1 <= budget_initial <= 32:
            raise ValueError(f"{eid}: invalid remaining reply budget")
        for variant in range(variants):
            local = _local_rng(config, split, episode, variant, rng)
            names = dict(zip(slots, local.sample(NAMES[split], len(slots)), strict=True))
            for target, source in episode.get("same_name", {}).items():
                if target not in names or source not in names:
                    raise ValueError(f"{eid}: unknown same-name slot")
                names[target] = names[source]
            ordered = [(s, f"{names[s]}；{_render(episode['cast'][s], names)}") for s in slots]
            local.shuffle(ordered)
            keys = {s: f"M{i}" for i, (s, _) in enumerate(ordered)}
            members = {f"M{i}": profile for i, (_, profile) in enumerate(ordered)}
            history = _background(episode["history_stress"], slots, names, keys) if episode.get("history_stress") else []
            for item in episode.get("prelude", []):
                author = item["author"]
                if author != "user" and author not in names:
                    raise ValueError(f"{eid}: unknown prelude author")
                history.append({"author_kind": "user" if author == "user" else "companion",
                                "author_id": "user" if author == "user" else keys[author],
                                "author_name": "用户" if author == "user" else names[author],
                                "text": _render(item["text"], names)})
            user = _render(episode["user"], names)
            pinned_index = len(history)
            history.append({"author_kind": "user", "author_id": "user", "author_name": "用户", "text": user})
            budget = budget_initial
            for ordinal, step in enumerate(episode["steps"]):
                if "heard" in step:
                    heard = step["heard"]
                    if heard["speaker"] not in names:
                        raise ValueError(f"{eid}: unknown heard speaker")
                    history.append({"author_kind": "companion", "author_id": keys[heard["speaker"]],
                                    "author_name": names[heard["speaker"]],
                                    "text": _render(heard["text"], names)})
                    budget -= 1
                if "new_user" in step:
                    user = _render(step["new_user"], names)
                    pinned_index = len(history)
                    history.append({"author_kind": "user", "author_id": "user", "author_name": "用户", "text": user})
                    budget = budget_initial
                if budget < 1:
                    raise ValueError(f"{eid}: budget exhausted before model decision")
                allowed = step.get("allowed_actions", list(ACTIONS))
                if not allowed or len(set(allowed)) != len(allowed) or set(allowed) - set(ACTIONS):
                    raise ValueError(f"{eid}: SDK-invalid allowed actions")
                action, target = step["action"], step.get("speaker")
                if action not in ACTIONS or action not in allowed:
                    raise ValueError(f"{eid}: invalid gold action")
                if action in ("respond", "clarify") and target not in keys:
                    raise ValueError(f"{eid}: unknown gold speaker")
                if action in ("wait", "finish") and target is not None:
                    raise ValueError(f"{eid}: silent action has speaker")
                move = f"{action}:{keys[target]}" if target else action
                prior = [dict(m) for i, m in enumerate(history[:-1]) if i != pinned_index]
                trigger = dict(history[-1])
                if len(history) - 1 == pinned_index:
                    trigger = {"author_kind": "user", "same_as_user_request": True}
                state = {
                    "user_request": user,
                    "scene_goal": _render(episode.get("goal", ""), names),
                    "candidates": members,
                    "allowed_actions": allowed,
                    "remaining_replies": budget,
                    "trigger": trigger,
                    "prior_public_messages": prior,
                }
                questions = scenario.build_questions({"moves": _options(members, allowed)})
                target_vector(questions["move"], {"gold": move})
                yield Record(
                    id=f"ip-team-v3/{eid}~v{variant:02d}-s{ordinal:02d}",
                    scenario="ip-team-v3", source="authored:ensemble-v3-text",
                    state=state, questions=questions, labels={"move": {"gold": move}},
                    tags=[episode["tag"], f"family:{eid}", f"mode:{episode['mode']}"],
                    split="eval" if split in ("comparison", "holdout") else split,
                    meta={"family": eid, "synthetic": True, "page_mode": episode["mode"],
                          "pair_id": episode.get("pair_id"), "branch": episode.get("branch"),
                          "history_stress": bool(episode.get("history_stress")),
                          "brief": _render(step.get("brief", ""), names)},
                )
