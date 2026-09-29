"""c 系列剧本（LABELING.md、PLAN.md §2）：户型 + 上下文 + 意图家族 → 金标。

写手只写 utterance；选项、状态和金标都在这里定。同一个上下文出 1–3 个不同家族的剧本（同组切分），
得到“同一上下文、不同话语”的对照。用法（在 laya/ 下）：

    .venv/bin/python train/scenarios/smart-home-continuation/frames.py --out train/data/continuation

写出 frames-{train,dev,test}.jsonl（含金标）和 writer/{split}-NN.jsonl（给写手，不含金标）。
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

LAYA = Path(__file__).resolve().parents[3]
HOME_DIRS = (LAYA / "train/data/homes/smart-home", LAYA / "evals/smart-home/homes")
SPLIT_HOMES = {
    "train": ("brand", "countryside", "duplex", "family3", "kids", "minimal", "northflat",
              "office", "rental", "studio", "townhouse"),
    "dev": ("elder", "pets", "southyard"),
    "test": ("apartment", "house", "mega"),
}
SIZES = {"train": 3000, "dev": 600, "test": 600}
SEEDS = {"train": 101, "dev": 202, "test": 303}
PICK_SHARE = 0.6
WRITER_BATCH = 150

# SDK 设备类型的中文名（eidolon_sdk.biz.smarthome.DEVICE_TYPES 的 label），Agent 的选项说明用它。
KIND_LABEL = {"light": "灯", "switch": "开关", "climate": "空调", "water_heater": "热水器",
              "cover": "窗帘", "fan": "风扇/净化/加湿", "media": "影音", "appliance": "家电",
              "camera": "摄像头"}
# 候选必须同类型才会一起出现在一次歧义里的 SDK 类型（“把加湿器打开”不会和净化器歧义）。
SAME_TYPE = {"fan", "media", "appliance", "switch"}
KIND_NOUN = {"light": "灯", "climate": "空调", "cover": "窗帘", "water_heater": "热水器",
             "camera": "摄像头"}

ON, OFF, UP, DOWN, SET, PAUSE = ("打开或启动", "关闭或停止", "调高或增大", "调低或减小",
                                 "设为指定的数值或模式", "暂停")
VERBS = (ON, OFF, UP, DOWN, SET, PAUSE)
# 单句 action 题的原说明（scenarios/smart-home/scenario.yaml），不含“上锁”。
VERB_DESC = {ON: "开启、启动、开始运行", OFF: "关掉、停止、断电、收起",
             UP: "调亮、升温、调大音量或档位", DOWN: "调暗、降温、调小、降下",
             SET: "设到具体的温度、档位、百分比或模式", PAUSE: "暂时停下，之后再继续"}
_STEP = {"light", "climate", "water_heater", "media"}
VERB_KINDS = {ON: set(KIND_LABEL), OFF: set(KIND_LABEL), UP: _STEP, DOWN: _STEP,
              SET: {"light", "climate", "water_heater", "cover", "fan", "media"}, PAUSE: {"appliance"}}
CANCEL, REDO = "取消", "重新理解"


def kind_of(category: str, typ: str) -> str | None:
    """户型清单的品类 / 类型 → SDK 设备类型；门锁、传感器、车库门等不进续接。"""
    if category == "照明":
        return "light"
    if category == "空调温控":
        return "climate"
    if category == "窗帘遮阳":
        return "cover"
    if category in ("空气环境", "风扇"):
        return "fan"
    if category == "影音娱乐":
        return "media"
    if category == "清洁电器":
        return "appliance"
    if category == "厨房电器":
        if "油烟机" in typ:
            return "fan"
        return "appliance" if typ in ("电饭煲", "烤箱", "洗碗机", "胶囊咖啡机") else None
    if category == "热水卫浴":
        return "water_heater" if "热水器" in typ else "switch"
    if category == "插座开关":
        return "switch"
    if category == "安防门禁":
        return "camera" if typ in ("摄像头", "可视门铃") else None
    if category == "宠物":
        return {"自动喂食器": "appliance", "宠物摄像头": "camera"}.get(typ)
    return None


@dataclass(frozen=True)
class Device:
    name: str
    room: str
    type: str
    kind: str


def load_home(name: str) -> list[Device]:
    for d in HOME_DIRS:
        p = d / f"{name}.json"
        if p.exists():
            raw = json.loads(p.read_text("utf-8"))
            break
    else:
        raise FileNotFoundError(name)
    devs = []
    for x in raw["devices"]:
        kind = kind_of(x["category"], x["type"])
        if kind is not None:
            devs.append(Device(x["name"], x["room"], x["type"], kind))
    return devs


def verbs_for(kinds) -> list[str]:
    return [v for v in VERBS if all(k in VERB_KINDS[v] for k in kinds)]


# ------------------------------------------------------------------ 动作的中文写法


def set_value(kind: str, rng: random.Random) -> str:
    if kind == "light":
        return f"{rng.choice([20, 30, 40, 50, 60, 70, 80, 100])}%"
    if kind == "climate":
        if rng.random() < 0.3:
            return rng.choice(["制冷", "制热", "自动", "除湿", "送风"])
        return f"{rng.randint(18, 28)}°C"
    if kind == "water_heater":
        return f"{rng.choice([40, 42, 45, 48, 50, 55])}°C"
    if kind == "cover":
        return f"{rng.choice([30, 50, 70])}%"
    if kind == "fan":
        return f"{rng.choice([30, 50, 80])}%"
    return str(rng.choice([10, 15, 20, 25, 30, 40]))  # media 音量


def action_phrase(kind: str, verb: str, value: str | None = None) -> str:
    """Agent 状态里 `待执行` 的写法。"""
    if verb == ON:
        return {"appliance": "启动", "cover": "打开"}.get(kind, "打开")
    if verb == OFF:
        return {"appliance": "停止", "cover": "关上"}.get(kind, "关闭")
    if verb == UP:
        return {"light": "调亮", "media": "调大音量"}.get(kind, "调高温度")
    if verb == DOWN:
        return {"light": "调暗", "media": "调小音量"}.get(kind, "调低温度")
    if verb == PAUSE:
        return "暂停"
    assert value is not None
    if kind == "light":
        return f"亮度调到{value}"
    if kind == "cover":
        return f"开到{value}"
    if kind == "fan":
        return f"风速调到{value}"
    if kind == "media":
        return f"音量调到{value}"
    if value.endswith("°C"):
        return f"设为{value}"
    return f"切换到{value}模式"


def done_phrase(names: list[str], kind: str, verb: str, value: str | None, rng: random.Random) -> str:
    """Agent 执行成功后回给用户的话（domain/smarthome/messages.py 的 done_phrase 写法）。"""
    subject = "、".join(names)
    if verb == ON:
        return f"{subject} 已启动" if kind == "appliance" else f"已打开{subject}"
    if verb == OFF:
        return f"{subject} 已停止" if kind == "appliance" else f"已关闭{subject}"
    if verb == PAUSE:
        return f"{subject} 已暂停"
    if verb in (UP, DOWN):
        parts = []
        for n in names:
            if kind == "light":
                parts.append(f"{n} 已调到 {rng.choice([30, 40, 50, 60, 70, 80, 90])}%")
            elif kind == "media":
                parts.append(f"{n} 音量已调到 {rng.choice([10, 15, 20, 30, 40])}")
            else:
                lo, hi = (40, 55) if kind == "water_heater" else (18, 28)
                parts.append(f"{n} 已调到 {rng.randint(lo, hi)}°C")
        return "；".join(parts)
    assert value is not None
    if kind in ("light", "cover"):
        return f"{subject} 已调到 {value}"
    if kind == "fan":
        return f"{subject} 风速已调到 {value}"
    if kind == "media":
        return f"{subject} 音量已调到 {value}"
    if value.endswith("°C"):
        return f"{subject} 已设为 {value}"
    return f"{subject} 已切换到{value}"


def query_phrase(dev: Device, rng: random.Random) -> str:
    on = rng.random() < 0.6
    if dev.kind == "appliance":
        return f"{dev.name}{rng.choice(['待机中', '运行中', '已暂停'])}"
    if dev.kind == "cover":
        return f"{dev.name}开了{rng.choice([30, 50, 100])}%" if on else f"{dev.name}关着"
    if not on:
        return f"{dev.name}关着"
    if dev.kind == "light":
        return f"{dev.name}开着，亮度{rng.choice([30, 50, 60, 80, 100])}%"
    if dev.kind == "climate":
        return f"{dev.name}开着，{rng.choice(['制冷', '制热', '自动'])} {rng.randint(18, 28)}°C"
    if dev.kind == "water_heater":
        return f"{dev.name}开着，{rng.choice([42, 45, 50])}°C"
    if dev.kind == "media":
        return f"{dev.name}开着，音量{rng.choice([10, 20, 30])}"
    return f"{dev.name}开着"


def generic_noun(devs: list[Device]) -> str:
    return family_of(devs[0])


PICK_PREV = {
    ON: ["把{n}打开", "打开{n}", "{n}开一下", "帮我开一下{n}", "开{n}"],
    OFF: ["把{n}关了", "关掉{n}", "{n}关一下", "帮我关一下{n}", "关{n}"],
    UP: ["{n}调高一点", "把{n}调大一些", "{n}再高一点"],
    DOWN: ["{n}调低一点", "把{n}调小一些", "{n}再低一点"],
    SET: ["把{n}{p}", "{n}{p}"],
    PAUSE: ["{n}暂停一下", "先把{n}暂停"],
}
FOLLOW_PREV = {
    ON: ["把{n}打开", "打开{n}", "{n}开一下", "开一下{n}"],
    OFF: ["把{n}关了", "关掉{n}", "{n}关一下", "关{n}"],
    UP: ["{n}调高一点", "把{n}调大些", "{n}再高点"],
    DOWN: ["{n}调低一点", "把{n}调小些", "{n}再低点"],
    SET: ["把{n}{p}", "{n}{p}"],
    PAUSE: ["{n}暂停一下", "把{n}暂停"],
}
QUERY_PREV = ["{n}开着吗", "{n}现在什么状态", "看看{n}", "{n}现在怎么样"]


def prev_text(table: dict, verb: str, noun: str, phrase: str, rng: random.Random) -> str:
    return rng.choice(table[verb]).format(n=noun, p=phrase)


# ------------------------------------------------------------------ 家族（比例见 PLAN.md §2）

PICK_FAMILIES = {  # 代号: (比例, 金标类型)
    "P01": (0.24, "target"), "P02": (0.05, "target"), "P03": (0.06, "target"),
    "P04": (0.07, "target"), "P05": (0.04, "target"), "P06": (0.03, "target"),
    "P07": (0.03, "target"),
    "P08": (0.03, REDO), "P09": (0.07, REDO), "P10": (0.04, REDO), "P11": (0.06, REDO),
    "P12": (0.05, REDO), "P13": (0.02, REDO), "P14": (0.12, CANCEL), "P15": (0.04, REDO),
    "P16": (0.02, REDO), "P17": (0.04, REDO), "P18": (0.03, REDO),
}
FOLLOW_FAMILIES = {
    "F01": (0.28, "verb"), "F02": (0.12, "verb"), "F03": (0.10, "verb"), "F04": (0.06, "verb"),
    "F05": (0.03, "verb"),
    "F06": (0.05, REDO), "F07": (0.08, REDO), "F08": (0.03, REDO), "F09": (0.07, REDO),
    "F10": (0.03, REDO), "F11": (0.05, REDO), "F12": (0.06, REDO), "F13": (0.07, REDO),
}
HOMONYM_FAMILIES = {"P07", "P13"}


def _weighted(rng: random.Random, fams: dict, exclude=()) -> str:
    items = [(k, w) for k, (w, _g) in fams.items() if k not in exclude]
    x = rng.random() * sum(w for _k, w in items)
    for k, w in items:
        x -= w
        if x <= 0:
            return k
    return items[-1][0]


def _n_candidates(rng: random.Random) -> int:
    x = rng.random()
    return 2 if x < 0.55 else 3 if x < 0.85 else 4 if x < 0.97 else rng.choice([5, 6])


def family_of(d: Device) -> str:
    """一句话会一起指到的品类：“开灯”指所有灯，“开空调”不会指地暖，“开窗帘”不会指晾衣架。"""
    if d.kind == "light":
        return "灯"
    if d.kind == "climate":
        return "空调" if "空调" in d.type or d.type in ("风管机", "挂机") else d.type
    if d.kind == "cover":
        return "晾衣架" if "晾衣" in d.type else "遮阳篷" if "遮阳篷" in d.type else "窗帘"
    if d.kind in SAME_TYPE:
        return d.type
    return KIND_NOUN.get(d.kind, d.type)


def _groups(devs: list[Device]) -> list[list[Device]]:
    """能一起构成一次歧义的设备组：同一品类（family_of）。"""
    by: dict[str, list[Device]] = {}
    for d in devs:
        by.setdefault(family_of(d), []).append(d)
    return [g for g in by.values() if len(g) >= 2]


HOMONYM_NAME = {"灯": ["主灯", "吸顶灯", "灯"]}


def pick_context(home: str, devs: list[Device], rng: random.Random, homonym: bool) -> dict | None:
    groups = _groups(devs)
    if homonym:  # 人为重名：房间必须两两不同，才有“有线索”的一面
        groups = [g for g in groups if len({d.room for d in g}) >= 2]
    if not groups:
        return None
    group = rng.choice(groups)
    if homonym:
        seen, uniq = set(), []
        for d in rng.sample(group, len(group)):
            if d.room not in seen:
                seen.add(d.room)
                uniq.append(d)
        group = uniq
    n = min(_n_candidates(rng), len(group), 8)
    cands = rng.sample(group, n)
    kind = cands[0].kind
    verbs = verbs_for([kind])
    weights = {ON: 0.38, OFF: 0.32, UP: 0.08, DOWN: 0.08, SET: 0.10, PAUSE: 0.04}
    verb = rng.choices(verbs, [weights[v] for v in verbs])[0]
    value = set_value(kind, rng) if verb == SET else None
    if homonym:
        shared = rng.choice(HOMONYM_NAME.get(family_of(cands[0]), [family_of(cands[0])]))
        names = [shared] * n
    else:
        names = [d.name for d in cands]
    labels = []
    for d, nm in zip(cands, names, strict=True):
        labels.append(f"{nm}（{d.room}）" if names.count(nm) > 1 else nm)
    phrase = action_phrase(kind, verb, value)
    noun = generic_noun(cands)
    return {
        "question": "pick", "home": home, "kind": kind, "homonym": homonym,
        "devices": [d.__dict__ for d in cands], "labels": labels, "verb": verb, "value": value,
        "context": {"上一句": prev_text(PICK_PREV, verb, noun, phrase, rng),
                    "Agent": f"{'、'.join(names)}，要哪一个？", "待执行": phrase},
        "options": {lab: f"{d.room}·{KIND_LABEL[kind]}" for lab, d in zip(labels, cands, strict=True)}
                   | {CANCEL: None, REDO: None},
    }


def follow_context(home: str, devs: list[Device], rng: random.Random, multi: bool | None = None) -> dict | None:
    if multi is True or (multi is None and rng.random() < 0.2):
        groups = [g for g in _groups(devs)]
        if not groups:
            return None
        g = rng.choice(groups)
        focus = rng.sample(g, min(len(g), rng.choice([2, 2, 3])))
    else:
        focus = [rng.choice(devs)]
    kind = focus[0].kind
    verbs = verbs_for([d.kind for d in focus])
    names = [d.name for d in focus]
    noun = names[0] if len(names) == 1 else "和".join(names)
    if rng.random() < 0.15 and len(focus) == 1:
        last, value = "查询", None
        prev = rng.choice(QUERY_PREV).format(n=noun)
        agent = query_phrase(focus[0], rng)
    else:
        weights = {ON: 0.40, OFF: 0.30, UP: 0.08, DOWN: 0.08, SET: 0.10, PAUSE: 0.04}
        last = rng.choices(verbs, [weights[v] for v in verbs])[0]
        value = set_value(kind, rng) if last == SET else None
        prev = prev_text(FOLLOW_PREV, last, noun, action_phrase(kind, last, value), rng)
        agent = done_phrase(names, kind, last, value, rng)
    return {
        "question": "follow", "home": home, "kind": kind, "devices": [d.__dict__ for d in focus],
        "last": last, "value": value,
        "context": {"上一句": prev, "Agent": agent, "设备": names},
        "options": {v: VERB_DESC[v] for v in verbs} | {REDO: None},
    }


# ------------------------------------------------------------------ 家族 → 写手要求 + 金标


def _ord(k: int, n: int) -> str:
    extra = "（也是最后一个）" if k == n else ""
    return f"第{k}个{extra}"


def pick_spec(fam: str, ctx: dict, devs: list[Device], rng: random.Random) -> tuple[str, str] | None:
    cands = [Device(**d) for d in ctx["devices"]]
    labels, n, phrase = ctx["labels"], len(cands), ctx["context"]["待执行"]
    i = rng.randrange(n)
    gold, dev = labels[i], cands[i]
    shared = labels[i].split("（")[0]  # 人为重名时所有候选共用的名字
    j = rng.choice([k for k in range(n) if k != i])
    other = cands[j]
    if fam in HOMONYM_FAMILIES and not ctx["homonym"]:
        return None
    if ctx["homonym"] and fam not in HOMONYM_FAMILIES | {"P03", "P09", "P10", "P12", "P14", "P15",
                                                          "P16", "P17", "P18"}:
        return None
    shared_room = sum(c.room == dev.room for c in cands) > 1
    who = f"「{shared}」" if ctx["homonym"] else f"「{dev.name}」"  # 写手要念的名字
    if fam == "P01":
        if shared_room:  # 同房间还有别的候选：只说房间会指到两台
            return f"用设备名（可以简称）指出「{dev.name}」，不改变动作；同房间还有别的候选，不要只说房间。", gold
        return f"用房间或设备名（可以简称）指出{dev.room}的「{dev.name}」，不改变动作。", gold
    if fam == "P02":
        same = [c for c in cands if c.type == dev.type]
        if len(same) > 1:
            return None
        return (f"不说房间，用它的类型或名字里的特征词指出「{dev.name}」（类型：{dev.type}；"
                f"其他候选的类型：{'、'.join(c.type for c in cands if c != dev)}），不改变动作。"), gold
    if fam == "P03":
        return (f"用顺序指出候选：选 Agent 那句里的{_ord(i + 1, n)}（共{n}个）。"
                f"只用顺序或位置（第几个、前面 / 后面 / 最后那个），不说房间和名字。"), gold
    if fam == "P04":
        how = f"「{dev.name}」（同房间还有别的候选，要说出名字）" if shared_room else f"{dev.room}的「{dev.name}」"
        return f"选{how}，同时带附和语气（嗯、对、好）或把动作「{phrase}」再说一遍。", gold
    if fam == "P05":
        return (f"先否定{other.room}的「{other.name}」，再明确说要{dev.room}的「{dev.name}」，"
                f"动作不变。"), gold
    if fam == "P06":
        return (f"句内改口：先说成「{other.name}」，马上改口，最后落在「{dev.name}」。"), gold
    if fam == "P07":
        return (f"候选同名（都叫「{shared}」）。用房间或位置线索指出{dev.room}的那一台，不改变动作。"), gold
    if fam == "P08":
        return (f"只否定{other.room}的「{other.name}」，不说要哪一台（不要说“另一个 / 那就剩下的”）。"), REDO
    if fam == "P09":
        alt = [v for v in verbs_for([dev.kind]) if v != ctx["verb"] and v != SET]
        if ctx["verb"] == SET and rng.random() < 0.5:
            change = f"换一个数值（原来是「{phrase}」）"
        elif alt:
            v = rng.choice(alt)
            change = f"动作改成「{action_phrase(dev.kind, v)}」"
        else:
            return None
        return f"选{dev.room}的{who}，但{change}。", REDO
    if fam == "P10":
        return f"要不止一台：全部、两个都要，或其中几台，动作「{phrase}」不变。", REDO
    if fam == "P11":
        pool = [d for d in devs if d not in cands and d.room not in {c.room for c in cands}]
        if not pool:
            return None
        out = rng.choice(pool)
        return (f"说一台不在候选里的设备：{out.room}的「{out.name}」（家里有，但这次没列出来）。"
                f"可以是纠正（“是…”）或直接点名。"), REDO
    if fam == "P12":
        return "只附和、委托或表示随意，不指出是哪一台（好的 / 嗯 / 随便 / 你决定 / 哪个都行……）。", REDO
    if fam == "P13":
        return f"候选同名（都叫「{shared}」）。只说这个名字或“{shared}那个”，不给房间或位置线索。", REDO
    if fam == "P14":
        return "明确不要这次操作了，也不提别的要求（可以委婉，可以说自己去弄）。", CANCEL
    if fam == "P15":
        return "不要这次操作了，同时提出一个新的要求（家里别的设备，或者别的事）。", REDO
    if fam == "P16":
        return "暂缓：让 Agent 等一下，或说自己想一想，不做决定。", REDO
    if fam == "P17":
        return "提一个问题（关于这些候选、它们的状态，或别的），不做选择。", REDO
    if fam == "P18":
        return "说一句跟这次选择无关的话（闲聊、别的话题），不做选择。", REDO
    raise ValueError(fam)


VERB_SAY = {ON: "打开 / 启动", OFF: "关闭 / 停止", UP: "调高 / 调亮 / 调大", DOWN: "调低 / 调暗 / 调小",
            PAUSE: "暂停"}
COMPLAINT = {"light": "太亮、太暗、刺眼", "climate": "冷、热、闷", "water_heater": "水太烫、不够热",
             "cover": "太晒、太暗", "fan": "太吵、风太大、空气闷", "media": "太吵、听不清",
             "appliance": "太吵、还没好", "switch": "太热、太冷", "camera": "看不清"}


def follow_spec(fam: str, ctx: dict, devs: list[Device], rng: random.Random) -> tuple[str, str] | None:
    focus = [Device(**d) for d in ctx["devices"]]
    kind, opts, last = ctx["kind"], [v for v in ctx["options"] if v != REDO], ctx["last"]
    names = "、".join(d.name for d in focus)
    if fam == "F01":
        simple = [v for v in opts if v in (ON, OFF, PAUSE)]
        agent = ctx["context"]["Agent"]
        if last == "查询":
            off = "关着" in agent or "待机" in agent or "已暂停" in agent
        else:
            off = last in (OFF, PAUSE)
        natural = ON if off else (PAUSE if PAUSE in simple and rng.random() < 0.3 else OFF)
        v = natural if rng.random() < 0.9 else rng.choice(simple)  # 少量照字面的“怪”要求，金标不变
        return f"不说设备名（用代词或省略宾语），要求「{action_phrase(kind, v)}」。", v
    if fam == "F02":
        steps = [v for v in opts if v in (UP, DOWN)]
        if not steps:
            return None
        v = rng.choice(steps)
        amount = {"light": "10%", "media": "一格", "climate": "两度", "water_heater": "两度"}[kind]
        return (f"用程度词要求「{action_phrase(kind, v)}」（再…一点、…些，可以带幅度如“{amount}”），"
                f"不说具体的目标值，不说设备名。"), v
    if fam == "F03":
        if SET not in opts:
            return None
        val = set_value(kind, rng)
        return f"要求设到一个具体的值或模式：「{action_phrase(kind, SET, val)}」。可以省略设备名。", SET
    if fam == "F04":
        v = rng.choice(opts)
        val = set_value(kind, rng) if v == SET else None
        target = focus[0].name if len(focus) == 1 else f"这几台（{names}）"
        return f"说出焦点设备本身（{target}，可以简称），要求「{action_phrase(kind, v, val)}」。", v
    if fam == "F05":
        if last not in (UP, DOWN):
            return None
        return (f"不说方向词（不出现高 / 低 / 亮 / 暗 / 大 / 小），要求把刚才的调节「{action_phrase(kind, last)}」"
                f"再来一次（比如“再来一点”“再一次”）。"), last
    if fam == "F06":
        bad = [v for v in VERBS if v not in opts] + ["上锁"]
        v = rng.choice(bad)
        say = "上锁" if v == "上锁" else VERB_SAY.get(v, "设到某个具体数值")
        return f"要求一个这台设备做不到的操作：{say}（它是{focus[0].type}）。", REDO
    if fam == "F07":
        pool = [d for d in devs if d not in focus]
        if not pool:
            return None
        o = rng.choice(pool)
        how = rng.choice(["换成", "也扩展到"])
        return f"把操作对象{how}另一台设备：{o.room}的「{o.name}」，可以带一个动作。", REDO
    if fam == "F08":
        if len(focus) < 2:
            return None
        sub = rng.choice(focus)
        return f"只针对焦点里的一部分：「{sub.name}」，带一个动作。", REDO
    if fam == "F09":
        if last == "查询":
            return None
        return "要求撤销 / 取消刚才的操作，不说要把设备具体怎样。", REDO
    if fam == "F10":
        return "表示保持现状、别再动它。", REDO
    if fam == "F11":
        return f"只抱怨一个感受或现象（{COMPLAINT[kind]}之类），不说要做什么操作。", REDO
    if fam == "F12":
        return "提一个问题（设备的状态、读数或别的），不要求操作。", REDO
    if fam == "F13":
        return "感谢、确认收到、闲聊，或提一个与家居无关的新要求。", REDO
    raise ValueError(fam)


def make_split(split: str) -> list[dict]:
    rng = random.Random(SEEDS[split])
    homes = {h: load_home(h) for h in SPLIT_HOMES[split]}
    target = SIZES[split]
    frames: list[dict] = []
    ctx_no = 0
    while len(frames) < target:
        home = rng.choice(list(homes))
        devs = homes[home]
        is_pick = rng.random() < PICK_SHARE
        k = rng.choices([1, 2, 3], [0.3, 0.4, 0.3])[0]
        if is_pick:
            fam0 = _weighted(rng, PICK_FAMILIES)
            ctx = pick_context(home, devs, rng, homonym=fam0 in HOMONYM_FAMILIES or rng.random() < 0.05)
            fams, spec_fn = PICK_FAMILIES, pick_spec
        else:
            fam0 = _weighted(rng, FOLLOW_FAMILIES)
            ctx = follow_context(home, devs, rng)
            fams, spec_fn = FOLLOW_FAMILIES, follow_spec
        if ctx is None:
            continue
        chosen: list[tuple[str, str, str]] = []
        tries = 0
        fam = fam0
        while len(chosen) < k and tries < 30:
            tries += 1
            if fam not in {c[0] for c in chosen}:
                got = spec_fn(fam, ctx, devs, rng)
                if got is not None:
                    chosen.append((fam, *got))
            fam = _weighted(rng, fams)
        if not chosen:
            continue
        ctx_no += 1
        cid = f"{split}-{ctx['question']}-{ctx_no:04d}"
        for j, (fam, spec, gold) in enumerate(chosen):
            frames.append({"frame_id": f"{cid}-{j}", "context_id": cid, "split": split, "family": fam,
                           "spec": spec, "gold": gold, **ctx})
    return frames[:target]


# c4：F08（焦点多台只动一部分）在训练池里只有 4 个剧本，c3 在 c-dev 上唯一的高置信错误执行就是它。
# 只给训练池补，只用焦点多台的上下文；每个上下文配一个“作用于全部焦点”的对照。
EXTRA_F08 = {"F08": (0.50, REDO), "F01": (0.20, "verb"), "F04": (0.15, "verb"), "F07": (0.15, REDO)}


def make_extra_f08(n: int = 200, seed: int = 404) -> list[dict]:
    rng = random.Random(seed)
    homes = {h: load_home(h) for h in SPLIT_HOMES["train"]}
    frames: list[dict] = []
    ctx_no = 0
    while len(frames) < n:
        home = rng.choice(list(homes))
        ctx = follow_context(home, homes[home], rng, multi=True)
        if ctx is None or len(ctx["devices"]) < 2:
            continue
        chosen = []
        for fam in ["F08", _weighted(rng, {k: v for k, v in EXTRA_F08.items() if k != "F08"})]:
            got = follow_spec(fam, ctx, homes[home], rng)
            if got is not None:
                chosen.append((fam, *got))
        if not chosen:
            continue
        ctx_no += 1
        cid = f"train-f08-follow-{ctx_no:04d}"
        for j, (fam, spec, gold) in enumerate(chosen):
            frames.append({"frame_id": f"{cid}-{j}", "context_id": cid, "split": "train", "family": fam,
                           "spec": spec, "gold": gold, **ctx})
    return frames[:n]


def writer_view(f: dict) -> dict:
    """写手看到的：状态、选项和要求，没有金标。"""
    return {"frame_id": f["frame_id"], "题目": f["question"], "context": f["context"],
            "选项": list(f["options"]), "要求": f["spec"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", choices=["f08"], help="只生成补充剧本（不动已有的 frames-{train,dev,test}）")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "writer").mkdir(parents=True, exist_ok=True)
    if args.extra == "f08":
        frames = make_extra_f08()
        with (out / "frames-train-f08.jsonl").open("w", encoding="utf-8") as fh:
            for f in frames:
                fh.write(json.dumps(f, ensure_ascii=False) + "\n")
        for b in range(0, len(frames), 100):
            with (out / "writer" / f"train-f08-{b // 100:02d}.jsonl").open("w", encoding="utf-8") as fh:
                for f in frames[b : b + 100]:
                    fh.write(json.dumps(writer_view(f), ensure_ascii=False) + "\n")
        print("extra f08", len(frames), dict(sorted(__import__("collections").Counter(f["family"] for f in frames).items())))
        return
    for split in SIZES:
        frames = make_split(split)
        with (out / f"frames-{split}.jsonl").open("w", encoding="utf-8") as fh:
            for f in frames:
                fh.write(json.dumps(f, ensure_ascii=False) + "\n")
        for b in range(0, len(frames), WRITER_BATCH):
            with (out / "writer" / f"{split}-{b // WRITER_BATCH:02d}.jsonl").open("w", encoding="utf-8") as fh:
                for f in frames[b : b + WRITER_BATCH]:
                    fh.write(json.dumps(writer_view(f), ensure_ascii=False) + "\n")
        fams: dict[str, int] = {}
        golds: dict[str, int] = {}
        for f in frames:
            fams[f["family"]] = fams.get(f["family"], 0) + 1
            g = f["gold"] if f["gold"] in (CANCEL, REDO) else "执行"
            golds[g] = golds.get(g, 0) + 1
        print(split, len(frames), "contexts", len({f["context_id"] for f in frames}), golds,
              dict(sorted(fams.items())))


if __name__ == "__main__":
    main()
