"""模板生成器：按"硬骨头"切片造智能家居用例（明确指令 / 隐含意图 / 状态查询 / 非命令 / 多设备 / 家里没有）。

措辞来自表格里的模板 × 户型里的设备 × 随机的口语前后缀；金标由模板决定。这是最便宜的一层，
用来保证每个切片都有覆盖；语言多样性靠 llm 生成器补。
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path

from eidolon_laya_train.records import Record, record_id

PREFIX = ["", "", "", "帮我", "麻烦", "小易，", "喂，", "那个，", "能不能", "请"]
SUFFIX = ["", "", "", "吧", "好吗", "呗", "谢谢", "一下"]

# (模板, 动作金标, 适用设备类型的关键词；None = 任意)
CONTROL = [
    ("把{d}打开", "打开或启动", None),
    ("开一下{d}", "打开或启动", None),
    ("{d}关了", "关闭或停止", None),
    ("关掉{d}", "关闭或停止", None),
    ("{d}调亮一点", "调高或增大", "灯"),
    ("{d}暗一点", "调低或减小", "灯"),
    ("{d}温度调高两度", "调高或增大", "空调"),
    ("{d}调到二十六度", "设为指定的数值或模式", "空调"),
    ("{d}开到最大档", "设为指定的数值或模式", "风扇"),
    ("{d}声音小点", "调低或减小", "音箱"),
    ("{d}先暂停", "暂停", "扫地机器人"),
    ("把{d}锁上", "上锁", "门锁"),
    ("{d}拉上", "关闭或停止", "窗帘"),
    ("{d}开始洗", "打开或启动", "洗衣机"),
]
IMPLICIT = [
    ("好热啊", "空调", ["打开或启动", "调低或减小"]),
    ("有点冷", "空调", ["打开或启动", "调高或增大"]),
    ("太暗了看不清", "灯", ["打开或启动", "调高或增大"]),
    ("太亮了刺眼", "灯", ["关闭或停止", "调低或减小"]),
    ("地上好脏", "扫地机器人", ["打开或启动"]),
    ("空气有点闷", "新风", ["打开或启动"]),
    ("外面太吵了", "窗", ["关闭或停止"]),
]
QUERY = [
    "{d}现在开着吗",
    "{d}是什么温度",
    "{d}还有多久",
    "{d}关了没有",
    "{d}现在几档",
]
NONCOMMAND = [
    "{d}一般一个月电费多少",
    "我昨天买的{d}还没到",
    "{d}是哪个牌子好",
    "刚才{d}的声音吓我一跳",
    "今天天气真不错",
    "帮我讲个笑话",
    "明天几点上班来着",
    "你觉得{d}好用吗",
]
MULTI = [
    ("我要睡了", ["关闭或停止"]),
    ("我出门了", ["关闭或停止", "上锁"]),
    ("回家模式", ["打开或启动"]),
    ("全都关掉", ["关闭或停止"]),
    ("客厅的都打开", ["打开或启动"]),
]
MISSING = [  # 家里可能没有的设备
    ("把电视打开", "电视", "打开或启动"),
    ("车库门关上", "车库门", "关闭或停止"),
    ("鱼缸灯开一下", "鱼缸灯", "打开或启动"),
    ("烤箱预热到二百度", "烤箱", "设为指定的数值或模式"),
    ("加湿器开一下", "加湿器", "打开或启动"),
]


def _homes(root: Path) -> dict[str, list[dict]]:
    raw = {p.stem: json.loads(p.read_text("utf-8")) for p in (root / "homes").glob("*.json")}
    out = {}
    for name, home in raw.items():
        devices = list(home.get("devices", []))
        for inc in home.get("include", []):
            devices.extend(raw[inc]["devices"])
        out[name] = devices
    return out


def _wrap(rng: random.Random, text: str) -> str:
    return rng.choice(PREFIX) + text + rng.choice(SUFFIX)


def generate(scenario, config: dict, rng: random.Random) -> Iterator[Record]:
    homes = _homes((scenario.root / config.get("homes", "../../../evals/smart-home")).resolve())
    n_per_slice = int(config.get("per_slice", 60))
    exits = scenario.questions["device"].exits
    exit_multi, exit_none = list(exits)[0], list(exits)[1]

    def make(home: str, text: str, intent: str, device, action, tag: str) -> Record:
        devs = {d["name"]: f"{d['room']}·{d['type']}" for d in homes[home]}
        labels = {"intent": {"gold": intent}}
        if device is not None:
            labels["device"] = {"gold": device}
        if action is not None:
            labels["action"] = {"gold": action}
        return Record(
            id=record_id(scenario.name, "rules", home, text),
            scenario=scenario.name,
            source="template:rules",
            state={"utterance": text},
            questions=scenario.build_questions({"devices": devs}),
            labels=labels,
            tags=[tag, f"home:{home}"],
            meta={"home": home},
        )

    home_names = sorted(homes)
    for _ in range(n_per_slice):
        home = rng.choice(home_names)
        devs = homes[home]
        # 明确指令
        tpl, action, kind = rng.choice(CONTROL)
        pool = [d for d in devs if kind is None or kind in d["type"] or kind in d["name"]]
        if pool:
            d = rng.choice(pool)
            yield make(
                home,
                _wrap(rng, tpl.format(d=d["name"])),
                "控制",
                d["name"],
                action,
                "explicit-control",
            )
        # 隐含意图
        text, kind, actions = rng.choice(IMPLICIT)
        pool = [d for d in devs if kind in d["type"] or kind in d["name"]]
        if pool:
            yield make(
                home,
                _wrap(rng, text),
                "控制",
                [d["name"] for d in pool] if len(pool) > 1 else pool[0]["name"],
                actions,
                "implicit-intent",
            )
        # 状态查询
        d = rng.choice(devs)
        yield make(
            home,
            _wrap(rng, rng.choice(QUERY).format(d=d["name"])),
            "查询",
            d["name"],
            None,
            "status-query",
        )
        # 非命令
        d = rng.choice(devs)
        yield make(
            home, rng.choice(NONCOMMAND).format(d=d["name"]), "无关", None, None, "non-command"
        )
        # 多设备 / 场景
        text, actions = rng.choice(MULTI)
        yield make(home, _wrap(rng, text), "控制", exit_multi, actions, "multi-device")
        # 家里没有
        text, kind, action = rng.choice(MISSING)
        if not any(kind in d["type"] or kind in d["name"] for d in devs):
            yield make(home, _wrap(rng, text), "控制", exit_none, action, "device-not-in-home")
