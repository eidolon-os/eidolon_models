"""c9 shopping/prohibition/reporting contrasts; no evaluation text is used."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from frames import OFF, ON, SET, SPLIT_HOMES, load_home  # noqa: E402
from target_pairs import dev_homes  # noqa: E402

from eidolon_laya_train.records import Record, write_jsonl  # noqa: E402
from eidolon_laya_train.scenario import Scenario  # noqa: E402


def examples(d, novel=False):
    n = d.name
    if novel:
        return [
            (f"准备淘汰{n}，下次挑个带遥控的", None, "shopping"),
            (f"{n}先不要启动，我只是在研究说明书", None, "prohibition"),
            (f"听维修员讲，{n}昨晚停过一次", None, "reported"),
            (f"新买的设备最好能代替{n}，还要支持定时", None, "shopping"),
            (f"不用担心耗电，现在就启动{n}", ON, "explicit-control"),
            (f"别继续运行了，请关闭{n}", OFF, "negated-control"),
            (f"检查结束，请把{n}开启", ON, "explicit-control"),
            (f"不用等我，立即关掉{n}", OFF, "negated-control"),
        ]
    out = [
        (f"打算买个新款替换{n}，要能远程控制的", None, "shopping"),
        (f"我在挑{n}的替代品，最好支持自动关闭", None, "shopping"),
        (f"想给{n}换个能预约启动的型号，你觉得值不值", None, "shopping"),
        (f"考虑把{n}升级成带触控的新机器，先看看价格", None, "shopping"),
        (f"先别打开{n}，保持现在这样", None, "prohibition"),
        (f"不要启动{n}，我还没准备好", None, "prohibition"),
        (f"{n}不要关，别改变它的状态", None, "prohibition"),
        (f"不用去开启{n}，暂时不需要", None, "prohibition"),
        (f"师傅说{n}昨天已经关了", None, "reported"),
        (f"孩子提起昨晚打开{n}的事，我只是复述一下", None, "reported"),
        (f"{n}现在打开，我要用了", ON, "explicit-control"),
        (f"先把{n}关闭，然后再说", OFF, "explicit-control"),
        (f"别让{n}继续运行了，马上关掉", OFF, "negated-control"),
        (f"不要再等了，把{n}打开", ON, "negated-control"),
        (f"不是让你买新的，现在打开{n}", ON, "shopping-control"),
        (f"{n}暂时不换，先关掉它", OFF, "shopping-control"),
    ]
    settings = {"light": "亮度设置到60%", "media": "音量设置到20%",
                "climate": "温度设置到24度", "water_heater": "温度设置到42度",
                "cover": "开度设置到30%", "fan": "风速设置到40%"}
    if d.kind in settings:
        out.extend([(f"不用换设备，把{n}的{settings[d.kind]}", SET, "shopping-control"),
                    (f"现在把{n}的{settings[d.kind]}，就这个数值", SET, "numeric-control")])
    return out


def build(homes, novel=False, *, examples_factory=examples, version="c9"):
    scenario = Scenario.load(HERE.parent / "smart-home")
    rows = []
    for home, devices in homes.items():
        questions = scenario.build_questions({"devices": {d.name: f"{d.room}·{d.type}" for d in devices}})
        for d in devices:
            group = hashlib.sha256(f"{home}/{d.name}".encode()).hexdigest()[:12]
            for utterance, action, family in examples_factory(d, novel):
                uid = hashlib.sha256(f"{home}/{utterance}".encode()).hexdigest()[:16]
                labels = {"intent": {"gold": "控制" if action else "无关"}}
                if action:
                    labels.update(device={"gold": d.name}, action={"gold": action})
                rows.append(Record(id=f"smart-home/{version}-{uid}", scenario="smart-home",
                                   source="authored-intent-contrast-v1", state={"utterance": utterance},
                                   questions=questions, labels=labels,
                                   tags=[family, f"{version}-intent-contrast", f"home:{home}"],
                                   meta={"paired_group": group, "home": home, "target_name": d.name,
                                         "label_basis": "shopping/reporting/no-state-change vs explicit state change"}))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    pool = build({h: load_home(h) for h in SPLIT_HOMES["train"]})
    dev = build(dev_homes(), novel=True)
    write_jsonl(a.out / "intent-pool.jsonl", pool)
    write_jsonl(a.out / "intent-dev.jsonl", dev)
    print(json.dumps({"pool": len(pool), "dev": len(dev)}))


if __name__ == "__main__":
    main()
