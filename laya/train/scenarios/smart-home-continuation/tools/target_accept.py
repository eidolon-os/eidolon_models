"""Freeze authored acceptance cases before c5 training; never imported by its generator.

New layouts and natural rewrites are evaluation only. This is a small synthetic
holdout, not a production-distribution or independently human-labelled test.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from build import SCENARIO, to_record  # noqa: E402
from frames import DOWN, OFF, ON, REDO, SET, UP, VERB_DESC, Device, verbs_for  # noqa: E402

from eidolon_laya_train.records import Record, record_id, write_jsonl  # noqa: E402
from eidolon_laya_train.scenario import Scenario  # noqa: E402

HOMES = {
    "c5-seaside-accept": [
        ("琴房落地灯", "琴房", "light"), ("备餐间灯带", "备餐间", "light"),
        ("影音间电视", "影音间", "media"), ("榻榻米房音箱", "榻榻米房", "media"),
        ("琴房空调", "琴房", "climate"), ("榻榻米房空调", "榻榻米房", "climate"),
    ],
    "c5-courtyard-accept": [
        ("南廊壁灯", "南廊", "light"), ("北屋阅读灯", "北屋", "light"),
        ("北屋投影仪", "北屋", "media"), ("西厢智能音箱", "西厢", "media"),
        ("北屋空调", "北屋", "climate"), ("西厢空调", "西厢", "climate"),
    ],
}

# target index, same-type distractor, cross-type distractor, action, literal utterance.
CASES = {
    "c5-seaside-accept": [
        (0, 1, 3, OFF, "琴房落地灯关一下呗"),
        (0, 1, 2, ON, "开开琴房落地灯"),
        (0, 1, 4, UP, "琴房落地灯稍微调亮些"),
        (0, 1, 3, DOWN, "把琴房落地灯的亮度往下调一点"),
        (0, 1, 2, SET, "琴房落地灯亮度设定在六成"),
        (1, 0, 2, OFF, "备餐间灯带给关了"),
        (1, 0, 4, ON, "帮我开着备餐间灯带"),
        (1, 0, 3, DOWN, "备餐间灯带再减点亮度"),
        (2, 3, 0, OFF, "关下影音间电视"),
        (2, 3, 4, ON, "影音间电视，开起来吧"),
        (2, 3, 1, UP, "影音间电视声音往上调五格"),
        (2, 3, 0, DOWN, "影音间电视音量给小点儿"),
        (2, 3, 4, SET, "把影音间电视的音量定到四十"),
        (3, 2, 1, OFF, "榻榻米房音箱关了吧"),
        (3, 2, 5, ON, "开启一下榻榻米房音箱"),
        (3, 2, 0, DOWN, "请给榻榻米房音箱降一点音量"),
        (4, 5, 2, OFF, "琴房空调停掉吧"),
        (4, 5, 0, ON, "琴房空调帮忙开起来"),
        (4, 5, 3, UP, "琴房空调升温一度"),
        (4, 5, 1, DOWN, "琴房空调往下调两度"),
        (5, 4, 3, SET, "榻榻米房空调定在二十三度"),
        (5, 4, 0, OFF, "请关榻榻米房空调"),
    ],
    "c5-courtyard-accept": [
        (0, 1, 2, OFF, "南廊壁灯关一下就好"),
        (0, 1, 4, ON, "南廊壁灯给开了吧"),
        (0, 1, 3, UP, "帮南廊壁灯加一点亮度"),
        (0, 1, 2, SET, "把南廊壁灯亮度定为七成"),
        (1, 0, 3, OFF, "关掉北屋阅读灯吧"),
        (1, 0, 5, ON, "北屋阅读灯打开一下嘛"),
        (1, 0, 2, DOWN, "北屋阅读灯的光调暗些"),
        (2, 3, 0, OFF, "北屋投影仪现在关了吧"),
        (2, 3, 4, ON, "启动一下北屋投影仪"),
        (2, 3, 1, UP, "北屋投影仪音量加十"),
        (2, 3, 5, DOWN, "北屋投影仪音量减五"),
        (3, 2, 1, OFF, "请把西厢智能音箱关一下"),
        (3, 2, 0, ON, "开一下西厢智能音箱呗"),
        (3, 2, 4, SET, "西厢智能音箱音量设置成百分之二十"),
        (3, 2, 0, OFF, "西厢音箱关一下"),
        (4, 5, 2, OFF, "北屋空调关闭一下"),
        (4, 5, 1, UP, "北屋空调的温度增加一度"),
        (4, 5, 3, DOWN, "北屋空调降温两度"),
        (5, 4, 0, ON, "西厢空调重新开一下"),
        (5, 4, 2, SET, "把西厢空调温度定为二十七度"),
        (5, 4, 1, OFF, "西厢空调帮我关下"),
        (5, 4, 3, DOWN, "西厢空调再降低一度"),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    single = Scenario.load(HERE.parent / "smart-home/scenario.yaml")
    follow, singles, controls = [], [], []
    for home, defs in HOMES.items():
        devs = [Device(n, room, kind, kind) for n, room, kind in defs]
        for i, (target, same, cross, verb, utterance) in enumerate(CASES[home]):
            group = f"c5-accept/{home}/{i}"
            for role, index in (("same-focus", target), ("switch-same-type", same), ("switch-cross-type", cross)):
                d = devs[index]
                f = {"question": "follow", "context_id": group, "frame_id": f"{group}/{role}",
                     "family": "F14-target-identity", "home": home,
                     "context": {"上一句": f"打开{d.name}", "Agent": f"已打开{d.name}", "设备": [d.name]},
                     "options": {v: VERB_DESC[v] for v in verbs_for([d.kind])}}
                follow.append(to_record(f, utterance, verb if index == target else REDO,
                                        source="authored-acceptance-20261002", extra_tags=[role, f"action:{verb}"],
                                        meta={"target_name": devs[target].name, "eval_only": True}))
            questions = single.build_questions({"devices": {d.name: f"{d.room}·{'灯' if d.kind == 'light' else '影音' if d.kind == 'media' else '空调'}" for d in devs}})
            singles.append(Record(id=record_id("c5-accept-single", home, utterance), scenario=single.name,
                                  source="authored-acceptance-20261002", state={"utterance": utterance},
                                  questions=questions, labels={"intent": {"gold": "控制"},
                                  "device": {"gold": devs[target].name}, "action": {"gold": verb}},
                                  tags=["explicit-control", f"home:{home}"], meta={"eval_only": True}))
        # Explicit negation, reported speech and pronouns with exactly one focus.
        d = devs[0]
        for i, (u, gold) in enumerate([
            ("关了它呗", OFF), ("再给它打开", ON), ("稍微调暗些", DOWN),
            ("亮度再增加10%", UP), ("亮度设为70%", SET),
            ("别把它关掉，保持现在这样", REDO), ("我刚刚跟家里人说让他关灯", REDO),
            (f"昨天我把{d.name}关了，现在没让你操作", REDO),
            (f"如果关掉{d.name}，这里会不会太暗", REDO), ("谢谢，就这样", REDO),
        ]):
            f = {"question": "follow", "context_id": f"accept-controls/{home}/{i}", "frame_id": f"accept-controls/{home}/{i}",
                 "family": "accept-regression", "home": home,
                 "context": {"上一句": f"打开{d.name}", "Agent": f"已打开{d.name}", "设备": [d.name]},
                 "options": {v: VERB_DESC[v] for v in verbs_for([d.kind])}}
            controls.append(to_record(f, u, gold, source="authored-acceptance-20261002"))
        names = [devs[0].name, devs[1].name]
        q = SCENARIO.build_questions({"candidates": {n: "照明·灯" for n in names}}, only=["pick"])
        for i, (u, gold) in enumerate([
            ("前面那个", names[0]), ("第二盏", names[1]), (names[1], names[1]),
            ("两个都不开了", "取消"), ("不需要了，我手动来", "取消"),
            (f"不是{names[0]}", REDO), (f"{names[0]}调暗点", REDO),
            ("先不选了，帮我查一下天气", REDO), ("都打开吧", REDO), ("你看哪个合适", REDO),
        ]):
            state = {"utterance": u, "context": {"上一句": "把灯打开", "Agent": f"{'、'.join(names)}，要哪一个？", "待执行": "打开"}}
            controls.append(Record(record_id("c5-accept-pick", home, i), SCENARIO.name,
                                   "authored-acceptance-20261002", state, q, {"pick": {"gold": gold}},
                                   ["pick-regression", "pick", "accept-regression"], meta={"home": home, "eval_only": True}))
    outputs = {"target-accept.jsonl": follow, "target-accept-single.jsonl": singles,
               "controls-accept.jsonl": controls}
    manifest = {"scope": "new synthetic layouts and authored rewrites; frozen before training; no independent human label audit", "files": {}}
    for name, rows in outputs.items():
        p = a.out / name
        if p.exists():
            raise FileExistsError(p)
        write_jsonl(p, rows)
        manifest["files"][name] = {"n": len(rows), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (a.out / "acceptance-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
