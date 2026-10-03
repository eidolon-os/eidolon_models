"""Build c5 target-identity contrast pairs using the existing c-series Record contract.

Each utterance is paired with the named target as focus and a different focus.
Pairs, word orders and action variants share a split group. No service is called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from build import to_record  # noqa: E402
from frames import (  # noqa: E402
    DOWN,
    OFF,
    ON,
    PAUSE,
    REDO,
    SET,
    SPLIT_HOMES,
    UP,
    VERB_DESC,
    Device,
    action_phrase,
    done_phrase,
    load_home,
    verbs_for,
)

from eidolon_laya_train.records import write_jsonl  # noqa: E402


def context(d: Device, mode: int) -> dict:
    verb = ON if mode % 2 == 0 else OFF
    if mode % 3 == 2:
        return {"上一句": f"{d.name}开着吗", "Agent": f"{d.name}开着", "设备": [d.name]}
    phrase = action_phrase(d.kind, verb)
    return {"上一句": f"{phrase}{d.name}",
            "Agent": done_phrase([d.name], d.kind, verb, None, random.Random(7)), "设备": [d.name]}


def utterances(d: Device, verb: str, name: str, novel: bool = False) -> list[str]:
    if verb == ON:
        return ([f"先开一下{name}吧", f"{name}给我开启一下", f"麻烦让{name}启动"] if novel else
                [f"打开{name}", f"{name}打开", f"把{name}打开", f"再开{name}"])
    if verb == OFF:
        return ([f"先关一下{name}吧", f"{name}给我关闭一下", f"麻烦让{name}停下来"] if novel else
                [f"关闭{name}", f"{name}关掉", f"把{name}关掉", f"再关{name}"])
    if verb in (UP, DOWN):
        attr, inc, dec, amount = {
            "light": ("亮度", "调亮", "调暗", "10%"),
            "media": ("音量", "调大", "调小", "10%"),
            "climate": ("温度", "调高", "调低", "两度"),
            "water_heater": ("温度", "调高", "调低", "两度"),
        }[d.kind]
        act = inc if verb == UP else dec
        return ([f"{name}的{attr}再{'高' if verb == UP else '低'}一点", f"帮忙将{name}的{attr}{act}{amount}"] if novel else
                [f"{act}{name}", f"{name}{act}一点", f"把{name}{act}{amount}", f"再{act}{name}的{attr}"])
    if verb == SET:
        attr, value = {"light": ("亮度", "40%"), "media": ("音量", "30%"),
                       "climate": ("温度", "25度"), "water_heater": ("温度", "45度"),
                       "cover": ("开度", "50%"), "fan": ("风速", "50%")}[d.kind]
        return ([f"{name}的{attr}要{value}", f"请将{name}的{attr}设在{value}"] if novel else
                [f"{name}{attr}调到{value}", f"把{name}{attr}设为{value}",
                 f"设定{name}{attr}为{value}", f"{name}，{attr}设成{value}"])
    assert verb == PAUSE
    return [f"暂停{name}", f"{name}暂停", f"把{name}暂停", f"{name}先暂停，待会再继续"]


def names(d: Device, other: Device) -> list[str]:
    # Only transparent name fragments; arbitrary user aliases are not supplied by v1.
    out = [d.name]
    for fragment in ("电视", "音箱", "投影仪", "扫地机", "洗衣机", "热水器", "加湿器", "净化器"):
        if fragment in d.name and fragment not in other.name and fragment != d.name:
            out.append(fragment)
            break
    return out


def pairs(home: str, devs: list[Device], seed: int, count: int, novel=False):
    rng = random.Random(seed)
    candidates = [(a, b) for a in devs for b in devs if a != b and a.name != b.name
                  and a.name not in b.name and b.name not in a.name]
    same = [p for p in candidates if p[0].kind == p[1].kind]
    cross = [p for p in candidates if p[0].kind != p[1].kind]
    rng.shuffle(same)
    rng.shuffle(cross)
    chosen = same[:count // 2] + cross[:count - count // 2]
    records = []
    for i, (target, other) in enumerate(chosen):
        # All utterance/action variants for this target stay together, including
        # repeated positive controls paired with different distractor focuses.
        group = f"c5-{home}-" + hashlib.sha256(target.name.encode()).hexdigest()[:12]
        common = [v for v in verbs_for([target.kind, other.kind]) if v != PAUSE]
        for verb in common:
            for name in names(target, other):
                for syntax, utt in enumerate(utterances(target, verb, name, novel)):
                    for is_same, focus in ((True, target), (False, other)):
                        frame = {"question": "follow", "context": context(focus, i), "context_id": group,
                                 "frame_id": f"{group}/{verb}/{name}/{syntax}/{is_same}",
                                 "family": "F14-target-identity", "home": home,
                                 "options": {v: VERB_DESC[v] for v in verbs_for([focus.kind])}}
                        gold = verb if is_same else REDO
                        r = to_record(frame, utt, gold, source="authored-target-pair-v1",
                                      extra_tags=["same-focus" if is_same else "switch-focus",
                                                  "same-type" if target.kind == other.kind else "cross-type",
                                                  f"action:{verb}", f"syntax:{syntax}",
                                                  "visible-short-name" if name != target.name else "full-name"],
                                      meta={"target_name": target.name, "focus_name": focus.name,
                                            "label_basis": "LABELING sections 2/4; named target equals focus iff executable",
                                            "paired_group": group})
                        records.append(r)
    return list({r.id: r for r in records}.values())


def dev_homes():
    # New synthetic layouts, disjoint from c1-c4 and the acceptance layouts.
    return {
        "c5-riverside": [Device(n, r, k, k) for n, r, k in [
            ("茶室壁灯", "茶室", "light"), ("走廊筒灯", "走廊", "light"),
            ("茶室智能音箱", "茶室", "media"), ("休息区电视", "休息区", "media"),
            ("茶室空调", "茶室", "climate"), ("休息区空调", "休息区", "climate"),
            ("茶室遮光帘", "茶室", "cover"), ("走廊净化器", "走廊", "fan")]],
        "c5-hillside": [Device(n, r, k, k) for n, r, k in [
            ("画室轨道灯", "画室", "light"), ("门廊吊灯", "门廊", "light"),
            ("画室投影仪", "画室", "media"), ("门廊音箱", "门廊", "media"),
            ("画室空调", "画室", "climate"), ("阁楼空调", "阁楼", "climate"),
            ("阁楼窗帘", "阁楼", "cover"), ("画室加湿器", "画室", "fan")]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "target-pool.jsonl": [r for i, home in enumerate(SPLIT_HOMES["train"])
                              for r in pairs(home, load_home(home), 5100 + i, 12)],
        "target-dev.jsonl": [r for i, (home, devs) in enumerate(dev_homes().items())
                             for r in pairs(home, devs, 5200 + i, 12, novel=True)],
    }
    stats = {}
    for name, rows in outputs.items():
        p = a.out / name
        if p.exists():
            raise FileExistsError(p)
        write_jsonl(p, rows)
        stats[name] = {"n": len(rows), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                       "gold": dict(Counter(r.labels["follow"]["gold"] for r in rows)),
                       "tags": dict(Counter(t for r in rows for t in r.tags)),
                       "homes": sorted({r.meta["home"] for r in rows})}
    (a.out / "target-manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
