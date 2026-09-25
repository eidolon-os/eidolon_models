#!/usr/bin/env python3
"""智能家居控制决策评测：调用 laya 服务（/v1/systemone），并与关键词规则基线对照。

    python3 run_eval.py --url http://127.0.0.1:8771 --label mac-torch-mps [--server-pid PID]

每条用例一次请求问三道题：
  intent  是控制 / 查询 / 无关
  device  是家里哪台设备（设备清单来自 homes/，外加「多个设备或整屋」「没有对应的设备」）
  action  做什么操作（只对金标给了 action 的用例计分）

结果写到 results/<label>/：predictions.jsonl（逐条）与 summary.json（汇总）。只依赖标准库。
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MULTI = "多个设备或整屋"
NONE = "没有对应的设备"

INTENT_Q = {
    "type": "choice",
    "instructions": "`utterance` 是在让智能家居做什么？",
    "criteria": {
        "控制": "要求改变家里设备的状态：打开、关闭、调节、设定、启动、暂停、上锁",
        "查询": "询问家里设备的状态或读数，不改变任何设备",
        "无关": "与控制或查询家里的设备无关：闲聊、常识、陈述、购物",
    },
}
ACTION_Q = {
    "type": "choice",
    "instructions": "`utterance` 要对设备做什么操作？",
    "criteria": {
        "打开或启动": "开启、启动、开始运行",
        "关闭或停止": "关掉、停止、断电、收起",
        "调高或增大": "调亮、升温、调大音量或档位",
        "调低或减小": "调暗、降温、调小、降下",
        "设为指定的数值或模式": "设到具体的温度、档位、百分比或模式",
        "暂停": "暂时停下，之后再继续",
        "上锁": "锁门、上锁",
    },
}


# ---------------------------------------------------------------------------- 数据


def load_homes() -> dict[str, dict]:
    raw = {p.stem: json.loads(p.read_text("utf-8")) for p in (HERE / "homes").glob("*.json")}
    homes = {}
    for name, home in raw.items():
        devices = list(home.get("devices", []))
        for inc in home.get("include", []):
            devices += raw[inc]["devices"]
        homes[name] = {**home, "devices": devices}
    return homes


def load_cases(only: set[str] | None) -> list[dict]:
    cases = []
    for sdir in sorted((HERE / "scenarios").iterdir()):
        if not sdir.is_dir() or (only and sdir.name[:2] not in only):
            continue
        for line in (sdir / "cases.jsonl").read_text("utf-8").splitlines():
            if line.strip():
                cases.append({**json.loads(line), "scenario": sdir.name})
    return cases


def device_question(home: dict) -> dict:
    crit = {d["name"]: f"{d['room']}·{d['type']}" for d in home["devices"]}
    crit[MULTI] = "同时涉及多台设备，或整屋的场景模式"
    crit[NONE] = "家里没有能满足这个要求的设备"
    return {
        "type": "choice",
        "instructions": "`utterance` 说的是家里的哪台设备？",
        "criteria": crit,
    }


def as_set(value) -> set[str] | None:
    if value is None:
        return None
    return set(value) if isinstance(value, list) else {value}


def category_of(case: dict, homes: dict) -> str:
    gold = case["gold"]
    if gold["intent"] == "无关":
        return "非设备"
    devices = as_set(gold.get("device")) or set()
    if devices & {MULTI}:
        return "多设备/场景"
    if devices & {NONE}:
        return "家中没有"
    by_name = {d["name"]: d for d in homes[case["home"]]["devices"]}
    return by_name[sorted(devices)[0]]["category"] if devices else "?"


# ---------------------------------------------------------------------------- 客户端


def post(url: str, body: dict, timeout: float = 120) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        url + "/v1/systemone",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get(url: str, path: str) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url + path, timeout=30) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------- 规则基线

TYPE_WORDS = {  # 设备类型 → 常见说法（一个通用词表，不针对具体用例）
    "吸顶灯": ["灯"],
    "台灯": ["台灯", "灯"],
    "吊灯": ["吊灯", "灯"],
    "空调": ["空调"],
    "电动窗帘": ["窗帘"],
    "空气净化器": ["净化器"],
    "加湿器": ["加湿器"],
    "电视": ["电视"],
    "音箱": ["音箱"],
    "扫拖机器人": ["扫地机"],
    "洗衣机": ["洗衣机"],
    "晾衣架": ["晾衣架"],
    "电饭煲": ["电饭煲"],
    "燃气热水器": ["热水器"],
    "门锁": ["门锁", "锁门", "门锁上"],
    "摄像头": ["摄像头"],
    "温湿度传感器": ["温度", "湿度", "多少度"],
    "落地扇": ["风扇", "落地扇"],
    "地暖": ["地暖"],
    "新风系统": ["新风"],
    "冰箱": ["冰箱"],
    "洗碗机": ["洗碗机"],
    "烤箱": ["烤箱"],
    "油烟机": ["油烟机"],
    "浴霸": ["浴霸"],
    "投影仪": ["投影"],
    "可视门铃": ["门铃"],
    "安防报警系统": ["安防", "布防", "警报"],
    "车库门": ["车库门"],
    "灌溉水阀": ["水阀", "浇水"],
    "电动车充电桩": ["充电桩"],
    "自动喂食器": ["喂食", "喂"],
    "智能插座": ["插座"],
}
ROOM_WORDS = {
    "主卧": ["主卧", "卧室"],
    "客厅": ["客厅"],
    "书房": ["书房"],
    "餐厅": ["餐厅"],
    "厨房": ["厨房"],
    "卫生间": ["卫生间"],
    "阳台": ["阳台"],
    "车库": ["车库"],
}
ALL_TYPE_WORDS = {w for words in TYPE_WORDS.values() for w in words}
MULTI_WORDS = ["全部", "所有", "都", "模式", "出门", "回来", "睡觉", "起床", "断电", "全屋"]
QUERY_WORDS = ["吗", "没有", "了没", "多少", "几度", "多久", "是不是", "还是", "谁", "状态"]
CONTROL_WORDS = [
    "开",
    "关",
    "调",
    "设",
    "拉",
    "锁",
    "降",
    "升",
    "暂停",
    "启动",
    "开始",
    "预热",
    "断电",
    "布防",
    "喂",
    "浇",
    "充电",
    "煮",
    "播放",
    "静音",
    "模式",
    "档",
]


def rule_baseline(text: str, home: dict) -> dict:
    scored, categories = [], set()
    for d in home["devices"]:
        if any(w in text for w in TYPE_WORDS.get(d["type"], [d["type"]])):
            room_hit = any(w in text for w in ROOM_WORDS.get(d["room"], [d["room"]]))
            scored.append((1 + room_hit, d["name"]))
            if d["category"] != "传感器":
                categories.add(d["category"])
    if any(w in text for w in MULTI_WORDS) or len(categories) >= 2:
        device = MULTI
    elif scored:
        device = max(scored, key=lambda x: x[0])[1]
    elif any(w in text for w in ALL_TYPE_WORDS):
        device = NONE
    else:
        device = None
    if device and any(w in text for w in QUERY_WORDS):
        intent = "查询"
    elif device and any(w in text for w in CONTROL_WORDS):
        intent = "控制"
    else:
        intent = "无关"
    for words, act in (
        (["暂停"], "暂停"),
        (["锁上", "锁门", "门锁上"], "上锁"),
        (["调到", "设", "模式", "档", "一半", "静音", "度"], "设为指定的数值或模式"),
        (["调高", "调亮", "高一点", "大一点", "亮一", "升", "强"], "调高或增大"),
        (["调低", "调暗", "小一点", "低一点", "暗", "降"], "调低或减小"),
        (["关", "断电", "拉上", "停", "回去充电"], "关闭或停止"),
        (["开", "启动", "开始", "预热", "喂", "浇", "充电", "煮"], "打开或启动"),
    ):
        if any(w in text for w in words):
            action = act
            break
    else:
        action = None
    return {"intent": intent, "device": device or NONE, "action": action}


# ---------------------------------------------------------------------------- 评测


def score(case: dict, pred: dict) -> dict:
    gold = case["gold"]
    res = {"intent_ok": pred["intent"] == gold["intent"]}
    if (devices := as_set(gold.get("device"))) is not None:
        res["device_ok"] = pred["device"] in devices
    if (actions := as_set(gold.get("action"))) is not None:
        res["action_ok"] = pred["action"] in actions
    if gold["intent"] == "控制":
        res["e2e_ok"] = (
            res["intent_ok"] and res.get("device_ok", True) and res.get("action_ok", True)
        )
    return res


def rate(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows if key in r]
    return {"n": len(vals), "acc": round(sum(vals) / len(vals), 4) if vals else None}


def metrics(rows: list[dict], score_key: str) -> dict:
    scored = [r[score_key] for r in rows]
    out = {k: rate(scored, k) for k in ("intent_ok", "device_ok", "action_ok", "e2e_ok")}
    tp = sum(
        r["gold"]["intent"] == "控制" and r[score_key.replace("score", "pred")]["intent"] == "控制"
        for r in rows
    )
    pp = sum(r[score_key.replace("score", "pred")]["intent"] == "控制" for r in rows)
    gp = sum(r["gold"]["intent"] == "控制" for r in rows)
    p = tp / pp if pp else 0.0
    rc = tp / gp if gp else 0.0
    out["control_detection"] = {
        "precision": round(p, 4),
        "recall": round(rc, 4),
        "f1": round(2 * p * rc / (p + rc), 4) if p + rc else 0.0,
    }
    return out


def pct(values: list[float], q: float) -> float | None:
    s = sorted(values)
    if not s:  # servers other than ours report no server-side timings
        return None
    return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8771")
    ap.add_argument("--label", required=True, help="结果目录名，例如 mac-torch-mps")
    ap.add_argument("--scenarios", help="只跑这些场景编号，逗号分隔，如 01,03")
    ap.add_argument("--server-pid", type=int, help="服务进程 PID，用来记录常驻内存")
    args = ap.parse_args()

    homes = load_homes()
    cases = load_cases(set(args.scenarios.split(",")) if args.scenarios else None)
    try:
        info = get(args.url, "/v1/info")  # our service; other Jev-protocol servers lack it
    except urllib.error.HTTPError:
        info = {}
    dq = {name: device_question(h) for name, h in homes.items()}
    for _ in range(3):  # 预热
        post(
            args.url,
            {
                "state": {"utterance": "打开客厅的灯"},
                "questions": {"intent": INTENT_Q, "device": dq["apartment"], "action": ACTION_Q},
            },
        )

    rows = []
    for case in cases:
        body = {
            "state": {"utterance": case["text"]},
            "questions": {"intent": INTENT_Q, "device": dq[case["home"]], "action": ACTION_Q},
        }
        t0 = time.perf_counter()
        res = post(args.url, body)
        wall = (time.perf_counter() - t0) * 1000
        a = res["answers"]
        pred = {
            "intent": a["intent"]["choice"],
            "device": a["device"]["choice"],
            "action": a["action"]["choice"],
        }
        top = {k: a[k]["probabilities"][a[k]["choice"]] for k in ("intent", "device", "action")}
        base = rule_baseline(case["text"], homes[case["home"]])
        rows.append(
            {
                "id": case["id"],
                "scenario": case["scenario"],
                "home": case["home"],
                "category": category_of(case, homes),
                "text": case["text"],
                "gold": case["gold"],
                "pred": pred,
                "p_top": top,
                "score": score(case, pred),
                "base_pred": base,
                "base_score": score(case, base),
                "n_device_options": len(dq[case["home"]]["criteria"]),
                "latency_ms": {
                    "client": round(wall, 1),
                    "server": res.get("timing_ms", {}).get("total"),
                    "forward": res.get("timing_ms", {}).get("forward"),
                },
                "input_tokens": res.get("usage", {}).get("input_tokens"),
            }
        )

    out = HERE / "results" / args.label
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_scn, by_cat, by_opts = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        by_scn[r["scenario"]].append(r)
        by_cat[r["category"]].append(r)
        by_opts[r["n_device_options"]].append(r)
    confusion = Counter((r["gold"]["intent"], r["pred"]["intent"]) for r in rows)
    calib = {}
    for q in ("intent", "device"):
        key = f"{q}_ok"
        pairs = [(r["p_top"][q], r["score"][key]) for r in rows if key in r["score"]]
        calib[q] = {
            str(th): {
                "coverage": round(sum(p >= th for p, _ in pairs) / len(pairs), 4),
                "acc": round(
                    sum(ok for p, ok in pairs if p >= th) / max(1, sum(p >= th for p, _ in pairs)),
                    4,
                ),
            }
            for th in (0.0, 0.5, 0.7, 0.9)
        }
    lat = [r["latency_ms"]["client"] for r in rows]
    summary = {
        "label": args.label,
        "url": args.url,
        "host": platform.node(),
        "machine": platform.machine(),
        "server": info.get("engine") or {"model": args.label},
        "n_cases": len(rows),
        "server_rss_mb": (
            int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(args.server_pid)])) // 1024
        )
        if args.server_pid
        else None,
        "laya": metrics(rows, "score"),
        "baseline": metrics(rows, "base_score"),
        "by_scenario": {
            k: {"laya": metrics(v, "score"), "baseline": metrics(v, "base_score")}
            for k, v in sorted(by_scn.items())
        },
        "by_category": {
            k: {"laya": metrics(v, "score"), "baseline": metrics(v, "base_score")}
            for k, v in sorted(by_cat.items())
        },
        "intent_confusion": {f"{g}->{p}": n for (g, p), n in sorted(confusion.items())},
        "calibration": calib,
        "latency_ms": {
            "client_p50": pct(lat, 0.5),
            "client_p95": pct(lat, 0.95),
            "client_mean": round(statistics.mean(lat), 1),
            "server_p50": pct([r["latency_ms"]["server"] for r in rows if r["latency_ms"]["server"]], 0.5),
            "forward_p50": pct([r["latency_ms"]["forward"] for r in rows if r["latency_ms"]["forward"]], 0.5),
            "by_device_options": {
                str(k): {
                    "n": len(v),
                    "client_p50": pct([r["latency_ms"]["client"] for r in v], 0.5),
                    "tokens_mean": round(statistics.mean(r["input_tokens"] for r in v)),
                }
                for k, v in sorted(by_opts.items())
            },
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    m, b = summary["laya"], summary["baseline"]
    print(
        f"[{args.label}] {len(rows)} cases | intent {m['intent_ok']['acc']} (base {b['intent_ok']['acc']})"
        f" | device {m['device_ok']['acc']} (base {b['device_ok']['acc']})"
        f" | action {m['action_ok']['acc']} | e2e {m['e2e_ok']['acc']} (base {b['e2e_ok']['acc']})"
        f" | p50 {summary['latency_ms']['client_p50']} ms p95 {summary['latency_ms']['client_p95']} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
