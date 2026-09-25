"""Import adapters for public datasets, used as replay data (see train/data/README.md).

Each adapter reads a copy under ``train/data/external/<name>/`` and yields Records whose gold
answers are always legal options of the scenario's questions (``target_vector`` is checked on
every record). Rows that cannot be mapped are skipped and counted; the counts are printed to
stderr at the end of the generator so a run reports what it dropped.

- ``zhihao_smarthome``: zhihao666/smartHome (HF, Apache-2.0). Each row carries its own device
  list and a ``control_device`` tool call.
- ``scenic``: huluk98/SCENIC (GitHub, MIT). Prompt -> Chinese confirmation; the confirmation is
  regular enough to parse room / device / action back out of it with rules.
- ``massive_zh``: MASSIVE zh-CN (CC-BY-4.0, via mteb/amazon_massive_intent). ``iot`` sentences
  become control cases on our apartment home; every other scenario becomes a ``无关`` negative.
"""

from __future__ import annotations

import gzip
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from .records import Record, record_id, target_vector
from .scenario import Scenario

EXIT_MULTI = "多个设备或整屋"
EXIT_NONE = "没有对应的设备"

ON, OFF, UP, DOWN, SET, PAUSE, LOCK = (
    "打开或启动",
    "关闭或停止",
    "调高或增大",
    "调低或减小",
    "设为指定的数值或模式",
    "暂停",
    "上锁",
)


class Skips:
    """Per-reason skip counter that reports itself on stderr."""

    def __init__(self, name: str):
        self.name = name
        self.counts: Counter[str] = Counter()
        self.notes: Counter[str] = Counter()
        self.kept = 0

    def skip(self, reason: str) -> None:
        self.counts[reason] += 1

    def note(self, reason: str) -> None:
        """A kept record that lacks something (e.g. no action label); not a skip."""
        self.notes[reason] += 1

    def report(self) -> None:
        total = self.kept + sum(self.counts.values())
        detail = ", ".join(f"{k}={v}" for k, v in self.counts.most_common()) or "none"
        notes = ", ".join(f"{k}={v}" for k, v in self.notes.most_common()) or "none"
        print(
            f"{self.name}: kept {self.kept}/{total}; skipped: {detail}; notes: {notes}",
            file=sys.stderr,
        )


def _checked(questions: dict, labels: dict, where: str) -> dict:
    for qid, lab in labels.items():
        try:
            target_vector(questions[qid], lab)
        except ValueError as e:
            raise ValueError(f"{where}: {qid}: {e}") from e
    return labels


def _load_homes(homes_dir: Path) -> dict[str, dict[str, str]]:
    from .generators import _load_homes

    return _load_homes(homes_dir)


def _resolve(scenario: Scenario, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (scenario.root / path).resolve()


# ------------------------------------------------------------------ zhihao666/smartHome

ZHIHAO_ACTIONS = {
    "TurnOn": ON,
    "TurnOff": OFF,
    "Open": ON,
    "Close": OFF,
    "Pause": PAUSE,
    "SetTemperature": SET,
    "SetLevel": SET,
}
_ZH_USER = re.compile(
    r"用户指令[:：]\s*(?P<utt>.*?)\s*;\s*设备列表如下[:：]\s*(?P<devs>\[.*\])", re.S
)
_ZH_CALL = re.compile(
    r'<tool_call name="control_device">\s*(?P<calls>\[.*?\])\s*</tool_call>', re.S
)


def parse_zhihao_row(row: dict) -> dict | None:
    """``{utterance, devices: [{deviceName, deviceTypeName, roomName, floorName}], calls: [...]}``
    or ``None`` when the row does not have the expected shape."""
    msgs = {m["role"]: m["content"] for m in row.get("messages", [])}
    mu = _ZH_USER.search(msgs.get("user", ""))
    mc = _ZH_CALL.search(msgs.get("assistant", ""))
    if not mu or not mc:
        return None
    try:
        devices = json.loads(mu.group("devs"))
        calls = json.loads(mc.group("calls"))
    except json.JSONDecodeError:
        return None
    if not isinstance(devices, list) or not isinstance(calls, list) or not calls:
        return None
    return {"utterance": mu.group("utt").strip(), "devices": devices, "calls": calls}


def zhihao_device_map(devices: list[dict]) -> tuple[dict[str, str], list[str]]:
    """Option names for a row's device list: ``deviceName``, prefixed with the room when the
    same name occurs more than once (and with the floor when room+name still collides).
    Returns ``(criteria, option_name_per_device)``."""
    names = Counter(d.get("deviceName", "") for d in devices)
    keyed: list[str] = []
    for d in devices:
        n = d.get("deviceName", "")
        keyed.append(n if names[n] == 1 else f"{d.get('roomName', '')}{n}")
    counts = Counter(keyed)
    for i, d in enumerate(devices):
        if counts[keyed[i]] > 1:
            keyed[i] = f"{d.get('floorName', '')}{keyed[i]}"
    crit: dict[str, str] = {}
    for d, k in zip(devices, keyed, strict=True):
        crit.setdefault(k, f"{d.get('roomName', '')}·{d.get('deviceTypeName', '')}")
    return crit, keyed


def zhihao_gold_device(parsed: dict, keyed: list[str]) -> tuple[str | list[str] | None, bool]:
    """The device gold for a row: the exit for several calls; otherwise the called device.
    When the called name (or, failing that, its type) matches several devices and the
    utterance does not name the room, every same-name (same-type) device is acceptable —
    the sentence alone cannot single one out. Returns ``(gold, ambiguous)``."""
    devices = parsed["devices"]
    calls = parsed["calls"]
    if len(calls) > 1:
        return EXIT_MULTI, False
    call = calls[0]
    name = call.get("deviceName")
    utt = parsed["utterance"]
    cands = [i for i, d in enumerate(devices) if d.get("deviceName") == name]
    if not cands:
        return None, False
    if len(cands) == 1:
        return keyed[cands[0]], False
    loc = str(call.get("location") or "")
    rooms_named = [i for i in cands if devices[i].get("roomName") and devices[i]["roomName"] in utt]
    if len(rooms_named) == 1:
        return keyed[rooms_named[0]], False
    in_loc = [i for i in cands if devices[i].get("roomName") and devices[i]["roomName"] in loc]
    if rooms_named and len(in_loc) == 1 and in_loc[0] in rooms_named:
        return keyed[in_loc[0]], False
    return [keyed[i] for i in cands], True


def import_zhihao_smarthome(scenario: Scenario, config: dict) -> Iterator[Record]:
    root = _resolve(scenario, config["path"])
    files = [root / f for f in config.get("files", ["train.jsonl"])]
    slot = config.get("device_slot", "devices")
    base_tags = list(config.get("tags") or ["external:zhihao", "explicit-control"])
    skips = Skips("zhihao_smarthome")
    seen: set[str] = set()
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            where = f"{f.name}:{n}"
            parsed = parse_zhihao_row(json.loads(line))
            if parsed is None:
                skips.skip("unparsable")
                continue
            if not parsed["utterance"]:
                skips.skip("empty-utterance")
                continue
            crit, keyed = zhihao_device_map(parsed["devices"])
            if not crit:
                skips.skip("empty-device-list")
                continue
            gold, ambiguous = zhihao_gold_device(parsed, keyed)
            if gold is None:
                skips.skip("called-device-not-in-list")
                continue
            key = f"{parsed['utterance']}|{sorted(crit)}"
            if key in seen:
                skips.skip("duplicate")
                continue
            seen.add(key)
            questions = scenario.build_questions({slot: crit})
            labels: dict[str, dict] = {"intent": {"gold": "控制"}, "device": {"gold": gold}}
            actions = {ZHIHAO_ACTIONS.get(str(c.get("action"))) for c in parsed["calls"]}
            if len(actions) == 1 and None not in actions:
                labels["action"] = {"gold": actions.pop()}
            else:
                skips.note("no-action-label")
            _checked(questions, labels, where)
            tags = base_tags + (["ambiguous-name"] if ambiguous else [])
            if gold == EXIT_MULTI:
                tags.append("multi-device")
            skips.kept += 1
            yield Record(
                id=record_id(scenario.name, "zhihao", parsed["utterance"], sorted(crit)),
                scenario=scenario.name,
                source=f"import:{config.get('name', 'zhihao_smarthome')}",
                state={"utterance": parsed["utterance"]},
                questions=questions,
                labels=labels,
                tags=tags,
                meta={"file": where, "raw_action": [c.get("action") for c in parsed["calls"]]},
            )
    skips.report()


# ------------------------------------------------------------------ SCENIC

SCENIC_ROOMS = ("客厅", "卧室", "书房", "厨房", "浴室", "阳台", "玄关")
# Words that follow the device name inside a confirmation (an attribute or a sub-function),
# used only to cut the device name off the rest of the segment.
SCENIC_ATTRS = (
    "温度",
    "风速",
    "风向",
    "亮度",
    "音量",
    "输入源",
    "画面模式",
    "频道",
    "加湿量",
    "除湿强度",
    "静音",
    "字幕",
    "播放",
    "清扫",
    "摇头",
    "录像",
    "隐私模式",
)
_SC_ON = ("打开", "开启", "开始", "启动", "恢复", "继续播放", "继续")
_SC_OFF = ("关闭", "关掉", "停止")
_SC_VERB = "|".join(sorted(_SC_ON + _SC_OFF + ("暂停",), key=len, reverse=True))
_SC_ATTR = "|".join(sorted(SCENIC_ATTRS, key=len, reverse=True))
_SC_ROOM = "|".join(SCENIC_ROOMS)
_SC_ADJUST = re.compile(
    r"^(?P<dev>.+?)(?:(?P<attr>" + _SC_ATTR + r"))?"
    r"(?P<op>调高|调大|调亮|升高|调低|调小|调暗|降低|调到|调整到|调至|设置为|设为|切换到|切换至|切到|调为|开到|关闭一半|打开一半)"
    r"(?P<val>.*)$"
)
_SC_TIMED = re.compile(r"^(?P<dev>.+?)设置(?P<when>.+?)(?P<op>开启|打开|关闭|关掉)$")
_SC_SIMPLE = re.compile(r"^(?P<verb>" + _SC_VERB + r")(?P<rest>.+)$")


def scenic_segments(response: str) -> list[str]:
    """Split a confirmation into its ``已…`` segments (one per device action)."""
    r = response.strip().rstrip("。").strip()
    r = re.sub(r"^(?:好的|好|收到)[，,]\s*", "", r)
    return [s.strip() for s in re.split(r"[；;]", r) if s.strip()]


def _split_room(s: str) -> tuple[str, str]:
    for room in SCENIC_ROOMS:
        if s.startswith(room):
            return room, s[len(room) :]
    return "", s


def _cut_attr(s: str) -> str:
    """Cut a trailing attribute / sub-function off a device string (``电视静音`` -> ``电视``)."""
    for a in sorted(SCENIC_ATTRS, key=len, reverse=True):
        if s.endswith(a) and len(s) > len(a):
            return s[: -len(a)]
    return s


def parse_scenic_segment(seg: str) -> dict | None:
    """``{room, device, action, kind}`` for one ``已…`` segment; ``None`` when no rule fits.
    ``action`` is one of the scenario's action names or ``None`` (the segment is a device
    control whose operation has no option, e.g. unlocking)."""
    if not seg.startswith("已"):
        return None
    body = seg[1:]
    if body.startswith("将"):
        room, rest = _split_room(body[1:])
        m = _SC_ADJUST.match(rest)
        if not m:
            return None
        op = m.group("op")
        if op in ("调高", "调大", "调亮", "升高"):
            action = UP
        elif op in ("调低", "调小", "调暗", "降低"):
            action = DOWN
        else:
            action = SET
        return {
            "room": room,
            "device": _cut_attr(m.group("dev")),
            "action": action,
            "kind": "adjust",
        }
    if body.startswith("为"):
        room, rest = _split_room(body[1:])
        m = _SC_TIMED.match(rest)
        if not m:
            return None
        action = ON if m.group("op") in ("开启", "打开") else OFF
        return {
            "room": room,
            "device": _cut_attr(m.group("dev")),
            "action": action,
            "kind": "timed",
        }
    if body.startswith("让"):
        rest = body[1:]
        m = re.match(r"^(?P<room>" + _SC_ROOM + r")?(?P<dev>.+?)返回充电$", rest)
        if m:
            return {
                "room": m.group("room") or "",
                "device": m.group("dev"),
                "action": OFF,
                "kind": "robot",
            }
        m = re.match(r"^(?P<dev>.+?)清扫(?P<room>" + _SC_ROOM + r")$", rest)
        if m:
            return {
                "room": m.group("room"),
                "device": m.group("dev"),
                "action": ON,
                "kind": "robot",
            }
        return None
    if body.startswith("锁好") or body.startswith("锁上"):
        room, rest = _split_room(body[2:])
        return {
            "room": room,
            "device": "门锁" if rest in ("门", "门锁") else rest,
            "action": LOCK,
            "kind": "lock",
        }
    if body.startswith("解锁"):
        room, rest = _split_room(body[2:])
        return {
            "room": room,
            "device": "门锁" if rest in ("门", "门锁") else rest,
            "action": None,
            "kind": "lock",
        }
    if body.startswith("检查"):
        room, rest = _split_room(body[2:])
        rest = re.sub(r"状态$", "", rest)
        return {
            "room": room,
            "device": "门锁" if rest in ("门", "门锁") else rest,
            "action": None,
            "kind": "query",
        }
    m = _SC_SIMPLE.match(body)
    if not m:
        return None
    verb, rest = m.group("verb"), m.group("rest")
    if verb == "继续播放" and rest.startswith(SCENIC_ROOMS + ("电视", "音箱")):
        room, rest = _split_room(rest)
    else:
        room, rest = _split_room(rest)
    if rest.startswith("播放"):  # "已打开播放音箱"
        rest = rest[2:]
    device = _cut_attr(rest)
    if not device:
        return None
    action = ON if verb in _SC_ON else OFF if verb in _SC_OFF else PAUSE
    return {"room": room, "device": device, "action": action, "kind": "simple"}


def parse_scenic_response(response: str) -> list[dict] | None:
    """All segments of a confirmation, or ``None`` if any of them fails to parse."""
    out = []
    for seg in scenic_segments(response):
        p = parse_scenic_segment(seg)
        if p is None:
            return None
        out.append(p)
    return out or None


def scenic_home(parsed_rows: Iterable[list[dict]], min_count: int = 1) -> dict[str, str]:
    """The fixed "SCENIC home": every (room, device) pair the confirmations mention, plus the
    devices that never come with a room, as ``{name: "room·device"}``. Device words seen fewer
    than ``min_count`` times are dropped (they are parser noise, not devices)."""
    counts: Counter[str] = Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    for segs in parsed_rows:
        for s in segs:
            counts[s["device"]] += 1
            pairs[(s["room"], s["device"])] += 1
    vocab = {d for d, c in counts.items() if c >= min_count}
    home: dict[str, str] = {}
    for room in SCENIC_ROOMS:
        for dev in sorted(vocab):
            if pairs[(room, dev)]:
                home[f"{room}{dev}"] = f"{room}·{dev}"
    for dev in sorted(vocab):
        if not any(pairs[(room, dev)] for room in SCENIC_ROOMS):
            home[dev] = f"全屋·{dev}"
    return home


def scenic_gold_device(segs: list[dict], home: dict[str, str]) -> str | list[str] | None:
    """Gold device for a parsed confirmation against the fixed home. A bare device name
    (no room) is acceptable as any of its room variants."""
    if len(segs) > 1:
        return EXIT_MULTI
    s = segs[0]
    if s["room"]:
        name = f"{s['room']}{s['device']}"
        return name if name in home else None
    if s["device"] in home:
        return s["device"]
    variants = [f"{r}{s['device']}" for r in SCENIC_ROOMS if f"{r}{s['device']}" in home]
    return variants or None


def import_scenic(scenario: Scenario, config: dict) -> Iterator[Record]:
    path = _resolve(scenario, config["path"])
    slot = config.get("device_slot", "devices")
    base_tags = list(config.get("tags") or ["external:scenic"])
    skips = Skips("scenic")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("data") or raw.get("examples") or []
    rows: list[tuple[str, str, list[dict]]] = []
    for i, x in enumerate(raw):
        prompt = (x.get("prompt") or x.get("instruction") or "").strip()
        response = (x.get("response") or "").strip()
        if not prompt or not response:
            skips.skip("empty")
            continue
        segs = parse_scenic_response(response)
        if segs is None:
            skips.skip("unparsed-response")
            continue
        rows.append((f"{path.name}:{i}", prompt, segs))
    home = config.get("home") or scenic_home(
        (s for _, _, s in rows), min_count=int(config.get("min_device_count", 3))
    )
    # The fixed home has ~70 devices; putting all of them in every question overflows the
    # head budget and truncates each option to a few tokens. Each record instead gets the gold
    # devices plus ``subset_options`` random others (deterministic per prompt), so list length
    # and order vary the way real homes do.
    subset = int(config.get("subset_options", 11))
    seen: set[str] = set()
    for where, prompt, segs in rows:
        if prompt in seen:
            skips.skip("duplicate")
            continue
        seen.add(prompt)
        kinds = {s["kind"] for s in segs}
        device = scenic_gold_device(segs, home)
        if device is None:
            skips.skip("device-not-in-home")
            continue
        if subset and len(home) > subset + 1:
            gold = [device] if isinstance(device, str) else list(device)
            if device == EXIT_MULTI:  # keep the devices the segments name, so the list is honest
                gold = [n for n in (f"{g['room']}{g['device']}" for g in segs) if n in home]
            r = random.Random(prompt)
            others = [n for n in home if n not in gold]
            keep = set(gold) | set(r.sample(others, min(subset, len(others))))
            names = [n for n in home if n in keep]
            r.shuffle(names)
            questions = scenario.build_questions({slot: {n: home[n] for n in names}})
        else:
            questions = scenario.build_questions({slot: home})
        if kinds == {"query"}:
            labels: dict[str, dict] = {"intent": {"gold": "查询"}, "device": {"gold": device}}
            tags = base_tags + ["status-query"]
        else:
            labels = {"intent": {"gold": "控制"}, "device": {"gold": device}}
            actions = {s["action"] for s in segs}
            if len(actions) == 1 and None not in actions:
                labels["action"] = {"gold": actions.pop()}
            else:
                skips.note("no-action-label")
            tags = base_tags + ["multi-device" if len(segs) > 1 else "explicit-control"]
            if "timed" in kinds:
                tags.append("timed")
        _checked(questions, labels, where)
        skips.kept += 1
        yield Record(
            id=record_id(scenario.name, "scenic", prompt),
            scenario=scenario.name,
            source=f"import:{config.get('name', 'scenic')}",
            state={"utterance": prompt},
            questions=questions,
            labels=labels,
            tags=tags,
            meta={"file": where, "parsed": segs},
        )
    skips.report()
    print(f"scenic: home has {len(home)} devices", file=sys.stderr)


# ------------------------------------------------------------------ MASSIVE zh-CN

MASSIVE_IOT = {
    # label -> (device gold, action gold)
    "iot_hue_lightoff": (["客厅主灯", "主卧灯", "床头灯"], OFF),
    "iot_hue_lighton": (["客厅主灯", "主卧灯", "床头灯"], ON),
    "iot_hue_lightdim": (["客厅主灯", "主卧灯", "床头灯"], DOWN),
    "iot_hue_lightup": (["客厅主灯", "主卧灯", "床头灯"], UP),
    "iot_hue_lightchange": (["客厅主灯", "主卧灯", "床头灯"], SET),
    "iot_cleaning": ("扫地机器人", ON),
    "iot_wemo_on": (EXIT_NONE, ON),
    "iot_wemo_off": (EXIT_NONE, OFF),
    "iot_coffee": (EXIT_NONE, ON),
}


def read_massive(path: Path, lang: str = "zh-CN") -> Iterator[dict]:
    """Rows of a MASSIVE file (``.json.gz`` / ``.jsonl`` / ``.json``), keeping ``lang`` only.
    Handles both the mteb layout (``label_text``) and the AmazonScience layout (``scenario`` +
    ``intent``, ``utt``)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        text = fh.read()
    rows = (
        json.loads(text)
        if text.lstrip().startswith("[")
        else [json.loads(x) for x in text.splitlines() if x.strip()]
    )
    for r in rows:
        if r.get("lang", r.get("locale", lang)) != lang:
            continue
        label = r.get("label_text") or r.get("intent") or r.get("label")
        if not isinstance(label, str):
            continue
        yield {
            "id": str(r.get("id", "")),
            "text": (r.get("text") or r.get("utt") or "").strip(),
            "label": label,
            "scenario": r.get("scenario")
            if isinstance(r.get("scenario"), str)
            else label.split("_", 1)[0],
        }


def massive_record_labels(row: dict) -> tuple[dict, list[str]] | None:
    """``(labels, tags)`` for one MASSIVE row; ``None`` when the row is not usable."""
    if not row["text"]:
        return None
    if row["scenario"] == "iot":
        m = MASSIVE_IOT.get(row["label"])
        if m is None:
            return None
        device, action = m
        return (
            {"intent": {"gold": "控制"}, "device": {"gold": device}, "action": {"gold": action}},
            ["external:massive", "explicit-control", f"massive:{row['label']}"],
        )
    return {"intent": {"gold": "无关"}}, [
        "external:massive",
        "non-command",
        f"massive:{row['scenario']}",
    ]


def import_massive_zh(
    scenario: Scenario, config: dict, rng: random.Random | None = None
) -> Iterator[Record]:
    rng = rng or random.Random(int(config.get("seed", 7)))
    root = _resolve(scenario, config["path"])
    lang = config.get("lang", "zh-CN")
    splits = config.get("splits", ["train", "validation", "test"])
    files = [root / s / f"{lang}.json.gz" for s in splits]
    files = [f for f in files if f.exists()] or sorted(root.glob(f"**/{lang}.json*"))
    if not files:
        raise FileNotFoundError(f"no {lang} files under {root}")
    homes = _load_homes(
        _resolve(scenario, config.get("homes", "../../../evals/smart-home")) / "homes"
    )
    home = config.get("home_name", "apartment")
    slot = config.get("device_slot", "devices")
    questions = scenario.build_questions({slot: homes[home]})
    per_scenario = int(config.get("negatives_per_scenario", 300))
    # audio (volume up/down/mute), play (radio, music, podcasts) and music (settings) are
    # commands to a speaker, and our homes have 智能音箱: labelling them 无关 would teach the
    # model to ignore real commands. Excluded unless a config says otherwise.
    exclude = set(config.get("exclude_scenarios", ["audio", "play", "music"]))
    skips = Skips("massive_zh")
    positives: list[tuple[dict, dict, list[str]]] = []
    negatives: dict[str, list[tuple[dict, dict, list[str]]]] = {}
    seen: set[str] = set()
    for f in files:
        for row in read_massive(f, lang):
            if row["text"] in seen:
                skips.skip("duplicate")
                continue
            seen.add(row["text"])
            if row["scenario"] in exclude:
                skips.skip(f"excluded:{row['scenario']}")
                continue
            got = massive_record_labels(row)
            if got is None:
                skips.skip("unmapped-iot-intent" if row["scenario"] == "iot" else "empty")
                continue
            labels, tags = got
            if row["scenario"] == "iot":
                positives.append((row, labels, tags))
            else:
                negatives.setdefault(row["scenario"], []).append((row, labels, tags))
    chosen = list(positives)
    for scn in sorted(negatives):
        pool = negatives[scn]
        if per_scenario and len(pool) > per_scenario:
            skips.counts[f"sampled-out:{scn}"] += len(pool) - per_scenario
            pool = rng.sample(pool, per_scenario)
        chosen.extend(pool)
    for row, labels, tags in chosen:
        _checked(questions, labels, f"massive:{row['id']}")
        skips.kept += 1
        yield Record(
            id=record_id(scenario.name, "massive", lang, row["text"]),
            scenario=scenario.name,
            source=f"import:{config.get('name', 'massive_zh')}",
            state={"utterance": row["text"]},
            questions=questions,
            labels=labels,
            tags=tags + [f"home:{home}"],
            meta={"home": home, "massive_id": row["id"], "massive_label": row["label"]},
        )
    skips.report()
