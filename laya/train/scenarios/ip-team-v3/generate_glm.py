"""Generate review-only, text-only IP Team episode drafts with GLM-5.3-Flash.

No draft is added to a training split automatically. Each paid request is recorded
before sending so an interrupted process will not silently repeat a charge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import yaml
from episodes import ACTIONS
from episodes import generate as expand_episode
from transformers import AutoTokenizer

from eidolon_laya_train.generators import ChatClient
from eidolon_laya_train.model import to_internal
from eidolon_laya_train.scenario import Scenario
from eidolon_models_laya.vendor.laya.common import build_sequence, serialize_state

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parents[1]
DEFAULT_OUT = TRAIN / "private/ip-team-v3-glm-flash-v2"
VALID_MODES = {"solo", "named", "group", "discuss", "interrupt", "quiet"}
EPISODE_FIELDS = {"mode", "cast", "user", "steps", "prelude", "goal", "budget", "same_name"}
STEP_FIELDS = {"action", "speaker", "heard", "brief"}
OFFICIAL_HOSTS = {"api.z.ai", "open.bigmodel.cn"}
ENSEMBLE_THEMES = ("森林旅伴", "旧书屋伙伴", "海岛小队", "星船同行者", "山城邻居", "古风旅伴", "夜市动物朋友")


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{number}: expected NAME=value")
        name, value = (piece.strip() for piece in line.split("=", 1))
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise ValueError(f"{path}:{number}: invalid environment variable name")
        if not os.environ.get(name):
            os.environ[name] = value.strip("\"'")


def load_plan(path: Path) -> dict:
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or not isinstance(plan.get("version"), str):
        raise ValueError("generation plan needs a version")
    slices = plan.get("slices")
    if plan.get("domain", "creative_staff") not in {"creative_staff", "ensemble_characters"}:
        raise ValueError("unsupported data-generation domain")
    if not isinstance(slices, list) or not slices:
        raise ValueError("generation plan needs slices")
    ids = set()
    for spec in slices:
        if not isinstance(spec, dict) or not re.fullmatch(r"[a-z][a-z0-9_]*", str(spec.get("id", ""))):
            raise ValueError("each slice needs a safe id")
        if spec["id"] in ids:
            raise ValueError(f"duplicate slice {spec['id']}")
        ids.add(spec["id"])
        if spec.get("mode") not in VALID_MODES or spec.get("candidates") not in (2, 3, 4):
            raise ValueError(f"{spec['id']}: invalid mode or candidate count")
        if not isinstance(spec.get("quota"), int) or not 1 <= spec["quota"] <= 20:
            raise ValueError(f"{spec['id']}: quota must be 1–20 paid calls")
        if not isinstance(spec.get("focus"), str) or not spec["focus"].strip():
            raise ValueError(f"{spec['id']}: missing focus")
        if spec["id"] == "wait_or_finish" and spec.get("decisions") != ["wait", "finish"]:
            raise ValueError("wait_or_finish must alternate wait and finish")
    return plan


def schedule(plan: dict, out: Path, count: int) -> list[tuple[dict, str]]:
    calls = out / "calls"
    used = {spec["id"]: 0 for spec in plan["slices"]}
    for path in calls.glob("*.json") if calls.exists() else ():
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("plan_version") != plan["version"]:
            raise ValueError(f"{path}: different plan version; choose a new --out directory")
        sid = item.get("slice")
        if sid not in used:
            raise ValueError(f"{path}: unknown slice")
        if item.get("status") == "pending":
            raise ValueError(f"{path}: interrupted paid call; inspect before retrying")
        used[sid] += 1
    selected = []
    while len(selected) < count:
        progress = False
        for spec in plan["slices"]:
            sid = spec["id"]
            if used[sid] >= spec["quota"]:
                continue
            used[sid] += 1
            current = dict(spec)
            current["domain"] = plan.get("domain", "creative_staff")
            if current["domain"] == "ensemble_characters":
                current["theme"] = ENSEMBLE_THEMES[(sum(used.values()) - 1) % len(ENSEMBLE_THEMES)]
            if sid == "wait_or_finish":
                current["decision"] = spec["decisions"][(used[sid] - 1) % 2]
            selected.append((current, f"{sid}-{used[sid]:03d}"))
            progress = True
            if len(selected) == count:
                break
        if not progress:
            break
    return selected


def prompt_for(spec: dict) -> str:
    """Ask for scenario facts only; the local adapter owns the decision schema.

    A nested episode/steps JSON request proved unreliable in the live smoke test.
    The model now supplies short natural text and role profiles, while the slice
    fixes the number and ordering of decisions. Semantic review is still required.
    """
    mode = spec["id"]
    ensemble = spec.get("domain") == "ensemble_characters"
    slots = [chr(ord("a") + i) for i in range(spec["candidates"])]
    fields = {f"role_{slot}": (f"{slot} 的性格、说话习惯、与其他伙伴关系，8-35字，不能写专长职业" if ensemble
                               else f"{slot} 的具体专长，8-35字") for slot in slots}
    fields["user"] = "原创自然的用户键入文本，15-100字"
    if mode in {"speaker_handoff", "same_member", "counterfactual", "named_finish", "wait_after_peer", "finish_after_peer"}:
        fields["heard_a"] = "成员 a 已公开说出的具体内容，15-100字"
    if mode == "counterfactual":
        fields["heard_b"] = "成员 a 在另一分支说出的内容，仅改变一个关键事实，15-100字"
    if mode == "user_override":
        fields["old_speech"] = "成员 a 关于旧目标已公开说出的内容，15-100字"
    extra = ({
        "named_unique": "user 必须包含字面量 {a}，明确只想听这位角色；其性格与当下情绪或问题自然契合，其他角色不插话。",
        "speaker_handoff": "user 请几位角色讨论一个日常选择；a 已说出具体看法并在 heard_a 中直接点名 {b} 邀请其补充不同角度，下一位应是 b，而不是机械轮流。",
        "same_member": "user 必须包含字面量 {a}，明确请这位角色连续分两段讲一件事；heard_a 只是第一段，第二段应由同一角色立即继续，不能等用户确认。",
        "user_override": "old_speech 是角色 a 对旧话题的真实公开发言；user 明确打断旧话题并点名 {b} 回应一个新的日常问题，不得继续旧发言。",
        "wait_or_finish": "user 明确要求" + ("所有角色暂停，等用户下一条输入；本轮仍开放。" if spec.get("decision") == "wait" else "这一轮已经结束，所有角色都不再发言。"),
        "named_finish": "user 必须包含字面量 {a}，明确只要这位角色一次完整回答、不让别人补话也不让同一角色续讲；请用多样的自然口语表达，不要机械写‘答完本轮就结束’。heard_a 是符合其性格的完整发言，不留待补内容。",
        "wait_after_peer": "user 必须包含字面量 {a}，明确只请 a 先说一句或一段，其他人不要接话、说完所有人等用户下一条指示；heard_a 已完成所要求的内容，不得要求其他角色继续。",
        "finish_direct": "user 直接明确表示本轮已结束，所有角色立刻停止发言；不需要先让任何角色回答。写自然口语，避免只重复一种‘到此为止’措辞。",
        "finish_after_peer": "user 必须包含字面量 {a}，明确请 a 一次回答完整，并原话说‘答完本轮就结束’；其他角色绝不补充。heard_a 是 a 已公开的完整回答，不能追问或邀请任何人继续。",
    } if ensemble else {
        "named_unique": "user 必须包含字面量 {a}，明确点名 a；a 的专长和请求匹配。只写需要 a 回答的请求。",
        "speaker_handoff": "user 同时需要 a 和 b 的不同专长；a 的 heard_a 已完成第一部分，接着唯一有用的是 b 的不同部分。",
        "same_member": "user 必须包含字面量 {a}，明确要求 a 分两段连续说完；heard_a 只完成第一段，下一段仍须 a 立即继续。不得写‘等我确认/等我输入/先停一下’。",
        "user_override": "old_speech 是关于 IP 角色创作的旧目标，user 明确改为另一项 IP 角色创作目标且唯独适合 b；不要写企业年会等无关活动。",
        "clarify": "user 缺少一个完成任务所必需的具体信息（如没指出是哪个角色或哪版草稿），a 最适合问清该信息；不得写已有足够信息即可给建议的请求。",
        "wait_or_finish": "user 明确要求" + ("暂时不发言、等用户再输入；此轮仍开放。" if spec.get("decision") == "wait" else "本轮已经完成，所有成员都停止发言。"),
        "same_name": "a 和 b 的展示名相同，user 必须包含字面量 {a}，但没有任何可区分 a/b 的线索；c 负责问用户要哪位。",
        "counterfactual": "user 需要先听 a 的事实，再选择 b 或 c；两个 heard 只改变一个具体事实，heard_a 应使 b 接力，heard_b 应使 c 接力。",
        "named_finish": "user 必须包含字面量 {a}，明确只请 a 一次说清，并在回答后结束本轮；heard_a 是 a 的完整回答，不能留下待补问题。",
        "wait_after_peer": "user 必须包含字面量 {a}，要求 a 先说明一件具体事，之后所有成员暂停，等用户下一条指示；heard_a 已完成这件事，不得提出新的问题。",
    })[mode]
    domain_intro = (
        "场景是 Eidolon One 上的故事角色团：用户与几位住在各自设备中的虚构角色本人聊天。"
        "角色像伙伴一样有性格、关系和各自的说话方式，彼此能根据已经公开的文字接话。"
        "cast 只写人物个性和人际关系，例如‘急性子，遇到新鲜事就想先试，常和 {b} 拌嘴’；"
        "不要写‘擅长/负责/熟悉/专注’某项工作，不能把他们写成专业顾问、行程策划师或内容制作团队。"
        f"cast 若提及另一位成员，只能用这些占位符：{', '.join('{' + slot + '}' for slot in slots)}；不能写裸字母或不存在的成员。"
        "请写角色本人面对用户的日常自由对话，比如半日出行、休息、陪伴、选择活动、观点讨论。"
        f"这次原创故事背景：{spec.get('theme', '日常伙伴')}。使用这个背景增加多样性，但用户问题仍是自然日常对话。"
        "避免总是‘急性子/慢性子/打圆场’三人组，也避免总聊河边散步。"
        "role_a/b/c 是人物性格与彼此关系，不是编剧、设计师、运营或制作 IP 的员工。"
        "只研究文本调度：谁先说、谁接话、何时安静；不写设备权限、声线、表情或 ASR。"
        if ensemble else "为 IP 角色团文本决策模型写一条原创中文自由对话素材。"
    )
    return (
        domain_intro + "只返回一个 JSON 对象，"
        "字段恰好如下，不写数组、Markdown、解释或嵌套步骤：\n"
        + json.dumps(fields, ensure_ascii=False)
        + f"\n场景约束：{extra}\n"
        + f"切片重点：{spec['focus']}\n"
        + "人物描述须有不同能力边界；用户文本要像真实键入，且下一步只有一个合理选择。"
        f"不要出现 ASR、语音、音频、智能家居设备控制。只可使用这些占位符：{', '.join('{' + slot + '}' for slot in slots)}；"
        "用户文本与公开发言中不要出现裸露的槽位字母 a、b、c。"
    )


def materialize_flat(text: str, spec: dict) -> list[dict]:
    raw = json.loads(text.strip())
    slots = [chr(ord("a") + i) for i in range(spec["candidates"])]
    mode = spec["id"]
    expected = {f"role_{slot}" for slot in slots} | {"user"}
    if mode in {"speaker_handoff", "same_member", "counterfactual", "named_finish", "wait_after_peer", "finish_after_peer"}:
        expected.add("heard_a")
    if mode == "counterfactual":
        expected.add("heard_b")
    if mode == "user_override":
        expected.add("old_speech")
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f"expected flat JSON keys {sorted(expected)}")
    for label, value in raw.items():
        if not isinstance(value, str):
            raise ValueError(f"{label}: expected text")
        unknown = set(re.findall(r"\{([a-z])\}", value)) - set(slots)
        if unknown:
            raise ValueError(f"{label}: references non-candidate slots {sorted(unknown)}")
    cast = {slot: _text(raw[f"role_{slot}"], f"role_{slot}", 4, 100) for slot in slots}
    user = _text(raw["user"], "user", 6, 220)
    if spec.get("domain") == "ensemble_characters":
        staff_terms = ("编剧", "策划", "设计师", "运营", "顾问", "制作人", "文案", "IP角色", "角色团", "周边", "人设", "擅长", "负责", "熟悉", "专注")
        if any(term in value for value in [*cast.values(), user] for term in staff_terms):
            raise ValueError("Ensemble cast/user describes IP production rather than in-world characters")
        if any(re.search(r"(?<!\{)[abc](?!\})", value) for value in cast.values()):
            raise ValueError("Ensemble cast contains an unrendered member slot")
        if mode == "wait_after_peer" and not any(term in user for term in ("等我", "先停", "别接", "不要接", "别说", "不说")):
            raise ValueError("wait_after_peer lacks an explicit user request to pause after a speaks")
        if mode in {"finish_direct", "finish_after_peer"} and not any(
            term in user for term in ("结束", "这轮就到", "聊到这", "先到这", "就到这", "收工", "散了", "不用再", "不再聊")
        ):
            raise ValueError("finish slice lacks an explicit user request to end the round")
    base = {"mode": spec["mode"], "cast": cast, "user": user, "prelude": []}
    if mode == "named_unique":
        base["steps"] = [{"action": "respond", "speaker": "a", "brief": "用户明确点名 a"}]
    elif mode == "speaker_handoff":
        if spec.get("domain") == "ensemble_characters":
            base["steps"] = [{"action": "respond", "speaker": "b",
                              "heard": {"speaker": "a", "text": raw["heard_a"]},
                              "brief": "a 已公开发言并点名 b 接话"}]
        else:
            base["steps"] = [
                {"action": "respond", "speaker": "a", "brief": "先由 a 处理其专长部分"},
                {"action": "respond", "speaker": "b", "heard": {"speaker": "a", "text": raw["heard_a"]},
                 "brief": "a 已公开完成自己的部分，交给 b"},
            ]
    elif mode == "same_member":
        base["steps"] = [
            {"action": "respond", "speaker": "a", "brief": "用户先指定 a 发言"},
            {"action": "respond", "speaker": "a", "heard": {"speaker": "a", "text": raw["heard_a"]},
             "brief": "用户要求 a 继续第二段"},
        ]
    elif mode == "user_override":
        base["prelude"] = [{"author": "a", "text": raw["old_speech"]}]
        base["steps"] = [{"action": "respond", "speaker": "b", "brief": "用户更新目标，应按新目标选择 b"}]
    elif mode == "clarify":
        base["steps"] = [{"action": "clarify", "speaker": "a", "brief": "缺少用户必须提供的信息"}]
    elif mode == "wait_or_finish":
        base["steps"] = [{"action": spec["decision"], "brief": "用户明确要求保持安静"}]
    elif mode == "finish_direct":
        base["steps"] = [{"action": "finish", "brief": "用户明确结束本轮"}]
    elif mode == "same_name":
        base["same_name"] = {"b": "a"}
        base["steps"] = [{"action": "clarify", "speaker": "c", "brief": "同名成员不可区分，先问用户"}]
    elif mode == "counterfactual":
        a = dict(base, steps=[{"action": "respond", "speaker": "b",
                               "heard": {"speaker": "a", "text": raw["heard_a"]},
                               "brief": "a 给出的事实使 b 适合接力"}])
        b = dict(base, steps=[{"action": "respond", "speaker": "c",
                               "heard": {"speaker": "a", "text": raw["heard_b"]},
                               "brief": "改变的事实使 c 适合接力"}])
        return [a, b]
    elif mode in {"named_finish", "wait_after_peer", "finish_after_peer"}:
        second = "wait" if mode == "wait_after_peer" else "finish"
        base["steps"] = [
            {"action": "respond", "speaker": "a", "brief": "用户明确先让 a 完成回答"},
            {"action": second, "heard": {"speaker": "a", "text": raw["heard_a"]},
             "brief": "a 已完成回答，按用户要求结束" if second == "finish" else "a 已完成回答，等用户下一条指示"},
        ]
    else:
        raise ValueError(f"unsupported slice {mode}")
    return [base]


def prompt_for_legacy(spec: dict) -> str:
    pair = bool(spec.get("pair"))
    count = 2 if pair else 1
    slots = [chr(ord("a") + i) for i in range(spec["candidates"])]
    shape = {"episodes": [{"mode": spec["mode"],
                           "cast": {slot: f"填写{slot}的具体专长" for slot in slots},
                           "user": "填写一条具体用户请求",
                           "prelude": [],
                           "steps": [{"action": "respond", "speaker": slots[0],
                                      "brief": "填写可审查的选择理由"}]}
                          for _ in range(count)]}
    pair_rule = (
        "两个 episodes 的 cast、user、mode、prelude 完全相同；都只有一个 step。"
        "两个 step 的 heard.speaker 相同，heard.text 只改变一个关键事实，"
        "且正确的 action/speaker 必须随之改变。"
        if pair else "一个 episode 可以有 1–4 个决策 step；下一步只能有一个合理金标。"
    )
    if spec["id"] == "named_unique":
        pair_rule = "本次只写一个 step：用户点名后的唯一下一步是 respond，不能追加 finish；未听到该成员真实发言前，不存在第二个决策。"
    return f"""你在为 IP 角色团的小型文本决策模型撰写离线训练草稿。只写原创、自然的中文键入文本；不得出现语音、ASR、音频或设备控制。模型仅决定下一步动作及一位候选成员，不生成实际回答。

本次切片：{spec['id']}。模式必须是 {spec['mode']}，候选必须恰好 {spec['candidates']} 位。
覆盖重点：{spec['focus']}
返回严格 JSON 对象，顶层只含 episodes 数组，恰好 {count} 项。不写解释、Markdown 或额外字段。以下是**字段形状**，所有“填写”内容都要换成原创具体文本，动作和说话成员按场景改：
{json.dumps(shape, ensure_ascii=False)}

每个 episode 的字段：mode、cast、user、steps；可选 prelude、goal、budget、same_name。
mode 的值只能是 {spec['mode']}，绝不能写切片 ID {spec['id']}。cast 必须是 JSON 对象，键恰好是 {', '.join(slots)}，不得写成数组；值是可区分的角色专长描述，不能写展示姓名。需要点名时写 {{a}}、{{b}} 这样的槽位占位符。
steps 每项字段只有 action、可选 speaker、可选 heard、可选 brief。action 只能是 respond、clarify、wait、finish；respond/clarify 必须有 speaker 槽位，wait/finish 不得有 speaker。brief 是给人工审稿人的一句理由，不是模型输入。heard 是已公开的角色发言，必须是与 action 同级的对象，格式 {{"speaker":"a","text":"具体发言"}}；若前一步安排了角色发言，下一步的 heard 必须来自刚安排的成员。prelude 是当前用户请求之前已公开的消息数组，每项格式 {{"author":"user 或槽位","text":"内容"}}。如有同名成员，same_name 用 {{"b":"a"}} 表示 b 展示名与 a 相同。
{pair_rule}
金标要由用户约束、角色专长与已公开文本共同支持；避免多个候选同样合理、机械轮流、重复答复和依赖外部事实。若用户要求等待或结束，就不要安排发言。最多四步，预算 1–8，公开文本要像真实自由对话，不能填充无关旧话题来凑长度。
"""


def parse_response(text: str) -> list[dict]:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1] if "\n" in value else ""
        value = value.rsplit("```", 1)[0].strip()
    result = json.loads(value)
    if not isinstance(result, dict) or set(result) != {"episodes"} or not isinstance(result["episodes"], list):
        raise ValueError("expected JSON object containing only an episodes array")
    return result["episodes"]


def _text(value: object, label: str, low: int = 1, high: int = 240) -> str:
    if not isinstance(value, str) or not low <= len(value.strip()) <= high:
        raise ValueError(f"{label}: expected {low}–{high} characters of text")
    return value.strip()


def validate_episode(item: dict, spec: dict) -> dict:
    if not isinstance(item, dict) or set(item) - EPISODE_FIELDS:
        raise ValueError("episode has unsupported fields")
    if item.get("mode") != spec["mode"]:
        raise ValueError("episode mode differs from planned slice")
    cast = item.get("cast")
    slots = [chr(ord("a") + i) for i in range(spec["candidates"])]
    if not isinstance(cast, dict) or set(cast) != set(slots):
        raise ValueError(f"cast must contain slots {slots}")
    for slot, description in cast.items():
        _text(description, f"cast.{slot}", 4, 100)
    _text(item.get("user"), "user", 6, 220)
    if spec["id"] in {"named_unique", "same_member", "same_name", "named_finish", "wait_after_peer", "finish_after_peer"} and not any(
        "{" + slot + "}" in item["user"] for slot in slots
    ):
        raise ValueError("named slice requires a slot reference in the user text")
    prelude = item.get("prelude", [])
    if not isinstance(prelude, list) or len(prelude) > 6:
        raise ValueError("prelude must have at most six messages")
    for msg in prelude:
        if not isinstance(msg, dict) or set(msg) != {"author", "text"} or msg["author"] not in {"user", *slots}:
            raise ValueError("invalid prelude message")
        _text(msg["text"], "prelude.text", 4, 260)
    if spec["id"] == "user_override" and not prelude:
        raise ValueError("user override needs prior public text")
    if "goal" in item:
        _text(item["goal"], "goal", 4, 180)
    if "budget" in item and (not isinstance(item["budget"], int) or not 1 <= item["budget"] <= 8):
        raise ValueError("budget must be 1–8")
    same_name = item.get("same_name", {})
    if not isinstance(same_name, dict) or any(k not in slots or v not in slots or k == v for k, v in same_name.items()):
        raise ValueError("invalid same_name mapping")
    if bool(same_name) != bool(spec.get("same_name")):
        raise ValueError("same_name mapping does not match planned slice")
    steps = item.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 4:
        raise ValueError("episode needs 1–4 steps")
    if spec["id"] == "named_unique" and (len(steps) != 1 or steps[0].get("action") != "respond"):
        raise ValueError("named_unique needs exactly one respond decision")
    if spec["id"] in {"named_finish", "wait_after_peer", "finish_after_peer"} and (
        len(steps) != 2 or [s.get("action") for s in steps] != ["respond", "wait" if spec["id"] == "wait_after_peer" else "finish"]
    ):
        raise ValueError("targeted slice needs respond then the specified silent action")
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or set(step) - STEP_FIELDS:
            raise ValueError("step has unsupported fields")
        action, speaker = step.get("action"), step.get("speaker")
        if action not in ACTIONS:
            raise ValueError("invalid action")
        if action in {"respond", "clarify"} and speaker not in slots:
            raise ValueError("speaking action needs a valid speaker")
        if action in {"wait", "finish"} and speaker is not None:
            raise ValueError("silent action cannot name a speaker")
        if "brief" in step:
            _text(step["brief"], "step.brief", 4, 180)
        heard = step.get("heard")
        if heard is not None:
            if not isinstance(heard, dict) or set(heard) != {"speaker", "text"} or heard["speaker"] not in slots:
                raise ValueError("invalid heard message")
            _text(heard["text"], "heard.text", 4, 260)
        if index:
            prior = steps[index - 1]
            if prior["action"] not in {"respond", "clarify"} or not heard or heard["speaker"] != prior["speaker"]:
                raise ValueError("next step must hear the previously selected speaker")
    if spec["id"] in {"clarify", "same_name"} and not any(s["action"] == "clarify" for s in steps):
        raise ValueError("clarification slice lacks a clarify decision")
    if spec["id"] == "wait_or_finish" and not any(s["action"] in {"wait", "finish"} for s in steps):
        raise ValueError("quiet slice lacks a silent decision")
    if spec["id"] == "same_member" and not (
        len(steps) >= 2 and steps[0]["action"] == steps[1]["action"] == "respond"
        and steps[0]["speaker"] == steps[1]["speaker"]
    ):
        raise ValueError("same-member slice needs consecutive turns by one member")
    if spec["id"] == "speaker_handoff" and not (
        (spec.get("domain") == "ensemble_characters" and len(steps) == 1
         and steps[0]["action"] == "respond" and steps[0]["speaker"] == "b"
         and steps[0].get("heard", {}).get("speaker") == "a")
        or (spec.get("domain") != "ensemble_characters" and len(steps) >= 2
            and steps[0]["action"] == steps[1]["action"] == "respond"
            and steps[0]["speaker"] != steps[1]["speaker"])
    ):
        raise ValueError("handoff slice needs two different consecutive speakers")
    return dict(item, cast={slot: cast[slot] for slot in slots}, tag=spec["id"])


def validate_batch(items: list[dict], spec: dict, call_id: str) -> list[dict]:
    pair = bool(spec.get("pair"))
    if len(items) != (2 if pair else 1):
        raise ValueError("wrong number of episodes")
    episodes = [validate_episode(item, spec) for item in items]
    if pair:
        a, b = episodes
        if any(a.get(key) != b.get(key) for key in ("cast", "user", "mode", "prelude", "goal", "same_name")):
            raise ValueError("counterfactual branches changed their shared context")
        x, y = a["steps"], b["steps"]
        if len(x) != 1 or len(y) != 1:
            raise ValueError("counterfactual branches need one decision each")
        if not x[0].get("heard") or not y[0].get("heard"):
            raise ValueError("counterfactual branches need heard text")
        if x[0]["heard"]["speaker"] != y[0]["heard"]["speaker"] or x[0]["heard"]["text"] == y[0]["heard"]["text"]:
            raise ValueError("counterfactual must change one speaker's latest text")
        if (x[0]["action"], x[0].get("speaker")) == (y[0]["action"], y[0].get("speaker")):
            raise ValueError("counterfactual branches need different gold decisions")
    for index, item in enumerate(episodes):
        item["id"] = f"glm53_flash_{call_id.replace('-', '_')}_{index}"
        if pair:
            item["pair_id"] = f"glm53_flash_pair_{call_id.replace('-', '_')}"
            item["branch"] = "a" if index == 0 else "b"
    return episodes


def normalized_user(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(c for c in value if c.isalnum() or c in "{}")


def existing_users(out: Path) -> set[str]:
    paths = list((TRAIN / "data/authored").glob("ip-team*/*.yaml"))
    paths.extend((TRAIN / "data/generated/ip-team-v4").glob("*/train.yaml"))
    paths.extend((out / "drafts").glob("*.yaml"))
    users = set()
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            users.update(normalized_user(item["user"]) for item in data if isinstance(item, dict) and isinstance(item.get("user"), str))
    return users


def validate_pipeline(episodes: list[dict], tokenizer, scenario: Scenario) -> int:
    with tempfile.TemporaryDirectory(prefix="ip-team-glm-check-") as directory:
        source = Path(directory) / "episodes.yaml"
        source.write_text(yaml.safe_dump(episodes, allow_unicode=True, sort_keys=False), encoding="utf-8")
        records = list(expand_episode(scenario, {"path": str(source), "split_name": "train", "variants": 1, "pair_seed": 1357}, random.Random(7)))
    if len(records) != sum(len(item["steps"]) for item in episodes):
        raise ValueError("scenario adapter dropped decision steps")
    maximum = 0
    for record in records:
        q = record.questions["move"]
        head, markers = build_sequence(tokenizer, "", to_internal(q), 2048, 256)
        state_tokens = tokenizer(serialize_state(record.state), add_special_tokens=False)["input_ids"]
        total = len(head) + len(state_tokens)
        if len(markers) != len(q["criteria"]) or total > 2048:
            raise ValueError(f"{record.id}: options incomplete or {total} tokens exceed 2048")
        maximum = max(maximum, total)
    return maximum


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(value)
        temp = Path(handle.name)
    os.replace(temp, path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE / "generation-plan.yaml")
    parser.add_argument("--env-file", type=Path, default=HERE / ".env")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=1, help="paid API calls this run (1–8)")
    parser.add_argument("--dry-run", action="store_true", help="print schedule; no key, files or API calls")
    args = parser.parse_args()
    if not 1 <= args.count <= 8:
        parser.error("--count must be 1–8 to bound spend per invocation")
    plan = load_plan(args.plan)
    selected = schedule(plan, args.out, args.count)
    if not selected:
        print("Plan quotas exhausted; no calls scheduled.")
        return 0
    if args.dry_run:
        print(json.dumps({"paid_calls": 0, "would_call": [call_id for _, call_id in selected],
                          "model": "glm-5.3-flash", "drafts_written": 0}, ensure_ascii=False, indent=2))
        return 0

    load_env(args.env_file)
    key = os.environ.get("EIDOLON_IP_DATA_API_KEY", "")
    base_url = os.environ.get("EIDOLON_IP_DATA_BASE_URL", "")
    model = os.environ.get("EIDOLON_IP_DATA_MODEL", "")
    effort = os.environ.get("EIDOLON_IP_DATA_REASONING_EFFORT", "low")
    parsed = urlparse(base_url)
    if not key:
        parser.error(f"set EIDOLON_IP_DATA_API_KEY in {args.env_file} before generation")
    if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_HOSTS or not parsed.path.endswith("/api/paas/v4"):
        parser.error("use an official Z.ai/BigModel regular HTTPS API base URL ending /api/paas/v4")
    if model != "glm-5.3-flash" or effort not in {"low", "high", "max"}:
        parser.error("this plan requires model glm-5.3-flash and a valid reasoning effort")
    tokenizer_path = TRAIN / "runs/r14/checkpoint/tokenizer"
    if not tokenizer_path.is_dir():
        parser.error(f"missing tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    scenario = Scenario.load(HERE)
    users = existing_users(args.out)
    client = ChatClient({"base_url": base_url, "model": model,
                         "api_key_env": "EIDOLON_IP_DATA_API_KEY", "timeout": 180,
                         "max_tokens": 4096, "thinking": {"type": "enabled"},
                         "reasoning_effort": effort,
                         "response_format": {"type": "json_object"}})
    accepted = 0
    for spec, call_id in selected:
        prompt = prompt_for(spec)
        metadata_path = args.out / "calls" / f"{call_id}.json"
        response_path = args.out / "responses" / f"{call_id}.txt"
        draft_path = args.out / "drafts" / f"{call_id}.yaml"
        meta = {"plan_version": plan["version"], "slice": spec["id"], "call_id": call_id,
                "model": model, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "pending"}
        atomic_json(metadata_path, meta)
        try:
            response = client.complete_with_meta(prompt, temperature=1.0)
        except Exception as exc:
            meta.update(status="api_error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
            atomic_json(metadata_path, meta)
            print(f"{call_id}: API request failed; inspect {metadata_path}")
            return 1
        atomic_text(response_path, response["text"])
        meta.update(response_id=response.get("id"), response_model=response.get("model"),
                    usage=response.get("usage", {}), response_sha256=hashlib.sha256(response["text"].encode()).hexdigest())
        try:
            episodes = validate_batch(materialize_flat(response["text"], spec), spec, call_id)
            fresh = {normalized_user(item["user"]) for item in episodes}
            if users & fresh:
                raise ValueError("user request duplicates an existing authored or draft family")
            maximum = validate_pipeline(episodes, tokenizer, scenario)
        except Exception as exc:
            meta.update(status="rejected", error=f"{type(exc).__name__}: {str(exc)[:300]}")
            atomic_json(metadata_path, meta)
            print(f"{call_id}: API reachable, draft rejected by data validation; inspect {metadata_path} and {response_path}")
            return 1
        atomic_text(draft_path, yaml.safe_dump(episodes, allow_unicode=True, sort_keys=False))
        users.update(fresh)
        meta.update(status="draft_for_review", episode_count=len(episodes), max_input_tokens=maximum,
                    draft_path=str(draft_path))
        atomic_json(metadata_path, meta)
        accepted += len(episodes)
        print(f"{call_id}: {len(episodes)} structurally valid episode draft(s); human review required")
    print(f"Completed {len(selected)} paid call(s), {accepted} draft(s); no training split changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
