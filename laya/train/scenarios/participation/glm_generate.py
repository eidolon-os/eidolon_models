"""Scripted episodes written by GLM: the plan (and so the gold) comes from this file, GLM only writes the words.

    uv run --extra torch --extra train python train/scenarios/participation/glm_generate.py plan --n 1000 --seed 1 --out <plan.jsonl>
    uv run --extra torch --extra train python train/scenarios/participation/glm_generate.py run --plan <plan.jsonl> \
        --per-call 2 --calls 5 --out <dir>          # at most --calls new calls; resumable; ledger in <dir>/calls/

Every call is written to the ledger before it is sent and counts against the shared cap (CAP) even if it
fails; there is no automatic retry. Accepted episodes land in <dir>/episodes.jsonl in WRITING.md format with
the gold from the plan; rejected ones in <dir>/rejected.jsonl with the reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))

from episodes import validate  # noqa: E402

CAP = 600  # user-approved GLM calls for this line of work (2026-09-28), across every use
LEDGER = HERE.parents[1] / "data" / "participation" / "glm-ledger"  # shared by every run
ENV = HERE.parent / "ip-team-v3" / ".env"
MODEL = "glm-5.3-flash"
BASE = "https://open.bigmodel.cn/api/paas/v4"
META_PHRASES = ("我说完了", "我讲完了", "说完了", "讲完了", "还没说完", "还没讲完", "没说完", "没讲完", "轮到你", "该你了", "到你了",
                "我的部分结束", "下面请", "第一部分", "第二部分", "第三部分")

TOPICS = [
    "周末去哪玩", "晚饭吃什么", "一道家常菜的做法", "养猫的烦恼", "要不要换工作", "备考复习计划", "一本好看的小说", "雨天的心情",
    "春节回家", "减肥和运动", "学吉他", "给妈妈挑生日礼物", "露营装备", "失眠怎么办", "一部老电影", "城市里的小公园", "早起的习惯",
    "和室友的矛盾", "第一次做饭", "旅行攻略", "宠物狗生病", "游戏里的关卡", "画画入门", "面试紧张", "搬家整理", "咖啡和茶", "种花",
    "童年的零食", "考驾照", "理财入门", "朋友过生日", "天文和星星", "恐龙知识", "历史小故事", "写作文", "打篮球", "包饺子",
    "海边旅行", "爬山", "雪天", "月饼的口味", "桌游规则", "怎么安慰朋友", "时间管理", "学英语", "做手工", "宇宙里的黑洞",
    "一首喜欢的歌", "奶奶的老房子", "新买的耳机", "夏天的西瓜", "二手书店", "夜市小吃", "地铁上的趣事", "猫为什么怕黄瓜",
]
STYLES = ["口语化、随意", "简短", "有点着急", "很客气", "带一点语音识别式的小错字", "说得有点绕", "撒娇的口吻", "认真正式"]


# --- plan: families -> events with a brief for the writer and the gold for the decision after it ---

def R(*s): return {"action": "respond", "speaker": [f"r{i}" for i in s]}
def C(about, *s): return {"action": "clarify", "speaker": [f"r{i}" for i in s], "clarify_about": about}
def W(): return {"action": "wait"}
def F(): return {"action": "finish"}
def U(brief, gold, **check): return {"kind": "user", "brief": brief, "gold": gold, **check}
def S(r, brief, gold): return {"kind": "say", "by": f"r{r}", "brief": brief, "gold": gold}


def team_plan(fam: str, rng: random.Random, topic: str) -> tuple[int, list[dict], dict]:
    n = rng.randint(3 if fam in ("T08", "T14") else 2, 5)
    idx = list(range(n)); rng.shuffle(idx)
    a, b, c = idx[0], idx[1 % n], idx[2 % n]
    every = list(range(n))
    extra: dict = {}
    if fam == "T01":
        ev = [U(f"用户点名 r{a}，请他就「{topic}」说一件事或给一个建议（只要他说一次）", R(a), name=a),
              S(a, "完整地回答用户", F())]
    elif fam == "T02":
        parts = rng.choice([2, 3])
        ev = [U(f"用户点名 r{a}，要他把「{topic}」分 {parts} 次讲，先讲第一部分", R(a), name=a)]
        for p in range(1, parts):
            ev.append(S(a, f"接着讲这个话题的第 {p} 块内容（内容上只讲一部分，但不要说'还没讲完''第几部分'之类的话）", R(a)))
        ev.append(S(a, "把剩下的内容讲完（不要说'讲完了'之类的话）", F()))
    elif fam == "T03":
        ev = [U(f"用户点名 r{a} 和 r{b}，让他们按这个顺序各说一句对「{topic}」的看法", R(a), name=a),
              S(a, "说自己的看法", R(b)), S(b, "说自己的看法", F())]
    elif fam == "T04":
        ev = [U(f"用户向大家问一个关于「{topic}」的开放问题，不点任何人的名字，谁都能回答", R(*every), unnamed=True),
              S(a, "回答这个问题", F()),
              U(f"用户接着点名 r{b}：「那你呢？」之类，问他的看法", R(b), name=b),
              S(b, "说自己的看法", F())]
    elif fam == "T05":
        ev = [U(f"用户让 r{a} 先说说对「{topic}」的看法，再让 r{b} 回应 r{a} 的观点", R(a), name=a),
              S(a, "提出一个具体、明确的观点", R(b)),
              S(b, f"针对 r{a} 刚才的具体观点回应（引用他的说法）", F())]
    elif fam == "T06":
        extra["specialty"] = a
        ev = [U(f"用户不点名，问一个明显属于 r{a} 专长领域的问题（和「{topic}」相关也可以）", R(a), unnamed=True),
              S(a, "用专业知识回答", F())]
    elif fam == "T07":
        ev = [U(f"用户点名 r{a} 聊「{topic}」", R(a), name=a),
              S(a, "回答，并提到一个具体细节（地点、数字、做法等）", F()),
              U("用户追问刚才提到的那个细节，用「你刚说的……」这样的说法，不点名", R(a), unnamed=True),
              S(a, "解释那个细节", F())]
    elif fam == "T08":
        ev = [U(f"用户让 r{a} 和 r{b} 都说说「{topic}」", R(a, b), name=a),
              S(a, "说自己的想法", R(b)), S(b, "说自己的想法", F()),
              U("用户说「他刚才说的我没听清，让他再说一遍」这类话，不指明是哪一位（两位都刚说过话）", C("指代不明", *every), unnamed=True),
              S(c, "问用户想让哪一位再说一遍", W()),
              U(f"用户说明是 r{b}（用名字或他说过的内容来指明）", R(b)),
              S(b, "把刚才的话再说一遍（换个说法）", F())]
    elif fam == "T09":
        ev = [U("用户点名一个根本不在候选里的名字（编一个和候选都不一样的名字），要他说话", C("对象不在场", *every), unnamed=True),
              S(c, "告诉用户这个人不在，问用户想找在场的哪一位", W()),
              U(f"用户改口点名 r{a}", R(a), name=a),
              S(a, "回应用户", F())]
    elif fam == "T10":
        extra["same_name"] = [a, b]
        if rng.random() < 0.5:
            ev = [U(f"用户用名字加一个区分线索（r{a} 的性格或职业）点名 r{a}；r{a} 和 r{b} 同名", R(a)),
                  S(a, "回应用户", F())]
        else:
            ev = [U(f"用户只叫这个名字（r{a} 和 r{b} 共用的名字），没有任何能区分是哪一位的线索", C("指代不明", *every)),
                  S(c, "问用户说的是哪一位（可以描述两位的区别）", W()),
                  U(f"用户用区分线索说明是 r{b}", R(b)),
                  S(b, "回应用户", F())]
    elif fam == "T11":
        ev = [U(f"用户点名 r{a} 问一个关于「{topic}」的问题", R(a), name=a),
              S(a, "回答", F()),
              U("用户说「你们先别说话，等我一下 / 我想想」这类话", W(), unnamed=True),
              U(f"用户回来，请 r{a} 接着说刚才的话题", R(a), name=a),
              S(a, "接着说", F())]
    elif fam == "T12":
        ev = [U(f"用户点名 r{a} 聊「{topic}」", R(a), name=a),
              S(a, "回答", F()),
              U("用户明确结束这一轮，比如「好了，这个就聊到这」", F(), unnamed=True),
              U(f"用户换一个新话题，点名 r{b}", R(b), name=b),
              S(b, "回答", F())]
    elif fam == "T13":
        ev = [U(f"用户点名 r{a}，要他分两次讲「{topic}」", R(a), name=a),
              S(a, "开始讲这个话题的一部分内容（不要说'还没讲完'之类的话）", R(a)),
              U(f"用户打断：不用讲了，改让 r{b} 说说别的", R(b), name=b),
              S(b, "回应用户的新要求", F())]
    elif fam == "T14":
        order = idx[:]
        ev = [U(f"用户让在场的每个人都用一句话说说「{topic}」", R(*every), unnamed=True)]
        for i, r in enumerate(order):
            rest = order[i + 1:]
            ev.append(S(r, "用一句话说", R(*rest) if rest else F()))
    elif fam == "T15":
        ev = [U(f"用户提到 r{a} 之前说过的某件事（带出 r{a} 的名字），但实际是点名问 r{b} 的看法", R(b), name=b),
              S(b, "回答", F())]
    elif fam == "T16":
        ev = [U("用户说一句含糊的请求，比如「那个再来一次」「就按刚才那样弄」，前面没有能对应的内容", C("要求不明", *every), unnamed=True),
              S(c, "问用户具体想要什么", W()),
              U(f"用户说明具体想要什么（关于「{topic}」），并点名 r{a}", R(a), name=a),
              S(a, "回应", F())]
    else:
        raise ValueError(fam)
    return n, ev, extra


def companion_plan(fam: str, rng: random.Random, topic: str) -> tuple[int, list[dict], dict]:
    if fam == "C01":
        ev = [U(f"用户和陪伴角色分享一件关于「{topic}」的事", R(0)), S(0, "回应用户", F()),
              U("用户接着这个话题再说几句", R(0)), S(0, "回应", F())]
    elif fam == "C02":
        ev = [U("用户说自己很累，请角色安静陪着就好，但可以说一句让人安心的话", R(0)), S(0, "说一句简短的安慰", F())]
    elif fam == "C03":
        ev = [U("用户说「先别说话，我想静一静 / 等我一下」", W()),
              U(f"用户回来了，想聊聊「{topic}」", R(0)), S(0, "回应", F())]
    elif fam == "C04":
        ev = [U(f"用户和角色聊「{topic}」", R(0)), S(0, "回应", F()),
              U("用户明确结束这一轮，比如「好啦，今天就聊到这，晚安」", F()),
              U("过了一会儿用户开了一个新话题", R(0)), S(0, "回应", F())]
    elif fam == "C05":
        ev = [U("用户说一句含糊的请求，比如「那个再来一个」，前面没有能对应的内容", C("要求不明", 0)),
              S(0, "问用户具体想要什么", W()),
              U(f"用户说明想要什么（关于「{topic}」）", R(0)), S(0, "回应", F())]
    elif fam == "C06":
        k = rng.choice([2, 3])
        ev = [U(f"用户请角色讲 {k} 个关于「{topic}」的小故事 / 列 {k} 条建议，明确说了数量", R(0))]
        for p in range(1, k):
            ev.append(S(0, "只讲其中一个（不要说'第几个''还有'之类的话）", R(0)))
        ev.append(S(0, "讲最后一个（不要说'最后一个''讲完了'之类的话）", F()))
    elif fam == "C07":
        ev = [U("用户倾诉一个烦恼或难过的事", R(0)), S(0, "温柔地回应", F()),
              U("用户说「不用安慰我了，让我自己静静」", W())]
    elif fam == "C08":
        if rng.random() < 0.5:
            ev = [U(f"用户问一个关于「{topic}」的问题", R(0)), S(0, "回答", F()),
                  U("用户只回一句纯确认或收尾的话，比如「好的」「嗯嗯就这样」", F())]
        else:
            ev = [U(f"用户问一个关于「{topic}」的问题", R(0)), S(0, "回答", F()),
                  U("用户道谢，但接着问了一个新的相关问题（「谢谢，那……？」）", R(0)), S(0, "回答", F())]
    else:
        raise ValueError(fam)
    return 1, ev, {}


def make_plan(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    team = [f"T{i:02d}" for i in range(1, 17)]
    comp = [f"C{i:02d}" for i in range(1, 9)]
    out = []
    for i in range(n):
        mode = "team" if rng.random() < 0.55 else "companion"
        fam = rng.choice(team if mode == "team" else comp)
        topic = rng.choice(TOPICS)
        k, ev, extra = (team_plan if mode == "team" else companion_plan)(fam, rng, topic)
        out.append({"id": f"G{seed}-{i:05d}", "mode": mode, "family": fam, "n": k, "topic": topic,
                    "style": rng.choice(STYLES), "events": ev, **extra})
    return out


# --- prompt / parse / check ---

RULES = """你是对话数据写手，为"智能陪伴 / IP 角色团"写虚构的对话文字。每个剧本给出角色数量、话题和一串事件，每个事件有一句写作要求。
你只负责：(1) 为 r0、r1… 每个角色编一个名字（2–4 个字）和一句角色设定（职业、性格、爱好各不相同，可以是现代、古风、奇幻、动物拟人等 IP 风格）；
(2) 按顺序为每个事件写一句自然的中文：用户事件写用户说的话，角色事件写该角色说的话。
要求：
- 严格按写作要求写，不增加、不删除、不调换事件；用户点名某角色时要用他的名字（或昵称里包含名字），要求"不点名"时不能出现任何角色的名字。
- 角色说话要符合自己的设定，自然口语，10–80 个字；不要说"我说完了""轮到你了"之类的话。用户的话按给定语气写，3–60 个字。
- 各剧本之间的人物和说法都要不同。
只输出 JSON：{"episodes": [{"id": 剧本id, "candidates": [{"slot": "r0", "name": "...", "role": "..."}, ...], "texts": ["事件1的文字", "事件2的文字", ...]}]}
注意：texts 必须是字符串数组，元素个数等于该剧本的事件数，每个元素只写一个事件的文字；每个剧本都要输出。"""


def prompt(batch: list[dict]) -> str:
    parts = [RULES, "", "剧本："]
    for p in batch:
        lines = [f"【{p['id']}】{'团队' if p['mode'] == 'team' else '单聊陪伴'}，{p['n']} 个角色（r0–r{p['n'] - 1}），话题：{p['topic']}，用户语气：{p['style']}"]
        if p.get("same_name"):
            x, y = p["same_name"]
            lines.append(f"  注意：r{x} 和 r{y} 名字完全相同，但设定不同（性格或职业能区分）。")
        if "specialty" in p:
            lines.append(f"  注意：r{p['specialty']} 的专长要和问题相关，其他角色的专长明显不相关。")
        for j, ev in enumerate(p["events"], 1):
            who = "用户" if ev["kind"] == "user" else ev["by"]
            lines.append(f"  {j}. {who}：{ev['brief']}")
        parts.extend(lines)
    return "\n".join(parts)


def called(name: str, text: str) -> bool:
    """The full name, or its last two characters (the usual nickname: 林知遥 -> 知遥)."""
    return name in text or (len(name) >= 3 and name[-2:] in text)


def check(p: dict, got: dict) -> tuple[dict | None, str]:
    cands = got.get("candidates") or []
    texts = got.get("texts")
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        return None, "texts is not a list of strings"
    if [c.get("slot") for c in cands] != [f"r{i}" for i in range(p["n"])]:
        return None, "candidate slots"
    if len(texts) != len(p["events"]):
        return None, f"{len(texts)} texts for {len(p['events'])} events"
    names = [str(c.get("name", "")).strip() for c in cands]
    if not all(names) or not all(str(c.get("role", "")).strip() for c in cands):
        return None, "empty name or role"
    if p.get("same_name"):
        x, y = p["same_name"]
        if names[x] != names[y]:
            return None, "same_name not honoured"
        if len(set(names)) != len(names) - 1:
            return None, "unexpected duplicate names"
    elif len(set(names)) != len(names):
        return None, "duplicate names"
    for ev, t in zip(p["events"], texts, strict=True):
        t = str(t).strip()
        if not t or len(t) > 200:
            return None, "empty or overlong text"
        if ev["kind"] == "say" and any(m in t for m in META_PHRASES):
            return None, "meta phrase in a role's line"
        if ev["kind"] == "user" and "name" in ev and not called(names[ev["name"]], t):
            return None, f"named user turn lacks the name {names[ev['name']]}"
        if ev["kind"] == "user" and ev.get("unnamed") and any(called(nm, t) for nm in names):
            return None, "unnamed user turn mentions a candidate"
    ep = {"episode": p["id"], "family": p["family"], "mode": p["mode"], "scene_goal": "", "source": "glm",
          "candidates": [{"id": c["slot"], "name": names[i], "role": str(c["role"]).strip()} for i, c in enumerate(cands)],
          "events": []}
    for ev, t in zip(p["events"], texts, strict=True):
        e = {"user": t.strip()} if ev["kind"] == "user" else {"say": ev["by"], "text": t.strip()}
        e["gold"] = ev["gold"]
        ep["events"].append(e)
    errs = validate(ep)
    return (None, "; ".join(errs)) if errs else (ep, "ok")


def load_env() -> None:
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def run(plan_path: Path, out: Path, per_call: int, calls: int) -> None:
    from eidolon_laya_train.generators import ChatClient

    load_env()
    if os.environ.get("EIDOLON_IP_DATA_BASE_URL", BASE) != BASE:
        raise SystemExit("unexpected endpoint")
    client = ChatClient({"base_url": BASE, "model": MODEL, "api_key_env": "EIDOLON_IP_DATA_API_KEY", "timeout": 240,
                         "max_tokens": 8000, "thinking": {"type": "enabled"}, "reasoning_effort": "low",
                         "response_format": {"type": "json_object"}})
    plan = [json.loads(x) for x in plan_path.read_text("utf-8").splitlines() if x.strip()]
    out.mkdir(parents=True, exist_ok=True)
    (LEDGER / "calls").mkdir(parents=True, exist_ok=True)
    done = set()
    for f in ("episodes.jsonl", "rejected.jsonl"):
        if (out / f).exists():
            done |= {json.loads(x)["episode"] for x in (out / f).read_text("utf-8").splitlines() if x.strip()}
    todo = [p for p in plan if p["id"] not in done]
    made = 0
    while todo and made < calls:
        used = len(list((LEDGER / "calls").glob("*.json")))
        if used >= CAP:
            print(f"cap reached: {used}/{CAP} calls", flush=True)
            return
        batch, todo = todo[:per_call], todo[per_call:]
        text = prompt(batch)
        cid = f"{time.strftime('%Y%m%dT%H%M%S')}-{hashlib.sha1(text.encode()).hexdigest()[:8]}"
        rec = {"id": cid, "model": MODEL, "plan": [p["id"] for p in batch], "prompt_sha1": hashlib.sha1(text.encode()).hexdigest(),
               "status": "sent", "out": str(out)}
        (LEDGER / "calls" / f"{cid}.json").write_text(json.dumps(rec, ensure_ascii=False))  # counted before sending
        made += 1
        t0 = time.time()
        try:
            meta = client.complete_with_meta(text, temperature=0.95)
            rec.update(status="ok", seconds=round(time.time() - t0, 1), usage=meta.get("usage"))
            raw = meta["text"]
            (out / "raw").mkdir(exist_ok=True)
            (out / "raw" / f"{cid}.json").write_text(raw)
            got = {e.get("id"): e for e in json.loads(raw).get("episodes", [])}
        except Exception as exc:  # noqa: BLE001 - counted and recorded, never retried
            rec.update(status="error", error=f"{type(exc).__name__}: {exc}"[:300], seconds=round(time.time() - t0, 1))
            got = {}
        (LEDGER / "calls" / f"{cid}.json").write_text(json.dumps(rec, ensure_ascii=False))
        for p in batch:
            ep, why = check(p, got[p["id"]]) if p["id"] in got else (None, rec.get("error", "missing in response"))
            with (out / ("episodes.jsonl" if ep else "rejected.jsonl")).open("a", encoding="utf-8") as f:
                f.write(json.dumps(ep if ep else {"episode": p["id"], "family": p["family"], "reason": why}, ensure_ascii=False) + "\n")
        ok = sum(1 for p in batch if (out / "episodes.jsonl").exists() and p["id"] in (out / "episodes.jsonl").read_text())
        print(f"call {cid} {rec['status']} {rec.get('seconds')}s accepted {ok}/{len(batch)} (ledger {used + 1}/{CAP})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--n", type=int, required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    r = sub.add_parser("run"); r.add_argument("--plan", required=True); r.add_argument("--out", required=True)
    r.add_argument("--per-call", type=int, default=2); r.add_argument("--calls", type=int, required=True)
    pv = sub.add_parser("preview"); pv.add_argument("--plan", required=True); pv.add_argument("--per-call", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "plan":
        rows = make_plan(a.n, a.seed)
        Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), "utf-8")
        print(f"{len(rows)} planned episodes -> {a.out}")
    elif a.cmd == "preview":
        rows = [json.loads(x) for x in Path(a.plan).read_text("utf-8").splitlines() if x.strip()]
        print(prompt(rows[: a.per_call]))
    else:
        run(Path(a.plan), Path(a.out), a.per_call, a.calls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
