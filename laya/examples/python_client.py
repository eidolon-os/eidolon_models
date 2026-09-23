"""Calling the laya decision API from Python: six request shapes, via ``laya_client``.

    python examples/python_client.py                       # the ECS demo
    LAYA_URL=http://127.0.0.1:8771 python examples/python_client.py   # a local `serve`

The checkpoint is zero-shot: these show the request shapes, not tuned accuracy.
See customer_service_router.py for answers turned into a routing decision.
"""

from __future__ import annotations

from laya_client import LayaClient

client = LayaClient()
decide = client.decide

TEAM = {
    "唐僧": "师父，也叫御弟哥哥、玄奘、唐三藏",
    "孙悟空": "大师兄，也叫悟空、猴哥、齐天大圣、弼马温",
    "猪八戒": "二师兄，也叫八戒、悟能、呆子、天蓬元帅",
    "沙僧": "三师弟，也叫沙师弟、悟净、卷帘大将",
}


def example_addressee():
    """1. 这句话说给谁听 —— 一个 choice，别名写进 criteria。"""
    r = decide(
        {"utterance": "猴哥，前面那座山有没有妖怪？", "speaker": "主人"},
        {
            "addressee": {
                "type": "choice",
                "instructions": "`utterance` 这句话是说给谁听的？",
                "criteria": {**TEAM, "所有人": "对在场的所有人说，或没有指定具体的人"},
            }
        },
    )
    a = r["answers"]["addressee"]
    return f"{a['choice']}  p={a['probabilities'][a['choice']]}  ({r['timing_ms']['total']} ms)"


def example_multi_addressee():
    """2. 可能同时点了几个人 —— 每个成员一个 noul（问题越多越慢：每个都重算一遍 state）。"""
    questions = {
        name: {"type": "noul", "instructions": f"`utterance` 是不是对{name}说的？（{name}{alias}）"}
        for name, alias in TEAM.items()
    }
    r = decide({"utterance": "悟空和八戒，你们俩一起去探路。", "speaker": "主人"}, questions)
    picked = {name: a["noul"] for name, a in r["answers"].items()}
    return f"{picked}  ({r['timing_ms']['total']} ms)"


def example_next_step():
    """3. 下一步该做什么 —— 参与决策的 respond / clarify / wait / finish。"""
    r = decide(
        {
            "utterance": "嗯……那个……",
            "speaker": "主人",
            "history": [{"speaker": "唐僧", "text": "徒儿们，今晚在哪里歇息？"}],
        },
        {
            "next_step": {
                "type": "choice",
                "instructions": "团队收到 `utterance` 后下一步应该怎么做？",
                "criteria": {
                    "respond": "有明确的问题或请求，应该有人回应",
                    "clarify": "意图不明确，需要追问",
                    "wait": "话还没说完，先等一等",
                    "finish": "对话已经结束，不需要回应",
                },
            }
        },
    )
    a = r["answers"]["next_step"]
    return f"{a['choice']}  {a['probabilities']}"


def example_urgency():
    """4. 有序打分 —— score 返回等级的期望值（0..n-1）。"""
    r = decide(
        {"utterance": "快！师父被妖怪抓走了！", "speaker": "沙僧"},
        {
            "urgency": {
                "type": "score",
                "instructions": "`utterance` 有多紧急？",
                "criteria": ["不急，随口一说", "需要尽快处理", "十万火急，马上行动"],
            }
        },
    )
    a = r["answers"]["urgency"]
    return f"score={a['score']} / 2  {a['probabilities']}"


def example_interrupt():
    """5. 播报中被插话 —— 和 channel 打断分类器同一组标签 keep/stop/reply/wait。"""
    r = decide(
        {
            "assistant_text": "今天北京晴，最高气温二十六度，傍晚有三到四级北风……",
            "user_text": "等等，那上海呢？",
        },
        {
            "intent": {
                "type": "choice",
                "instructions": (
                    "用户在 `assistant_text` 播放时说了 `user_text`，希望怎么处理当前播报？"
                ),
                "criteria": {
                    "keep": "附和或希望继续",
                    "stop": "只停止，不需要新回答",
                    "reply": "停止并回应新的问题或纠正",
                    "wait": "话意还没形成",
                },
            }
        },
    )
    a = r["answers"]["intent"]
    return f"{a['choice']}  confidence={a['confidence']}"


def example_long_history():
    """6. 长历史 —— 超出 1024 token 时默认丢末尾；truncate_left=True 保留最近的对话。

    更稳的写法是把当前这句话放在 state 最前面。回复里的 `truncated` 列出被截断的问题。
    """
    history = [
        {"speaker": "猪八戒", "text": f"第{i}件事：我又饿了，想吃斋饭。"} for i in range(200)
    ]
    state = {"history": history, "utterance": "八戒，你到底想说什么？"}  # 当前这句在最后
    q = {
        "addressee": {
            "type": "choice",
            "instructions": "`utterance` 是说给谁听的？",
            "criteria": list(TEAM) + ["所有人"],
        }
    }
    head = decide(state, q)
    tail = decide(state, q, truncate_left=True)
    return (
        f"默认: truncated={head['truncated']} → {head['answers']['addressee']['choice']} | "
        f"truncate_left: truncated={tail['truncated']} → {tail['answers']['addressee']['choice']}"
    )


if __name__ == "__main__":
    for fn in (
        example_addressee,
        example_multi_addressee,
        example_next_step,
        example_urgency,
        example_interrupt,
        example_long_history,
    ):
        print(f"{fn.__doc__.strip().splitlines()[0]}\n    {fn()}")
