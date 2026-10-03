"""c10 authored clause composition contrasts, without importing evaluation records."""
from intent_pairs import OFF, ON, SET
from intent_pairs import build as build_records


def examples(d, novel=False):
    n = d.name
    if novel:
        return [
            (f"{n}不用开，咱们出去散散步", None, "prohibition"),
            (f"{n}先别关，我得找人商量件事", None, "prohibition"),
            (f"等发奖金就给{n}换一种支持语音的，省得老找遥控器", None, "shopping"),
            (f"原来的{n}准备淘汰，打算挑带自动感应的，晚上省事", None, "shopping"),
            (f"我记得早晨已经开启{n}了，具体时间记不清", None, "reported"),
            (f"{n}开启吧，我去楼下买水果", ON, "explicit-control"),
            (f"{n}关掉吧，接下来我要看资料", OFF, "explicit-control"),
            (f"别让{n}停下来，给我启动它，我要用了", ON, "negated-control"),
            (f"先不买新的，把{n}关闭，我要出门", OFF, "shopping-control"),
        ]
    out = []
    clauses = ["我们去喝杯茶", "帮我讲一个故事", "我想聊聊今天的新闻", "等会儿有人来检修"]
    prohibitions = [f"别启动{n}了", f"不要开启{n}", f"{n}不用关了", f"先别关闭{n}"]
    for first, second in zip(prohibitions, clauses, strict=True):
        out.append((f"{first}，{second}", None, "prohibition"))
    for action, verb in [(ON, "打开"), (OFF, "关掉")]:
        for second in clauses[:2]:
            out.append((f"{n}{verb}吧，{second}", action, "explicit-control"))
    features = ["自动感应", "更省电", "远程定时", "不发出噪音"]
    for feature, purpose in zip(features, ["用起来方便", "省点电费", "出门也放心", "不影响休息"], strict=True):
        out.extend([
            (f"以后把{n}换一个能{feature}的型号，{purpose}", None, "shopping"),
            (f"准备给{n}换成带新功能的，最好{feature}，{purpose}", None, "shopping"),
        ])
    out.extend([
        (f"{n}刚才打开过，我只是告诉你一声", None, "reported"),
        (f"昨天有人关了{n}，我不知道是谁", None, "reported"),
        (f"不是要关{n}，我要你现在开启它，然后我去休息", ON, "negated-control"),
        (f"别再开着{n}了，立刻关闭，然后聊点别的", OFF, "negated-control"),
        (f"买新款的事以后再说，先开启{n}，今天还得用", ON, "shopping-control"),
        (f"{n}先不换了，现在关闭它，我准备出门", OFF, "shopping-control"),
    ])
    settings = {"light": "亮度调到70%", "media": "音量调到30%", "climate": "温度设为25度",
                "water_heater": "温度设为45度", "cover": "开度调到40%", "fan": "风速调到50%"}
    if d.kind in settings:
        out.append((f"暂时不更换{n}，把它的{settings[d.kind]}，我继续用", SET, "shopping-control"))
    return out


def build(homes, novel=False):
    return build_records(homes, novel, examples_factory=examples, version="c10")
