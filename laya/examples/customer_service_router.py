"""智能客服工单路由：laya 给概率，规则决定怎么派单。

    python examples/customer_service_router.py                  # 跑内置的几条样例工单
    python examples/customer_service_router.py "我的账号被锁了"   # 路由一条自定义消息

一次请求问 5 个问题（部门 / 紧急度 / 要不要退款 / 会不会流失 / 要不要人工），
laya 在一次前向里全部回答；派到哪个队列、什么优先级、自动还是转人工分诊，
由下面几条明确的规则和阈值决定——模型建议不等于派单。

注意：服务端是 zero-shot 的 laya-multilingual，这里演示请求形状和决策流程；
阈值要用自己的工单数据校准，准确率要靠微调。每个问题都会把工单完整编码一次，
在 1 核的 ECS 上 5 个问题约 3 秒。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from laya_client import LayaClient

# ---- 要问模型的问题 --------------------------------------------------------------

DEPARTMENTS = {
    "billing": "账单与退款：扣费、重复扣款、退款、发票、续费",
    "technical": "技术支持：报错、闪退、故障、接口异常、无法使用",
    "sales": "销售咨询：价格、套餐、采购、试用、合同",
    "account": "账号与安全：登录、密码、验证码、账号被锁、被盗",
    "complaint": "投诉与建议：对服务不满、投诉、表扬、建议",
    "other": "其他：寒暄、致谢、与以上都无关",
}

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "`message` 应该由哪个部门处理？",
        "criteria": DEPARTMENTS,
    },
    "urgency": {
        "type": "score",
        "instructions": "`message` 有多紧急？",
        "criteria": ["不急，可以排队处理", "需要当天处理", "业务中断或有明确的截止时间，马上处理"],
    },
    "refund": {
        "type": "noul",
        "instructions": "客户在 `message` 里是否明确要求退款或退费？",
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "客户在 `message` 里是否表示要取消、不再续费或换用别家？",
    },
    "wants_human": {
        "type": "noul",
        "instructions": "客户在 `message` 里是否要求转人工客服或投诉？",
    },
}

# ---- 派单规则（阈值要用真实工单校准）--------------------------------------------

AUTO_ROUTE_MIN_P = 0.60  # 部门最高概率低于这个 → 不自动派，进人工分诊
FLAG_MIN_P = 0.50  # 是/否类信号的判定阈值
QUEUES = {
    "billing": "账单组",
    "technical": "技术支持组",
    "sales": "销售组",
    "account": "账号安全组",
    "complaint": "客户关系组",
    "other": "自助机器人",
}


@dataclass
class Routing:
    queue: str
    priority: str
    auto: bool
    department: str
    department_p: float
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


def route(client: LayaClient, message: str, *, customer: dict | None = None) -> Routing:
    # 当前这条消息放在 state 最前面：超长时 laya 默认丢掉末尾。
    state = {"message": message, **({"customer": customer} if customer else {})}
    result = client.decide(state, QUESTIONS)
    a = result["answers"]

    dept = a["department"]["choice"]
    dept_p = a["department"]["probabilities"][dept]
    urgency = a["urgency"]["score"]  # 0..2 的期望等级
    flags = [
        name for name in ("refund", "churn_risk", "wants_human") if a[name]["noul"] >= FLAG_MIN_P
    ]

    reasons = [f"部门={dept} p={dept_p:.2f}", f"紧急度={urgency:.2f}/2"]
    reasons += [f"{name}={a[name]['noul']:.2f}" for name in flags]

    if urgency >= 1.5 or "churn_risk" in flags:
        priority = "P1"
    elif urgency >= 0.8:
        priority = "P2"
    else:
        priority = "P3"

    if "wants_human" in flags:
        queue, auto = "人工坐席", False
        reasons.append("客户要求人工")
    elif dept_p < AUTO_ROUTE_MIN_P:
        queue, auto = "人工分诊", False
        reasons.append(f"部门置信不足（<{AUTO_ROUTE_MIN_P}）")
    else:
        queue, auto = QUEUES[dept], True

    return Routing(
        queue, priority, auto, dept, dept_p, flags, reasons, result["timing_ms"]["total"]
    )


SAMPLES = [
    "我上个月的账单被重复扣了两次费，麻烦今天退一下，不然我就不续费了。",
    "App 一打开就闪退，更新到最新版本还是不行，急用！",
    "你们企业版怎么收费？我们公司大概 50 个人想采购。",
    "我的账号被锁了，收不到验证码，没法登录。",
    "你们的服务太烂了，我要投诉，给我找个人工客服！",
    "请问发票抬头可以修改吗？",
    "Your API has been returning 500 errors for the last hour and our production is down.",
    "谢谢，问题已经解决了。",
]


def main(argv: list[str]) -> int:
    client = LayaClient()
    if not client.ready():
        print(f"服务不可用：{client.base_url}", file=sys.stderr)
        return 1
    messages = argv or SAMPLES
    for message in messages:
        r = route(client, message)
        mode = "自动" if r.auto else "转人工"
        print(f"[{r.priority}] {r.queue:<6} {mode:<3} | {message}")
        print(f"        {'；'.join(r.reasons)}  ({r.latency_ms:.0f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
