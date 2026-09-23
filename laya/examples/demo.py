#!/usr/bin/env python3
"""laya 决策 API 演示：智能客服工单路由。单文件，只依赖 Python 标准库（3.8+）。

    python3 demo.py                                   # 跑内置样例工单
    python3 demo.py "我的账号被锁了，收不到验证码"        # 路由自定义消息（可以给多条）
    python3 demo.py --raw "发票开错了"                  # 同时打印模型原始回答
    python3 demo.py --url http://127.0.0.1:8771       # 指定服务地址（也可用环境变量 LAYA_URL）

流程：一次请求问 5 个问题（部门 / 紧急度 / 要不要退款 / 会不会流失 / 要不要人工），
laya 一次前向全部回答；派到哪个队列、什么优先级、自动还是转人工分诊，由下面的
阈值规则决定——模型只给概率，不直接派单。

服务端是 zero-shot 的 laya-multilingual：这里演示请求形状和决策流程，阈值要用
真实工单校准，准确率要靠微调。每个问题都会把工单完整编码一次，1 核 ECS 上约 2 秒/单。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_URL = "http://8.141.101.214:8771"


# =============================================================================
# 客户端：POST /v1/systemone
# =============================================================================


class LayaError(RuntimeError):
    """服务端返回非 2xx。``code`` 是服务端错误码，如 invalid_question / busy。"""

    def __init__(self, status: int, code: str | None, message: str | None):
        super().__init__(f"HTTP {status} {code}: {message}")
        self.status, self.code, self.message = status, code, message


class LayaClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        retries: int = 3,
        use_system_proxy: bool = False,
    ):
        self.base_url = (base_url or os.environ.get("LAYA_URL") or DEFAULT_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("LAYA_API_KEY")
        self.timeout = timeout
        self.retries = retries
        # 本机的 HTTP 代理（如 Clash 7890）对一个直连得到的服务器只是多一跳。
        handlers = [] if use_system_proxy else [urllib.request.ProxyHandler({})]
        self._opener = urllib.request.build_opener(*handlers)

    def decide(
        self, state: Any, questions: dict[str, dict], *, truncate_left: bool = False
    ) -> dict:
        """POST /v1/systemone.

        questions: {问题id: {"type": "choice" | "score" | "noul",
                             "instructions": ..., "criteria": ...}}
        """
        body = {"state": state, "questions": questions, "options": {"truncate_left": truncate_left}}
        return self._call("POST", "/v1/systemone", body)

    def ready(self) -> bool:
        try:
            return self._call("GET", "/readyz").get("status") == "ready"
        except (LayaError, OSError):
            return False

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        for attempt in range(self.retries):
            try:
                with self._opener.open(request, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as err:
                # 503：服务端排队已满，按它给的 Retry-After 再来。
                if err.code == 503 and attempt + 1 < self.retries:
                    time.sleep(float(err.headers.get("Retry-After", "1")))
                    continue
                try:
                    detail = json.loads(err.read() or b"{}").get("error", {})
                except ValueError:
                    detail = {}
                raise LayaError(err.code, detail.get("code"), detail.get("message")) from None
        raise AssertionError("unreachable")


# =============================================================================
# 路由：要问模型的问题
# =============================================================================

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
    "refund": {"type": "noul", "instructions": "客户在 `message` 里是否明确要求退款或退费？"},
    "churn_risk": {
        "type": "noul",
        "instructions": "客户在 `message` 里是否表示要取消、不再续费或换用别家？",
    },
    "wants_human": {
        "type": "noul",
        "instructions": "客户在 `message` 里是否要求转人工客服或投诉？",
    },
}

# =============================================================================
# 路由：派单规则（阈值要用真实工单校准）
# =============================================================================

AUTO_ROUTE_MIN_P = 0.60  # 部门最高概率低于这个 → 不自动派，进人工分诊
FLAG_MIN_P = 0.50  # 是/否类信号的判定阈值
P1_URGENCY = 1.5  # 紧急度期望值（0..2）≥ 这个 → P1
P2_URGENCY = 0.8  # ≥ 这个 → P2，否则 P3
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
    urgency: float
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)


def route(client: LayaClient, message: str, *, customer: dict | None = None) -> Routing:
    # 当前消息放在 state 最前面：超出 token 预算时服务端默认丢掉末尾。
    state: dict[str, Any] = {"message": message}
    if customer:
        state["customer"] = customer
    result = client.decide(state, QUESTIONS)
    a = result["answers"]

    dept = a["department"]["choice"]
    dept_p = a["department"]["probabilities"][dept]
    urgency = a["urgency"]["score"]
    flags = [q for q in ("refund", "churn_risk", "wants_human") if a[q]["noul"] >= FLAG_MIN_P]

    reasons = [f"部门={dept} p={dept_p:.2f}", f"紧急度={urgency:.2f}/2"]
    reasons += [f"{q}={a[q]['noul']:.2f}" for q in flags]

    if urgency >= P1_URGENCY or "churn_risk" in flags:
        priority = "P1"
    elif urgency >= P2_URGENCY:
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
        queue,
        priority,
        auto,
        dept,
        dept_p,
        urgency,
        flags,
        reasons,
        result["timing_ms"]["total"],
        result,
    )


# =============================================================================
# 演示
# =============================================================================

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="laya 智能客服路由演示")
    parser.add_argument("messages", nargs="*", help="要路由的消息；不给则跑内置样例")
    parser.add_argument("--url", help=f"服务地址（默认 $LAYA_URL 或 {DEFAULT_URL}）")
    parser.add_argument("--raw", action="store_true", help="同时打印模型的原始回答")
    args = parser.parse_args(argv)

    client = LayaClient(args.url)
    if not client.ready():
        print(f"服务不可用：{client.base_url}", file=sys.stderr)
        return 1
    print(f"服务：{client.base_url}\n")
    for message in args.messages or SAMPLES:
        try:
            r = route(client, message)
        except LayaError as exc:
            print(f"[错误] {exc} | {message}")
            continue
        mode = "自动派单" if r.auto else "转人工"
        print(f"[{r.priority}] {r.queue} · {mode} | {message}")
        print(f"      {'；'.join(r.reasons)}  ({r.latency_ms:.0f} ms)")
        if args.raw:
            print(json.dumps(r.raw["answers"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
