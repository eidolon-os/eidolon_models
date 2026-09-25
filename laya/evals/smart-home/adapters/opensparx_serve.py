#!/usr/bin/env python3
"""把 OpenSparX/OAK-Decision-cabin-qwen3.5-0.8b 包成 Jev 协议的 `/v1/systemone`，供 run_eval.py 对比用。

作者只给了 `inference.py`（读一个 record 文件、打印 softmax），没有服务。这里原样调用作者仓库里的
`src/`（pack + ParallelDecisionModel），只做协议翻译，不改模型：

  state     `{"utterance": "..."}` → `用户语句：...`（作者训练时 state 的最后一行就是这个写法）
  choice    选项 = `名字: 描述`（有描述时），问题 = instructions；答案映射回名字
  noul      选项 = ["是", "否"]，返回 P(是)

分支预算：作者默认每题 ≤256 token（问题 + 全部选项），38 台设备加描述会超，这里放到 2048；
位置编码每个分支从 state 末尾重新计起，所以放大预算只是超出训练分布，不会越界。

    .venv/bin/python opensparx_serve.py --repo <adapter 目录> --base <Qwen3.5-0.8B 目录> --device mps --port 8772

依赖钉作者的 requirements.txt（transformers==5.8.1、peft==0.19.1）；aiohttp 是这个文件自己的。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from aiohttp import web


def render_state(state) -> str:
    if isinstance(state, dict) and set(state) == {"utterance"}:
        return f"用户语句：{state['utterance']}"
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_questions(questions: dict) -> tuple[list[dict], list[tuple[str, str, list]]]:
    """Jev 题目 → 作者 record 的 questions；同时记住每题的类型和选项名，用来还原答案。"""
    records, index = [], []
    for qid, q in questions.items():
        t = q.get("type", "choice")
        text = q.get("instructions") or "哪个答案符合上下文？"
        if t == "choice":
            crit = q["criteria"]
            names = list(crit) if isinstance(crit, dict) else [str(c) for c in crit]
            opts = [f"{n}: {crit[n]}" if isinstance(crit, dict) and crit[n] else n for n in names]
        elif t == "noul":
            names, opts = [True, False], ["是", "否"]
        elif t == "score":
            crit = q["criteria"]
            levels = [crit[k] for k in sorted(crit, key=float)] if isinstance(crit, dict) else crit
            names = list(range(len(levels)))
            opts = [f"{i}: {c}" for i, c in enumerate(levels)]
        else:
            raise web.HTTPUnprocessableEntity(text=f"unknown question type {t!r}")
        records.append({"id": qid, "type": t, "question": text, "options": opts})
        index.append((qid, t, names))
    return records, index


def certainty(p: list[float]) -> float:
    import math

    h = -sum(x * math.log(x) for x in p if x > 0)
    return max(0.0, 1.0 - h / math.log(len(p))) if len(p) > 1 else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo", required=True, help="adapter 仓库快照目录（含 src/、decision_config.json）"
    )
    ap.add_argument("--base", required=True, help="Qwen/Qwen3.5-0.8B 快照目录")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--port", type=int, default=8772)
    ap.add_argument("--max-state-tokens", type=int, default=512)
    ap.add_argument("--max-branch-tokens", type=int, default=2048)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.repo)))
    from src.model import ParallelDecisionModel  # noqa: E402  作者的代码
    from src.packing import pack  # noqa: E402
    from src.tokenizer import load_tokenizer  # noqa: E402

    tokenizer, _ = load_tokenizer(args.repo)
    model = ParallelDecisionModel.load(
        args.repo, dtype=torch.float32, device=args.device, base_model=args.base
    ).eval()
    name = "opensparx-cabin-0.8b"

    def decide(state, questions: dict) -> dict:
        t0 = time.perf_counter()
        records, index = render_questions(questions)
        packed = pack(
            {"state": render_state(state), "questions": records},
            tokenizer,
            max_state_tokens=args.max_state_tokens,
            max_question_branch_tokens=args.max_branch_tokens,
        ).to(args.device)
        t1 = time.perf_counter()
        with torch.inference_mode():
            scores = model(packed).float().softmax(-1).cpu()
        t2 = time.perf_counter()
        answers = {}
        for (qid, t, names), row in zip(index, scores, strict=True):
            p = row[: len(names)].tolist()
            s = sum(p) or 1.0
            p = [x / s for x in p]
            j = max(range(len(p)), key=p.__getitem__)
            if t == "noul":
                answers[qid] = {"type": "noul", "noul": round(p[0], 4)}
            elif t == "choice":
                answers[qid] = {
                    "type": "choice",
                    "choice": names[j],
                    "confidence": round(certainty(p), 4),
                    "probabilities": {n: round(x, 4) for n, x in zip(names, p, strict=True)},
                }
            else:
                answers[qid] = {
                    "type": "score",
                    "score": round(sum(i * x for i, x in enumerate(p)), 2),
                    "confidence": round(certainty(p), 4),
                    "probabilities": {str(n): round(x, 4) for n, x in zip(names, p, strict=True)},
                }
        return {
            "model": name,
            "answers": answers,
            "usage": {"input_tokens": int(packed.input_ids.shape[1]), "output_tokens": 0},
            "timing_ms": {
                "total": round((t2 - t0) * 1000, 1),
                "forward": round((t2 - t1) * 1000, 1),
                "branch_tokens": list(packed.branch_lengths),
            },
        }

    async def systemone(req: web.Request) -> web.Response:
        body = await req.json()
        try:
            out = decide(body["state"], body["questions"])
        except ValueError as e:  # pack() 拒绝超预算或选项重复
            raise web.HTTPUnprocessableEntity(text=str(e)) from e
        return web.json_response(out, dumps=lambda o: json.dumps(o, ensure_ascii=False))

    async def info(_):
        return web.json_response(
            {
                "engine": {
                    "model": name,
                    "backend": "torch",
                    "device": args.device,
                    "precision": "float32",
                    "max_state_tokens": args.max_state_tokens,
                    "max_branch_tokens": args.max_branch_tokens,
                }
            }
        )

    app = web.Application()
    app.add_routes(
        [
            web.post("/v1/systemone", systemone),
            web.get("/v1/info", info),
            web.get("/healthz", lambda _: web.Response(text="ok")),
            web.get("/readyz", lambda _: web.Response(text="ok")),
        ]
    )
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
