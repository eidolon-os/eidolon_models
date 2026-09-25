#!/usr/bin/env python3
"""不同 checkpoint 在同一套智能家居用例上的对比 → COMPARISON.md。只依赖标准库。

读 results/ 下这些运行（存在哪个用哪个）：
  mac-torch-mps                         laya-multilingual（官方，基线）
  laya-multilingual-h1024@mac-torch-mps 同一权重，头部预算放到 1024（隔离“选项预算”的影响）
  laya-cn-a@mac-torch-mps               Adkid/laya-cn-a（中文意图后训练）
  macjev-322m-4k@mac-torch-mps          chaoliangUNSW/MacJev-322M-4K-Laya（Mac agent 决策）
  laya-zh-v2@mac-torch-mps              zcgnull/laya-zh-v2（魔搭；中文分诊 + 客服 RLCD 微调）
  decider-0.8b@mac-mps                  Mapika/decider-0.8b（Qwen3.5-0.8B 小 LLM 路线，作者的 Jev 兼容服务）
  opensparx-cabin-0.8b@mac-mps          OpenSparX/OAK-Decision-cabin-qwen3.5-0.8b（车控意图）
人工结论写在 comparison_conclusions.md，生成时放在最前面。
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = [
    ("laya-multilingual", "mac-torch-mps"),
    ("laya-multilingual·head1024", "laya-multilingual-h1024@mac-torch-mps"),
    ("laya-cn-a", "laya-cn-a@mac-torch-mps"),
    ("MacJev-322M-4K", "macjev-322m-4k@mac-torch-mps"),
    ("laya-zh-v2", "laya-zh-v2@mac-torch-mps"),
    ("decider-0.8B", "decider-0.8b@mac-mps"),
    ("OpenSparX-cabin-0.8B", "opensparx-cabin-0.8b@mac-mps"),
]
DIMS = ("intent", "device", "action")


def load(label):
    d = HERE / "results" / label
    rows = [json.loads(x) for x in (d / "predictions.jsonl").read_text("utf-8").splitlines()]
    return json.loads((d / "summary.json").read_text("utf-8")), {r["id"]: r for r in rows}


def pct(v):
    return "—" if v is None else f"{v:.0%}"


def e2e_ok(row, pred):
    """End-to-end correctness of a control case for an arbitrary (intent, device, action) triple."""
    g = row["gold"]
    ok = pred["intent"] == g["intent"]
    for dim in ("device", "action"):
        if g.get(dim) is not None:
            allowed = g[dim] if isinstance(g[dim], list) else [g[dim]]
            ok = ok and pred[dim] in allowed
    return ok


def main() -> int:
    runs = [(name, *load(label)) for name, label in MODELS if (HERE / "results" / label).is_dir()]
    base_name, _, base = runs[0]
    ids = list(base)
    scenarios = sorted({r["scenario"] for r in base.values()})
    L = []
    w = L.append
    w("# 智能家居评测：不同 checkpoint 对比\n")
    w(
        f"同一套 182 条用例、同样的三道题与问法、同一台 Mac（torch + MPS）。模型：{'、'.join(n for n, *_ in runs)}。\n"
    )
    c = HERE / "comparison_conclusions.md"
    if c.is_file():
        w(c.read_text("utf-8").strip() + "\n")

    w("## 总体\n")
    w("| 指标 | " + " | ".join(n for n, *_ in runs) + " |")
    w("|---|" + "---|" * len(runs))
    rows_spec = [
        ("意图（三分类）", lambda s: pct(s["laya"]["intent_ok"]["acc"])),
        (
            "是否控制：精确率 / 召回率",
            lambda s: (
                f"{s['laya']['control_detection']['precision']:.0%} / {s['laya']['control_detection']['recall']:.0%}"
            ),
        ),
        ("设备", lambda s: pct(s["laya"]["device_ok"]["acc"])),
        ("动作", lambda s: pct(s["laya"]["action_ok"]["acc"])),
        ("**控制端到端**", lambda s: f"**{pct(s['laya']['e2e_ok']['acc'])}**"),
        (
            "设备 p≥0.9：覆盖 / 准确",
            lambda s: (
                f"{s['calibration']['device']['0.9']['coverage']:.0%} / {s['calibration']['device']['0.9']['acc']:.0%}"
            ),
        ),
        (
            "意图 p≥0.9：覆盖 / 准确",
            lambda s: (
                f"{s['calibration']['intent']['0.9']['coverage']:.0%} / {s['calibration']['intent']['0.9']['acc']:.0%}"
            ),
        ),
        (
            "延迟 p50 / p95",
            lambda s: (
                f"{s['latency_ms']['client_p50']:.0f} / {s['latency_ms']['client_p95']:.0f} ms"
            ),
        ),
        (
            "头部预算 / 最大长度",
            lambda s: f"{s['server'].get('head_max_len', '—')} / {s['server'].get('max_len', '—')}",
        ),
    ]
    for label, fn in rows_spec:
        w(f"| {label} | " + " | ".join(fn(s) for _, s, _ in runs) + " |")
    w(
        f"\n规则基线（同一套用例）：意图 {pct(runs[0][1]['baseline']['intent_ok']['acc'])}、设备 "
        f"{pct(runs[0][1]['baseline']['device_ok']['acc'])}、控制端到端 {pct(runs[0][1]['baseline']['e2e_ok']['acc'])}。\n"
    )

    for title, key in (
        ("分场景：意图", "intent_ok"),
        ("分场景：设备", "device_ok"),
        ("分场景：控制端到端", "e2e_ok"),
    ):
        w(f"## {title}\n")
        w("| 场景 | " + " | ".join(n for n, *_ in runs) + " | 规则基线 |")
        w("|---|" + "---|" * (len(runs) + 1))
        for scn in scenarios:
            cells = [pct(s["by_scenario"][scn]["laya"][key]["acc"]) for _, s, _ in runs]
            if all(x == "—" for x in cells):
                continue
            w(
                f"| `{scn}` | "
                + " | ".join(cells)
                + f" | {pct(runs[0][1]['by_scenario'][scn]['baseline'][key]['acc'])} |"
            )
        w("")

    w(f"## 相对 {base_name}：修好了多少、弄坏了多少\n")
    w("| 模型 | 意图 修好 / 弄坏 | 设备 修好 / 弄坏 | 动作 修好 / 弄坏 |")
    w("|---|---|---|---|")
    for name, _, rows in runs[1:]:
        cells = []
        for dim in DIMS:
            k = f"{dim}_ok"
            fixed = sum(
                1
                for i in ids
                if k in base[i]["score"] and not base[i]["score"][k] and rows[i]["score"][k]
            )
            broke = sum(
                1
                for i in ids
                if k in base[i]["score"] and base[i]["score"][k] and not rows[i]["score"][k]
            )
            cells.append(f"+{fixed} / −{broke}")
        w(f"| {name} | " + " | ".join(cells) + " |")
    w("")

    w("## 互补性：三道题各取一个模型的答案组合\n")
    control_ids = [i for i in ids if base[i]["gold"]["intent"] == "控制"]
    combos = []
    for pick in itertools.product(range(len(runs)), repeat=3):
        acc = sum(
            e2e_ok(
                base[i],
                {dim: runs[m][2][i]["pred"][dim] for dim, m in zip(DIMS, pick, strict=True)},
            )
            for i in control_ids
        ) / len(control_ids)
        combos.append((acc, pick))
    combos.sort(reverse=True)
    w("| 意图来自 | 设备来自 | 动作来自 | 控制端到端 |")
    w("|---|---|---|---|")
    seen = set()
    for acc, pick in combos:
        if len(seen) >= 5:
            break
        if pick in seen:
            continue
        seen.add(pick)
        w(f"| {runs[pick[0]][0]} | {runs[pick[1]][0]} | {runs[pick[2]][0]} | {acc:.0%} |")
    single = {runs[m][0]: acc for acc, (a, b, c2) in combos for m in [a] if a == b == c2}
    w(
        "\n（同一模型三道题都用自己的答案："
        + "，".join(f"{k} {v:.0%}" for k, v in single.items())
        + "）\n"
    )
    oracle = {
        dim: sum(
            any(r[i]["score"].get(f"{dim}_ok") for _, _, r in runs)
            for i in ids
            if f"{dim}_ok" in base[i]["score"]
        )
        / sum(1 for i in ids if f"{dim}_ok" in base[i]["score"])
        for dim in DIMS
    }
    w(
        "逐条取“任一模型答对就算对”的上限："
        + "，".join(f"{d} {v:.0%}" for d, v in oracle.items())
        + "。\n"
    )

    w("## 所有模型都答错的用例（新数据必须覆盖的“硬骨头”）\n")
    hard = defaultdict(list)
    for i in ids:
        for dim in ("intent", "device"):
            k = f"{dim}_ok"
            if k in base[i]["score"] and not any(r[i]["score"][k] for _, _, r in runs):
                hard[base[i]["scenario"]].append((dim, base[i]["text"]))
    w("| 场景 | 意图全错 | 设备全错 | 例子 |")
    w("|---|---|---|---|")
    for scn in scenarios:
        items = hard.get(scn, [])
        n_i = sum(d == "intent" for d, _ in items)
        n_d = sum(d == "device" for d, _ in items)
        eg = "；".join(dict.fromkeys(t for _, t in items[:3]))
        w(f"| `{scn}` | {n_i} | {n_d} | {eg} |")
    w("\n## 复现\n")
    w(
        "```bash\ncd laya\nEIDOLON_LAYA_MODEL_DIR=models/laya-cn-a/178eb2c0 scripts/eidolon-laya fetch\n"
        "EIDOLON_LAYA_MODEL_DIR=models/macjev-322m-4k/92b182e6 scripts/eidolon-laya fetch\n"
        "HEAD_MAX_LEN=1024 LABEL_PREFIX=laya-multilingual-h1024@ evals/smart-home/run_all.sh\n"
        "MODEL_DIR=models/laya-cn-a/178eb2c0 LABEL_PREFIX=laya-cn-a@ evals/smart-home/run_all.sh\n"
        "MODEL_DIR=models/macjev-322m-4k/92b182e6 LABEL_PREFIX=macjev-322m-4k@ evals/smart-home/run_all.sh\n"
        "EIDOLON_LAYA_MODEL_DIR=models/laya-zh-v2/92ae01f5 scripts/eidolon-laya fetch   # 魔搭\n"
        "MODEL_DIR=models/laya-zh-v2/92ae01f5 LABEL_PREFIX=laya-zh-v2@ evals/smart-home/run_all.sh\n```\n\n"
        "两个 Qwen 小模型不是 laya，各自在独立的 venv 里起 Jev 协议的服务，再用 run_eval.py 打它：\n\n"
        "```bash\n# decider：作者自带的服务（pip install 'decider-ai[serve]'；numpy<2，所以要单独 venv）\n"
        "DECIDER_MODEL=<Mapika/decider-0.8b 快照目录> DECIDER_DEVICE=mps uvicorn decider.serve:app --port 8773\n"
        "python3 evals/smart-home/run_eval.py --url http://127.0.0.1:8773 --label decider-0.8b@mac-mps\n"
        "# OpenSparX：只有 inference.py，用 adapters/opensparx_serve.py 包成服务（钉作者的 transformers==5.8.1、peft==0.19.1）\n"
        "python evals/smart-home/adapters/opensparx_serve.py --repo <adapter 快照> --base <Qwen3.5-0.8B 快照> --device mps --port 8772\n"
        "python3 evals/smart-home/run_eval.py --url http://127.0.0.1:8772 --label opensparx-cabin-0.8b@mac-mps\n```\n"
    )
    (HERE / "COMPARISON.md").write_text("\n".join(L), "utf-8")
    print(f"wrote {HERE / 'COMPARISON.md'} ({len(runs)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
