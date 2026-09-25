#!/usr/bin/env python3
"""把 results/*/summary.json 与 predictions.jsonl 汇总成 REPORT.md。只依赖标准库。

结论写在 conclusions.md（人工撰写），生成时原样放在报告最前面。
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIMARY = "mac-torch-mps"  # 准确率以这次运行为准（各后端答案一致，见“后端一致性”）
SCENARIO_NAMES = {
    "01-explicit-control": "明确指令（15 类设备逐一覆盖）",
    "02-implicit-intent": "隐含意图（“好热啊”）",
    "03-room-disambiguation": "同类设备分房间",
    "04-status-query": "状态查询",
    "05-non-command": "非命令（提到设备但不是指令）",
    "06-multi-device-scene": "多设备 / 场景",
    "07-asr-noise": "语音识别错字",
    "08-large-inventory": "大户型 38 个选项",
    "09-device-not-in-home": "家里没有该设备",
}


def acc(m: dict, key: str) -> str:
    v = m[key]["acc"]
    return "—" if v is None else f"{v:.0%}"


def load(label: str) -> tuple[dict, list[dict]]:
    d = HERE / "results" / label
    rows = [json.loads(line) for line in (d / "predictions.jsonl").read_text("utf-8").splitlines()]
    return json.loads((d / "summary.json").read_text("utf-8")), rows


def main() -> int:
    # Runs of other checkpoints are named <model>@<run>; they belong in COMPARISON.md.
    labels = sorted(
        p.name
        for p in (HERE / "results").iterdir()
        if (p / "summary.json").is_file() and "@" not in p.name
    )
    runs = {lab: load(lab) for lab in labels}
    s, rows = runs[PRIMARY]
    cpu = (
        subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        or s["machine"]
    )
    L = []
    w = L.append
    w("# laya 智能家居控制决策评测报告\n")
    w(
        f"生成：{dt.date.today()}｜模型：laya-multilingual（zero-shot，未微调）"
        f"@{json.loads((HERE.parent.parent / 'models/laya-multilingual/1c5edc17/manifest.json').read_text())['source']['revision'][:8]}"
        f"｜机器：{cpu}｜用例：{s['n_cases']} 条，9 个场景\n"
    )
    conclusions = HERE / "conclusions.md"
    if conclusions.is_file():
        w(conclusions.read_text("utf-8").strip() + "\n")

    w("## 测试设置\n")
    w(
        "- 每条用例一次请求、三道题：**意图**（控制 / 查询 / 无关）、**设备**（该户设备清单 + "
        "“多个设备或整屋” + “没有对应的设备”）、**动作**（打开 / 关闭 / 调高 / 调低 / 设定 / 暂停 / 上锁）。"
    )
    w(
        "- 户型：`apartment`、`house` 各 18 台设备（加两个特殊项共 20 个选项），合起来覆盖 15 类；"
        "`villa` 为两者合并（38 个选项）。见 `homes/`、`taxonomy.md`。"
    )
    w(
        "- 金标允许多个可接受答案（如“好热啊”开哪台空调都算对）。设备只对查询/控制用例计分，"
        "动作只对给了动作金标的用例计分；**端到端** = 控制用例的意图、设备、动作全部正确。"
    )
    w(
        "- 对照组：通用关键词规则基线（设备类型词表 + 房间词 + 动词/疑问词，`run_eval.py` 中 `rule_baseline`）。\n"
    )

    w("## 总体准确率\n")
    b = s["baseline"]
    m = s["laya"]
    w("| 指标 | laya | 规则基线 |")
    w("|---|---|---|")
    w(f"| 意图（三分类） | {acc(m, 'intent_ok')} | {acc(b, 'intent_ok')} |")
    cd, cb = m["control_detection"], b["control_detection"]
    w(
        f"| 是否控制命令：精确率 / 召回率 / F1 | {cd['precision']:.0%} / {cd['recall']:.0%} / {cd['f1']:.2f}"
        f" | {cb['precision']:.0%} / {cb['recall']:.0%} / {cb['f1']:.2f} |"
    )
    w(f"| 设备（n={m['device_ok']['n']}） | {acc(m, 'device_ok')} | {acc(b, 'device_ok')} |")
    w(f"| 动作（n={m['action_ok']['n']}） | {acc(m, 'action_ok')} | {acc(b, 'action_ok')} |")
    w(f"| 控制命令端到端（n={m['e2e_ok']['n']}） | {acc(m, 'e2e_ok')} | {acc(b, 'e2e_ok')} |\n")

    w("## 分场景\n")
    w("| 场景 | 用例 | 意图 | 设备 | 动作 | 端到端 | 基线意图 | 基线设备 | 基线端到端 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for k, v in s["by_scenario"].items():
        lm, bm = v["laya"], v["baseline"]
        w(
            f"| `{k}` {SCENARIO_NAMES.get(k, '')} | {lm['intent_ok']['n']} | {acc(lm, 'intent_ok')} | "
            f"{acc(lm, 'device_ok')} | {acc(lm, 'action_ok')} | {acc(lm, 'e2e_ok')} | "
            f"{acc(bm, 'intent_ok')} | {acc(bm, 'device_ok')} | {acc(bm, 'e2e_ok')} |"
        )
    w("")

    w("## 分设备品类\n")
    w("| 品类 | 用例 | 意图 | 设备 | 端到端 | 基线设备 | 基线端到端 |")
    w("|---|---|---|---|---|---|---|")
    for k, v in sorted(s["by_category"].items(), key=lambda kv: -kv[1]["laya"]["intent_ok"]["n"]):
        lm, bm = v["laya"], v["baseline"]
        w(
            f"| {k} | {lm['intent_ok']['n']} | {acc(lm, 'intent_ok')} | {acc(lm, 'device_ok')} | "
            f"{acc(lm, 'e2e_ok')} | {acc(bm, 'device_ok')} | {acc(bm, 'e2e_ok')} |"
        )
    w("")

    w("## 意图混淆矩阵（行 = 金标，列 = 预测）\n")
    labs = ["控制", "查询", "无关"]
    w("| 金标 \\ 预测 | " + " | ".join(labs) + " |")
    w("|---|---|---|---|")
    for g in labs:
        w(
            f"| {g} | "
            + " | ".join(str(s["intent_confusion"].get(f"{g}->{p}", 0)) for p in labs)
            + " |"
        )
    w("")

    w("## 置信度门控（只采纳最高概率 ≥ 阈值的判断）\n")
    w("| 阈值 | 意图覆盖率 | 意图准确率 | 设备覆盖率 | 设备准确率 |")
    w("|---|---|---|---|---|")
    for th in ("0.0", "0.5", "0.7", "0.9"):
        ci, cdv = s["calibration"]["intent"][th], s["calibration"]["device"][th]
        w(
            f"| ≥ {th} | {ci['coverage']:.0%} | {ci['acc']:.0%} | {cdv['coverage']:.0%} | {cdv['acc']:.0%} |"
        )
    w("\n注意：multilingual checkpoint 出厂未做温度校准，这里的概率是未校准的。\n")

    w("## 性能（Mac 本机，每条用例 = 3 道题 = 3 条序列，一次请求）\n")
    w("| 运行 | 后端 / 设备 | 线程 | 客户端 p50 | p95 | 平均 | 服务端前向 p50 | 服务常驻内存 |")
    w("|---|---|---|---|---|---|---|---|")
    for lab, (sm, _) in runs.items():
        eng, lt = sm["server"], sm["latency_ms"]
        w(
            f"| `{lab}` | {eng['backend']} / {eng['device']} | {eng['threads']} | {lt['client_p50']} ms | "
            f"{lt['client_p95']} ms | {lt['client_mean']} ms | {lt['forward_p50']} ms | "
            f"{sm['server_rss_mb']} MB |"
        )
    tok = s["latency_ms"]["by_device_options"]
    w(
        f"\n每条请求平均输入 {round(sum(v['tokens_mean'] * v['n'] for v in tok.values()) / s['n_cases'])} "
        "token（三条序列合计）。精度均为 FP32。\n"
    )

    w("## 设备清单变长（20 个选项 → 38 个选项）\n")
    by_id = {r["id"]: r for r in rows}
    src_ids = {}
    for line in (HERE / "scenarios/08-large-inventory/cases.jsonl").read_text("utf-8").splitlines():
        c = json.loads(line)
        src_ids[c["id"]] = c["same_as"]
    small = [by_id[v] for v in src_ids.values()]
    large = [by_id[k] for k in src_ids]
    f = lambda rs, k: f"{sum(r['score'][k] for r in rs) / len(rs):.0%}"  # noqa: E731
    w("同一批 20 条语句，分别放在 20 个选项（原户型）和 38 个选项（villa）下：\n")
    w("| 选项数 | 意图 | 设备 | 端到端 | 客户端 p50（" + PRIMARY + "） |")
    w("|---|---|---|---|---|")
    for name, rs, key in (("20", small, "20"), ("38", large, "38")):
        w(
            f"| {name} | {f(rs, 'intent_ok')} | {f(rs, 'device_ok')} | {f(rs, 'e2e_ok')} | "
            f"{tok[key]['client_p50']} ms |"
        )
    w(
        "\n头部预算固定 256 token，选项越多每个选项分到的 token 越少（38 项时约 6 个），所以序列长度几乎不变、"
        "延迟只略增；准确率风险来自选项描述被截短。\n"
    )

    w("## 后端一致性\n")
    base_pred = {r["id"]: r["pred"] for r in rows}
    for lab, (_, rs) in runs.items():
        if lab == PRIMARY:
            continue
        same = sum(r["pred"] == base_pred[r["id"]] for r in rs)
        w(f"- `{lab}` 与 `{PRIMARY}` 三道题的选择完全相同：{same}/{len(rs)}")
    w("")

    w("## 典型错误（" + PRIMARY + "）\n")
    w("| 场景 | 语句 | 金标 | laya 预测（最高概率） |")
    w("|---|---|---|---|")
    for scn in s["by_scenario"]:
        errs = [r for r in rows if r["scenario"] == scn and not all(r["score"].values())]
        for r in errs[:3]:
            g, p, t = r["gold"], r["pred"], r["p_top"]
            gd = g.get("device")
            gd = "/".join(gd) if isinstance(gd, list) else gd
            w(
                f"| `{scn[:2]}` | {r['text']} | {g['intent']} · {gd or '—'} | "
                f"{p['intent']}({t['intent']:.2f}) · {p['device']}({t['device']:.2f}) · {p['action']} |"
            )
    w("")
    w("## 复现\n")
    w(
        "```bash\ncd laya && uv sync --all-extras && scripts/eidolon-laya fetch && scripts/eidolon-laya export-onnx\n"
        "evals/smart-home/run_all.sh          # 依次起服务、跑评测、停服务，最后生成本报告\n```\n"
    )
    (HERE / "REPORT.md").write_text("\n".join(L), "utf-8")
    print(f"wrote {HERE / 'REPORT.md'} from {len(runs)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
