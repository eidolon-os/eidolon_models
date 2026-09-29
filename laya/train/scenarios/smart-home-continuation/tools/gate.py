"""PLAN.md §5 门槛的机械判定：读 post_train.sh 的输出，逐条打印 PASS / FAIL，全部通过时退出码 0。

    .venv/bin/python train/scenarios/smart-home-continuation/tools/gate.py train/runs/c2

门槛 1（续接，c-dev）：错误执行 0、错误取消 ≤ 1%、接管率 ≥ 50%、换序一致 ≥ 95%。
门槛 2（单句，三套开发集，对 train/runs/r14-reeval）：配对门槛通过；0.9 自动执行覆盖降 ≤ 3 个百分点、精度（已验证）降 ≤ 1 个百分点。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    r = Path(sys.argv[1])
    m = json.loads((r / "eval-c/c-dev.metrics.json").read_text("utf-8"))
    sw = json.loads((r / "eval-c/c-dev.swap.json").read_text("utf-8"))
    ss = json.loads((r / "eval/single-sentence.json").read_text("utf-8"))
    o = m["overall"]
    checks = [
        ("续接 错误执行 = 0", o["counts"]["错误执行"] == 0, o["counts"]["错误执行"]),
        ("续接 错误取消 ≤ 1%", o["wrong_cancel_rate"] <= 0.01, o["wrong_cancel_rate"]),
        ("续接 接管率 ≥ 50%", (o["takeover"] or 0) >= 0.5, o["takeover"]),
        ("续接 换序一致 ≥ 95%", sw["consistent"] >= 0.95, sw["consistent"]),
    ]
    for s, v in ss.items():
        cov_b, cov_c = v["auto_exec_coverage"]
        pre_b, pre_c = v["auto_exec_precision_verified"]
        checks += [
            (f"单句 {s} 配对门槛", v["paired_gate_passed"], v["acc"]),
            (f"单句 {s} 覆盖降 ≤ 3pt", cov_c >= cov_b - 0.03, [cov_b, cov_c]),
            (f"单句 {s} 精度降 ≤ 1pt", pre_b is None or (pre_c is not None and pre_c >= pre_b - 0.01), [pre_b, pre_c]),
        ]
    ok = all(c[1] for c in checks)
    for name, passed, value in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}  {value}")
    print(f"GATE {'PASS' if ok else 'FAIL'}  tau_exec={o['tau_exec']} tau_cancel={o['tau_cancel']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
