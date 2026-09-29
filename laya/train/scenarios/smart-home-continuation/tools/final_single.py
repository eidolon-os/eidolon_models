"""最后一道单句回归（PLAN.md §7 c3 登记）：v2-test 与 accept，对 train/runs/r14-reeval，标准同开发集门槛 2。
全过退出码 0。用法：final_single.py train/runs/c3
"""
import json
import sys
from pathlib import Path

r = Path(sys.argv[1])
ok = True
for s in ("locked-v2-test", "locked-accept"):
    c = json.loads((r / f"eval-final/{s}.json").read_text("utf-8"))
    b = json.loads(Path(f"train/runs/r14-reeval/{s}.json").read_text("utf-8"))
    g = json.loads((r / f"eval-final/{s}.gate.json").read_text("utf-8"))
    ca, ba = c["decision"]["auto_execute"], b["decision"]["auto_execute"]
    checks = [("配对门槛", g["passed"], [b["overall"]["acc"], c["overall"]["acc"]]),
              ("覆盖降 ≤ 3pt", ca["coverage"] >= ba["coverage"] - 0.03, [ba["coverage"], ca["coverage"]]),
              ("精度降 ≤ 1pt", ca["precision_verified"] >= ba["precision_verified"] - 0.01,
               [ba["precision_verified"], ca["precision_verified"]])]
    for name, passed, v in checks:
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  最终单句 {s} {name}  {v}")
    for chk in g["checks"]:
        if not chk["ok"]:
            print(f"      不过的切片：{chk['slice']} {chk['baseline']} → {chk['candidate']} 改错 {chk['broke']} / 改对 {chk['fixed']} p {chk['p']}")
print(f"FINAL-SINGLE {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
