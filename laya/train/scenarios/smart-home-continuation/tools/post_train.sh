#!/usr/bin/env bash
# 训练后的固定流程（PLAN.md §4–5）：校准 → 分任务温度 → c-dev / diag-44 → 单句三套开发集回归 → 指标 → 换序。
# 不跑 c-test。用法（在 laya/ 下）：train/scenarios/smart-home-continuation/tools/post_train.sh train/runs/c1 [device]
set -euo pipefail
R=${1:?run dir}
DEV=${2:-mps}
T=train/scenarios/smart-home-continuation/tools
E=evals/smart-home-continuation
S=train/scenarios/smart-home/eval
export HF_HUB_OFFLINE=1
BIN=.venv/bin/eidolon-laya-train

$BIN calibrate --checkpoint "$R/checkpoint" --calib "$R/dataset/calib.jsonl" --device "$DEV" > "$R/calibrate.log" 2>&1
.venv/bin/python $T/calib_split.py --checkpoint "$R/checkpoint" --calib "$R/dataset/calib.jsonl" --device "$DEV" \
  > "$R/calib-split.json" 2> "$R/calib-split.log"
$BIN eval --checkpoint "$R/checkpoint" --eval-set $E/c-dev.jsonl --eval-set $E/diag-44.jsonl \
  --out "$R/eval-c" --device "$DEV" > "$R/eval-c.log" 2>&1
$BIN eval --checkpoint "$R/checkpoint" --eval-set $S/locked-182.jsonl --eval-set $S/locked-v2-dev.jsonl \
  --eval-set $S/locked-v3-dev.jsonl --out "$R/eval" --baseline train/runs/r14/eval --alpha 0.05 \
  --device "$DEV" > "$R/eval.log" 2>&1 || true   # 门槛不过时 eval 以非零退出，报告照写
.venv/bin/python $T/cmetrics.py --report "$R/eval-c/c-dev.json" --out "$R/eval-c/c-dev.metrics.json"
.venv/bin/python $T/cmetrics.py --report "$R/eval-c/diag-44.json" --out "$R/eval-c/diag-44.metrics.json"
.venv/bin/python $T/swap.py --checkpoint "$R/checkpoint" --eval-set $E/c-dev.jsonl --device "$DEV" \
  > "$R/eval-c/c-dev.swap.json" 2> "$R/eval-c/swap.log"
.venv/bin/python - "$R" <<'EOF'
import json, sys
R = sys.argv[1]
out = {}
for s in ("locked-182", "locked-v2-dev", "locked-v3-dev"):
    c = json.load(open(f"{R}/eval/{s}.json")); b = json.load(open(f"train/runs/r14/eval/{s}.json"))
    g = json.load(open(f"{R}/eval/{s}.gate.json"))
    ca, ba = c["decision"]["auto_execute"], b["decision"]["auto_execute"]
    out[s] = {"acc": [b["overall"]["acc"], c["overall"]["acc"]],
              "auto_exec_coverage": [ba["coverage"], ca["coverage"]],
              "auto_exec_precision": [ba["precision"], ca["precision"]],
              "control_e2e": [b["decision"]["control_e2e"]["acc"], c["decision"]["control_e2e"]["acc"]],
              "paired_gate_passed": g["passed"]}
json.dump(out, open(f"{R}/eval/single-sentence.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False))
EOF
echo "post_train done: $R"
