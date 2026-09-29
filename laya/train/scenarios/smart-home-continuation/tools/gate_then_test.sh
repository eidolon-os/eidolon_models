#!/usr/bin/env bash
# 训练结束 → post_train.sh → gate.py → 最后单句回归（v2-test / accept）→ 全过才跑一次 c-test（阈值用 c-dev 上选出并冻结的值）。
# 用法（在 laya/ 下）：train/scenarios/smart-home-continuation/tools/gate_then_test.sh train/runs/c2 [device]
set -uo pipefail
R=${1:?run dir}
DEV=${2:-mps}
T=train/scenarios/smart-home-continuation/tools
export HF_HUB_OFFLINE=1

while pgrep -f "eidolon-laya-train train" >/dev/null; do sleep 30; done
[ -f "$R/checkpoint/model.safetensors" ] || { echo "NO CHECKPOINT in $R"; exit 2; }
$T/post_train.sh "$R" "$DEV" || { echo "post_train failed"; exit 3; }
.venv/bin/python $T/gate.py "$R" | tee "$R/gate.txt"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "GATE FAIL: c-test not run"
  exit 1
fi
# 最后一道单句回归：v2-test 与 accept（PLAN.md §7 c3 登记），不过就不跑 c-test
.venv/bin/eidolon-laya-train eval --checkpoint "$R/checkpoint" --eval-set train/scenarios/smart-home/eval/locked-v2-test.jsonl \
  --eval-set train/scenarios/smart-home/eval/locked-accept.jsonl --out "$R/eval-final" --baseline train/runs/r14-reeval \
  --alpha 0.05 --device "$DEV" > "$R/eval-final.log" 2>&1
.venv/bin/python $T/final_single.py "$R" | tee "$R/final-single.txt"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "FINAL-SINGLE FAIL: c-test not run"
  exit 1
fi
TE=$(.venv/bin/python -c "import json;print(json.load(open('$R/eval-c/c-dev.metrics.json'))['overall']['tau_exec'])")
TC=$(.venv/bin/python -c "import json;print(json.load(open('$R/eval-c/c-dev.metrics.json'))['overall']['tau_cancel'])")
.venv/bin/eidolon-laya-train eval --checkpoint "$R/checkpoint" --eval-set evals/smart-home-continuation/c-test.jsonl \
  --out "$R/eval-test" --device "$DEV" > "$R/eval-test.log" 2>&1
.venv/bin/python $T/cmetrics.py --report "$R/eval-test/c-test.json" --tau-exec "$TE" --tau-cancel "$TC" \
  --out "$R/eval-test/c-test.metrics.json"
.venv/bin/python $T/swap.py --checkpoint "$R/checkpoint" --eval-set evals/smart-home-continuation/c-test.jsonl \
  --device "$DEV" > "$R/eval-test/c-test.swap.json" 2> "$R/eval-test/swap.log"
.venv/bin/python -c "import json;d=json.load(open('$R/eval-test/c-test.swap.json'));print('c-test swap', d['n'], d['consistent'])"
echo "c-test done (frozen tau_exec=$TE tau_cancel=$TC)"
