#!/usr/bin/env bash
# Existing c-series stages, with raw logits retained for calibration/NPU replay.
# Run from laya/. New acceptance files are deliberately evaluated separately.
set -euo pipefail
R=${1:?run directory}
DEVICE=${2:-mps}
BASELINE=${3:-train/runs/c4}
T=train/scenarios/smart-home-continuation/tools
BIN=.venv/bin/eidolon-laya-train
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
test -f "$R/checkpoint/train_summary.json"
test ! -e "$R/calibration-complete.json"
test ! -e "$R/config-before-calibration.json"
mkdir -p "$R/items" "$R/torch-logits" "$R/eval-raw" "$R/eval" "$R/policy"
cp "$R/checkpoint/rl_agent_config.json" "$R/config-before-calibration.json"
SETS=(
  "$R/data/mined-regression.jsonl"
  "$R/data/mined-controls.jsonl"
  "$R/data/target-dev.jsonl"
  evals/smart-home-continuation/c-dev.jsonl
  evals/smart-home-continuation/diag-44.jsonl
  train/scenarios/smart-home/eval/locked-182.jsonl
  train/scenarios/smart-home/eval/locked-v2-dev.jsonl
  train/scenarios/smart-home/eval/locked-v3-dev.jsonl
  train/scenarios/smart-home/eval/locked-accept.jsonl
)
if test -f "$R/data/intent-dev.jsonl"; then
  SETS+=("$R/data/intent-dev.jsonl")
fi
ARGS=()
for s in "${SETS[@]}"; do ARGS+=(--eval-set "$s"); done
$BIN items --checkpoint "$R/checkpoint" "${ARGS[@]}" --out "$R/items" --device "$DEVICE" > "$R/items.log" 2>&1
.venv/bin/python - "$R" <<'PY'
import json, sys
from pathlib import Path
r = Path(sys.argv[1])
for p in (r / "items").glob("*.items.jsonl"):
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    out = r / "torch-logits" / p.name.replace(".items.jsonl", ".logits.jsonl")
    out.write_text("".join(json.dumps({"key": x["key"], "logits": x["ref_logits"]}) + "\n" for x in rows))
PY
$BIN eval --checkpoint "$R/checkpoint" "${ARGS[@]}" --logits "$R/torch-logits" --out "$R/eval-raw" --device "$DEVICE" > "$R/eval-raw.log" 2>&1
$BIN calibrate --checkpoint "$R/checkpoint" --calib "$R/dataset/calib.jsonl" --device "$DEVICE" > "$R/calibrate.log" 2>&1
cp "$R/checkpoint/rl_agent_config.json" "$R/config-after-calibration.json"
.venv/bin/python "$T/calib_split.py" --checkpoint "$R/checkpoint" --calib "$R/dataset/calib.jsonl" --device "$DEVICE" > "$R/calib-split.json" 2> "$R/calib-split.log"
$BIN eval --checkpoint "$R/checkpoint" "${ARGS[@]}" --logits "$R/torch-logits" --out "$R/eval" --device "$DEVICE" > "$R/eval.log" 2>&1
for s in "${SETS[@]}"; do
  name=$(basename "$s" .jsonl)
  baseline=""
  case "$name" in
    mined-regression|target-dev|intent-dev) baseline="$R/baseline-new/$name.json" ;;
    c-dev|diag-44) baseline="$BASELINE/eval-c/$name.json" ;;
    locked-accept) baseline="$BASELINE/eval-final/$name.json" ;;
    locked-*) baseline="$BASELINE/eval/$name.json" ;;
  esac
  REPORT_ARGS=(--report "$R/eval/$name.json" --out "$R/policy/$name.json")
  if test -n "$baseline"; then REPORT_ARGS+=(--baseline "$baseline"); fi
  .venv/bin/python "$T/target_report.py" "${REPORT_ARGS[@]}"
done
.venv/bin/python "$T/swap.py" --checkpoint "$R/checkpoint" --eval-set evals/smart-home-continuation/c-dev.jsonl --device "$DEVICE" > "$R/c-dev.swap.json" 2> "$R/swap.log"
.venv/bin/python - "$R" <<'PY'
import hashlib, json, sys
from pathlib import Path
r = Path(sys.argv[1])
paths = [r / "checkpoint/model.safetensors", r / "config-before-calibration.json", r / "config-after-calibration.json"]
(r / "calibration-complete.json").write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2) + "\n")
PY
