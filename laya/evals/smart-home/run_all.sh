#!/bin/sh
# 在本机起 laya 服务（默认 torch + MPS）、跑评测、停服务，然后生成 REPORT.md。
#
#   evals/smart-home/run_all.sh                  # 默认只跑 mac-torch-mps（Mac GPU）
#   evals/smart-home/run_all.sh mac-torch-cpu mac-onnx-cpu mac-onnx-cpu-t6   # 需要时再跑 CPU 后端
#   MODEL_DIR=models/laya-cn-a/178eb2c0 LABEL_PREFIX=laya-cn-a@ evals/smart-home/run_all.sh
#                                                # 换一个 checkpoint；结果目录 laya-cn-a@mac-torch-mps
#   HEAD_MAX_LEN=1024 LABEL_PREFIX=laya-multilingual-h1024@ evals/smart-home/run_all.sh
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
LAYA=$(CDPATH= cd -- "$HERE/../.." && pwd)
PORT=${EIDOLON_LAYA_PORT:-8771}
URL="http://127.0.0.1:$PORT"
RUNS=${*:-"mac-torch-mps"}

for run in $RUNS; do
  threads=0
  case "$run" in
    mac-torch-mps)   backend=torch; device=mps ;;
    mac-torch-cpu)   backend=torch; device=cpu ;;
    mac-onnx-cpu)    backend=onnx;  device=auto ;;
    # ORT's default pool also schedules onto the efficiency cores; pin it to the performance cores.
    mac-onnx-cpu-t6) backend=onnx;  device=auto; threads=6 ;;
    *) echo "unknown run $run" >&2; exit 2 ;;
  esac
  if curl -fs -m 2 "$URL/healthz" >/dev/null 2>&1; then
    echo "port $PORT is already serving; stop it first" >&2; exit 1
  fi
  label="${LABEL_PREFIX:-}$run"
  log="$HERE/results/$label.server.log"
  mkdir -p "$HERE/results"
  EIDOLON_LAYA_BACKEND=$backend EIDOLON_LAYA_DEVICE=$device EIDOLON_LAYA_PORT=$PORT \
    EIDOLON_LAYA_THREADS=$threads EIDOLON_LAYA_MODEL_DIR="${MODEL_DIR:-}" \
    EIDOLON_LAYA_HEAD_MAX_LEN="${HEAD_MAX_LEN:-}" \
    "$LAYA/scripts/eidolon-laya" serve >"$log" 2>&1 &
  pid=$!
  trap 'kill $pid 2>/dev/null || true' EXIT
  for _ in $(seq 1 120); do
    curl -fs -m 2 "$URL/readyz" >/dev/null 2>&1 && break
    sleep 1
  done
  python3 "$HERE/run_eval.py" --url "$URL" --label "$label" --server-pid "$pid"
  kill "$pid"; wait "$pid" 2>/dev/null || true
  trap - EXIT
done
python3 "$HERE/make_report.py"
python3 "$HERE/compare_models.py"
