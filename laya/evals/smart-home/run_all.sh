#!/bin/sh
# 在本机起 laya 服务（默认 torch + MPS）、跑评测、停服务，然后生成 REPORT.md。
#
#   evals/smart-home/run_all.sh                  # 默认只跑 mac-torch-mps（Mac GPU）
#   evals/smart-home/run_all.sh mac-torch-cpu mac-onnx-cpu mac-onnx-cpu-t6   # 需要时再跑 CPU 后端
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
  log="$HERE/results/$run.server.log"
  mkdir -p "$HERE/results"
  EIDOLON_LAYA_BACKEND=$backend EIDOLON_LAYA_DEVICE=$device EIDOLON_LAYA_PORT=$PORT \
    EIDOLON_LAYA_THREADS=$threads \
    "$LAYA/scripts/eidolon-laya" serve >"$log" 2>&1 &
  pid=$!
  trap 'kill $pid 2>/dev/null || true' EXIT
  for _ in $(seq 1 120); do
    curl -fs -m 2 "$URL/readyz" >/dev/null 2>&1 && break
    sleep 1
  done
  python3 "$HERE/run_eval.py" --url "$URL" --label "$run" --server-pid "$pid"
  kill "$pid"; wait "$pid" 2>/dev/null || true
  trap - EXIT
done
python3 "$HERE/make_report.py"
