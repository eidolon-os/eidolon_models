#!/bin/bash
# c4 on the NPU as a scratch home service (127.0.0.1:18771, deployed code, deployed placement = cores 1+2, max_pending 1),
# next to the live p4 participation service (8773, core 0). Waits for the offline NPU run to release its runtimes,
# then: readyz + NPU memory, per-request consistency against the torch eval (saving every reply), the four
# concurrency scenarios, the participation journal. Stops only the PID it started. Live 8771 / 8773 untouched.
set -u
cd /root/laya-npu/c4
while kill -0 2630196 2>/dev/null; do sleep 10; done
echo "npu run done $(date -Is)"
P4PID=$(systemctl show -p MainPID --value eidolon-laya-participation)
T0=$(date '+%Y-%m-%d %H:%M:%S')
env EIDOLON_LAYA_BACKEND=rknn EIDOLON_LAYA_MODEL_DIR=/root/laya-npu/c4-artifact EIDOLON_LAYA_PORT=18771 \
  EIDOLON_LAYA_HOST=127.0.0.1 EIDOLON_LAYA_ALLOW_NO_AUTH=1 \
  /opt/eidolon/current/eidolon_models/scripts/eidolon-laya serve > conc/service.log 2>&1 &
pid=$!
echo "c4 service pid $pid, p4 pid $P4PID, start $T0"
for i in $(seq 180); do curl -s --noproxy '*' http://127.0.0.1:18771/readyz | grep -q ready && break; sleep 1; done
curl -s --noproxy '*' http://127.0.0.1:18771/readyz > conc/readyz.json; echo; cat conc/readyz.json | head -c 400; echo
python3 /root/laya-npu/p4/npumem.py $pid > conc/npumem.txt 2>&1; tail -2 conc/npumem.txt
mkdir -p conc/responses
python3 service_check.py --url http://127.0.0.1:18771 --save conc/responses \
  --pair check/c-dev.jsonl check/c-dev.eval.json \
  --pair check/locked-v2-dev.jsonl check/locked-v2-dev.eval.json \
  --pair check/locked-v3-dev.jsonl check/locked-v3-dev.eval.json > conc/service_check.txt 2>&1
echo "service_check rc $?"; cat conc/service_check.txt | tail -6
H=http://127.0.0.1:18771/v1/systemone
for sc in home-alone part-alone both collide; do
  python3 conc/concurrency.py run conc $sc --home-url $H --pids $pid,$P4PID >> conc/run.log 2>&1
  tail -1 conc/run.log
done
journalctl -u eidolon-laya-participation --since "$T0" --no-pager > conc/participation.journal 2>&1
python3 conc/concurrency.py report conc --service-log conc/participation.journal > conc/report.txt 2>&1; cat conc/report.txt
systemctl is-active eidolon-laya eidolon-laya-participation
kill $pid; wait $pid 2>/dev/null; echo "stopped $pid $(date -Is)"
