#!/bin/bash
# r14 baseline for the c4 concurrency run: the deployed r14 RKNN artifact (read-only) as a scratch home service on 18771
# with the same launcher / placement / max_pending and the same request streams (inputs copied from conc/, checksummed).
# Guards: 18771 must be free before; after start the 18771 listener must be this PID with NPU memory mapped; re-checked
# before every scenario. Live 8771 / 8773 untouched; stops only the PID it started.
set -u
cd /root/laya-npu/c4
D=conc-r14b
if ss -ltn | grep -q ':18771 '; then echo "ABORT: 18771 already in use"; ss -ltnp | grep ':18771 '; exit 3; fi
mkdir -p $D && cp conc/home.jsonl conc/participation.jsonl conc/clarify_instructions.json conc/concurrency.py $D/
md5sum conc/home.jsonl conc/participation.jsonl conc/concurrency.py $D/home.jsonl $D/participation.jsonl $D/concurrency.py
P4PID=$(systemctl show -p MainPID --value eidolon-laya-participation)
T0=$(date '+%Y-%m-%d %H:%M:%S')
env EIDOLON_LAYA_BACKEND=rknn EIDOLON_LAYA_MODEL_DIR=/var/lib/eidolon/models/laya-smart-home-r14-rknn-45f3dedb EIDOLON_LAYA_PORT=18771 \
  EIDOLON_LAYA_HOST=127.0.0.1 EIDOLON_LAYA_ALLOW_NO_AUTH=1 \
  /opt/eidolon/current/eidolon_models/scripts/eidolon-laya serve > $D/service.log 2>&1 &
pid=$!
echo "r14 scratch service pid $pid, p4 pid $P4PID, start $T0"
mine() { ss -ltnp | grep ':18771 ' | grep -q "pid=$pid,"; }
for i in $(seq 180); do mine && curl -s --noproxy '*' http://127.0.0.1:18771/readyz | grep -q ready && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
if ! mine; then echo "ABORT: 18771 is not served by $pid"; tail -5 $D/service.log; kill $pid 2>/dev/null; exit 4; fi
curl -s --noproxy '*' http://127.0.0.1:18771/readyz > $D/readyz.json; cat $D/readyz.json | head -c 300; echo
python3 /root/laya-npu/p4/npumem.py $pid > $D/npumem.txt 2>&1; tail -1 $D/npumem.txt
grep -q "[1-9][0-9]* MiB" $D/npumem.txt || { echo "ABORT: no NPU memory mapped by $pid"; kill $pid; exit 5; }
H=http://127.0.0.1:18771/v1/systemone
for sc in home-alone part-alone both collide; do
  mine || { echo "ABORT before $sc: 18771 listener changed"; kill $pid 2>/dev/null; exit 6; }
  python3 $D/concurrency.py run $D $sc --home-url $H --pids $pid,$P4PID >> $D/run.log 2>&1
  tail -1 $D/run.log
done
journalctl -u eidolon-laya-participation --since "$T0" --no-pager > $D/participation.journal 2>&1
python3 $D/concurrency.py report $D --service-log $D/participation.journal > $D/report.txt 2>&1; cat $D/report.txt
systemctl is-active eidolon-laya eidolon-laya-participation
kill $pid; wait $pid 2>/dev/null; echo "stopped $pid $(date -Is)"
