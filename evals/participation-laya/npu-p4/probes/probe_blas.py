import json, os, resource, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from laya_npu import score_hidden
NPU = sys.argv[1]; items = [json.loads(x) for x in open(sys.argv[2]) if x.strip()]
sc = dict(np.load(f"{NPU}/scorer.npz"))
it = next(i for i in items if 256 < len(i["ids"]) <= 384)
h = np.random.default_rng(0).standard_normal((384, 768)).astype(np.float32)
def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
score_hidden(h, it["markers"], sc)
c, t = cpu(), time.perf_counter()
for _ in range(20): score_hidden(h, it["markers"], sc)
w = (time.perf_counter() - t) / 20
time.sleep(0.5); spin = cpu()
time.sleep(1.0); spin = cpu() - spin
print(json.dumps({"OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"), "threads": len(os.listdir("/proc/self/task")),
                  "score_wall_ms": round(w * 1000, 2), "score_cpu_ms": round((cpu() - c - spin) / 20 * 1000, 2),
                  "idle_cpu_ms_per_s_after": round(spin * 1000, 1)}))
try:
    import numpy.__config__ as nc; print([l for l in str(nc.show(mode="dicts")).split(",") if "openblas" in l.lower()][:2])
except Exception as e: print(e)
