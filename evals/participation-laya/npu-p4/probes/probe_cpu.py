"""CPU cost of NPU inference through the service's ctypes RknnRuntime (C API), per call."""
import json, os, resource, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rknn_runtime import RknnRuntime
from laya_npu import npu_inputs
NPU = sys.argv[1]; items = [json.loads(x) for x in open(sys.argv[2]) if x.strip()]
emb = np.load(f"{NPU}/tok_emb_fp16.npy", mmap_mode="r"); type_emb = np.load(f"{NPU}/type_emb.npy")
def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
out = {}
for L, lo in ((384, 256), (512, 384)):
    rt = RknnRuntime(f"{NPU}/hidden_l{L}.rknn", 0)
    it = next(i for i in items if lo < len(i["ids"]) <= L)
    x = npu_inputs(it["ids"], it["qtype"], L, emb, type_emb); rt.infer(x)
    c, t = cpu(), time.perf_counter()
    for _ in range(10): rt.infer(x)
    out[L] = {"wall_ms": round((time.perf_counter() - t) * 100, 1), "cpu_ms": round((cpu() - c) * 100, 1)}
    c, t = cpu(), time.perf_counter()
    for _ in range(10): npu_inputs(it["ids"], it["qtype"], L, emb, type_emb)
    out[L]["inputs_cpu_ms"] = round((cpu() - c) * 100, 1)
    rt.close()
print(json.dumps(out))
