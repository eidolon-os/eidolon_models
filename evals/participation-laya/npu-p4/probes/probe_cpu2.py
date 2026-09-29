"""Does CPU cost grow with the number of loaded contexts, or with parallel inference?"""
import ctypes as C, json, os, resource, sys, threading, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rknn_runtime import RknnRuntime, _check
from laya_npu import npu_inputs
NPU = sys.argv[1]; items = [json.loads(x) for x in open(sys.argv[2]) if x.strip()]
emb = np.load(f"{NPU}/tok_emb_fp16.npy", mmap_mode="r"); type_emb = np.load(f"{NPU}/type_emb.npy")
def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
def wrap(lib, v, core):
    r = object.__new__(RknnRuntime); r._lib = lib; r._ctx = C.c_uint64(v)
    _check(lib.rknn_set_core_mask(r._ctx, 1 << core), "mask")
    r._inputs = [r._attribute(i, 1) for i in range(3)]; r._output = r._attribute(0, 2); return r
A = RknnRuntime(f"{NPU}/hidden_l384.rknn", 0); lib = A._lib
lib.rknn_dup_context.argtypes = [C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]; lib.rknn_dup_context.restype = C.c_int
it = next(i for i in items if 256 < len(i["ids"]) <= 384); x = npu_inputs(it["ids"], it["qtype"], 384, emb, type_emb)
A.infer(x)
def measure(label, fn, n=6):
    c, t = cpu(), time.perf_counter()
    for _ in range(n): fn()
    return {"case": label, "wall_ms": round((time.perf_counter() - t) / n * 1000, 1), "cpu_ms": round((cpu() - c) / n * 1000, 1),
            "threads": len(os.listdir("/proc/self/task"))}
res = [measure("1 context", lambda: A.infer(x))]
dups = []
for core in (1, 2, 0, 1, 2, 0):
    d = C.c_uint64(); assert lib.rknn_dup_context(C.byref(A._ctx), C.byref(d)) == 0; dups.append(wrap(lib, d.value, core))
time.sleep(2)
c = cpu(); time.sleep(3); res.append({"case": "7 contexts, idle 3 s", "cpu_ms_total": round((cpu() - c) * 1000, 1), "threads": len(os.listdir("/proc/self/task"))})
res.append(measure("7 contexts, infer on 1", lambda: A.infer(x)))
trio = [A, dups[0], dups[1]]  # cores 0, 1, 2
def par():
    th = [threading.Thread(target=r.infer, args=(x,)) for r in trio]; [t.start() for t in th]; [t.join() for t in th]
res.append(measure("7 contexts, 3 in parallel on cores 0/1/2 (per decision of 3)", par))
for d in dups: d.close()
time.sleep(1)
res.append(measure("back to 1 context", lambda: A.infer(x)))
one_more = [wrap(lib, v, core) for v, core in []]
B2 = C.c_uint64(); assert lib.rknn_dup_context(C.byref(A._ctx), C.byref(B2)) == 0; B = wrap(lib, B2.value, 1)
def par2():
    th = [threading.Thread(target=r.infer, args=(x,)) for r in (A, B)]; [t.start() for t in th]; [t.join() for t in th]
res.append(measure("2 contexts, 2 in parallel on cores 0/1", par2))
print(json.dumps(res, ensure_ascii=False, indent=0))
