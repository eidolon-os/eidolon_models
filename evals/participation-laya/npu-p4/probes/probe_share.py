"""NPU weight sharing on RK3588: rknn_dup_context and RKNN_FLAG_SHARE_WEIGHT_MEM (standalone probe)."""
import ctypes as C, json, os, sys, threading, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rknn_runtime import RknnRuntime, _check
from laya_npu import npu_inputs

NPU = sys.argv[1]; ITEMS = sys.argv[2]
emb = np.load(f"{NPU}/tok_emb_fp16.npy", mmap_mode="r"); type_emb = np.load(f"{NPU}/type_emb.npy")
items = [json.loads(x) for x in open(ITEMS) if x.strip()]
def item_for(L, lo):
    return next(it for it in items if lo < len(it["ids"]) <= L)

def mapped():
    t = 0
    for line in open("/proc/self/maps"):
        if "card1" in line:
            a, b = line.split()[0].split("-"); t += int(b, 16) - int(a, 16)
    return round(t / 2**20)

class Ext(C.Structure):
    _fields_ = [("ctx", C.c_uint64), ("real_model_offset", C.c_int32), ("real_model_size", C.c_uint32),
                ("model_buffer_fd", C.c_int32), ("model_buffer_flags", C.c_uint32), ("reserved", C.c_uint8 * 112)]

def wrap(lib, ctxval, core):
    r = object.__new__(RknnRuntime); r._lib = lib; r._ctx = C.c_uint64(ctxval)
    _check(lib.rknn_set_core_mask(r._ctx, 1 << core), "core mask")
    r._inputs = [r._attribute(i, 1) for i in range(3)]; r._output = r._attribute(0, 2)
    return r

out = {"baseline_mapped_mib": mapped()}
A = RknnRuntime(f"{NPU}/hidden_l384.rknn", 0)
lib = A._lib
lib.rknn_dup_context.argtypes = [C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]; lib.rknn_dup_context.restype = C.c_int
lib.rknn_init.argtypes = [C.POINTER(C.c_uint64), C.c_void_p, C.c_uint32, C.c_uint32, C.c_void_p]
out["A_384_core0"] = mapped()
# 1. dup context of the same model onto core 1
d = C.c_uint64()
rc = lib.rknn_dup_context(C.byref(A._ctx), C.byref(d)); out["dup_rc"] = rc
if rc == 0:
    B = wrap(lib, d.value, 1); out["A+dup_core1"] = mapped()
it = item_for(384, 256); x = npu_inputs(it["ids"], it["qtype"], 384, emb, type_emb)
ref = A.infer(x)
if rc == 0:
    out["dup_max_abs_diff"] = float(np.abs(B.infer(x) - ref).max())
    # concurrent timing A (core0) || B (core1) vs A alone
    t = time.perf_counter(); [A.infer(x) for _ in range(5)]; out["A_alone_ms"] = round((time.perf_counter() - t) / 5 * 1000, 1)
    ts = {}
    def run(r, k):
        t0 = time.perf_counter(); [r.infer(x) for _ in range(5)]; ts[k] = round((time.perf_counter() - t0) / 5 * 1000, 1)
    th = [threading.Thread(target=run, args=(A, "A")), threading.Thread(target=run, args=(B, "B"))]
    [t_.start() for t_ in th]; [t_.join() for t_ in th]; out["A||dup_ms"] = ts
# 2. other buckets sharing A's weights
for L, lo in ((512, 384), (640, 512)):
    c = C.c_uint64(); ext = Ext(); ext.ctx = A._ctx.value
    before = mapped()
    rc2 = lib.rknn_init(C.byref(c), os.fsencode(f"{NPU}/hidden_l{L}.rknn"), 0, 0x20, C.byref(ext))
    out[f"share_{L}_rc"] = rc2
    if rc2 == 0:
        S_ = wrap(lib, c.value, 0); out[f"share_{L}_added_mib"] = mapped() - before
        it = item_for(L, lo); x2 = npu_inputs(it["ids"], it["qtype"], L, emb, type_emb)
        N = RknnRuntime(f"{NPU}/hidden_l{L}.rknn", 1)  # an ordinary context for reference
        out[f"share_{L}_max_abs_diff_vs_plain"] = float(np.abs(S_.infer(x2) - N.infer(x2)).max())
        t = time.perf_counter(); [S_.infer(x2) for _ in range(3)]; out[f"share_{L}_ms"] = round((time.perf_counter() - t) / 3 * 1000, 1)
        t = time.perf_counter(); [N.infer(x2) for _ in range(3)]; out[f"plain_{L}_ms"] = round((time.perf_counter() - t) / 3 * 1000, 1)
        out[f"after_plain_{L}_mapped"] = mapped()
        N.close()
out["final_mapped_mib"] = mapped()
print(json.dumps(out, indent=1))
