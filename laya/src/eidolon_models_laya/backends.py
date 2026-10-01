"""The two forward passes. Everything around them lives in ``engine`` and ``sequence``.

Each backend takes the numpy batch from ``sequence.collate`` and returns
``(logits, act_logits)`` as float32 numpy arrays. Heavy imports stay inside the
constructors so an ONNX-only install never imports torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .sequence import INPUT_NAMES


class Backend(Protocol):
    name: str

    def forward(self, batch: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]: ...

    def describe(self) -> dict: ...


class TorchBackend:
    """``laya.load`` for the weights; the forward pass only."""

    name = "torch"

    def __init__(self, torch_dir: Path, *, device: str = "auto", threads: int = 0):
        import torch

        from .vendor import laya

        if threads:
            torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)  # one forward per call; nothing to overlap
        except RuntimeError:
            pass  # already set, or parallel work already started in this process
        self._torch = torch
        agent = laya.load(str(torch_dir), device=None if device == "auto" else device)
        self._model = agent.model
        self._device = agent.device
        self._dtype = agent.dtype
        self._threads = torch.get_num_threads()
        del agent  # keep the model, drop laya's own transformers tokenizer

    def forward(self, batch: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        tensors = [torch.from_numpy(batch[k]).to(self._device) for k in INPUT_NAMES]
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self._device.type,
                dtype=self._dtype,
                enabled=self._device.type == "cuda",
            ),
        ):
            logits, act = self._model(*tensors)
        if self._device.type == "mps":
            torch.mps.synchronize()
        return logits.float().cpu().numpy(), act.float().cpu().numpy()

    def describe(self) -> dict:
        # The checkpoint ships fp16 weights; laya upcasts them to fp32 when it builds
        # the model, and only autocasts (to fp16/bf16) on CUDA.
        weights = str(next(self._model.parameters()).dtype).removeprefix("torch.")
        on_cuda = self._device.type == "cuda"
        compute = str(self._dtype).removeprefix("torch.") if on_cuda else weights
        return {
            "backend": self.name,
            "device": str(self._device),
            "precision": {"weights": weights, "compute": compute},
            "threads": self._threads,
            "torch": self._torch.__version__,
        }


class OnnxBackend:
    name = "onnx"

    def __init__(self, onnx_path: Path, *, threads: int = 0):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._ort = ort
        self._threads = threads
        self._session = ort.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )

    def forward(self, batch: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        logits, act = self._session.run(None, {k: batch[k] for k in INPUT_NAMES})
        return logits, act

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "device": "cpu",
            # export.py traces the fp32 CPU model; nothing is quantized.
            "precision": {"weights": "float32", "compute": "float32"},
            "threads": self._threads or "ort-default",
            "onnxruntime": self._ort.__version__,
        }


def limit_blas_threads(n: int = 1) -> list[str]:
    """Set numpy's OpenBLAS pool to ``n`` threads, what ``OPENBLAS_NUM_THREADS`` does at startup.

    On the NPU backend the CPU half is a few small matmuls per question (scorer, act head). With OpenBLAS's
    default pool — one thread per core — each costs ~20 ms and ~160 ms of CPU on the RK3588, and the pool
    spins between calls: 1.9 s of CPU per smart-home request, against 67 ms with one thread. Returns what was
    set (empty where numpy is not on OpenBLAS, e.g. macOS's Accelerate)."""
    import ctypes

    done = []
    try:
        maps = Path("/proc/self/maps").read_text()
    except OSError:
        return done
    paths = sorted({f[-1] for f in (line.split() for line in maps.splitlines())
                    if len(f) > 5 and f[-1].startswith("/") and "openblas" in Path(f[-1]).name.lower()})
    for path in paths:
        lib = ctypes.CDLL(path)
        for name in ("scipy_openblas_set_num_threads64_", "scipy_openblas_set_num_threads",
                     "openblas_set_num_threads64_", "openblas_set_num_threads"):
            function = getattr(lib, name, None)
            if function is not None:
                function.argtypes, function.restype = [ctypes.c_int], None
                function(n)
                done.append(f"{Path(path).name}:{name}({n})")
                break
    return done


def parse_placement(text: str) -> dict[int, tuple[int, ...]]:
    """``"0:128,256|1:256,384,512"`` -> {0: (128, 256), 1: (256, 384, 512)}."""
    out = {}
    for part in text.split("|"):
        core, _, buckets = part.partition(":")
        if not buckets or int(core) not in (0, 1, 2):
            raise ValueError(f"bad NPU placement {text!r}: want 'core:bucket,bucket|core:...' with cores 0-2")
        out[int(core)] = tuple(int(b) for b in buckets.split(","))
    return out


class RknnBackend:
    """RK3588 NPU via the pinned RKNN C Runtime, from what ``export-npu`` wrote and ``deploy/rk3588/laya_npu.py
    convert`` compiled: encoder + head layers per sequence bucket on the NPU; token-embedding lookup,
    marker gather, scorer and act head on the CPU (numpy).

    The items of one batch (the questions of one request) run in parallel, one queue per NPU core.
    Each core loads only the buckets in its placement and an item goes to the least-loaded core that has its
    bucket. A bucket placed on several cores is loaded once and duplicated onto the others (shared weights):
    the NPU address space is shared by every model on the board, so each copy of the weights counts.
    """

    name = "rknn"
    PLACEMENT = {0: (128, 256), 1: (256, 384, 512), 2: (128, 256)}

    def __init__(
        self,
        npu_dir: Path,
        *,
        placement: dict[int, tuple[int, ...]] | None = None,
        library: Path | None = None,
    ):
        import re
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from .rknn_runtime import RknnRuntime

        npu_dir = Path(npu_dir)
        files = {int(re.search(r"hidden_l(\d+)", p.name).group(1)): p for p in npu_dir.glob("hidden_l*.rknn")}
        if not files:
            raise FileNotFoundError(f"no hidden_l*.rknn in {npu_dir}; run deploy/rk3588/laya_npu.py convert")
        placement = placement or self.PLACEMENT
        self._placement = {c: tuple(L for L in bs if L in files) for c, bs in placement.items()}
        missing = set(files) - {L for bs in self._placement.values() for L in bs}
        if missing:
            raise ValueError(f"buckets {sorted(missing)} are on no core in placement {placement}")
        self._buckets = sorted(files)
        self.max_len = max(self._buckets)
        self._rt = {}
        self._duplicates = 0
        first = {}
        try:
            for core, bs in self._placement.items():
                for L in bs:
                    if L in first and hasattr(first[L], "dup"):
                        self._rt[(core, L)] = first[L].dup(core)
                        self._duplicates += 1
                    else:
                        first[L] = self._rt[(core, L)] = RknnRuntime(files[L], core, library=library)
        except BaseException:
            for runtime in reversed(list(self._rt.values())):  # duplicates before the originals
                runtime.close()
            raise
        self._blas = limit_blas_threads(1)
        self._emb = np.load(npu_dir / "tok_emb_fp16.npy", mmap_mode="r")
        self._type = np.load(npu_dir / "type_emb.npy")
        self._sc = dict(np.load(npu_dir / "scorer.npz"))
        self._act = dict(np.load(npu_dir / "act_head.npz"))
        self._pool = ThreadPoolExecutor(len(self._placement))
        self._lock = threading.Lock()  # one batch at a time: its items already fill the cores
        # (released by whichever worker finishes the batch's last item, so a plain Lock, not RLock)

    def _bucket(self, n: int) -> int:
        for b in self._buckets:
            if n <= b:
                return b
        raise ValueError(f"sequence of {n} tokens exceeds the largest NPU bucket {self._buckets[-1]}")

    def schedule(self, lengths: list[int]) -> dict[int, list[tuple[int, int]]]:
        """Item index -> core: longest first onto the least-loaded core that has the bucket (cost ~ L);
        each core then runs its items shortest first, so short questions (intent) finish earliest."""
        load = {c: 0 for c in self._placement}
        queues: dict[int, list[tuple[int, int]]] = {c: [] for c in self._placement}
        for i in sorted(range(len(lengths)), key=lambda i: -lengths[i]):
            L = self._bucket(lengths[i])
            core = min((c for c, bs in self._placement.items() if L in bs), key=lambda c: load[c])
            load[core] += L
            queues[core].append((i, L))
        return {c: sorted(q, key=lambda x: x[1]) for c, q in queues.items()}

    def submit(self, batch: dict[str, np.ndarray]) -> list:
        """Start every item of the batch; one future per item resolving to ``(logits[k], act[2])``.
        The NPU stays busy with this batch (the next one waits) until all of its items are done,
        even those nobody waits for."""
        import threading
        from concurrent.futures import Future

        from .export_npu import act_from_logits, npu_inputs, score_hidden

        n = batch["marker_mask"].shape[0]
        lengths = [int(batch["attention_mask"][i].sum()) for i in range(n)]
        futures = [Future() for _ in range(n)]
        self._lock.acquire()
        try:
            queues = self.schedule(lengths)
        except Exception:
            self._lock.release()
            raise
        left, guard = [n], threading.Lock()

        def finished():
            with guard:
                left[0] -= 1
                if left[0] == 0:
                    self._lock.release()

        def run(core, queue):
            for i, L in queue:
                try:
                    ids = batch["input_ids"][i, : lengths[i]].tolist()
                    markers = batch["marker_pos"][i][batch["marker_mask"][i].astype(bool)].tolist()
                    x = npu_inputs(ids, int(batch["qtype"][i]), L, self._emb, self._type)
                    h = np.asarray(self._rt[(core, L)].infer(x), np.float32)[0]
                    lo = score_hidden(h, markers, self._sc)
                    futures[i].set_result((lo, act_from_logits(h[0], lo, self._act)))
                except Exception as exc:  # noqa: BLE001 - surfaced through the future
                    futures[i].set_exception(exc)
                finally:
                    finished()

        if n == 0:
            self._lock.release()
        for core, queue in queues.items():
            if queue:
                self._pool.submit(run, core, queue)
        return futures

    def forward(self, batch: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        n, kmax = batch["marker_mask"].shape
        logits = np.full((n, kmax), -1e4, np.float32)
        act = np.zeros((n, 2), np.float32)
        for i, f in enumerate(self.submit(batch)):
            lo, a = f.result()
            logits[i, : len(lo)] = lo
            act[i] = a
        return logits, act

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "device": "rk3588-npu",
            "precision": {"weights": "float16", "compute": "float16"},
            "buckets": self._buckets,
            "placement": {str(c): list(bs) for c, bs in self._placement.items()},
            "weight_copies": len(self._rt) - self._duplicates,
            "cpu_blas": self._blas or "not OpenBLAS",
        }
