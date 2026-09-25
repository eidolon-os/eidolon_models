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
