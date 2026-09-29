"""Thin binding to Rockchip's RKNN C Runtime, independent of CPython wheel ABI.

ABI: the pinned 2.3.2 rknn_api.h already shipped with Models' TTS engine.
This adapter owns tensor buffers and context lifetime, not model semantics.
"""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np


class _Input(C.Structure):
    _fields_ = [
        ("index", C.c_uint32),
        ("buf", C.c_void_p),
        ("size", C.c_uint32),
        ("pass_through", C.c_uint8),
        ("type", C.c_int),
        ("fmt", C.c_int),
    ]


class _Output(C.Structure):
    _fields_ = [
        ("want_float", C.c_uint8),
        ("is_prealloc", C.c_uint8),
        ("index", C.c_uint32),
        ("buf", C.c_void_p),
        ("size", C.c_uint32),
    ]


class _Counts(C.Structure):
    _fields_ = [("n_input", C.c_uint32), ("n_output", C.c_uint32)]


class _Attr(C.Structure):
    _fields_ = [
        ("index", C.c_uint32),
        ("n_dims", C.c_uint32),
        ("dims", C.c_uint32 * 16),
        ("name", C.c_char * 256),
        ("n_elems", C.c_uint32),
        ("size", C.c_uint32),
        ("fmt", C.c_int),
        ("type", C.c_int),
        ("qnt_type", C.c_int),
        ("fl", C.c_int8),
        ("zp", C.c_int32),
        ("scale", C.c_float),
        ("w_stride", C.c_uint32),
        ("size_with_stride", C.c_uint32),
        ("pass_through", C.c_uint8),
        ("h_stride", C.c_uint32),
    ]


def _check(code: int, operation: str) -> None:
    if code != 0:
        raise RuntimeError(f"{operation} failed: RKNN status {code}")


class RknnRuntime:
    """One synchronous context; the backend serializes calls per NPU core."""

    def __init__(self, model: Path, core: int):
        if core not in (0, 1, 2):
            raise ValueError("NPU core must be 0, 1 or 2")
        self._ctx = C.c_uint64()
        self._lib = C.CDLL(os.environ.get("EIDOLON_LAYA_RKNN_LIBRARY", "librknnrt.so"))
        ctx = C.c_uint64
        declarations = {
            "rknn_init": [C.POINTER(ctx), C.c_void_p, C.c_uint32, C.c_uint32, C.c_void_p],
            "rknn_set_core_mask": [ctx, C.c_int],
            "rknn_query": [ctx, C.c_int, C.c_void_p, C.c_uint32],
            "rknn_inputs_set": [ctx, C.c_uint32, C.POINTER(_Input)],
            "rknn_run": [ctx, C.c_void_p],
            "rknn_outputs_get": [ctx, C.c_uint32, C.POINTER(_Output), C.c_void_p],
            "rknn_outputs_release": [ctx, C.c_uint32, C.POINTER(_Output)],
            "rknn_destroy": [ctx],
        }
        for name, args in declarations.items():
            function = getattr(self._lib, name)
            function.argtypes, function.restype = args, C.c_int
        try:
            _check(
                self._lib.rknn_init(C.byref(self._ctx), os.fsencode(model), 0, 0, None), "rknn_init"
            )
            _check(self._lib.rknn_set_core_mask(self._ctx, 1 << core), "rknn_set_core_mask")
            counts = _Counts()
            _check(
                self._lib.rknn_query(self._ctx, 0, C.byref(counts), C.sizeof(counts)),
                "rknn_query counts",
            )
            if (counts.n_input, counts.n_output) != (3, 1):
                raise ValueError("Laya hidden graph requires exactly three inputs and one output")
            self._inputs = [self._attribute(i, 1) for i in range(counts.n_input)]
            self._output = self._attribute(0, 2)
        except BaseException:
            self.close()
            raise

    def _attribute(self, index: int, query: int) -> _Attr:
        attr = _Attr(index=index)
        _check(
            self._lib.rknn_query(self._ctx, query, C.byref(attr), C.sizeof(attr)),
            "rknn_query tensor",
        )
        if not 0 < attr.n_dims <= 16 or not attr.n_elems:
            raise ValueError("invalid RKNN tensor dimensions")
        return attr

    def infer(self, inputs: list[np.ndarray]) -> np.ndarray:
        if not self._ctx.value:
            raise RuntimeError("RKNN context is closed")
        if len(inputs) != len(self._inputs):
            raise ValueError("wrong RKNN input count")
        arrays = [np.ascontiguousarray(a) for a in inputs]
        descriptors = (_Input * len(arrays))()
        for i, (array, attr) in enumerate(zip(arrays, self._inputs, strict=True)):
            if array.size != attr.n_elems or tuple(array.shape) != tuple(attr.dims[: attr.n_dims]):
                raise ValueError(f"input {i} shape differs from the RKNN graph")
            kind = {np.dtype("float32"): 0, np.dtype("int64"): 8}.get(array.dtype)
            if kind is None:
                raise ValueError(f"unsupported RKNN input dtype {array.dtype}")
            descriptors[i] = _Input(i, array.ctypes.data, array.nbytes, 0, kind, attr.fmt)
        _check(self._lib.rknn_inputs_set(self._ctx, len(arrays), descriptors), "rknn_inputs_set")
        _check(self._lib.rknn_run(self._ctx, None), "rknn_run")
        # Float conversion is the official runtime's job; keep the preallocated owner alive.
        output = np.empty(tuple(self._output.dims[: self._output.n_dims]), dtype=np.float32)
        descriptor = _Output(1, 1, 0, output.ctypes.data, output.nbytes)
        _check(
            self._lib.rknn_outputs_get(self._ctx, 1, C.byref(descriptor), None), "rknn_outputs_get"
        )
        try:
            return output
        finally:
            _check(
                self._lib.rknn_outputs_release(self._ctx, 1, C.byref(descriptor)),
                "rknn_outputs_release",
            )

    def dup(self, core: int) -> RknnRuntime:
        """Another context of the same model on ``core`` that shares this one's weights (``rknn_dup_context``).

        Every ``rknn_init`` of a Laya graph holds its own ~280 MiB of weights, and the NPU's address space is
        shared by every model on the board (about 14 such graphs fit on the RK3588). A duplicate adds ~36 MiB
        and returns bit-identical outputs (measured, participation EXPERIMENTS.md). Close duplicates first.
        """
        if core not in (0, 1, 2):
            raise ValueError("NPU core must be 0, 1 or 2")
        if not self._ctx.value:
            raise RuntimeError("RKNN context is closed")
        function = self._lib.rknn_dup_context
        function.argtypes = [C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]
        function.restype = C.c_int
        other = object.__new__(RknnRuntime)
        other._lib, other._ctx = self._lib, C.c_uint64()
        _check(function(C.byref(self._ctx), C.byref(other._ctx)), "rknn_dup_context")
        try:
            _check(self._lib.rknn_set_core_mask(other._ctx, 1 << core), "rknn_set_core_mask")
        except BaseException:
            other.close()
            raise
        other._inputs, other._output = self._inputs, self._output
        return other

    def close(self) -> None:
        if self._ctx.value:
            self._lib.rknn_destroy(self._ctx)
            self._ctx.value = 0
