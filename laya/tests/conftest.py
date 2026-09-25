from __future__ import annotations

import importlib.util

import pytest

from eidolon_models_laya.artifacts import Manifest, verify_onnx, verify_torch
from eidolon_models_laya.config import Settings


@pytest.fixture(scope="session")
def manifest() -> Manifest:
    return Manifest.load(Settings.from_env({}).model_dir)


@pytest.fixture(scope="session")
def torch_files(manifest: Manifest) -> Manifest:
    if verify_torch(manifest, checksums=False):
        pytest.skip("PyTorch checkpoint not fetched (scripts/eidolon-laya fetch)")
    return manifest


@pytest.fixture(scope="session")
def upstream_agent(torch_files: Manifest):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch extra not installed")
    from eidolon_models_laya.vendor import laya  # noqa: F401

    return laya.load(str(torch_files.torch_dir), device="cpu")


@pytest.fixture(scope="session")
def onnx_files(torch_files: Manifest) -> Manifest:
    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("onnx extra not installed")
    if verify_onnx(torch_files, checksums=False):
        pytest.skip("ONNX not exported (scripts/eidolon-laya export-onnx)")
    return torch_files
