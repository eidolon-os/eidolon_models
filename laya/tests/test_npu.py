"""The board-side runner duplicates the CPU half of the NPU split; keep it equal to the exporter's
and to the model's own scorer (no weights needed: random ones)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from eidolon_models_laya import export_npu

RUNNER = Path(__file__).resolve().parents[1] / "deploy" / "rk3588" / "laya_npu.py"


def _runner():
    spec = importlib.util.spec_from_file_location("laya_npu", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scorer(d=32, seed=0):
    rng = np.random.default_rng(seed)
    return {"ln_w": rng.normal(size=d).astype(np.float32), "ln_b": rng.normal(size=d).astype(np.float32),
            "w1": rng.normal(size=(d, d)).astype(np.float32) / np.sqrt(d), "b1": rng.normal(size=d).astype(np.float32),
            "w2": rng.normal(size=(1, d)).astype(np.float32) / np.sqrt(d), "b2": rng.normal(size=1).astype(np.float32)}


def test_runner_matches_exporter():
    run = _runner()
    sc = _scorer()
    h = np.random.default_rng(1).normal(size=(40, 32)).astype(np.float32)
    markers = [3, 9, 17, 30]
    np.testing.assert_array_equal(run.score_hidden(h, markers, sc), export_npu.score_hidden(h, markers, sc))
    emb = np.random.default_rng(2).normal(size=(50, 32)).astype(np.float16)
    te = np.random.default_rng(3).normal(size=(3, 32)).astype(np.float32)
    for a, b in zip(run.npu_inputs([5, 7, 9], 2, 16, emb, te), export_npu.npu_inputs([5, 7, 9], 2, 16, emb, te)):
        np.testing.assert_array_equal(a, b)


def test_scorer_matches_torch():
    torch = pytest.importorskip("torch")
    d = 32
    sc = _scorer(d)
    m = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, d), torch.nn.GELU(), torch.nn.Linear(d, 1))
    with torch.no_grad():
        m[0].weight.copy_(torch.from_numpy(sc["ln_w"])); m[0].bias.copy_(torch.from_numpy(sc["ln_b"]))
        m[1].weight.copy_(torch.from_numpy(sc["w1"])); m[1].bias.copy_(torch.from_numpy(sc["b1"]))
        m[3].weight.copy_(torch.from_numpy(sc["w2"])); m[3].bias.copy_(torch.from_numpy(sc["b2"]))
    h = np.random.default_rng(4).normal(size=(20, d)).astype(np.float32)
    markers = [1, 4, 11]
    ref = m(torch.from_numpy(h[markers])).squeeze(-1).detach().numpy()
    np.testing.assert_allclose(export_npu.score_hidden(h, markers, sc), ref, atol=1e-5)


def test_bucket_for():
    assert export_npu.bucket_for(100) == 128
    assert export_npu.bucket_for(128) == 128
    assert export_npu.bucket_for(129) == 256
    with pytest.raises(ValueError):
        export_npu.bucket_for(513)
