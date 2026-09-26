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


def test_act_head_matches_torch():
    torch = pytest.importorskip("torch")
    d = 32
    rng = np.random.default_rng(5)
    act = {"w1": rng.normal(size=(16, d + 4)).astype(np.float32) / 6, "b1": rng.normal(size=16).astype(np.float32),
           "w2": rng.normal(size=(2, 16)).astype(np.float32) / 4, "b2": rng.normal(size=2).astype(np.float32)}
    head = torch.nn.Sequential(torch.nn.Linear(d + 4, 16), torch.nn.GELU(), torch.nn.Linear(16, 2))
    with torch.no_grad():
        head[0].weight.copy_(torch.from_numpy(act["w1"])); head[0].bias.copy_(torch.from_numpy(act["b1"]))
        head[2].weight.copy_(torch.from_numpy(act["w2"])); head[2].bias.copy_(torch.from_numpy(act["b2"]))
    pooled = rng.normal(size=d).astype(np.float32)
    for logits in (np.array([2.0, -1.0, 0.5], np.float32), np.array([0.3], np.float32)):
        # the model's features (vendor/laya/common.py DecisionModel.forward)
        p = torch.softmax(torch.from_numpy(logits)[None], -1)
        k = torch.tensor([max(len(logits), 2)], dtype=torch.float32)
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top = torch.cat([p, torch.zeros(1, 1)], -1).topk(2, -1).values
        feats = torch.stack([top[:, 0], top[:, 0] - top[:, 1], ent, k / 255.0], -1)
        ref = head(torch.cat([torch.from_numpy(pooled)[None], feats], -1))[0].detach().numpy()
        np.testing.assert_allclose(export_npu.act_from_logits(pooled, logits, act), ref, atol=1e-5)


class _FakeRuntime:
    """h = inputs_embeds @ W + type_vec: checkable without an NPU. Records which core ran what."""

    ran: list = []

    def __init__(self, verbose=False):
        self.core = self.path = None

    def load_rknn(self, path):
        self.path = path
        return 0

    def init_runtime(self, core_mask=None):
        self.core = core_mask
        return 0

    def inference(self, inputs):
        x, mask, tv = inputs
        _FakeRuntime.ran.append((self.core, x.shape[1]))
        return [x @ _W + tv]

    def release(self):
        pass


_W = np.random.default_rng(9).normal(size=(32, 32)).astype(np.float32) / 6


@pytest.fixture
def rknn_dir(tmp_path, monkeypatch):
    import sys
    import types

    api = types.SimpleNamespace(RKNNLite=_FakeRuntime)
    for i, name in enumerate(("NPU_CORE_0", "NPU_CORE_1", "NPU_CORE_2")):
        setattr(_FakeRuntime, name, i)
    monkeypatch.setitem(sys.modules, "rknnlite", types.SimpleNamespace(api=api))
    monkeypatch.setitem(sys.modules, "rknnlite.api", api)
    rng = np.random.default_rng(11)
    np.save(tmp_path / "tok_emb_fp16.npy", rng.normal(size=(100, 32)).astype(np.float16))
    np.save(tmp_path / "type_emb.npy", rng.normal(size=(3, 32)).astype(np.float32))
    np.savez(tmp_path / "scorer.npz", **_scorer(32))
    np.savez(tmp_path / "act_head.npz", w1=rng.normal(size=(8, 36)).astype(np.float32), b1=np.zeros(8, np.float32),
             w2=rng.normal(size=(2, 8)).astype(np.float32), b2=np.zeros(2, np.float32))
    for L in (128, 256, 384, 512):
        (tmp_path / f"hidden_l{L}.rknn").write_bytes(b"")
    _FakeRuntime.ran = []
    return tmp_path


def test_rknn_backend_places_buckets_and_matches_the_cpu_half(rknn_dir):
    from eidolon_models_laya.backends import RknnBackend
    from eidolon_models_laya.sequence import collate

    be = RknnBackend(rknn_dir)
    loaded = sorted((r.core, Path(r.path).name) for r in be._rt.values())
    assert len(loaded) == 7 and (0, "hidden_l512.rknn") not in loaded  # placement, not every bucket everywhere
    rng = np.random.default_rng(3)
    items = [{"ids": rng.integers(0, 100, n).tolist(), "markers": [1, 3, 5][:k], "qtype": q}
             for n, k, q in ((90, 3, 0), (300, 3, 0), (140, 2, 2))]
    batch = collate(items, 0)
    logits, act = be.forward(batch)
    emb = np.load(rknn_dir / "tok_emb_fp16.npy").astype(np.float32)
    te, sc, ah = np.load(rknn_dir / "type_emb.npy"), dict(np.load(rknn_dir / "scorer.npz")), dict(np.load(rknn_dir / "act_head.npz"))
    for i, it in enumerate(items):
        L = be._bucket(len(it["ids"]))
        x, _, tv = export_npu.npu_inputs(it["ids"], it["qtype"], L, emb, te)
        h = (x @ _W + tv)[0]
        ref = export_npu.score_hidden(h, it["markers"], sc)
        np.testing.assert_allclose(logits[i, : len(ref)], ref, rtol=1e-5, atol=1e-5)
        assert (logits[i, len(ref):] == -1e4).all()
        np.testing.assert_allclose(act[i], export_npu.act_from_logits(h[0], ref, ah), rtol=1e-4, atol=1e-4)
    assert {c for c, _ in _FakeRuntime.ran} == {0, 1, 2}  # three questions, three cores
    assert (1, 384) in _FakeRuntime.ran  # the long one went to the only core with its bucket


def test_rknn_backend_refuses_a_bucket_on_no_core(rknn_dir):
    from eidolon_models_laya.backends import RknnBackend

    with pytest.raises(ValueError, match="on no core"):
        RknnBackend(rknn_dir, placement={0: (128,), 1: (256,), 2: (256,)})
