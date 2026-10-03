"""Board evaluation accounting must separate warmup and honor the caller's budget."""
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load(name):
    path = Path(__file__).resolve().parents[1] / 'deploy/rk3588' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_home_collision_without_participation_and_budget(tmp_path, monkeypatch):
    m = load('concurrency')
    item = {'id': 'a', 'body': {}, 'gold': {'follow': '关闭或停止'}}
    (tmp_path / 'home.jsonl').write_text(json.dumps(item)+'\n')
    barrier = threading.Barrier(2)

    def post(url, body, timeout):
        barrier.wait(timeout=2)
        return 200, {'answers': {'follow': {
            'choice': '关闭或停止', 'probabilities': {'关闭或停止': .99}}}}, None

    monkeypatch.setattr(m, 'post', post)
    args = SimpleNamespace(dir=tmp_path, scenario='home-collide', home_url='fake:home',
                           part_url='fake:part', limit=1, seed=7, pids='', gap=0,
                           pairs=3, gap_pairs=0)
    assert m.run(args) == 0
    rows = [json.loads(s) for s in (tmp_path / 'home-collide.jsonl').read_text().splitlines()]
    assert len(rows) == 6 and all(r['kind'] == 'home' and r['http'] == 200 for r in rows)
    row = rows[0] | {'ms': 900}
    assert m.home_outcome(row, item['gold'])['outcome'] == 'timeout'
    assert m.home_outcome(row, item['gold'], 1000) == {'outcome': 'decided', 'correct': True}
    assert m.report(SimpleNamespace(dir=tmp_path, home_budget_ms=1000, service_log=None)) == 0
    assert json.loads((tmp_path / 'summary.json').read_text())['budgets']['home_ms'] == 1000


def test_warmup_is_not_counted_as_evaluation(tmp_path, monkeypatch):
    m = load('laya_npu')
    calls = []

    class Runtime:
        NPU_CORE_0, NPU_CORE_1, NPU_CORE_2, NPU_CORE_0_1_2, NPU_CORE_AUTO = range(5)

        def __init__(self, **kwargs):
            pass

        def load_rknn(self, path):
            return 0

        def init_runtime(self, **kwargs):
            return 0

        def inference(self, inputs):
            calls.append(inputs)
            return [np.zeros((1, 128, 4), dtype=np.float32)]

        def release(self):
            pass

    monkeypatch.setitem(sys.modules, 'rknnlite.api', SimpleNamespace(RKNNLite=Runtime))
    monkeypatch.setattr(m, 'score_hidden', lambda h, markers, sc: np.array([1., 0.]))
    np.save(tmp_path / 'tok_emb_fp16.npy', np.zeros((10, 4), dtype=np.float16))
    np.save(tmp_path / 'type_emb.npy', np.zeros((3, 4), dtype=np.float32))
    np.savez(tmp_path / 'scorer.npz', ignored=np.zeros(1))
    (tmp_path / 'hidden_l128.rknn').touch()
    item = {'key': 'test/follow', 'ids': [0, 1], 'markers': [0, 1],
            'qtype': 0, 'ref_logits': [1., 0.]}
    (tmp_path / 'test.items.jsonl').write_text(json.dumps(item)+'\n')
    out = tmp_path / 'out'
    assert m.cmd_run(SimpleNamespace(npu_dir=tmp_path, items_dir=tmp_path,
                                     out_dir=out, core='2')) == 0
    summary = json.loads((out / 'run.json').read_text())
    assert len(calls) == 4
    assert len(summary['startup_by_bucket']['128']['warmup_ms']) == 3
    assert summary['sets']['test']['ran'] == 1
    assert summary['sets']['test']['npu_ms_by_bucket']['128']['n'] == 1
