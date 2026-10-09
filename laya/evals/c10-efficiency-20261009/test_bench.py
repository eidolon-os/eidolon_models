"""Safety properties of the bounded experimental scheduler, without an NPU."""
import importlib.util
from pathlib import Path
import random

spec = importlib.util.spec_from_file_location('efficiency_bench', Path(__file__).with_name('bench.py'))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_scheduler_preserves_items_and_never_worsens_estimated_critical_path():
    from eidolon_models_laya.backends import RknnBackend
    be = object.__new__(RknnBackend)
    be._placement = {1: (128,256,384,512), 2: (128,256)}
    be._buckets = [128,256,384,512]
    costs = {'1':{'128':123,'256':277,'384':441,'512':643},'2':{'128':117,'256':271}}
    rng = random.Random(20261009)
    for _ in range(100):
        lengths = [rng.randint(1,512) for _ in range(rng.randint(1,8))]
        optimized = bench.optimal(be,lengths,costs)
        baseline = be.schedule(lengths)
        assert sorted(i for queue in optimized.values() for i,_ in queue)==list(range(len(lengths)))
        for core, queue in optimized.items():
            for i,bucket in queue:
                assert bucket in be._placement[core] and bucket >= lengths[i]
        def critical(schedule):
            return max(sum(costs[str(c)][str(b)] for _,b in queue) for c,queue in schedule.items())
        assert critical(optimized)<=critical(baseline)


def test_long_context_moves_short_question_to_long_core():
    from eidolon_models_laya.backends import RknnBackend
    be=object.__new__(RknnBackend)
    be._placement={1:(128,256,384,512),2:(128,256)}
    be._buckets=[128,256,384,512]
    costs={'1':{'128':123,'256':277,'384':441,'512':643},'2':{'128':117,'256':271}}
    result=bench.optimal(be,[93,241,135,392,225],costs)
    assert set(result[1])=={(0,128),(3,512)}
    assert set(result[2])=={(1,256),(2,256),(4,256)}
