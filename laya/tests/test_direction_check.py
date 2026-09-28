import copy
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]/'train/scenarios/direction-check-v1'


def load(name):
    s=importlib.util.spec_from_file_location('direction_'+name,HERE/f'{name}.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_only_instructions_change_and_gold_has_no_effect():
    m=load('run')
    for _,r in m.c.records():
        original=m.c.payload(r)
        for variant in ['A','B']:
            modified=m.payload(r,variant)
            assert original[0]==modified[0]
            assert original[2:]==modified[2:]
            assert modified[1]['instructions']==original[1]['instructions']+'\n'+m.POLICIES[variant]
            poisoned=copy.deepcopy(r);poisoned['labels']={'move':{'gold':['ABSTAIN_SECRET']}};poisoned['meta']={'reason':'DO_NOT_INCLUDE'}
            assert m.payload(poisoned,variant)==modified
        assert m.c.payload(r)==original


def test_high_total_does_not_hide_failed_abstention_or_v7_regression():
    m=load('report')
    summary={'v7':{'correct':36},'v8':{'must_speak':{'n':24,'correct':24},'must_stop':{'n':24,'correct':24,'wrong_speech':0},'must_abstain':{'n':24,'correct':16}}}
    assert m.gate(summary,36)=={'passed':False,'failures':['must_abstain']}
    summary['v8']['must_abstain']['correct']=17
    assert m.gate(summary,36)['passed']
    summary['v7']['correct']=35
    assert m.gate(summary,36)=={'passed':False,'failures':['v7_regression']}
