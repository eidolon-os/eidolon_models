"""Family metrics and test gating, with test-only artificial fixtures."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]/'train/scenarios/ip-team-v8'
def load(name):
    s=importlib.util.spec_from_file_location('v8_test_'+name,ROOT/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_family_metric_preserves_correlated_semantic_states():
    records=[];rows=[]
    for group,gold in [('respond','respond:M0'),('stop','wait'),('abstain','abstain')]:
        for variant in (0,1):
            rid=f'{group}-{variant}'
            records.append(SimpleNamespace(id=rid,meta={'family':'one','snapshot':group,'group':group,'variant':variant,'slice':group,'batch':'one','members':2,'slot_to_source':{'M0':0,'M1':1}}))
            rows.append({'record_id':rid,'gold':[gold],'pred':gold,'correct':True,'p_top':.9})
    m=load('evaluate').metrics(records,rows)
    assert m['families']==1 and m['snapshots']==3
    assert m['all_six_correct_families']==1 and m['equivariant_snapshots']==3
    assert m['all_three_original_correct_families']==1
    assert m['action_macro_accuracy']==1
    rows[0].update(pred='wait',correct=False)
    m=load('evaluate').metrics(records,rows)
    assert m['all_six_correct_families']==0 and m['both_variants_correct_snapshots']==2
    assert m['all_three_original_correct_families']==0


def test_dev_gate_does_not_allow_aggregate_score_to_hide_stop_errors():
    gate=load('gate')
    s={'action_groups':{g:{'accuracy':.9} for g in ('respond','stop','abstain')},
       'must_stop':{'n':20,'wrong_speech':3},'must_abstain':{'n':20,'correct':18}}
    regression={'records':42,'correct':27,'must_stop':{'n':6,'wrong_speech':4}}
    assert not gate.passes(s,regression)
    s['must_stop']['wrong_speech']=2
    assert gate.passes(s,regression)
    s['action_groups']['respond']['accuracy']=.69
    assert not gate.passes(s,regression)
    s['action_groups']['respond']['accuracy']=.9
    regression['correct']=26
    assert not gate.passes(s,regression)
