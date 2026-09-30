import copy
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('decomposition_run',Path(__file__).with_name('run.py'))
run=importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_all_frozen_inputs_exclude_annotation_and_private_metadata():
    for row,base in zip(run.read(run.DATA),run.read(run.BASE)):
        p=run.proposal(base['probabilities'])
        public=run.inputs(row,p)
        poisoned=copy.deepcopy(row)
        for key in set(poisoned)-{'state','questions'}:
            poisoned[key]={'PRIVATE_LABEL_POISON':'never send'}
        assert run.inputs(poisoned,p)==public
        assert all(set(x)=={'state','questions'} for x in public)
        assert public[1]['state']['proposed_responder']==p.removeprefix('respond:')


def test_routing_never_turns_a_stop_into_speech_in_repair():
    for original in ('wait','finish','abstain'):
        for timing in ('respond','wait','finish','unclear'):
            for evidence in ('supported','ambiguous','other'):
                repair,decomposed=run.combine(original,'respond:M3',timing,evidence)
                assert repair==original
                if timing!='respond':
                    assert decomposed==('abstain' if timing=='unclear' else timing)
                else:
                    assert decomposed==('respond:M3' if evidence=='supported' else 'abstain')


def test_proposal_is_model_ranking_not_gold_or_nonresponse():
    assert run.proposal({'wait':.8,'respond:M1':.05,'respond:M2':.15})=='respond:M2'
    assert run.combine('respond:M1','respond:M1','respond','other')==('abstain','abstain')
