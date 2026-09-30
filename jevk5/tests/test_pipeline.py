import copy
import torch

from eidolon_models_jevk5.data import payload
from eidolon_models_jevk5.engine import LowRankLinear,acceptable_set_loss
from eidolon_models_jevk5.evaluate import gate
from eidolon_models_jevk5.prepare import project


def test_gold_and_metadata_never_enter_input():
    r={'state':{'user':'hello'},'questions':{'move':{'type':'choice','instructions':'choose',
       'criteria':{'respond:M0':'reply','wait':'wait'}}},'labels':{'move':{'gold':['wait']}},'meta':{}}
    before=payload(r);poison=copy.deepcopy(r)
    poison['labels']={'secret':'DO NOT SEND'};poison['meta']={'reason':'PRIVATE GOLD'}
    assert payload(poison)==before


def test_acceptable_set_loss_and_adapter_gradients():
    logits=torch.tensor([1.,2.,3.],requires_grad=True)
    loss=acceptable_set_loss(logits,[0,2])
    assert torch.allclose(loss,-logits.softmax(-1)[[0,2]].sum().log())
    layer=torch.nn.Linear(5,3);layer.requires_grad_(False);adapter=LowRankLinear(layer,2,4)
    x=torch.randn(2,5)
    assert torch.equal(layer(x),adapter(x))
    adapter(x).square().sum().backward()
    assert layer.weight.grad is None
    assert adapter.b.grad.abs().sum()>0


def test_gate_rejects_missing_task_and_bad_abstention():
    metrics={'v7':{'correct':38},'v8':{'groups':{'abstain':{'n':24,'correct':0}}}}
    result=gate(metrics)
    assert not result['passed']
    assert set(result['failures'])=={'v8_abstain','v8_respond_unmeasured','v8_stop_unmeasured','companion_missing','ip_team_missing'}


def test_reordering_preserves_target_identity_and_history():
    scene={'id':'s0','cast':[{'name':'甲','persona':'quiet'},{'name':'乙','persona':'active'}],
           'user':'请接着说','before':[{'speaker':1,'text':'hello'}]}
    plan={'id':'family','task':'ip_team','split':'train'}
    variants=[project(scene,['respond:1'],plan,v) for v in (0,1)]
    for r in variants:
        slot=r['labels']['move']['gold'][0].split(':')[1]
        assert r['state']['candidates'][slot]['name']=='乙'
        assert r['state']['prior_public_messages'][0]['author']==slot
        assert r['meta']['slot_to_source'][slot]==1
    assert variants[0]['labels']!=variants[1]['labels']
    assert variants[0]['meta']['family']==variants[1]['meta']['family']
