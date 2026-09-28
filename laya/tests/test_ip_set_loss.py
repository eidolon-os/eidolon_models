"""The set objective must not invent a uniform preference inside the gold set."""
import math
import pytest
import torch
from eidolon_laya_train.train import question_loss


def loss(probs, target, **kwargs):
    return question_loss(torch.tensor([probs]).log(),torch.tensor([target]),torch.tensor([[True]*len(probs)]),torch.tensor([0]),brier_weight=kwargs.pop('brier_weight',0.),proper_weight=0.,**kwargs)[0]


def test_set_mass_not_internal_distribution():
    a=loss([.8,.1,.1],[.5,.5,0.],loss_mode='acceptable_set')
    b=loss([.45,.45,.1],[.5,.5,0.],loss_mode='acceptable_set')
    assert a.item()==pytest.approx(-math.log(.9))
    assert a.item()==pytest.approx(b.item())
    assert loss([.8,.1,.1],[.5,.5,0.]).item()>loss([.45,.45,.1],[.5,.5,0.]).item()


def test_single_gold_ce_equivalence_and_mask():
    assert loss([.8,.1,.1],[1.,0.,0.],loss_mode='acceptable_set').item()==pytest.approx(loss([.8,.1,.1],[1.,0.,0.]).item())
    z=torch.tensor([[1.,2.,100.]],requires_grad=True)
    value,_=question_loss(z,torch.tensor([[1.,0.,0.]]),torch.tensor([[True,True,False]]),torch.tensor([0]),brier_weight=.5,proper_weight=0.,loss_mode='acceptable_set')
    value.backward()
    assert z.grad[0,0]<0 and z.grad[0,1]>0 and z.grad[0,2]==0


def test_illegal_empty_set_rejected():
    with pytest.raises(ValueError,match='nonempty'):
        loss([.5,.5],[0.,0.],loss_mode='acceptable_set')
