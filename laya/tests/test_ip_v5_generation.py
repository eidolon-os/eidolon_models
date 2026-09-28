"""Structural rejection tests; test fixtures never become training examples."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]/'train/scenarios/ip-team-v5'
spec=importlib.util.spec_from_file_location('ip_v5_generator',ROOT/'generate_glm.py')
gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)


def test_action_annotation_never_inferred_from_slice():
    assert gen.moves('respond:b,respond:a','ab')==['respond:a','respond:b']
    with pytest.raises(ValueError):gen.moves('a','ab')
    with pytest.raises(ValueError):gen.moves('respond:c','ab')
    with pytest.raises(ValueError):gen.moves('wait,wait','ab')


def test_flat_array_normalization_does_not_lose_duplicate_fields():
    assert gen.parse('[{"c1_user":"x"},{"c2_user":"y"}]')=={'c1_user':'x','c2_user':'y'}
    with pytest.raises(ValueError):gen.parse('[{"c1_user":"x"},{"c1_user":"y"}]')


def test_no_silent_change_or_retry_of_paid_call(tmp_path):
    path=tmp_path/'calls';path.mkdir()
    (path/'old.json').write_text(json.dumps({'status':'pending'}))
    class NeverCalled:
        def complete_with_meta(self,*args,**kwargs):raise AssertionError('network must not be called')
    with pytest.raises(ValueError,match='pending'):
        gen.paid(NeverCalled(),tmp_path,'new','no credentials',24)


def test_checkpoint_metric_rejects_invalid_or_missing_dev(tmp_path):
    from eidolon_laya_train.train import train
    with pytest.raises(ValueError,match='checkpoint_metric'):
        train({'checkpoint_metric':'made_up'},tmp_path,tmp_path/'out')
    with pytest.raises(ValueError,match='validation'):
        train({'checkpoint_metric':'accuracy'},tmp_path,tmp_path/'out')
