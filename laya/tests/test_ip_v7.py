"""Test-only synthetic fixtures, excluded from all training/holdout artifacts."""
import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]/'train/scenarios/ip-team-v7'
def module(name,file):
 s=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prep=module('v7_prep_test','prepare.py');metrics=module('v7_metric_test','evaluate.py')

def row():
 return {'id':'test-only','split':'val','slice':'direct_address','batch':'test-only-batch','generator_sha256':'test-only','rename_safe':True,
  'scene':{'cast':[{'name':'甲甲','persona':'温和，乙乙的朋友'},{'name':'乙乙','persona':'机灵，甲甲的朋友'}],
           'user':'乙乙，这次你来讲。','before':[],'after':[],'gold':['respond:1'],'why':'用户点名'}}

def test_matched_projections_and_identity_mapping():
 for v in (0,1):
  plain=prep.expand(row(),v,{'丙丙','丁丁'},False);named=prep.expand(row(),v,{'丙丙','丁丁'},True)
  assert plain.state==named.state and plain.labels==named.labels
  gold=named.labels['move']['gold'][0];slot=gold.split(':')[1]
  assert named.meta['slot_to_source'][slot]==1
  name=named.state['candidates'][slot]['name']
  assert name in named.questions['move']['criteria'][gold]
  assert name in named.state['user_request']
  assert name not in plain.questions['move']['criteria'][gold]

def test_equivariance_compares_people_not_slots():
 records=[prep.expand(row(),v,{'丙丙','丁丁'},False) for v in (0,1)]
 preds=[{'record_id':r.id,'pred':r.labels['move']['gold'][0],'gold':r.labels['move']['gold'],'correct':True,'p_top':.9} for r in records]
 assert preds[0]['pred']!=preds[1]['pred']
 result=metrics.metrics(records,preds)
 assert result['exact_equivariant_families']==1 and result['all_variants_correct_families']==1
 assert result['unique_member']=={'n':2,'correct':2,'wrong_member':0}
 assert result['confidence']['0.8']['errors']==0
 preds[1].update(pred=preds[0]['pred'],correct=False)
 result=metrics.metrics(records,preds)
 assert result['exact_equivariant_families']==0
 assert result['all_variants_correct_families']==0
 assert result['confidence']['0.8']['errors']==1

def test_nickname_scene_does_not_rename_without_audit():
 d=row();d['rename_safe']=False
 r=prep.expand(d,1,{'丙丙','丁丁'},True)
 assert '乙乙' in r.state['user_request'] and not r.meta['renamed']
