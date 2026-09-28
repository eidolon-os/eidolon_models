"""Structural/identity regression fixtures, not training data."""
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]/'train/scenarios/ip-team-v6'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
gen=module('v6_generation',ROOT/'generate_pairs.py')
prep=module('v6_preparation',ROOT/'prepare_pairs.py')
shared=module('v6_shared',ROOT/'generate_shared.py')

def fixture():
 return {'name_a':'甲甲','name_b':'乙乙','name_c':'丙丙','role_a':'温和，乙乙的朋友','role_b':'机灵，甲甲的朋友','role_c':'安静，喜欢听故事',
         'user_x':'甲甲，请你回答。','user_y':'乙乙，请你回答。','peer_author_x':'','peer_author_y':'','peer_text_x':'','peer_text_y':'',
         'acceptable_x':'respond:a','acceptable_y':'respond:b','reason_x':'用户点名甲甲','reason_y':'用户点名乙乙'}

def test_single_field_change_and_same_permutation():
 d=gen.validate(json.dumps(fixture(),ensure_ascii=False),'named')
 row={'pair':d,'id':'fixture','slice':'named','split':'val','review':{'reason_x':'点名甲甲','reason_y':'点名乙乙'},'generator_sha256':'test-only'}
 x,y=[prep.expand_pair(row,b,0) for b in 'xy']
 assert x.state['candidates']==y.state['candidates']
 assert x.questions==y.questions
 assert x.meta['family']==y.meta['family'] and x.id!=y.id
 assert not set(x.labels['move']['gold']) & set(y.labels['move']['gold'])
 for r in (x,y):
  selected=r.labels['move']['gold'][0].split(':')[1]
  assert r.state['candidates'][selected]['name'] in r.state['user_request']


def test_changed_history_not_allowed_in_user_only_pair():
 d=fixture();d.update(peer_author_y='a',peer_text_y='另一段公开内容')
 with pytest.raises(ValueError,match='changed'):gen.validate(json.dumps(d),'named')


def test_shared_wire_materializes_only_shared_fields():
 d=fixture()
 wire={k:v for k,v in d.items() if k in shared.fields_for('user')}
 result=shared.materialize(json.dumps(wire),'named','user')
 assert result['peer_text_x']==result['peer_text_y']==''
 assert result['acceptable_x']==['respond:a']
 assert result['acceptable_y']==['respond:b']
 wire['peer_text']='unexpected peer'
 with pytest.raises(ValueError,match='fields mismatch'):
  shared.materialize(json.dumps(wire),'named','user')
