import json
from analyze_audit import collect

def test_join_uses_id_and_keeps_cancellation_unknown(tmp_path):
    rows=[dict(source='agent',event='decision_validated',session_id='s',decision_id='b',timestamp='2'),
          dict(source='gateway',event='result_returned',session_id='s',decision_id='a',timestamp='1',result={'proposal':{'action':'respond'}}),
          dict(source='agent',event='decision_not_applied',session_id='s',decision_id='a',timestamp='3')]
    (tmp_path/'a.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n{')
    result=collect(tmp_path)
    a,b=result['decisions']
    assert a['decision_id']=='a' and not a['applied'] and a['application_status']=='not_applied'
    assert a['reply'] is None and b['applied'] and b['result'] is None
    assert result['malformed_lines']==['a.jsonl:4']
