"""Freeze a dev-only representation choice before sealed test inference."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--compact',type=Path,required=True);p.add_argument('--named',type=Path,required=True);p.add_argument('--base',type=Path,required=True);a=p.parse_args()
dest=a.dataset/'selection.json'
if dest.exists():raise ValueError('selection already frozen')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
candidates=[]
for mode,run in [('compact',a.compact),('named',a.named)]:
 s=json.loads((run/'val/summary.json').read_text())
 candidates.append({'projection':mode,'run':str(run.resolve()),'dev_macro':s['slice_macro_accuracy'],'dev_wrong_speech':s['must_stop']['wrong_speech'],'dev_summary_sha256':digest(run/'val/summary.json')})
chosen=max(candidates,key=lambda c:(c['dev_macro'],-c['dev_wrong_speech'],c['projection']=='compact'))
checkpoint=Path(chosen['run'])/'checkpoint'
selected={**chosen,'candidates':candidates,'selection_rule':'highest dev slice macro, ties fewer must-stop wrong responses, then compact','test_not_used_for_selection':True,
 'dataset_lineage_sha256':digest(a.dataset/'lineage.json'),'sealed_test_sha256':digest(a.dataset/'sealed'/chosen['projection']/'test.jsonl'),
 'allowed_test_checkpoints':{str(x.resolve()):digest(x/'model.safetensors') for x in [checkpoint,a.base]}}
dest.write_text(json.dumps(selected,indent=2));print(json.dumps(selected,indent=2))
