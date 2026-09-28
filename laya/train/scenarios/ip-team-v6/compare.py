"""Summarize frozen-run results without selecting thresholds or checkpoints."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
TRAIN=HERE.parents[1]
RUNS={
 'base_zero':'ip-ensemble-v6-base-zero',
 'v5_r1_zero':'ip-ensemble-v6-v5r1-zero',
 'distribution_r3':'ip-ensemble-v6-distribution-r3',
 'set_r4':'ip-ensemble-v6-set-r4',
}

def main():
 comparison={};reference=None
 for name,dirname in RUNS.items():
  run=TRAIN/'runs'/dirname
  summary=json.loads((run/'eval/summary.json').read_text())
  rows=json.loads((run/'eval/val-predictions.json').read_text())
  signature=[(r['record_id'],r['gold']) for r in rows]
  if reference is None:reference=signature
  if reference!=signature:raise ValueError('different evaluation records/labels')
  member=[r for r in rows if len(r['gold'])==1 and r['gold'][0].startswith('respond:')]
  member_stats={'n':len(member),'correct':sum(r['correct'] for r in member),
                'wrong_member':sum(r['pred'].startswith('respond:') and not r['correct'] for r in member),
                'missed_speech':sum(not r['pred'].startswith('respond:') for r in member)}
  confusion={}
  for r in rows:
   if len(r['gold'])==1:
    key=r['gold'][0]+' -> '+r['pred'];confusion[key]=confusion.get(key,0)+1
  paired=json.loads((run/'eval/paired-summary.json').read_text())
  old=json.loads((run/'old-regression/summary.json').read_text())
  out={'run':str(run),'train':summary['train'],'val':summary['val'],'paired':paired,
       'unique_member':member_stats,'unique_gold_confusion':confusion,
       'sdk_replay':summary['sdk_replay'],'old_v5_val':old['val'],
       'all_constant_baselines':{k:sum(k in r['gold'] for r in rows) for k in ('respond:M0','respond:M1','respond:M2','wait','finish','abstain')}}
  hist=run/'checkpoint/train_summary.json'
  if hist.exists():out['training']=json.loads(hist.read_text())
  comparison[name]=out
 (HERE/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2))
 for k,v in comparison.items():
  print(json.dumps({'run':k,'train_accuracy':v['train']['acceptable_accuracy'],'val_accuracy':v['val']['acceptable_accuracy'],
                    'both_correct':v['paired']['both_correct'],'paired_variants':v['paired']['paired_variants'],
                    'unique_member':v['unique_member'],'silent_wrong_speech':v['val']['silent_wrong_speech'],
                    'threshold':v['val']['uncalibrated_threshold_0.8'],'slices':v['val']['slices'],
                    'old_v5_accuracy':v['old_v5_val']['acceptable_accuracy']},ensure_ascii=False))
if __name__=='__main__':main()
