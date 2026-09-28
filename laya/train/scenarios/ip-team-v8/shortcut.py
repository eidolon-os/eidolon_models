"""Train-only user-length centroids: a diagnostic for synthetic format shortcuts.

No gold is changed. Test labels are never read. A strong score suggests that
semantic benchmarks may also be solved using a superficial generation artifact.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args()
    read=lambda split:[json.loads(x) for x in (a.dataset/'named'/f'{split}.jsonl').read_text().splitlines()]
    train=[r for r in read('train') if r['meta']['variant']==0]
    val=[r for r in read('val') if r['meta']['variant']==0]
    lengths=defaultdict(list)
    for r in train:lengths[r['meta']['group']].append(len(r['state']['user_request']))
    centers={g:float(np.mean(v)) for g,v in lengths.items()}
    predictions=[{'id':r['id'],'gold':r['meta']['group'],'pred':min(sorted(centers),key=lambda g:abs(len(r['state']['user_request'])-centers[g]))} for r in val]
    result={'method':'nearest training mean user character length, original variants only, no semantics',
            'train_mean_chars':centers,'train_range_chars':{g:[min(v),max(v)] for g,v in lengths.items()},
            'val_correct':sum(r['gold']==r['pred'] for r in predictions),'val_n':len(predictions),'predictions':predictions}
    (a.dataset/'length-shortcut.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='predictions'},ensure_ascii=False))


if __name__=='__main__':main()
