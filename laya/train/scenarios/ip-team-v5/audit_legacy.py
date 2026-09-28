"""Read-only source and truncation audit; no old model evaluation or label tuning."""
import json
from collections import Counter
from pathlib import Path
from eidolon_laya_train.records import read_jsonl
from eidolon_models_laya.sequence import Tokenizer, render_options, to_internal, serialize_state
HERE=Path(__file__).resolve().parent
TRAIN=HERE.parents[1]
tok=Tokenizer(TRAIN/'runs/r14/checkpoint/tokenizer')
result={}
for name in ('ip-ensemble-v4-characters-r1','ip-ensemble-v4-characters-r3'):
    dataset=TRAIN/'runs'/name/'dataset'
    splits={}
    for split in ('train','val'):
        path=dataset/f'{split}.jsonl'
        if not path.exists():continue
        rows=list(read_jsonl(path)); defects=Counter(); sources=Counter(); families=set(); source_families={}
        for r in rows:
            families.add(r.meta.get('family'));sources[r.source]+=1
            source_families.setdefault(r.source,set()).add(r.meta.get('family'))
            q=to_internal(r.questions['move']); opts=render_options(q)
            sizes=[len(tok.encode(' '+o)) for o in opts]
            ins=len(tok.encode('choice question: '+q['ins']))
            if max(sizes)>48: defects['option_over_48']+=1
            if sum(min(s,48)+1 for s in sizes)+ins>256: defects['head_over_256']+=1
            if len(tok.encode(serialize_state(r.state)))+sum(s+1 for s in sizes)+ins+4>2048: defects['full_input_over_2048']+=1
        splits[split]={'records':len(rows),'families':len(families),'sources':dict(sources),'source_families':{k:len(v) for k,v in source_families.items()},'potential_truncation':dict(defects)}
    result[name]=splits
(HERE/'legacy-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
