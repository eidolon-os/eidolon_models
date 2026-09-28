"""Report three-state families without treating their snapshots as independent."""
import copy
import importlib.util
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('v7_metrics',HERE.parent/'ip-team-v7/evaluate.py')
v7=importlib.util.module_from_spec(s);s.loader.exec_module(v7)


def metrics(records,rows):
    snapshots=copy.deepcopy(records)
    for r in snapshots:r.meta['family']=r.meta['snapshot']
    result=v7.metrics(snapshots,rows)
    # Base diagnostics use snapshot-level variants; family bootstrap below uses
    # complete correlated three-state families as the resampling unit.
    result['snapshots']=result.pop('families')
    result['equivariant_snapshots']=result.pop('exact_equivariant_families')
    result['both_variants_correct_snapshots']=result.pop('all_variants_correct_families')
    byid={r.id:r for r in records};families=defaultdict(list);groups=defaultdict(list)
    for p in rows:
        meta=byid[p['record_id']].meta
        families[meta['family']].append(p);groups[meta['group']].append(p)
    family_scores=np.array([np.mean([p['correct'] for p in ps]) for _,ps in sorted(families.items())])
    rng=np.random.default_rng(71)
    boots=[np.mean(rng.choice(family_scores,len(family_scores),replace=True)) for _ in range(2000)]
    result.update(families=len(families),family_accuracy=float(np.mean(family_scores)),
                  all_six_correct_families=sum(all(p['correct'] for p in ps) for ps in families.values()),
                  all_three_original_correct_families=sum(all(p['correct'] for p in ps if byid[p['record_id']].meta.get('variant',0)==0) for ps in families.values()),
                  action_groups={g:{'n':len(ps),'correct':sum(p['correct'] for p in ps),'accuracy':float(np.mean([p['correct'] for p in ps]))} for g,ps in groups.items()},
                  exploratory_family_bootstrap_95=np.quantile(boots,[.025,.975]).tolist(),
                  interval_caveat='Three-state family resampling only; teacher/template dependence remains; no release confidence bound.')
    result['action_macro_accuracy']=float(np.mean([g['accuracy'] for g in result['action_groups'].values()]))
    result['worst_action_accuracy']=min(g['accuracy'] for g in result['action_groups'].values())
    return result
