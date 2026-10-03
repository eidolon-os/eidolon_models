"""Append grouped intent contrasts without moving or removing any old record."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from intent_pairs import HERE, SPLIT_HOMES, build, dev_homes, load_home

from eidolon_laya_train.records import read_jsonl, target_vector, write_jsonl


def prepare(base: Path, out: Path, composition=False):
    builder = build
    if composition:
        from composition_pairs import build as builder
    out.mkdir(parents=True, exist_ok=False)
    for name in ('data', 'dataset', 'baseline-new'):
        shutil.copytree(base / name, out / name)
    old = {s: list(read_jsonl(base / 'dataset' / f'{s}.jsonl')) for s in ('train', 'val', 'calib')}
    blocked = {r.state['utterance'] for rows in old.values() for r in rows}
    laya = HERE.parents[2]
    locked = [*laya.glob('evals/**/cases.jsonl'), *laya.glob('evals/smart-home-continuation/*.jsonl'),
              *HERE.parent.glob('smart-home/eval/*.jsonl'), *HERE.glob('eval/*.jsonl'),
              *base.glob('data/*accept*.jsonl'),
              *base.glob('data/*dev*.jsonl'), *base.glob('data/mined*.jsonl')]
    for path in locked:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            text = row.get('state', {}).get('utterance', row.get('text'))
            if text:
                blocked.add(text)
    pool = builder({h: load_home(h) for h in SPLIT_HOMES['train']})
    added = {s: [] for s in old}
    for r in pool:
        if r.state['utterance'] in blocked:
            continue
        slot = int(hashlib.sha256(r.meta['target_name'].encode()).hexdigest()[:8], 16) % 10
        split = 'val' if slot == 0 else 'calib' if slot == 1 else 'train'
        added[split].append(r)
    dev = builder(dev_homes(), novel=True)
    assert not {r.state['utterance'] for r in dev} & blocked
    for split in old:
        for r in added[split]:
            for q, label in r.labels.items():
                target_vector(r.questions[q], label)
        write_jsonl(out / 'dataset' / f'{split}.jsonl', old[split] + added[split])
        write_jsonl(out / 'data' / f'intent-{split}.jsonl', added[split])
    prior_dev = list(read_jsonl(base / 'data/intent-dev.jsonl')) if composition else []
    write_jsonl(out / 'data/intent-dev.jsonl', prior_dev + dev)
    for a, b in [('train', 'val'), ('train', 'calib'), ('val', 'calib')]:
        assert not {r.state['utterance'] for r in added[a]} & {r.state['utterance'] for r in added[b]}
    assert not {r.state['utterance'] for r in dev} & {r.state['utterance'] for v in added.values() for r in v}
    audit = {'base': str(base), 'old_records_unchanged': {s: len(v) for s, v in old.items()},
             'added': {s: len(v) for s, v in added.items()}, 'pool': len(pool), 'dev': len(dev),
             'excluded_overlap': len(pool)-sum(map(len, added.values())),
             'split_group': 'visible device name across all homes and templates',
             'new_utterance_overlap_across_splits_and_dev': 0,
             'sha256': {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(out.rglob('*.jsonl'))}}
    shutil.copy2(base / 'dataset/manifest.json', out / 'dataset/parent-manifest.json')
    (out / 'dataset/manifest.json').write_text(json.dumps({
        'stage': 'append-intent-contrasts', 'audit': '../intent-data-audit.json',
        'parent_manifest': 'parent-manifest.json',
        'splits': {s: len(old[s]) + len(added[s]) for s in old},
        'sha256': {s: audit['sha256'][f'dataset/{s}.jsonl'] for s in old},
    }, indent=2)+'\n')
    (out / 'intent-data-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--composition', action='store_true')
    args = ap.parse_args()
    prepare(args.base, args.out, args.composition)
