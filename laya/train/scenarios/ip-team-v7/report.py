"""Collect frozen run outputs without running inference or choosing a model."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYA = HERE.parents[2]
DATA = LAYA / 'train/data/generated/ip-team-v7/generalization1'
RUNS = LAYA / 'train/runs'


def read(path):
    return json.loads(path.read_text())


def main():
    selection = read(DATA / 'selection.json')
    names = {
        'compact': 'ip-ensemble-v7-compact-r5',
        'named': 'ip-ensemble-v7-named-r6',
        'compact_base': 'ip-ensemble-v7-compact-zero',
        'named_base': 'ip-ensemble-v7-named-zero',
        'selected_test': 'ip-ensemble-v7-selected-test',
        'base_test': 'ip-ensemble-v7-base-test',
    }
    result = {'selection': selection, 'dataset': read(DATA / 'lineage.json'), 'runs': {}}
    for label, name in names.items():
        root = RUNS / name
        entry = {'path': str(root), 'lineage': read(root / 'lineage.json')}
        for split in ('train', 'val', 'test'):
            path = root / split / 'summary.json'
            if path.exists():
                entry[split] = read(path)
        if (root / 'checkpoint/train_summary.json').exists():
            entry['training'] = read(root / 'checkpoint/train_summary.json')
        result['runs'][label] = entry
    # Paired disagreement counts are descriptive; variants are not independent.
    chosen = read(RUNS / names['selected_test'] / 'test/predictions.json')
    base = {p['record_id']: p for p in read(RUNS / names['base_test'] / 'test/predictions.json')}
    result['paired_test'] = {
        'fixed': sum(p['correct'] and not base[p['record_id']]['correct'] for p in chosen),
        'broken': sum(not p['correct'] and base[p['record_id']]['correct'] for p in chosen),
        'both_correct': sum(p['correct'] and base[p['record_id']]['correct'] for p in chosen),
        'both_wrong': sum(not p['correct'] and not base[p['record_id']]['correct'] for p in chosen),
    }
    (HERE / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    files = sorted(HERE.glob('*.py')) + [
        LAYA / 'src/eidolon_models_laya/participation_text.py',
        LAYA / 'src/eidolon_laya_train/train.py',
        LAYA / 'tests/test_ip_v7.py',
    ]
    (HERE / 'code-manifest.json').write_text(json.dumps({
        str(p.relative_to(LAYA)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }, indent=2))
    for label, entry in result['runs'].items():
        for split in ('train', 'val', 'test'):
            if split in entry:
                s = entry[split]
                print(label, split, json.dumps({k: s[k] for k in (
                    'correct', 'records', 'slice_macro_accuracy', 'must_stop',
                    'unique_member', 'exact_equivariant_families', 'confidence'
                )}))


if __name__ == '__main__':
    main()
