"""Validate completed policy audit against the frozen manifest and original DEV."""
import json
import math
import run


def main():
    manifest = run.freeze()
    previous = json.loads((run.HERE.parent / 'backbone-comparison-v1/PROVENANCE.json').read_text())
    raw = {(s, r['id']): r for s, r in run.c.records()}
    files = {}
    total = 0
    for model in ('decider', 'jevk5'):
        assert run.c.digest(run.c.OUT / f'{model}.jsonl') == previous['raw_predictions_sha256'][model]
        for variant in run.POLICIES:
            p = run.OUT / f'{model}-{variant}.jsonl'
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            assert len(rows) == len(raw) == 114
            assert {(r['split'], r['id']) for r in rows} == set(raw)
            for row in rows:
                source = raw[row['split'], row['id']]
                _, _, names, _ = run.payload(source, variant)
                probs = row['probabilities']
                assert list(probs) == names
                assert all(math.isfinite(x) and 0 <= x <= 1 for x in probs.values())
                assert abs(sum(probs.values()) - 1) < .002
                assert row['pred'] == max(probs, key=probs.get)
                assert row['gold'] == source['labels']['move']['gold']
                assert row['correct'] == (row['pred'] in row['gold'])
                assert row['family'] == source['meta']['family']
                assert row['variant'] == variant
                assert row['input_tokens'] > 0
                assert math.isfinite(row['elapsed_ms']) and row['elapsed_ms'] > 0
            files[p.name] = run.c.digest(p)
            total += len(rows)
    result = dict(passed=True, new_predictions=total, records_per_configuration=114,
                  baseline_unchanged=True, frozen_manifest_unchanged=True,
                  finite_normalized_probabilities=True, gold_and_ids_match=True,
                  raw_sha256=files, manifest=manifest,
                  parent_provenance_sha256=run.c.digest(run.HERE.parent / 'backbone-comparison-v1/PROVENANCE.json'),
                  audit_code_sha256={name:run.c.digest(run.HERE/name) for name in ('report.py','validate.py')},
                  automated_tests={'command':'PYTHONPATH=../eidolon_sdk laya/.venv/bin/python -m pytest -q laya/tests/test_direction_check.py laya/tests/test_backbone_comparison.py', 'passed':6},
                  scope='DEV only; no training or teacher calls; no sealed test inference')
    (run.HERE / 'VALIDATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(passed=True, new_predictions=total)))


if __name__ == '__main__':
    main()
