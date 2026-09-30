"""Public-only decision inputs and group-safe data IO."""
import hashlib
import json
from pathlib import Path

from .vendor.jevk5.prompt import decision_options


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def payload(row):
    q = row['questions']['move']
    assert q['type'] == 'choice'
    options = decision_options(q)
    assert 2 <= len(options) <= 16
    # Match the established comparison's state representation exactly.
    return json.dumps(row['state'], ensure_ascii=False), q, options


def validate(row):
    _, _, options = payload(row)
    keys = [k for k, _ in options]
    gold = row['labels']['move']['gold']
    if not gold or len(set(gold)) != len(gold) or not set(gold) <= set(keys):
        raise ValueError('invalid acceptable action set')
    if not row['meta']['family'] or not row['id']:
        raise ValueError('missing lineage')
    return row


def group(gold):
    if all(g.startswith('respond:') for g in gold):
        return 'respond'
    if set(gold) <= {'wait', 'finish'}:
        return 'stop'
    if gold == ['abstain']:
        return 'abstain'
    return 'mixed'
