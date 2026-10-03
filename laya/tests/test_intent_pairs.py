"""Intent repair must preserve real commands containing negation and shopping words."""
import importlib
from pathlib import Path

import pytest

from eidolon_laya_train.records import target_vector


@pytest.mark.parametrize('variant', ['intent_pairs', 'composition_pairs'])
def test_contrasts_have_legal_labels_and_disjoint_development_utterances(monkeypatch, variant):
    tools = Path(__file__).resolve().parents[1] / 'train/scenarios/smart-home-continuation/tools'
    monkeypatch.syspath_prepend(str(tools))
    m = importlib.import_module('intent_pairs')
    build = importlib.import_module(variant).build
    homes = {h: m.load_home(h) for h in m.SPLIT_HOMES['train']}
    pool = build(homes)
    dev = build(m.dev_homes(), novel=True)
    assert not {r.state['utterance'] for r in pool} & {r.state['utterance'] for r in dev}
    for rows in (pool, dev):
        assert any(
            'negated-control' in r.tags and r.labels['intent']['gold'] == '控制' for r in rows
        )
        assert any('prohibition' in r.tags and r.labels['intent']['gold'] == '无关' for r in rows)
        for r in rows:
            for q, label in r.labels.items():
                target_vector(r.questions[q], label)
            if r.labels['intent']['gold'] == '无关':
                assert set(r.labels) == {'intent'}
