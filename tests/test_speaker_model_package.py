"""A release carries the same fixed CAM++ bytes that offline evaluation uses."""

import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'speaker_verification/campplus_zh_16k_common/624dcb0f'


def test_model_files_match_manifest_and_have_fixed_origin():
    manifest = json.loads((PACKAGE / 'manifest.json').read_text())
    assert manifest['source']['revision'] == '624dcb0f2e40153d7bbbbbcb02cb9f3518f225c3'
    assert manifest['runtime']['auto_download'] is False
    for item in manifest['artifacts']:
        path = PACKAGE / item['path']
        assert path.stat().st_size == item['size_bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']


def test_delivery_uses_only_exact_repository_files_and_includes_license():
    contract = tomllib.loads((ROOT / 'ops/component.toml').read_text())
    artifact = next(x for x in contract['artifacts'] if x.get('requires_capability') == 'local_speaker_verification')
    assert artifact['requires_capability'] == 'local_speaker_verification'
    digest = hashlib.sha256('\n'.join(f"{f['path']} {f['sha256']}" for f in sorted(artifact['files'], key=lambda f: f['path'])).encode()).hexdigest()
    assert artifact['install_root'] == '/var/lib/eidolon/models/campplus-zh-16k-' + digest[:8]
    assert artifact['id'] == Path(artifact['install_root']).name
    paths = {x['path'] for x in artifact['files']}
    assert paths >= {'campplus_cn_common.bin', 'config.yaml', 'configuration.json',
                     'manifest.json', 'LICENSE', 'README.md', 'upstream-README.md',
                     'identity-policy.json', 'identity-policy.md',
                     'identity-backend.json', 'identity-cohort.npz'}
    for item in artifact['files']:
        assert item['url'].startswith('repo://speaker_verification/campplus_zh_16k_common/624dcb0f/')
        path = ROOT / item['url'].removeprefix('repo://')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    assert 'Apache' in (PACKAGE / 'LICENSE').read_text()


def test_the_policy_and_calibration_name_the_delivered_weights_and_cohort():
    # Channel reads these by file name from voiceprint.model_dir. A declaration
    # that drops one turns recognition off on every Host that selects it,
    # silently (10-30: the recognition policy went missing this way).
    import numpy as np

    contract = tomllib.loads((ROOT / 'ops/component.toml').read_text())
    artifact = next(x for x in contract['artifacts'] if x.get('requires_capability') == 'local_speaker_verification')
    declared = {x['path']: x['sha256'] for x in artifact['files']}
    weights = declared['campplus_cn_common.bin']
    policy = json.loads((PACKAGE / 'identity-policy.json').read_text())
    backend = json.loads((PACKAGE / 'identity-backend.json').read_text())
    assert {'identity-policy.json', 'identity-backend.json', 'identity-cohort.npz'} <= declared.keys()
    assert policy['weights_hash'] == weights == backend['identity'][1]
    assert policy['model_id'] == backend['identity'][0] == 'campplus_zh_16k_common'
    assert policy['preprocessing_version'] == backend['identity'][2]
    assert policy['backend_version'] == backend['version']
    assert policy['min_speech_ms'] <= policy['min_unknown_speech_ms']
    uses = ('personal_memory', 'addressing', 'speaker_label', 'candidate_evidence', 'discovery')
    assert set(uses) <= policy.keys()  # a use is switched off explicitly, never by omission
    assert all(0.5 <= policy[u] < 1 for u in uses if policy[u] is not None)
    cohort = backend['cohort']
    assert declared[cohort['path']] == cohort['sha256']
    assert len(backend['weights']) == len(backend['features'])
    with np.load(PACKAGE / cohort['path'], allow_pickle=False) as npz:
        assert npz['cohort'].shape == (cohort['vectors'], npz['mean'].shape[0])
        assert backend['top_n'] < cohort['vectors']


def test_weights_are_not_duplicated_in_release_source_archive():
    attrs = (ROOT / '.gitattributes').read_text()
    assert 'speaker_verification/** export-ignore' in attrs
