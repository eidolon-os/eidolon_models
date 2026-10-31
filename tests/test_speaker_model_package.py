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
                     'recognition-controlled-trial.json', 'recognition-policy.md'}
    for item in artifact['files']:
        assert item['url'].startswith('repo://speaker_verification/campplus_zh_16k_common/624dcb0f/')
        path = ROOT / item['url'].removeprefix('repo://')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    assert 'Apache' in (PACKAGE / 'LICENSE').read_text()


def test_delivered_policies_name_the_delivered_weights():
    # Channel reads both policies from voiceprint.model_dir by file name. A
    # declaration that drops one turns that feature off on every Host that
    # selects it, silently (10-30: the recognition policy went missing this way).
    contract = tomllib.loads((ROOT / 'ops/component.toml').read_text())
    artifact = next(x for x in contract['artifacts'] if x.get('requires_capability') == 'local_speaker_verification')
    declared = {x['path'] for x in artifact['files']}
    weights = next(x['sha256'] for x in artifact['files'] if x['path'] == 'campplus_cn_common.bin')
    for name in ('admission-controlled-trial.json', 'recognition-controlled-trial.json'):
        assert name in declared
        policy = json.loads((PACKAGE / name).read_text())
        assert policy['model_id'] == 'campplus_zh_16k_common'
        assert policy['weights_hash'] == weights
        assert 0 < policy['threshold'] < 1
    recognition = json.loads((PACKAGE / 'recognition-controlled-trial.json').read_text())
    assert recognition['version'].startswith('campplus-person-recognition-')
    assert recognition['min_speech_ms'] > 0


def test_weights_are_not_duplicated_in_release_source_archive():
    attrs = (ROOT / '.gitattributes').read_text()
    assert 'speaker_verification/** export-ignore' in attrs
