"""Export an additional bucket without touching the released model directory."""
import argparse
from pathlib import Path
from eidolon_models_laya.artifacts import Manifest
from eidolon_models_laya.export_npu import export_npu
p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
(a.out/'torch').symlink_to((a.source/'torch').resolve(), target_is_directory=True)
(a.out/'manifest.json').write_bytes((a.source/'manifest.json').read_bytes())
export_npu(Manifest.load(a.out), buckets=(160,))
