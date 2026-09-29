"""校验一份单句 authored 文件（WRITING-single.md）：每条金标都要是该户型的合法选项。

    .venv/bin/python train/scenarios/smart-home-continuation/tools/check_single.py train/data/authored/smart-home-c3/b1.jsonl
"""
import collections
import sys
from pathlib import Path

from eidolon_laya_train.generators import _import_authored_cases
from eidolon_laya_train.scenario import Scenario

LAYA = Path(__file__).resolve().parents[4]
scenario = Scenario.load(LAYA / "train/scenarios/smart-home")
cfg = {"path": str(Path(sys.argv[1]).resolve()),
       "homes": [str(LAYA / "evals/smart-home"), str(LAYA / "train/data/homes/smart-home")]}
recs = list(_import_authored_cases(scenario, cfg))
print(f"OK {len(recs)} records", dict(collections.Counter(r.tags[0] for r in recs)))
