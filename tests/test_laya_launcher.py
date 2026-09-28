"""Release selection and backpressure without loading model weights."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("explicit_limit,expected", [(None, "1"), ("2", "2")])
def test_npu_launcher_uses_bounded_admission(tmp_path, explicit_limit, expected):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    launcher = scripts / "eidolon-laya"
    shutil.copyfile(Path(__file__).parents[1] / "scripts/eidolon-laya", launcher)
    binary = tmp_path / ".venv/bin/eidolon-laya"
    binary.parent.mkdir(parents=True)
    binary.write_text(
        '#!/bin/sh\nprintf "%s %s" "$EIDOLON_LAYA_BACKEND" "$EIDOLON_LAYA_MAX_PENDING"\n'
    )
    binary.chmod(0o755)
    model = tmp_path / "model"
    (model / "npu").mkdir(parents=True)
    (model / "manifest.json").touch()
    (model / "npu/hidden_l512.rknn").touch()
    env = {k: v for k, v in os.environ.items() if not k.startswith("EIDOLON_LAYA_")}
    env.update(EIDOLON_HOST_CAPABILITIES="local_laya,rknpu2", EIDOLON_LAYA_MODEL_DIR=str(model))
    if explicit_limit is not None:
        env["EIDOLON_LAYA_MAX_PENDING"] = explicit_limit
    result = subprocess.run(
        ["sh", str(launcher), "serve"], env=env, capture_output=True, text=True, check=True
    )
    assert result.stdout == f"rknn {expected}"
