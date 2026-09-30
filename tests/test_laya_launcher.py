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


def _fake_release(tmp_path, launcher_name):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    launcher = scripts / launcher_name
    shutil.copyfile(Path(__file__).parents[1] / "scripts" / launcher_name, launcher)
    binary = tmp_path / ".venv/bin/eidolon-laya"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nenv | grep -E '^(EIDOLON_LAYA_|OPENBLAS_)' | sort\n")
    binary.chmod(0o755)
    return launcher


def _run(launcher, env, *, check=True):
    base = {k: v for k, v in os.environ.items() if not k.startswith(("EIDOLON_", "OPENBLAS_"))}
    result = subprocess.run(["sh", str(launcher), "serve"], env=base | env, capture_output=True, text=True, check=check)
    return dict(line.split("=", 1) for line in result.stdout.splitlines()) if check else result


def test_home_launcher_keeps_npu_cores_1_and_2_and_one_blas_thread(tmp_path):
    launcher = _fake_release(tmp_path, "eidolon-laya")
    model = tmp_path / "model"
    (model / "npu").mkdir(parents=True)
    (model / "manifest.json").touch()
    (model / "npu/hidden_l512.rknn").touch()
    got = _run(launcher, {"EIDOLON_HOST_CAPABILITIES": "local_laya,rknpu2", "EIDOLON_LAYA_MODEL_DIR": str(model)})
    assert got["EIDOLON_LAYA_RKNN_PLACEMENT"] == "1:128,256,384,512|2:128,256"
    assert got["OPENBLAS_NUM_THREADS"] == "1"
    assert "EIDOLON_LAYA_ENABLE_PARTICIPATION" not in got


def _participation_model(tmp_path, *, profile=True):
    model = tmp_path / "participation"
    (model / "npu").mkdir(parents=True)
    (model / "onnx").mkdir()
    for name in ("manifest.json", "npu/hidden_l640.rknn", "onnx/model.onnx.data"):
        (model / name).touch()
    if profile:
        (model / "participation.json").touch()
    return model


def test_participation_launcher_inherits_none_of_the_home_services_settings(tmp_path):
    """host.env and cpu-allocation.env are shared by every unit: the home service's values must not leak."""
    launcher = _fake_release(tmp_path, "eidolon-laya-participation")
    model = _participation_model(tmp_path)
    home = {
        "EIDOLON_LAYA_MODEL_DIR": "/var/lib/eidolon/models/laya-smart-home-c4-rknn-7b695ba8",
        "EIDOLON_LAYA_RKNN_PLACEMENT": "1:128,256,384,512|2:128,256",
        "EIDOLON_LAYA_PORT": "8771", "EIDOLON_LAYA_API_KEY": "home-key", "EIDOLON_LAYA_DEVICE": "cpu",
        "EIDOLON_LAYA_SPECULATIVE": "1", "EIDOLON_LAYA_THREADS": "4",
    }
    got = _run(launcher, home | {
        "EIDOLON_HOST_CAPABILITIES": "local_laya,local_laya_participation,rknpu2",
        "EIDOLON_LAYA_PARTICIPATION_MODEL_DIR": str(model),
        "EIDOLON_LAYA_PARTICIPATION_RKNN_PLACEMENT": "0:384,512,640",
    })
    assert got["EIDOLON_LAYA_MODEL_DIR"] == str(model)
    assert got["EIDOLON_LAYA_BACKEND"] == "rknn"
    assert got["EIDOLON_LAYA_RKNN_LIBRARY"] == f"{model}/librknnrt.so"
    assert got["EIDOLON_LAYA_RKNN_PLACEMENT"] == "0:384,512,640"
    assert got["EIDOLON_LAYA_SPECULATIVE"] == "0"
    assert got["EIDOLON_LAYA_PORT"] == "8773" and got["EIDOLON_LAYA_HOST"] == "127.0.0.1"
    assert got["EIDOLON_LAYA_ENABLE_PARTICIPATION"] == "1" and got["EIDOLON_LAYA_MAX_PENDING"] == "1"
    assert got["OPENBLAS_NUM_THREADS"] == "1"
    for leaked in ("EIDOLON_LAYA_API_KEY", "EIDOLON_LAYA_DEVICE", "EIDOLON_LAYA_THREADS"):
        assert leaked not in got


def test_participation_launcher_picks_onnx_without_an_npu_and_needs_its_profile(tmp_path):
    launcher = _fake_release(tmp_path, "eidolon-laya-participation")
    model = _participation_model(tmp_path)
    got = _run(launcher, {"EIDOLON_HOST_CAPABILITIES": "local_laya_participation",
                          "EIDOLON_LAYA_PARTICIPATION_MODEL_DIR": str(model),
                          "EIDOLON_LAYA_PARTICIPATION_PORT": "8790"})
    assert got["EIDOLON_LAYA_BACKEND"] == "onnx" and got["EIDOLON_LAYA_PORT"] == "8790"
    assert "EIDOLON_LAYA_RKNN_PLACEMENT" not in got and "EIDOLON_LAYA_RKNN_LIBRARY" not in got
    bare = _participation_model(tmp_path / "bare", profile=False)
    result = _run(launcher, {"EIDOLON_LAYA_PARTICIPATION_MODEL_DIR": str(bare)}, check=False)
    assert result.returncode == 2 and "participation.json" in result.stderr
