"""The two Laya launchers hand every setting to the services file, and that file agrees with the contract.

A launcher only names its service and the file; it reads no ``EIDOLON_LAYA_*`` and sets none. The file
(``laya/deploy/services.toml``) is held here to what the component contract declares: the rknn model
directories are the pinned artifacts' install roots, the ports are the registered ports, and each
unit's launcher serves the table it is named for.
"""

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICES_FILE = REPO_ROOT / "laya" / "deploy" / "services.toml"
CONTRACT = tomllib.loads((REPO_ROOT / "ops" / "component.toml").read_text(encoding="utf-8"))
UNIT_SERVICE = {"eidolon-laya": "smart_home", "eidolon-laya-participation": "participation"}

sys.path.insert(0, str(REPO_ROOT / "laya" / "src"))
from eidolon_models_laya.config import Settings  # noqa: E402


def _fake_release(tmp_path: Path, launcher_name: str) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    launcher = scripts / launcher_name
    shutil.copyfile(REPO_ROOT / "scripts" / launcher_name, launcher)
    binary = tmp_path / ".venv/bin/eidolon-laya"
    binary.parent.mkdir(parents=True)
    binary.write_text('#!/bin/sh\necho "$@"\nenv | grep -E "^(EIDOLON_LAYA_|OPENBLAS_)" | sort\n')
    binary.chmod(0o755)
    return launcher


@pytest.mark.parametrize(("unit", "service"), sorted(UNIT_SERVICE.items()))
def test_a_launcher_names_its_service_and_the_file_and_nothing_else(tmp_path, unit, service):
    launcher = _fake_release(tmp_path, unit)
    base = {k: v for k, v in os.environ.items() if not k.startswith(("EIDOLON_", "OPENBLAS_"))}
    # A stray variable from an older Host must change nothing: none is read or passed on.
    env = base | {"EIDOLON_HOST_CAPABILITIES": "local_laya,rknpu2", "EIDOLON_LAYA_PORT": "9999"}
    out = subprocess.run(
        ["sh", str(launcher), "serve"], env=env, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert out[0] == f"--config {tmp_path}/laya/deploy/services.toml --service {service} serve"
    assert out[1:] == ["EIDOLON_LAYA_PORT=9999", "OPENBLAS_NUM_THREADS=1"]  # inherited, not used


def _artifact_roots(capability: str) -> set[str]:
    return {
        item["install_root"]
        for item in CONTRACT["artifacts"]
        if item.get("requires_capability") == capability and "-rknn-" in item["id"]
    }


@pytest.mark.parametrize(
    ("service", "capability", "port"),
    [("smart_home", "local_laya", "laya_api"), ("participation", "local_laya_participation", "laya_participation_api")],
)
def test_the_services_file_is_what_the_contract_pins(service, capability, port):
    npu = Settings.for_service(service, path=SERVICES_FILE, capabilities=frozenset({"rknpu2"}))
    assert {str(npu.model_dir)} == _artifact_roots(capability)
    assert npu.port == CONTRACT["ports"][port]["default"]
    unit = next(u for u in CONTRACT["units"] if u.get("requires_capability") == capability)
    assert UNIT_SERVICE[unit["id"]] == service and unit["exec"] == f"scripts/{unit['id']}"


@pytest.mark.parametrize("unit", sorted(UNIT_SERVICE))
def test_the_units_carry_no_laya_settings(unit):
    text = (REPO_ROOT / "deploy" / "systemd" / f"{unit}.service").read_text(encoding="utf-8")
    assert "EIDOLON_LAYA" not in text and "cpu-allocation" not in text
    assert "EnvironmentFile=/etc/eidolon/host.env" in text  # EIDOLON_HOST_CAPABILITIES picks the backend
