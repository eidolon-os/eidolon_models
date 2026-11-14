"""A release carries the same fixed face models and score policy the offline checks used."""

import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "face_recognition/sface_yunet_opencv_zoo/48b10fa5"
CAPABILITY = "local_face_identity"


def _artifact() -> dict:
    contract = tomllib.loads((ROOT / "ops/component.toml").read_text())
    selected = [x for x in contract["artifacts"] if x.get("requires_capability") == CAPABILITY]
    # Ops resolves the capability's artifact, not a named one: a second entry is an invalid release.
    assert len(selected) == 1
    return selected[0]


def test_model_files_match_manifest_and_have_fixed_origin():
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    assert manifest["source"]["revision"] == "48b10fa505aeb51fbd127fc60a5c660040e823b7"
    assert manifest["runtime"]["auto_download"] is False
    for item in manifest["artifacts"]:
        path = PACKAGE / item["path"]
        assert path.stat().st_size == item["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_delivery_uses_only_exact_repository_files_and_includes_licenses():
    artifact = _artifact()
    assert artifact["id"] == "face-sface-yunet-48b10fa5"
    assert artifact["install_root"] == "/var/lib/eidolon/models/" + artifact["id"]
    assert "activation" not in artifact and "requires_capabilities" not in artifact
    paths = {x["path"] for x in artifact["files"]}
    assert paths == {
        "manifest.json", "face-policy.json",
        "face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx",
        "LICENSE-yunet", "LICENSE-sface", "README.md",
    }
    for item in artifact["files"]:
        assert item["url"] == f"repo://face_recognition/sface_yunet_opencv_zoo/48b10fa5/{item['path']}"
        assert hashlib.sha256((PACKAGE / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert "MIT" in (PACKAGE / "LICENSE-yunet").read_text()
    assert "Apache" in (PACKAGE / "LICENSE-sface").read_text()


def test_the_manifest_and_policy_name_the_delivered_weights():
    # Vision reads these by file name from one directory and refuses weights the manifest does
    # not name, so a declaration that drops or repins one turns face identity off on the Host.
    declared = {x["path"]: x["sha256"] for x in _artifact()["files"]}
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    for item in manifest["artifacts"]:
        assert declared[item["path"]] == item["sha256"]
    policy = json.loads((PACKAGE / "face-policy.json").read_text())
    assert policy["model_id"] == manifest["model_alias"]


def test_weights_are_not_duplicated_in_release_source_archive():
    attrs = (ROOT / ".gitattributes").read_text()
    assert "face_recognition/** export-ignore" in attrs
