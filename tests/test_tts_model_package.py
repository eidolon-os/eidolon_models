"""The committed synthesis package, held to the service and to its upstream pin.

These weights used to be a hand-placed copy on the board: the unit started
because somebody had once run `scp`, and a new board came up mute. They are Git
content now, so what can go wrong has moved — the tree can drift from what the
engine loads, or from the upstream revision it claims to be. Both are checked
here, and neither needs an NPU.

The bytes are committed once, here. They were briefly also declared as an Ops
artifact pointing at the publishing repository, so a Host could re-fetch and
re-check them without going through Git — a second 1.5 GB copy on the board
proving what these tests already prove. That declaration is gone; the property
it defended is kept below, by holding this manifest's own provenance to
something a machine can resolve.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from eidolon_models_asr.artifacts import verify_artifacts
from eidolon_models_tts.config import Settings

_ROOT = Path(__file__).resolve().parents[1]
_PKG = _ROOT / "tts" / "cosyvoice2-rk3588" / "2026-07-21"
_MODEL = _PKG / "model"
_MANIFEST = _PKG / "manifest.json"

def _manifest() -> dict:
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


def test_the_committed_package_matches_its_own_manifest() -> None:
    """Every file, byte for byte, against the digest recorded beside it.

    Reuses the ASR verifier rather than a second one: the manifest schema is
    the same, so a package that verifies there verifies here.
    """

    result = verify_artifacts(_MANIFEST, _MODEL)

    assert result["ok"] is True
    assert result["model_id"] == "Sariel00/cosyvoice2_rknn"
    assert result["revision"] == "47e9a3ea6724b0a65f8c433c77281258ac393f5a"
    assert "qwen2_body_w8a8_c2_ctx2048.rkllm" in result["checked_files"]
    assert "flow_estimator_cache200_fp16.rknn" in result["checked_files"]
    assert len(result["checked_files"]) == 24


def test_the_manifest_covers_the_tree_and_nothing_else() -> None:
    """A file on disk that no digest covers is a file nothing would notice
    changing, and a digest with no file is a manifest that verifies vacuously."""

    recorded = set(json.loads(_MANIFEST.read_text(encoding="utf-8"))["files"])
    on_disk = {str(path.relative_to(_MODEL)) for path in _MODEL.rglob("*") if path.is_file()}

    assert recorded == on_disk


def test_the_package_holds_everything_the_engine_refuses_to_start_without() -> None:
    """``required_paths()`` is the service's own list, so this asks the service
    rather than restating it. The engine itself is built on the Host and is not
    part of the package, so it is the one entry excluded."""

    settings = Settings(model_root=_MODEL, engine=Path("/nonexistent/engine"))

    missing = [
        path for path in settings.required_paths() if path != settings.engine and not path.is_file()
    ]

    assert missing == [], f"the committed package is missing: {missing}"


def test_the_upstream_pin_resolves_to_a_fetchable_location() -> None:
    """What the removed Ops declaration was for, kept without the second copy.

    It carried the same bytes from the publishing repository so they could be
    re-fetched and re-checked outside Git. The checking is what mattered, and it
    does not need the fetching: the manifest names the repository, an immutable
    revision, and each file's path within it, so a URL for any file here is
    composable — by a person, or by whoever declares this as an artifact later.

    Asserted rather than assumed, because a `file_map` whose entries did not
    compose would leave the provenance readable and unusable.
    """

    manifest = _manifest()
    source = manifest["source"]

    assert source["git_url"].startswith("https://huggingface.co/")
    assert len(source["revision"]) == 40, "a branch can move under a digest"
    assert source["url"].rstrip("/") in source["git_url"]

    base = source["git_url"].removesuffix(".git")
    for name, upstream in source["file_map"].items():
        url = f"{base}/resolve/{source['revision']}/{upstream}"
        assert url.startswith(f"{base}/resolve/{source['revision']}/")
        assert not upstream.startswith("/"), name
        assert ".." not in upstream, name
        assert name in manifest["files"], name


def test_no_file_here_is_without_an_upstream_to_point_at() -> None:
    """This used to allow exactly two exceptions, recorded in an
    `unpublished_files` section: the conversion run's own `manifest.json` in the
    text-frontend and voice roots. Recording a gap honestly was right; having
    one was not.

    They were carried because `required_paths()` demanded them, and the engine
    reads neither — `TextFrontend`'s constructor opens five files and no
    manifest. Dropping that requirement dropped the exception, and adding a
    voice became a copy from the pinned revision instead of a conversion run.

    So the assertion is now the stronger one: every file is verifiable against
    a published digest, with no exception list to grow."""

    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))

    assert "unpublished_files" not in manifest["source"]
    assert set(manifest["source"]["file_map"]) == set(manifest["files"])


def test_this_component_declares_no_artifact_for_the_synthesis_weights() -> None:
    """They ride the release as Git content, and declaring them as well put a
    second 1.5 GB copy on the board proving what these tests already prove.

    The declaration also had to aim at a non-servable path, because
    `install-component-artifact` rmtree's its destination and the declaration
    could not carry the complete set. Both halves of that are gone: the two
    files it could not carry turned out to be ones the engine never reads.
    """

    contract = tomllib.loads((_ROOT / "ops" / "component.toml").read_text(encoding="utf-8"))
    declared = {entry["id"] for entry in contract.get("artifacts", ())}

    assert not {name for name in declared if name.startswith("cosyvoice2")}
    # The chat model's weights are a different case and stay declared: unsloth
    # publishes that gguf and Git does not hold it.
    assert "qwen3-1.7b-q4_0" in declared


def test_the_launcher_points_at_the_committed_package() -> None:
    """The path is spelled in a shell script, so nothing but a test connects it
    to the directory that actually exists. The ASR service learned this the
    expensive way: a model root that resolved to the wrong place restarted it
    114 times."""

    launcher = (_ROOT / "scripts" / "eidolon-tts").read_text(encoding="utf-8")
    relative = _MODEL.relative_to(_ROOT)

    assert f"$REPO_ROOT/{relative}" in launcher
    assert "/var/lib/eidolon/models/cosyvoice2}" not in launcher


def test_the_unit_leaves_the_model_root_to_the_launcher() -> None:
    """Two places naming one path is how one of them goes stale — and the unit
    is the copy an operator does not rebuild when the package version changes."""

    unit = (_ROOT / "deploy" / "systemd" / "eidolon-tts.service").read_text(encoding="utf-8")

    assert "Environment=EIDOLON_TTS_MODEL_ROOT=" not in unit
    assert "Environment=EIDOLON_TTS_ENGINE=" in unit
