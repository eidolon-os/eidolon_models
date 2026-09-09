"""The engine's build inputs, held to what the board actually links against.

The engine is compiled on the Host — it links the board's NPU runtime — so the
things that can make that build wrong are all in this repository, and this is
where they are checked. Nothing here needs an NPU.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

_CPP = Path(__file__).resolve().parents[1] / "src/eidolon_models_tts/cpp"

#: What the board's own copies hash to, compared byte for byte against the
#: headers running there. Recorded here as well as in vendor/README.md because
#: a prose table is not a check.
_VENDORED = {
    "rknn_api.h": "c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a",
    "rkllm.h": "80596a578f7f8e70df6eda1c2cbead3bfced14623a190258f2bd009a3d1f72cf",
}


def test_the_vendored_headers_are_the_ones_the_board_links_against() -> None:
    """An ABI mismatch does not fail the build, it makes the runtime behave
    differently — which is why this is pinned by digest and not by version."""

    for name, digest in _VENDORED.items():
        path = _CPP / "vendor" / name
        assert path.is_file(), f"{name} is not vendored"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


def test_the_headers_are_reachable_without_anything_hand_placed() -> None:
    """They used to live under /root on the board, which the `eidolon` user
    cannot read: a release failed on the real Host with `fatal error: rkllm.h:
    No such file or directory`. The build now defaults to the copy the source
    carries."""

    cmake = (_CPP / "CMakeLists.txt").read_text(encoding="utf-8")

    assert 'EIDOLON_TTS_RUNTIME_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/vendor"' in cmake
    assert "/root/npu" not in cmake
    assert "/root/rkllm" not in cmake


def test_changing_a_vendored_header_rebuilds_the_engine() -> None:
    """The build is skipped by a digest of its inputs. A header left out of
    that digest would mean an ABI change that never triggers a rebuild."""

    script = (
        Path(__file__).resolve().parents[1] / "scripts/eidolon-tts-build"
    ).read_text(encoding="utf-8")

    assert '"$SOURCE_DIR"/vendor/*.h' in script
    assert '"$SOURCE_DIR"/src/*.cpp' in script
    assert '"$SOURCE_DIR/CMakeLists.txt"' in script


def test_the_engine_source_the_release_carries_is_the_whole_build() -> None:
    """The pipeline textually includes two other translation units. A release
    that carried only the file named in CMakeLists would fail to compile on the
    Host — which is how this was found."""

    pipeline = (_CPP / "src/cosyvoice2_streaming_pipeline.cpp").read_text(encoding="utf-8")

    included = {
        line.split('"')[1]
        for line in pipeline.splitlines()
        if line.startswith('#include "') and line.rstrip().endswith('.cpp"')
    }
    assert included, "the pipeline no longer includes translation units textually"
    for name in included:
        assert (_CPP / "src" / name).is_file(), f"{name} is included but not carried"


def test_a_changed_source_does_not_reuse_the_previous_build_tree() -> None:
    """CMake caches every option in the tree, and the tree survives releases.

    `set(... CACHE ...)` does not override an existing entry, so a release that
    stopped passing an option kept the previous release's value: the vendored
    headers landed on the Host and the build still looked for them in
    /usr/include, because that is what the first release had cached. Found on
    the board, twice.
    """

    script = (
        Path(__file__).resolve().parents[1] / "scripts/eidolon-tts-build"
    ).read_text(encoding="utf-8")

    lines = script.splitlines()
    discard = next(i for i, line in enumerate(lines) if line.startswith('rm -rf "$BUILD_ROOT/build"'))
    configure = next(i for i, line in enumerate(lines) if line.startswith("cmake -S "))
    assert discard < configure, "the stale build tree must go before configuring"


def test_the_required_assets_are_the_ones_the_engine_opens() -> None:
    """Derived from the engine's own source, so the two cannot drift.

    `required_paths()` was written from a directory listing rather than from
    what the engine loads, and it asked for a `manifest.json` in the text
    frontend and voice roots — files nothing reads. That turned two
    conversion-run artifacts into a startup precondition: the package had to
    carry them, and they were its only files with no upstream to point at.

    Only the frontend's roots are compared. The RKNN and RKLLM paths are
    composed by the pipeline from positional arguments and shape numbers, which
    is not a literal this can read.
    """

    from eidolon_models_tts.config import Settings

    frontend = (_CPP / "src/cosyvoice2_text_frontend.cpp").read_text(encoding="utf-8")
    opened: dict[str, set[str]] = {"model_root": set(), "voice_profile_root": set()}
    for root in opened:
        for line in frontend.splitlines():
            if root + ' / "' not in line:
                continue
            opened[root].add(line.split(root + ' / "', 1)[1].split('"', 1)[0])

    assert opened["model_root"], "the frontend no longer names its files as literals"
    assert opened["voice_profile_root"]

    settings = Settings(model_root=Path("/m"), engine=Path("/e"))
    required = set(settings.required_paths())
    for name in opened["model_root"]:
        assert settings.text_frontend_root / name in required, name
    for name in opened["voice_profile_root"]:
        assert settings.voice_profile_root / name in required, name

    # And nothing extra asked of those two roots: an asset required but unread
    # is one the package must carry for no reason.
    for path in required:
        if path.parent in (settings.text_frontend_root, settings.voice_profile_root):
            root = (
                "model_root"
                if path.parent == settings.text_frontend_root
                else "voice_profile_root"
            )
            assert path.name in opened[root], f"{path.name} is required but never opened"


def test_every_carried_asset_has_an_upstream_to_point_at() -> None:
    """No file in the package without a line in `file_map`.

    The package briefly carried two conversion-run `manifest.json` files that
    upstream does not publish — recorded honestly in an `unpublished_files`
    section, which is the right way to record a gap and the wrong way to have
    one. They were required by `required_paths()` and read by nothing, so
    removing that requirement removed the gap. This is what keeps it removed:
    a new asset has to name where it came from.
    """

    import hashlib
    import json

    root = Path(__file__).resolve().parents[1] / "tts/cosyvoice2-rk3588/2026-07-21"
    if not root.is_dir():
        pytest.skip("the synthesis package is not checked out")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    carried = {
        str(path.relative_to(root / "model"))
        for path in (root / "model").rglob("*")
        if path.is_file()
    }
    assert carried == set(manifest["files"]), "a carried file is not in `files`"
    assert carried == set(manifest["source"]["file_map"]), (
        "a carried file has no upstream path in `file_map`"
    )
    assert "unpublished_files" not in manifest["source"], (
        "an asset with no upstream is back; requiring it is what to question first"
    )
    assert manifest["source"]["revision"], "the upstream revision is what makes this reproducible"

    # And the digests are the files, not a table beside them. Cheap: the large
    # ones are LFS pointers unless this checkout has them, and either way the
    # bytes on disk are what the release ships.
    for name, digest in manifest["files"].items():
        blob = (root / "model" / name).read_bytes()
        if blob.startswith(b"version https://git-lfs"):
            continue
        assert hashlib.sha256(blob).hexdigest() == digest, name
