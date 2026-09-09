"""The engine's build inputs, held to what the board actually links against.

The engine is compiled on the Host — it links the board's NPU runtime — so the
things that can make that build wrong are all in this repository, and this is
where they are checked. Nothing here needs an NPU.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

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
