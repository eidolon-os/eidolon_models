"""Fetch pinned public research inputs; never reads product state or uploads audio."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
import wave
from pathlib import Path

HF_REV = "0743f301363dec56491a490f6d6cbc9d67f9a3bf"
DATA_REV = "bbe295d530192a4cd41644b711c9aecd087df653"
CODE_REV = "c5bbe0b76e37df698c403f8844e41304ceab6307"
SPEAKERS = [f"S{n:04d}" for n in (2, 3, 4, 5, 10, 11, 12, 13, 18, 19, 20, 21)]


def get(url):
    return urllib.request.urlopen(url, timeout=60)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    root = parser.parse_args().output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / "assets.json"
    manifest = json.loads(record_path.read_text()) if record_path.exists() else {
        "models": [], "code": [], "samples": [], "dataset_revision": DATA_REV,
        "code_revision": CODE_REV, "export_revision": HF_REV,
        "dataset": "AISHELL/AISHELL-1", "real_capture_sessions_known": False,
        "training_overlap_excluded": False, "production_ready": False,
    }

    def save():
        record_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        record_path.chmod(0o600)

    def fetch(url, relative, category, expected=None):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with get(url) as response:
                data = response.read(120_000_001)
            if len(data) > 120_000_000:
                raise ValueError("asset exceeded 120 MB research limit")
            if expected and digest(data) != expected:
                raise ValueError("upstream asset checksum mismatch")
            target.write_bytes(data)
        data = target.read_bytes()
        if expected and digest(data) != expected:
            raise ValueError("cached checksum mismatch")
        row = dict(path=relative, source=url, sha256=digest(data), bytes=len(data))
        manifest[category] = [x for x in manifest[category] if x["path"] != relative] + [row]
        save()
        print(relative, len(data), flush=True)

    tree_url = f"https://huggingface.co/api/models/csukuangfj/speaker-embedding-models/tree/{HF_REV}"
    tree = json.load(get(tree_url))
    for name in ("3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
                 "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"):
        meta = next(x for x in tree if x["path"] == name)
        fetch(f"https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/{HF_REV}/{name}",
              "weights/" + name, "models", meta.get("lfs", {}).get("oid"))

    release = json.load(get("https://api.github.com/repos/PalabraAI/redimnet2/releases/tags/v1.0.0"))
    for name in ("b2-vox2-lm.pt", "b3-vb2+vox2+cnc2_v0-lm.pt"):
        asset = next(x for x in release["assets"] if x["name"] == name)
        expected = (asset.get("digest") or "").removeprefix("sha256:") or None
        fetch(asset["browser_download_url"], "weights/" + name, "models", expected)

    tree = json.load(get(f"https://api.github.com/repos/PalabraAI/redimnet2/git/trees/{CODE_REV}?recursive=1"))
    for item in tree["tree"]:
        path = item["path"]
        if (path.startswith("redimnet2/") and path.endswith(".py")) or path in ("LICENSE", "README.md"):
            fetch(f"https://raw.githubusercontent.com/PalabraAI/redimnet2/{CODE_REV}/{path}",
                  "redimnet-source/" + path, "code")
    fetch(f"https://huggingface.co/datasets/AISHELL/AISHELL-1/raw/{DATA_REV}/README.md",
          "corpus/README.md", "code")

    for speaker in SPEAKERS:
        if sum(x["speaker"] == speaker for x in manifest["samples"]) == 12:
            continue
        url = f"https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/{DATA_REV}/data_aishell/wav/{speaker}.tar.gz"
        rows = []
        # Only read the beginning of each archive; no 15 GB corpus download and no tar extraction.
        with get(url) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                if member.size > 3_000_000:
                    continue
                data = archive.extractfile(member).read()
                with wave.open(io.BytesIO(data)) as audio:
                    duration = audio.getnframes() / audio.getframerate()
                    if audio.getframerate() != 16000 or audio.getnchannels() != 1 or duration < 3.2:
                        continue
                relative = "corpus/" + speaker + "/" + Path(member.name).name
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(0o600)
                rows.append(dict(id=target.stem, speaker=speaker, path=relative,
                                 source=url, archive_member=member.name,
                                 sha256=digest(data), duration_s=duration,
                                 capture_session=None, device="aishell-high-fidelity-mic"))
                if len(rows) == 12:
                    break
        if len(rows) != 12:
            raise ValueError(f"not enough eligible recordings for {speaker}")
        manifest["samples"] = [x for x in manifest["samples"] if x["speaker"] != speaker] + rows
        save()
        print(speaker, "12 independent files; recording sessions unknown", flush=True)


if __name__ == "__main__":
    main()
