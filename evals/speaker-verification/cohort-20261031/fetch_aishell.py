"""Fetch the first eligible recordings of every AISHELL-1 speaker.

Reads only the beginning of each speaker archive (no 15 GB download, no tar
extraction to disk) and keeps the audio outside git. Speakers already fetched
are skipped, so the script can be rerun after an interruption.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import threading
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA_REV = "bbe295d530192a4cd41644b711c9aecd087df653"
BASE = f"https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/{DATA_REV}/data_aishell/wav/"
TREE = f"https://huggingface.co/api/datasets/AISHELL/AISHELL-1/tree/{DATA_REV}/data_aishell/wav"
PER_SPEAKER = 8
MIN_SECONDS = 4.0


def get(url):
    return urllib.request.urlopen(url, timeout=120)


def speakers() -> list[str]:
    names = [Path(x["path"]).name for x in json.load(get(TREE))]
    return sorted(n.removesuffix(".tar.gz") for n in names if n.endswith(".tar.gz"))


def fetch(root: Path, speaker: str) -> list[dict]:
    rows = []
    url = BASE + speaker + ".tar.gz"
    with get(url) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".wav") or member.size > 3_000_000:
                continue
            data = archive.extractfile(member).read()
            with wave.open(io.BytesIO(data)) as audio:
                seconds = audio.getnframes() / audio.getframerate()
                if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
                    continue
            if seconds < MIN_SECONDS:
                continue
            relative = f"corpus/{speaker}/{Path(member.name).name}"
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o600)
            rows.append(dict(speaker=speaker, path=relative, archive_member=member.name,
                             sha256=hashlib.sha256(data).hexdigest(), seconds=round(seconds, 3)))
            if len(rows) == PER_SPEAKER:
                break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--skip", nargs="*", default=[], help="speakers already on disk elsewhere")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    record = root / "assets.json"
    manifest = json.loads(record.read_text()) if record.exists() else {
        "dataset": "AISHELL/AISHELL-1", "dataset_revision": DATA_REV,
        "license": "apache-2.0 (dataset card)", "per_speaker": PER_SPEAKER,
        "min_seconds": MIN_SECONDS, "samples": [], "short": [],
    }
    lock = threading.Lock()
    done = {x["speaker"] for x in manifest["samples"]} | set(manifest["short"])
    todo = [s for s in speakers() if s not in done and s not in set(args.skip)]
    print(f"{len(todo)} speakers to fetch", flush=True)

    def one(speaker):
        try:
            rows = fetch(root, speaker)
        except Exception as exc:  # network hiccup: leave it for the next run
            print(speaker, "failed:", exc, flush=True)
            return
        with lock:
            if len(rows) < PER_SPEAKER:
                manifest["short"].append(speaker)
            manifest["samples"].extend(rows)
            record.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
            print(speaker, len(rows), flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, todo))


if __name__ == "__main__":
    main()
