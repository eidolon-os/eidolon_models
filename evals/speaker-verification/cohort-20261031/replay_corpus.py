"""Replay public speakers through this computer's speaker (design §8.2, 1c-a).

The phone, next to the speaker, records through the app's own capture path
("声纹环境校准"), so the embeddings are of public voices as the household's
phone channel hears them. Speakers play one block each (their eight recordings
back to back, short gaps), with silence between blocks; every block's
wall-clock span is logged so captures can be assigned to a single speaker
and captures straddling two speakers dropped.

    python3 -I eidolon_models/evals/speaker-verification/cohort-20261031/replay_corpus.py \\
        --corpus .eidolon/speaker-cohort-20261031 --log .eidolon/speaker-cohort-20261031/replay/<batch>.jsonl

Uses macOS ``afplay``. Nothing is recorded here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

PER_SPEAKER = 8
GAP_S = 0.6
BETWEEN_SPEAKERS_S = 3.0
LEVEL_RMS = 0.06


def corpus(root: Path) -> dict[str, list[Path]]:
    record = json.loads((root / "assets.json").read_text())
    speakers: dict[str, list[Path]] = {}
    for row in record["samples"]:
        if row.get("seconds", row.get("duration_s", 0)) < 4.0:
            continue
        files = speakers.setdefault(row["speaker"], [])
        if len(files) < PER_SPEAKER and (root / row["path"]).is_file():
            files.append(root / row["path"])
    return {s: f for s, f in sorted(speakers.items()) if len(f) == PER_SPEAKER}


def block(paths: list[Path]) -> np.ndarray:
    parts = []
    for path in paths:
        with wave.open(str(path)) as audio:
            x = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float64)
        gain = LEVEL_RMS * 32768 / max(float(np.sqrt(np.mean(x * x))), 1.0)
        gain = min(gain, 0.9 * 32767 / max(float(np.abs(x).max()), 1.0))
        parts += [x * gain, np.zeros(int(GAP_S * 16000))]
    return np.clip(np.concatenate(parts), -32768, 32767).astype("<i2")


def write_wav(path: Path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--speakers", nargs="*", help="only these speakers (default: all)")
    parser.add_argument("--volume", type=float, default=1.0, help="afplay volume (0-1+)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    speakers = corpus(args.corpus)
    chosen = [s for s in speakers if not args.speakers or s in args.speakers]
    total = sum(len(block(speakers[s])) for s in chosen) / 16000 + BETWEEN_SPEAKERS_S * len(chosen)
    print(f"{len(chosen)} speakers, about {total / 60:.0f} minutes", flush=True)
    if args.dry_run:
        return
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, open(args.log, "a") as log:
        for n, speaker in enumerate(chosen, 1):
            path = Path(tmp) / f"{speaker}.wav"
            write_wav(path, block(speakers[speaker]))
            start = time.time()
            subprocess.run(["afplay", "-v", str(args.volume), str(path)], check=True)
            log.write(json.dumps({"speaker": speaker, "start": start, "end": time.time()}) + "\n")
            log.flush()
            print(f"{n}/{len(chosen)} {speaker}", flush=True)
            time.sleep(BETWEEN_SPEAKERS_S)


if __name__ == "__main__":
    main()
