"""Embed the 3D-Speaker test set with Channel's own extractor (``embeddings.json`` beside the corpus).

Each utterance as captured: the extractor measures voiced speech at the phone
app's level and embeds the audio unchanged. Utterances under 600 ms of voiced
speech are skipped, as the runtime rejects them.

    eidolon_channel/.venv/bin/python -I \\
        eidolon_models/evals/speaker-verification/identity-20261101/embed_corpus.py \\
        --corpus /Users/manson/ai/data/speaker-corpora/3d-speaker \\
        --model-dir eidolon_models/speaker_verification/campplus_zh_16k_common/624dcb0f
"""

from __future__ import annotations

import argparse
import json
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eidolon.livekit.plugins.speaker_verification.reference_extractor import (
    InvalidSpeakerAudio,
    ReferenceSpeakerExtractor,
)

MIN_SPEECH_MS = 600
_EXTRACTOR = None


def _start(model_dir):
    global _EXTRACTOR
    _EXTRACTOR = ReferenceSpeakerExtractor(model_dir)
    _EXTRACTOR.warm_up()


def _embed(job):
    out = {}
    for utt, path in job:
        with wave.open(path) as audio:
            assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
            pcm = audio.readframes(audio.getnframes())
        try:
            sample = _EXTRACTOR.extract(pcm, MIN_SPEECH_MS)
        except InvalidSpeakerAudio:
            continue
        out[utt] = {"vector": [round(float(v), 6) for v in sample.vector], "speech_ms": sample.speech_ms}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out", type=Path, help="default: embeddings.json beside the corpus")
    args = parser.parse_args()
    files = args.corpus / "files-x" / "files"
    wav = dict(line.split(maxsplit=1) for line in (files / "test_wav.scp").read_text().splitlines())
    items = sorted((u, str(args.corpus / "test-x" / p.strip())) for u, p in wav.items())
    chunks = [items[i:i + 200] for i in range(0, len(items), 200)]
    done = {}
    with ProcessPoolExecutor(args.jobs, initializer=_start, initargs=(args.model_dir,)) as pool:
        for n, part in enumerate(pool.map(_embed, chunks), 1):
            done.update(part)
            print(f"embedded {n}/{len(chunks)}", flush=True)
    (args.out or args.corpus / "embeddings.json").write_text(json.dumps(done))
    print(json.dumps({"utterances": len(items), "embedded": len(done),
                      "speech_ms_median": float(np.median([e["speech_ms"] for e in done.values()]))}))


if __name__ == "__main__":
    main()
