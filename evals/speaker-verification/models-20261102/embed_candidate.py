"""Embed the 3D-Speaker test set with a candidate speaker model, as Channel embeds a turn.

The utterances and their voiced speech are those of the delivered model's
``embeddings.json`` (voice activity does not depend on the speaker model);
only the vector is recomputed, from the utterance's PCM unchanged, through
ModelScope's own speaker-verification pipeline for the candidate.

    eidolon_channel/.venv/bin/python -I \\
        eidolon_models/evals/speaker-verification/models-20261102/embed_candidate.py \\
        --corpus /Users/manson/ai/data/speaker-corpora/3d-speaker \\
        --reference /Users/manson/ai/data/speaker-corpora/3d-speaker/embeddings.json \\
        --model-dir <ModelScope model directory> --out <embeddings file>
"""

from __future__ import annotations

import argparse
import json
import os
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_PIPELINE = None


def _start(model_dir, threads, device):
    global _PIPELINE
    import torch
    from modelscope.pipelines import pipeline

    torch.set_num_threads(threads)
    _PIPELINE = pipeline(task="speaker-verification", model=str(model_dir))
    if device != "cpu":  # the same vectors (cosine 1.000000 against CPU), faster
        _PIPELINE.model.device = torch.device(device)
        _PIPELINE.model.embedding_model.to(device)


def _embed(job):
    out = {}
    for utt, path, speech_ms in job:
        with wave.open(path) as audio:
            assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) \
                == (16000, 1, 2)
            pcm = audio.readframes(audio.getnframes())
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        embedding = _PIPELINE([samples], output_emb=True)["embs"][0]
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        assert norm > 0 and np.isfinite(vector).all(), utt
        out[utt] = {"vector": [round(float(v), 6) for v in vector / norm], "speech_ms": speech_ms}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="the delivered model's embeddings.json")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--device", default="cpu", help="cpu, or mps with --jobs 1")
    args = parser.parse_args()
    files = args.corpus / "files-x" / "files"
    wav = dict(line.split(maxsplit=1) for line in (files / "test_wav.scp").read_text().splitlines())
    reference = json.loads(args.reference.read_text())
    items = sorted((u, str(args.corpus / "test-x" / wav[u].strip()), e["speech_ms"])
                   for u, e in reference.items())
    chunks = [items[i:i + 200] for i in range(0, len(items), 200)]
    threads = max(1, (os.cpu_count() or 1) // args.jobs)
    done = {}
    start = (args.model_dir.resolve(), threads, args.device)
    with ProcessPoolExecutor(args.jobs, initializer=_start, initargs=start) as pool:
        for n, part in enumerate(pool.map(_embed, chunks), 1):
            done.update(part)
            print(f"embedded {n}/{len(chunks)}", flush=True)
    args.out.write_text(json.dumps(done))
    print(json.dumps({"utterances": len(items), "embedded": len(done),
                      "dim": len(next(iter(done.values()))["vector"])}))


if __name__ == "__main__":
    main()
