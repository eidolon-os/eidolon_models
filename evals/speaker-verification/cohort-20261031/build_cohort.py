"""Embed public speakers exactly as the Host embeds registration and probes.

Every recording is passed through two simulated channels, then through
Channel's own ReferenceSpeakerExtractor (``extract_enrollment``: the
utterance plus the same windows a registration gets):

- ``phone``: the registration path; device-like level, Opus 32 kbps.
- ``device``: a far-field board; 150–4000 Hz band, a synthetic room
  impulse response (exponentially decaying noise, RT60 0.3–0.6 s), light
  background noise, lower level, Opus 16 kbps.

Speakers are split once, by a hash of their id, into the cohort pool
(which may ship with the model) and a held-out set used only for
evaluation. Codec, level and channel noise are seeded per recording, so a
rebuild is identical.

    eidolon_channel/.venv/bin/python -I \
        eidolon_models/evals/speaker-verification/cohort-20261031/build_cohort.py \
        --model-dir .eidolon/mac-product/state/models/<campplus artifact> \
        --corpus .eidolon/speaker-cohort-20261031 \
        --out .eidolon/speaker-cohort-20261031/build

Output (audio never leaves .eidolon/): ``embeddings.json`` with the model
identity, the split and every speaker's samples per channel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from scipy import signal

from eidolon.livekit.plugins.speaker_verification.reference_extractor import (
    InvalidSpeakerAudio,
    ReferenceSpeakerExtractor,
)

PER_SPEAKER = 8
COHORT_SHARE = 0.7
# AISHELL is recorded quiet (RMS ~0.015); device microphones run AGC. The
# extractor's voiced-speech measure depends on level (it saturates near 0.06).
PHONE = {"levels": (0.04, 0.06, 0.08), "bitrate": 32000}
DEVICE = {"levels": (0.02, 0.04, 0.06), "bitrate": 16000, "band": (150, 4000),
          "rt60": (0.3, 0.6), "snr_db": (20, 30)}


def corpus(roots: list[Path]) -> dict[str, list[Path]]:
    speakers: dict[str, list[Path]] = {}
    for root in roots:
        record = json.loads((root / "assets.json").read_text())
        for row in record["samples"]:
            if row.get("seconds", row.get("duration_s", 0)) < 4.0:
                continue
            files = speakers.setdefault(row["speaker"], [])
            if len(files) < PER_SPEAKER and (root / row["path"]).is_file():
                files.append(root / row["path"])
    return {s: f for s, f in sorted(speakers.items()) if len(f) == PER_SPEAKER}


def split(speakers) -> tuple[list[str], list[str]]:
    ordered = sorted(speakers, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    cut = round(len(ordered) * COHORT_SHARE)
    return sorted(ordered[:cut]), sorted(ordered[cut:])


def read_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path)) as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
        return np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")


def level(samples: np.ndarray, rms: float) -> np.ndarray:
    """Scale to the target RMS without letting the peak pass 90% of full scale."""
    x = samples.astype(np.float64)
    gain = rms * 32768 / max(float(np.sqrt(np.mean(x * x))), 1.0)
    gain = min(gain, 0.9 * 32767 / max(float(np.abs(x).max()), 1.0))
    return np.clip(x * gain, -32768, 32767).astype("<i2")


def far_field(samples: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = samples.astype(np.float64)
    low, high = DEVICE["band"]
    x = signal.sosfilt(signal.butter(2, [low, high], btype="band", fs=16000, output="sos"), x)
    rt60 = rng.uniform(*DEVICE["rt60"])
    n = int(rt60 * 16000)
    rir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)  # -60 dB at rt60
    rir[0] = 4.0 * np.abs(rir).max()  # keep a direct path
    wet = signal.fftconvolve(x, rir / np.abs(rir).sum() * 4)[: len(x)]
    snr = rng.uniform(*DEVICE["snr_db"])
    wet += rng.standard_normal(len(x)) * np.sqrt(np.mean(wet * wet) / 10 ** (snr / 10))
    return np.clip(wet, -32768, 32767).astype("<i2")


def opus_round_trip(pcm: np.ndarray, bitrate: int) -> np.ndarray:
    """16 kHz PCM16 -> Opus 48 kHz 20 ms frames -> 16 kHz PCM16, aligned."""
    up = av.AudioResampler(format="s16", layout="mono", rate=48000)
    down = av.AudioResampler(format="s16", layout="mono", rate=16000)
    encoder = av.CodecContext.create("libopus", "w")
    encoder.sample_rate, encoder.layout, encoder.format = 48000, "mono", "s16"
    encoder.bit_rate = bitrate
    encoder.time_base = Fraction(1, 48000)
    decoder = av.CodecContext.create("libopus", "r")
    decoder.sample_rate, decoder.layout = 48000, "mono"
    frame = av.AudioFrame.from_ndarray(pcm[None, :], format="s16", layout="mono")
    frame.sample_rate = 16000
    out = []
    for upsampled in up.resample(frame) + up.resample(None):
        for packet in encoder.encode(upsampled):
            for decoded in decoder.decode(packet):
                out += [f.to_ndarray()[0] for f in down.resample(decoded)]
    for packet in encoder.encode(None):
        for decoded in decoder.decode(packet):
            out += [f.to_ndarray()[0] for f in down.resample(decoded)]
    out += [f.to_ndarray()[0] for f in down.resample(None)]
    decoded = np.concatenate(out).astype("<i2")
    # Opus adds a fixed look-ahead delay; align by cross-correlation on a slice.
    probe = pcm[:16000].astype(np.float64)
    window = decoded[:16000 + 4000].astype(np.float64)
    lag = int(np.argmax(np.correlate(window, probe, mode="valid")))
    aligned = decoded[lag:lag + len(pcm)]
    return np.pad(aligned, (0, len(pcm) - len(aligned)))


def channel(raw: np.ndarray, name: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if name == "phone":
        return opus_round_trip(level(raw, rng.choice(PHONE["levels"])), PHONE["bitrate"])
    return opus_round_trip(level(far_field(raw, rng), rng.choice(DEVICE["levels"])), DEVICE["bitrate"])


_EXTRACTOR = None


def _start(model_dir):
    global _EXTRACTOR
    _EXTRACTOR = ReferenceSpeakerExtractor(model_dir)
    _EXTRACTOR.warm_up()


def _speaker(job):
    speaker, paths = job
    rows = []
    for path in paths:
        raw = read_pcm(path)
        for name in ("phone", "device"):
            seed = int(hashlib.sha256(f"{path.name}:{name}".encode()).hexdigest()[:8], 16)
            try:
                sample = _EXTRACTOR.extract_enrollment(channel(raw, name, seed).tobytes(), 1500)
            except InvalidSpeakerAudio:
                continue
            rows.append(dict(file=path.name, channel=name, sample=sample.model_dump(mode="json")))
    return speaker, rows, _EXTRACTOR.model_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    speakers = corpus(args.corpus)
    cohort_ids, held_ids = split(speakers)
    print(f"{len(speakers)} speakers: {len(cohort_ids)} cohort, {len(held_ids)} held out", flush=True)
    embedded, identities = {}, set()
    jobs = [(s, speakers[s]) for s in sorted(speakers)]
    with ProcessPoolExecutor(args.jobs, initializer=_start, initargs=(args.model_dir,)) as pool:
        for speaker, rows, identity in pool.map(_speaker, jobs):
            embedded[speaker] = rows
            identities.add(identity)
            print(speaker, len(rows), flush=True)
    (identity,) = identities
    (args.out / "embeddings.json").write_text(json.dumps({
        "dataset": "AISHELL/AISHELL-1",
        "dataset_revision": "bbe295d530192a4cd41644b711c9aecd087df653",
        "identity": list(identity), "phone": PHONE, "device": DEVICE,
        "cohort": cohort_ids, "held_out": held_ids, "speakers": embedded,
    }))


if __name__ == "__main__":
    main()
