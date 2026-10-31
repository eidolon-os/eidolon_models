"""Embed public speakers exactly as the Host embeds a conversation probe.

Every recording goes through an Opus round trip (what LiveKit does to
microphone audio), then through Channel's own ReferenceSpeakerExtractor:
the whole utterance plus the same windows a registration gets. Speakers are
split once, by a hash of their id, into the cohort that ships with the model
and a held-out set that only the acceptance evaluation uses.

Run with the Channel environment (it owns the extractor):

    eidolon_channel/.venv/bin/python -I \
        eidolon_models/evals/speaker-verification/cohort-20261031/build_cohort.py \
        --model-dir .eidolon/mac-product/state/models/campplus-zh-16k-7959bae7 \
        --corpus .eidolon/speaker-cohort-20261031 \
        --corpus .eidolon/anonymous-speaker-eval-20261028 \
        --out .eidolon/speaker-cohort-20261031/build

Outputs (the corpus audio never leaves .eidolon/):
    impostor-cohort.npz    cohort speakers' vectors, voiced lengths, speaker index
    impostor-cohort.json   provenance and per-band counts
    held-out.json          held-out speakers' samples and windows, for evaluate.py
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

from eidolon.livekit.plugins.speaker_verification.reference_extractor import (
    InvalidSpeakerAudio,
    ReferenceSpeakerExtractor,
)

PER_SPEAKER = 8
COHORT_SHARE = 0.7
# Microphone audio reaches the Host as Opus at 48 kHz; bitrates span what
# WebRTC voice and the boards' encoders use.
CODEC_BITRATES = (24000, 32000)
# AISHELL is recorded quiet (RMS ~0.015); device microphones run AGC. The
# extractor's voiced-speech measure depends on level (it saturates near 0.06),
# so recordings are brought to device-like levels before the codec.
LEVELS_RMS = (0.04, 0.06, 0.08)


def corpus(roots: list[Path]) -> dict[str, list[Path]]:
    speakers: dict[str, list[Path]] = {}
    for root in roots:
        record = json.loads((root / "assets.json").read_text())
        for row in record["samples"]:
            seconds = row.get("seconds", row.get("duration_s", 0))
            if seconds < 4.0:
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


def level(pcm: np.ndarray, rms: float) -> np.ndarray:
    """Scale to the target RMS without letting the peak pass 90% of full scale."""
    samples = pcm.astype(np.float64)
    gain = rms * 32768 / max(float(np.sqrt(np.mean(samples * samples))), 1.0)
    gain = min(gain, 0.9 * 32767 / max(float(np.abs(samples).max()), 1.0))
    return np.clip(samples * gain, -32768, 32767).astype("<i2")


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


def embed(extractor, pcm: np.ndarray) -> dict | None:
    try:
        sample = extractor.extract_enrollment(pcm.tobytes(), min_speech_ms=1500)
    except InvalidSpeakerAudio:
        return None
    return sample.model_dump(mode="json")


_EXTRACTOR = None


def _start(model_dir):
    global _EXTRACTOR
    _EXTRACTOR = ReferenceSpeakerExtractor(model_dir)
    _EXTRACTOR.warm_up()


def _speaker(job):
    n, speaker, paths = job
    rows = []
    for i, path in enumerate(paths):
        # Codec and level vary deterministically, so a rebuild is identical.
        bitrate = CODEC_BITRATES[(n + i) % len(CODEC_BITRATES)]
        rms = LEVELS_RMS[(n + i) % len(LEVELS_RMS)]
        sample = embed(_EXTRACTOR, opus_round_trip(level(read_pcm(path), rms), bitrate))
        if sample is not None:
            rows.append(dict(file=path.name, bitrate=bitrate, rms=rms, sample=sample))
    return speaker, rows, _EXTRACTOR.model_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    speakers = corpus(args.corpus)
    cohort_ids, held_ids = split(speakers)
    print(f"{len(speakers)} speakers: {len(cohort_ids)} cohort, {len(held_ids)} held out", flush=True)

    embedded: dict[str, list[dict]] = {}
    identities = set()
    jobs = [(n, s, speakers[s]) for n, s in enumerate(sorted(speakers))]
    with ProcessPoolExecutor(args.jobs, initializer=_start, initargs=(args.model_dir,)) as pool:
        for speaker, rows, identity in pool.map(_speaker, jobs):
            embedded[speaker] = rows
            identities.add(identity)
            print(speaker, len(rows), flush=True)
    (identity,) = identities

    vectors, speech_ms, owners = [], [], []
    for index, speaker in enumerate(cohort_ids):
        for row in embedded[speaker]:
            windows = [row["sample"]] + row["sample"]["crops"]
            for w in windows:
                vectors.append(w["vector"])
                speech_ms.append(w["speech_ms"])
                owners.append(index)
    npz = args.out / "impostor-cohort.npz"
    with open(npz, "wb") as f:
        np.savez(f, vectors=np.asarray(vectors, dtype=np.float32),
                 speech_ms=np.asarray(speech_ms, dtype=np.int32),
                 speakers=np.asarray(owners, dtype=np.int16),
                 model_id=np.array(identity[0]), weights_hash=np.array(identity[1]),
                 preprocessing_version=np.array(identity[2]))
    lengths = np.asarray(speech_ms)
    record = {
        "dataset": "AISHELL/AISHELL-1",
        "dataset_revision": "bbe295d530192a4cd41644b711c9aecd087df653",
        "dataset_license": "apache-2.0 (dataset card)",
        "contains": "speaker embeddings only; no audio, transcripts or speaker names",
        "model_id": identity[0], "weights_hash": identity[1], "preprocessing_version": identity[2],
        "codec": {"name": "opus", "sample_rate": 48000, "bitrates": list(CODEC_BITRATES)},
        "level_rms": list(LEVELS_RMS),
        "speakers": len(cohort_ids), "held_out_speakers": len(held_ids),
        "vectors": len(vectors),
        "voiced_ms": {"lt_1200": int((lengths < 1200).sum()),
                      "1200_2500": int(((lengths >= 1200) & (lengths < 2500)).sum()),
                      "ge_2500": int((lengths >= 2500).sum())},
        "sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
    }
    (args.out / "impostor-cohort.json").write_text(json.dumps(record, indent=1))
    (args.out / "held-out.json").write_text(json.dumps(
        {"identity": list(identity), "speakers": {s: embedded[s] for s in held_ids}}))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
