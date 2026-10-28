"""Offline powerset overlap screening; never grants Person admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assets = json.loads(args.assets.read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    model = ort.InferenceSession(str(args.model), options, providers=['CPUExecutionProvider'])
    meta = model.get_modelmeta().custom_metadata_map
    assert (meta['num_classes'], meta['num_speakers'], meta['powerset_max_classes']) == ('7', '3', '2')
    rate, window = int(meta['sample_rate']), int(meta['window_size'])
    receptive, shift = int(meta['receptive_field_size']), int(meta['receptive_field_shift'])
    samples = {}
    for row in assets['samples']:
        if row['speaker'] in samples:
            continue
        path = args.assets.parent / row['path']
        assert digest(path) == row['sha256']
        with wave.open(str(path)) as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (rate, 1, 2)
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
        samples[row['speaker']] = (row, audio)
    rows = []

    def measure(audio, kind, sources, seconds):
        # Right padding matches upstream example. Exclude padded and partial receptive fields.
        padded = np.pad(audio, (0, window - len(audio)))[None, None, :]
        start = time.perf_counter()
        logits = model.run(None, {model.get_inputs()[0].name: padded})[0][0]
        elapsed = (time.perf_counter() - start) * 1000
        valid = np.arange(len(logits)) * shift + receptive <= len(audio)
        classes = logits[valid].argmax(axis=-1)
        overlap = classes >= 4
        rows.append(dict(kind=kind, sources=sources, seconds=seconds,
                         valid_frames=len(classes), speech_frames=int((classes > 0).sum()),
                         overlap_frames=int(overlap.sum()), overlap_ms=float(overlap.sum() * shift / rate * 1000),
                         inference_ms=elapsed))

    entries = list(samples.values())
    model.run(None, {model.get_inputs()[0].name: np.zeros((1, 1, window), dtype=np.float32)})
    for i, (row, audio) in enumerate(entries):
        other, second = entries[(i + 1) % len(entries)]
        for seconds in (1.5, 3.0):
            n = int(seconds * rate)
            a = audio[(len(audio)-n)//2: (len(audio)-n)//2+n]
            b = second[(len(second)-n)//2: (len(second)-n)//2+n]
            measure(a, 'single', [row['id']], seconds)
            for db in (0, 10):
                scaled = b * (np.sqrt(np.mean(a*a)) / max(np.sqrt(np.mean(b*b)), 1e-8)) * 10**(-db/20)
                mixed = a + scaled
                mixed = mixed / max(1., float(np.abs(mixed).max()) / .95)
                measure(mixed, f'mixture_{db}db', [row['id'], other['id']], seconds)
    for seconds in (1.5, 3.0):
        measure(np.zeros(int(seconds * rate), dtype=np.float32), 'silence', [], seconds)
    summary = {}
    for kind in sorted({r['kind'] for r in rows}):
        for seconds in (1.5, 3.0):
            group = [r for r in rows if r['kind'] == kind and r['seconds'] == seconds]
            summary[f'{kind}_{seconds}s'] = dict(n=len(group), any_overlap=sum(r['overlap_frames'] > 0 for r in group),
                overlap_at_least_100ms=sum(r['overlap_ms'] >= 100 for r in group),
                p95_inference_ms=float(np.percentile([r['inference_ms'] for r in group], 95)))
    result = dict(model_sha256=digest(args.model), script_sha256=digest(Path(__file__)),
                  assets_sha256=digest(args.assets), metadata=meta, runtime=ort.__version__,
                  dataset_revision=assets['dataset_revision'], production_ready=False,
                  limitation='Synthetic mixtures of public AISHELL train speech; no frame truth; training overlap possible; no release threshold.',
                  summary=summary, measurements=rows)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
