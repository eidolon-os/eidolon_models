"""Small signal stress probes: embeddings alone must not imply valid speaker evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from analyze import calibrate, open_set, split_samples, unit, vectors_for
from extract import build


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("model")
    p.add_argument("--native-weights", type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    os.umask(0o077)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    output = root / "controls" / (args.model + ".json")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(exist_ok=True)
    manifest = json.loads((root / "assets.json").read_text())
    data = json.loads((root / "embeddings" / (args.model + ".json")).read_text())
    groups, cal, test = split_samples(manifest)
    ec, pc, _ = vectors_for(data, groups, cal, 2.0)
    policy = calibrate(ec, pc)
    et, _, _ = vectors_for(data, groups, test, 2.0)
    paths = {x["id"]: root / x["path"] for x in manifest["samples"]}

    def clip(sid):
        audio, sr = sf.read(paths[sid], dtype="float32")
        assert sr == 16000
        start = (len(audio) - 32000) // 2
        return audio[start:start + 32000].copy()

    infer, _ = build(root, args.model, args.native_weights)
    rng = np.random.default_rng(42)
    scenarios = []
    for person in test:
        for sid in groups[person][3:5]:
            audio = clip(sid)
            sigma = float(np.sqrt(np.mean(audio**2) / 10))
            noisy = audio + rng.normal(0, sigma, len(audio)).astype(np.float32)
            scenarios.append(("noise10db", sid, person, noisy, None))
            other = test[(test.index(person) + 1) % len(test)]
            partner = clip(groups[other][3])
            a = audio / max(float(np.sqrt(np.mean(audio**2))), 1e-8)
            b = partner / max(float(np.sqrt(np.mean(partner**2))), 1e-8)
            overlap = (a + b) * 0.05
            scenarios.append(("equal_energy_overlap", sid, person, overlap.astype(np.float32), other))
    scenarios.extend([
        ("digital_silence", "silence", None, np.zeros(32000, dtype=np.float32), None),
        ("white_noise", "noise", None, rng.normal(0, 0.03, 32000).astype(np.float32), None),
    ])
    records, noise_rows = [], []
    with torch.inference_mode():
        for condition, sid, speaker, audio, other in scenarios:
            audio /= max(float(np.max(np.abs(audio))) / 0.99, 1.0)
            record = dict(condition=condition, source_id=sid, primary_speaker=speaker,
                          other_speaker=other)
            vector = None
            try:
                vector = unit(infer(audio))
                scores = sorted([float(vector @ v) for v in et.values()], reverse=True)
                record.update(status="ok", max_gallery_score=scores[0],
                              accepted_by_embedding_only=bool(scores[0] >= policy["known"] and
                                                               scores[0]-scores[1] >= policy["margin"]))
            except Exception as exc:
                record.update(status="unavailable", error=type(exc).__name__)
            records.append(record)
            if condition == "noise10db":
                noise_rows.append(dict(id=sid, speaker=speaker, vector=vector))
    output.write_text(json.dumps(dict(
        model=args.model, policy_from_clean_calibration=policy,
        noisy_open_set=open_set(et, noise_rows, policy), records=records,
        production_ready=False,
        note="Synthetic signal perturbations of held-out recordings; no VAD/overlap detector. "
             "Overlap has two real speakers and no valid single-person truth. "
             "Accepted-by-embedding-only counts are diagnostics, not product authorization failures."
    ), ensure_ascii=False, indent=2))
    print(args.model, "control cases", len(records), flush=True)


if __name__ == "__main__":
    main()
