"""Stranger scores at scale: one image per LFW identity, every pair compared.

``pairs.txt`` holds 3000 different-person pairs, too few to see false-accept
rates below about one in a thousand. Here every identity contributes its first
image and all identity pairs are scored (about 16.5 million), so the tail is
visible. Genuine scores come from identities with at least two images (first
image against the second). Same environment and code path as lfw_pairs_eval.py.
Writes aggregate numbers only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from eidolon_vision.adapters.faces.opencv_zoo import OpenCvZooFaceEncoder
from eidolon_vision.runtime.types import Frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--lfw-home", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--probe-scale", type=float, nargs="*", default=[1.0],
                        help="shrink the probe image (the enrolled one stays full size)")
    args = parser.parse_args()
    encoder = OpenCvZooFaceEncoder(args.model_dir)
    root = Path(args.lfw_home) / "lfw_funneled"

    def embed(path: Path, scale: float = 1.0):
        image = cv2.imread(str(path))
        if scale != 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
        faces = encoder.encode(Frame(pixels=image, width=w, height=h))
        if not faces:
            return None
        face = min(faces, key=lambda f: (f.box[0] + f.box[2] / 2 - w / 2) ** 2
                   + (f.box[1] + f.box[3] / 2 - h / 2) ** 2)
        vector = np.asarray(face.vector)
        return vector / np.linalg.norm(vector), min(face.box[2], face.box[3]), face.score

    people = sorted(p for p in root.iterdir() if p.is_dir())
    enrolled, second = [], {}
    for index, person in enumerate(people):
        images = sorted(person.glob("*.jpg"))
        first = embed(images[0])
        if first is None:
            continue
        enrolled.append(first[0])
        if len(images) > 1:
            second[len(enrolled) - 1] = images[1]
    gallery = np.stack(enrolled)
    report = {"model_id": encoder.model_id, "preprocessing_version": encoder.preprocessing_version,
              "dataset": "LFW funneled, first image of every identity as the enrolled photo",
              "identities": int(len(gallery)), "identities_with_a_second_image": len(second), "runs": []}

    for scale in args.probe_scale:
        # Probes: the second image of each identity that has one, shrunk by `scale`.
        rows, owners, sizes, det_scores, undetected = [], [], [], [], 0
        for owner, path in second.items():
            probe = embed(path, scale)
            if probe is None:
                undetected += 1
                continue
            rows.append(probe[0]); owners.append(owner); sizes.append(probe[1]); det_scores.append(probe[2])
        probes, owners = np.stack(rows), np.asarray(owners)
        scores = probes @ gallery.T                      # probe x enrolled
        genuine = scores[np.arange(len(owners)), owners]
        mask = np.ones_like(scores, dtype=bool)
        mask[np.arange(len(owners)), owners] = False
        impostor = scores[mask]
        # Household view: each probe against a random household of N other people;
        # a false accept is any of them reaching the threshold.
        rng = np.random.default_rng(20261114)
        household = {}
        for size in (1, 4, 8):
            best = np.empty(len(owners))
            for i in range(len(owners)):
                others = rng.choice(np.delete(np.arange(len(gallery)), owners[i]), size=size, replace=False)
                best[i] = scores[i, others].max()
            household[size] = best

        def at(t: float) -> dict:
            return {"threshold": t,
                    "stranger_pairs_accepted": int((impostor >= t).sum()),
                    "stranger_pair_rate": float(f"{(impostor >= t).mean():.3g}"),
                    "own_photo_accepted_rate": round(float((genuine >= t).mean()), 4),
                    **{f"stranger_named_in_household_of_{n}": float(f"{(household[n] >= t).mean():.3g}")
                       for n in household}}

        report["runs"].append({
            "probe_scale": scale, "probes": int(len(owners)), "probes_without_a_detected_face": undetected,
            "stranger_pairs": int(impostor.size),
            "probe_face_px": {str(p): round(float(np.percentile(sizes, p)), 1) for p in (5, 50, 95)},
            "probe_detector_score_p5": round(float(np.percentile(det_scores, 5)), 3),
            "genuine_percentiles": {str(p): round(float(np.percentile(genuine, p)), 4) for p in (1, 5, 10, 25, 50)},
            "stranger_percentiles": {str(p): round(float(np.percentile(impostor, p)), 4)
                                     for p in (50, 99, 99.9, 99.99, 99.999)},
            "stranger_max": round(float(impostor.max()), 4),
            "at_thresholds": [at(t) for t in (0.30, 0.363, 0.40, 0.45, 0.50, 0.55, 0.60)],
        })
        print(json.dumps(report["runs"][-1], indent=1), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
