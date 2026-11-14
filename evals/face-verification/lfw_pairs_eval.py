"""Score the LFW verification pairs through eidolon_vision's own face encoder.

Run with eidolon_vision's environment so the code measured is the code served:

    cd ../eidolon_vision && uv run --no-sync python \
        ../eidolon_models/evals/face-verification/lfw_pairs_eval.py \
        --model-dir ../eidolon_models/face_recognition/sface_yunet_opencv_zoo/48b10fa5 \
        --lfw-home <dir holding lfw_funneled/ and pairs.txt> --out results.json

Every image goes through detection, alignment and embedding exactly as a probe
frame would; the face nearest the image centre is taken (LFW centres its
subject). ``--scale`` shrinks every image first, to see what small faces cost.
Reads images only; writes aggregate numbers, never an image, name or vector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from eidolon_vision.adapters.faces.opencv_zoo import OpenCvZooFaceEncoder
from eidolon_vision.runtime.types import Frame


def read_pairs(path: Path) -> list[tuple[str, str, bool, int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    folds, per_fold = (int(v) for v in lines[0].split())
    pairs = []
    for index, line in enumerate(lines[1:]):
        fold = index // (2 * per_fold)
        parts = line.split()
        if len(parts) == 3:
            name, a, b = parts
            pairs.append((f"{name}/{name}_{int(a):04d}.jpg", f"{name}/{name}_{int(b):04d}.jpg", True, fold))
        else:
            n1, a, n2, b = parts
            pairs.append((f"{n1}/{n1}_{int(a):04d}.jpg", f"{n2}/{n2}_{int(b):04d}.jpg", False, fold))
    assert len(pairs) == folds * per_fold * 2
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--lfw-home", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale", type=float, nargs="*", default=[1.0])
    parser.add_argument("--one-side", action="store_true",
                        help="shrink only the second image of each pair (enrolled photo stays full size)")
    args = parser.parse_args()

    encoder = OpenCvZooFaceEncoder(args.model_dir)
    home = Path(args.lfw_home)
    pairs = read_pairs(home / "pairs.txt")
    report = {"model_id": encoder.model_id, "preprocessing_version": encoder.preprocessing_version,
              "dataset": "LFW funneled, pairs.txt (10 folds, 6000 pairs)", "one_side": args.one_side,
              "runs": []}

    for scale in args.scale:
        cache: dict[tuple[str, float], tuple[np.ndarray, float] | None] = {}

        def embed(rel: str, s: float):
            key = (rel, s)
            if key not in cache:
                image = cv2.imread(str(home / "lfw_funneled" / rel))
                if s != 1.0:
                    image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                h, w = image.shape[:2]
                faces = encoder.encode(Frame(pixels=image, width=w, height=h))
                if not faces:
                    cache[key] = None
                else:
                    face = min(faces, key=lambda f: (f.box[0] + f.box[2] / 2 - w / 2) ** 2
                               + (f.box[1] + f.box[3] / 2 - h / 2) ** 2)
                    cache[key] = (np.asarray(face.vector), min(face.box[2], face.box[3]))
            return cache[key]

        scores, same, fold_of, sizes, missed = [], [], [], [], 0
        for a, b, is_same, fold in pairs:
            ea, eb = embed(a, 1.0 if args.one_side else scale), embed(b, scale)
            if ea is None or eb is None:
                missed += 1
                continue
            va, vb = ea[0], eb[0]
            scores.append(float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb))))
            same.append(is_same)
            fold_of.append(fold)
            sizes.append(eb[1])
        scores, same, fold_of = np.asarray(scores), np.asarray(same), np.asarray(fold_of)
        genuine, impostor = scores[same], scores[~same]

        def at(threshold: float) -> dict:
            return {"threshold": round(threshold, 4),
                    "false_accept": int((impostor >= threshold).sum()),
                    "false_accept_rate": round(float((impostor >= threshold).mean()), 5),
                    "true_accept_rate": round(float((genuine >= threshold).mean()), 4)}

        # Standard LFW protocol: threshold chosen on nine folds, accuracy on the tenth.
        grid = np.arange(0.0, 0.8, 0.005)
        accuracies = []
        for fold in range(10):
            train, test = fold_of != fold, fold_of == fold
            best = max(grid, key=lambda t: ((scores[train] >= t) == same[train]).mean())
            accuracies.append(float(((scores[test] >= best) == same[test]).mean()))

        ordered = np.sort(impostor)[::-1]
        report["runs"].append({
            "scale": scale,
            "pairs_scored": int(len(scores)), "pairs_without_a_detected_face": missed,
            "genuine": int(len(genuine)), "impostor": int(len(impostor)),
            "probe_face_px_median": round(float(np.median(sizes)), 1),
            "probe_face_px_p5": round(float(np.percentile(sizes, 5)), 1),
            "accuracy_10fold_mean": round(float(np.mean(accuracies)), 4),
            "accuracy_10fold_std": round(float(np.std(accuracies)), 4),
            "genuine_percentiles": {str(p): round(float(np.percentile(genuine, p)), 4) for p in (1, 5, 25, 50)},
            "impostor_percentiles": {str(p): round(float(np.percentile(impostor, p)), 4) for p in (50, 95, 99, 99.9)},
            "impostor_top5": [round(float(v), 4) for v in ordered[:5]],
            "at_thresholds": [at(t) for t in (0.30, 0.363, 0.40, 0.45, 0.50, 0.55, 0.60)],
        })
        print(json.dumps(report["runs"][-1], indent=1))

    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
