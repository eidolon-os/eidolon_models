"""Compare stills taken by a device camera with enrolled reference photos.

Same environment and code path as the LFW scripts: eidolon_vision's own encoder,
enrollment gate and comparison, with the policy that ships beside the weights.

    cd ../eidolon_vision && uv run --no-sync python \
        ../eidolon_models/evals/face-verification/device_stills_eval.py \
        --model-dir ../eidolon_models/face_recognition/sface_yunet_opencv_zoo/48b10fa5 \
        --enrolled NAME=photo.jpg [NAME=photo.jpg ...] \
        --stills DIR --subject NAME --out rows.csv

``--subject`` is who really stood in front of the camera in these stills (or
``-`` for someone not enrolled). Prints, per still: faces found, the largest
face's size, and its score and band against each enrolled person; then a
summary by face size. Images are read only; the CSV holds numbers and the
labels given on the command line, never an image or a feature.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from eidolon_vision.adapters.faces.opencv_zoo import OpenCvZooFaceEncoder
from eidolon_vision.runtime.faces import FaceTemplate, compare, enroll, load_policy
from eidolon_vision.runtime.types import Frame


def frame(path: Path) -> Frame:
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"unreadable image: {path}")
    return Frame(pixels=image, width=image.shape[1], height=image.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--enrolled", nargs="+", required=True, metavar="NAME=PHOTO")
    parser.add_argument("--stills", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    encoder = OpenCvZooFaceEncoder(args.model_dir)
    policy = load_policy(Path(args.model_dir) / "face-policy.json")
    templates = []
    for item in args.enrolled:
        name, _, photo = item.partition("=")
        result = enroll(encoder, policy, frame(Path(photo)))
        print(f"enroll {name}: accepted={result.accepted} rejection={result.rejection} "
              f"faces={result.faces} face_px={result.face_px}")
        if result.accepted:
            templates.append(FaceTemplate(ref=name, vector=result.vector, model_id=encoder.model_id,
                                          preprocessing_version=encoder.preprocessing_version))
    names = [t.ref for t in templates]

    rows = []
    for path in sorted(Path(args.stills).glob("*.jpg")):
        still = frame(path)
        result = compare(encoder, policy, still, templates)
        row = {"still": path.name, "width": still.width, "faces": len(result.faces),
               "brightness": round(float(still.pixels.mean()), 1)}
        if result.faces:
            face = result.faces[0]  # largest
            scores = {c.ref: c for c in face.comparisons}
            row |= {"face_px": face.face_px, "quality": face.quality, "center_x": face.center_x,
                    "center_y": face.center_y, "distinct": face.distinct}
            for name in names:
                row[f"score_{name}"] = scores[name].score if name in scores else ""
                row[f"band_{name}"] = scores[name].band if name in scores else ""
        rows.append(row)

    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    seen = [r for r in rows if r["faces"]]
    print(f"\nstills={len(rows)} with_a_face={len(seen)} subject={args.subject} "
          f"policy={policy.version} strong>={policy.strong} weak>={policy.weak} "
          f"min_probe_face_px={policy.min_probe_face_px}")
    buckets = [(0, 28), (28, 40), (40, 60), (60, 90), (90, 10_000)]
    print("face_px      n   " + "   ".join(f"{n}: median / min / strong / weak" for n in names))
    for low, high in buckets:
        group = [r for r in seen if low <= r["face_px"] < high]
        if not group:
            continue
        line = f"{low:>3}-{high if high < 10_000 else '+':<5} {len(group):>3}   "
        for name in names:
            values = sorted(r[f"score_{name}"] for r in group if r.get(f"score_{name}") != "")
            if not values:
                line += f"{name}: (not compared)   "
                continue
            strong = sum(r.get(f"band_{name}") == "strong" for r in group)
            weak = sum(r.get(f"band_{name}") == "weak" for r in group)
            line += (f"{name}: {values[len(values) // 2]:.3f} / {values[0]:.3f} / "
                     f"{strong} / {weak}   ")
        print(line)


if __name__ == "__main__":
    main()
