"""Explicit offline evaluation, with private output files and no policy writes."""

import argparse
import json
import os
from pathlib import Path

from .evaluation import calibration_curve, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--partition", choices=["calibration", "test"], required=True)
    parser.add_argument("--threshold", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text())
    if args.partition == "test" and len(args.threshold) != 1:
        parser.error("test accepts one frozen threshold; tune on calibration only")
    report = (
        evaluate(data, partition="test", threshold=args.threshold[0])
        if args.partition == "test"
        else calibration_curve(data, thresholds=args.threshold)
    )
    # Refuse overwriting previous evidence. Reports contain no embeddings/PCM.
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as out:
        json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    print(f"Offline report saved: {args.output}; production readiness not established.")


if __name__ == "__main__":
    main()
