"""Side-by-side acceptance of speaker models on the same held-out pipeline (design §8.2 B2 trigger).

Each run directory holds ``identity/report.json`` (``identity_train.py``) and
``tracks/report.json`` (``track_eval.py``, with that run's own policy). The
criteria were fixed before any candidate was run:

- far cold start, a member's turns on a track of three or more segments,
  phone template only: ``personal_memory`` >= 50 %, ``addressing`` >= 70 %;
- strangers named on the far device: ``personal_memory`` <= 0.5 %,
  ``addressing`` <= 2 %;
- the phone (``near_same``, single turn) no worse than the delivered model.

    eidolon_channel/.venv/bin/python -I \\
        eidolon_models/evals/speaker-verification/models-20261102/compare.py \\
        delivered=<run dir> <name>=<run dir> ... --out results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FAR_COLD = {"personal_memory": 0.50, "addressing": 0.70}
STRANGER = {"personal_memory": 0.005, "addressing": 0.02}
USES = ("personal_memory", "addressing")


def correct(tally: dict) -> dict:
    return {u: tally.get(f"{u}:correct", 0.0) for u in USES}


def summary(run: Path) -> dict:
    identity = json.loads((run / "identity" / "report.json").read_text())
    tracks = json.loads((run / "tracks" / "report.json").read_text())
    phone = identity["households"]["near_same"]
    out = {
        "eer_k1": {c: identity["held_out"][c]["k1"]["eer"] for c in identity["held_out"]},
        "cllr_k1": {c: identity["held_out"][c]["k1"]["cllr"] for c in identity["held_out"]},
        "policy": {c: {u: identity["policy"][c][u] for u in USES} for c in identity["policy"]},
        "phone_single_turn": correct(phone["own/k1"]),
        "phone_stranger": {u: phone["stranger/all"].get(f"{u}:wrong", 0.0) for u in USES},
        "far_cold_track": correct(tracks["own"]["cold/k3+"]),
        "far_cold_single_turn": correct(tracks["own"]["cold/k1"]),
        "far_warm_single_turn": correct(tracks["own"]["warm/k1"]),
        "far_stranger": {u: max(tracks[who][k].get(f"{u}:wrong", 0.0)
                                for who in ("stranger", "member_as_stranger") for k in tracks[who])
                         for u in USES},
        "attributed_conversations": tracks["events"].get("attributed", 0),
        "conversations": sum(tracks["events"].values()),
        "best_coverage_k3_17": {u: [tracks["cold_start_accumulation"][b][f"{u}:best_coverage"]
                                    for b in ("k3-5", "k6-11", "k12-17")] for u in USES},
        "continuation_at_1pct": tracks["track_continuation"]["other_joins_0.01"]["own_continues"],
    }
    return out


def verdict(s: dict, delivered: dict) -> dict:
    checks = {
        **{f"far_cold_{u}": s["far_cold_track"][u] >= FAR_COLD[u] for u in USES},
        **{f"far_stranger_{u}": s["far_stranger"][u] <= STRANGER[u] for u in USES},
        **{f"phone_{u}": s["phone_single_turn"][u] >= delivered["phone_single_turn"][u] - 0.01
           for u in USES},
    }
    return {"checks": checks, "passes": all(checks.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+",
                        help="name=run directory; the first is the delivered model")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = dict(r.split("=", 1) for r in args.runs)
    summaries = {name: summary(Path(path)) for name, path in runs.items()}
    delivered = summaries[next(iter(runs))]
    report = {"criteria": {"far_cold_track_min": FAR_COLD, "far_stranger_max": STRANGER,
                           "phone": "no worse than the delivered model by more than 1 point"},
              "models": {name: {**s, **verdict(s, delivered)} for name, s in summaries.items()}}
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
