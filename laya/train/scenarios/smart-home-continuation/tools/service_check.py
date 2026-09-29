"""Send eval records through a running Laya service and compare with the offline eval, question by question.

    python3 train/scenarios/smart-home-continuation/tools/service_check.py --url http://127.0.0.1:18771 \
        --pair evals/smart-home-continuation/c-dev.jsonl train/runs/c4/eval-c/c-dev.json \
        --pair train/scenarios/smart-home/eval/locked-v2-dev.jsonl train/runs/c4/eval/locked-v2-dev.json

Each record's state and questions go to POST /v1/systemone exactly as stored (no ask_if, as the Agent adapter
sends them). A different choice means the service does not see what the model was evaluated on; the largest
probability difference shows numeric drift (device / backend). Prints one line per pair and exits 1 on any
choice mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("RECORDS", "EVAL_JSON"))
    a = ap.parse_args()
    bad = 0
    for records, report in a.pair:
        want = {(r["record_id"], r["qid"]): r for r in json.load(open(report, encoding="utf-8"))["rows"]}
        n = diff = 0
        worst = 0.0
        for line in open(records, encoding="utf-8"):
            rec = json.loads(line)
            body = json.dumps({"state": rec["state"], "questions": rec["questions"]}).encode()
            req = urllib.request.Request(a.url.rstrip("/") + "/v1/systemone", body, {"content-type": "application/json"})
            got = json.loads(urllib.request.urlopen(req, timeout=30).read())["answers"]
            for qid, ans in got.items():
                w = want.get((rec["id"], qid))
                if w is None:  # not scored offline (e.g. device for a 无关 record)
                    continue
                n += 1
                if ans["choice"] != w["pred"]:
                    diff += 1
                    print(f"  choice differs {rec['id']} {qid}: service {ans['choice']} vs eval {w['pred']}")
                worst = max(worst, max(abs(ans["probabilities"][k] - p) for k, p in w["probabilities"].items()))
        bad += diff
        print(f"{records}: {n} answers, {diff} choice differences, max |dp| {worst:.4f}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
