"""Board-local parity and contention measurements; no actuator or LLM calls."""

import argparse
import concurrent.futures
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def post(url, body):
    start = time.perf_counter()
    req = urllib.request.Request(
        url + "/v1/systemone", json.dumps(body).encode(), {"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            status = response.status
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        status = exc.code
        payload = json.load(exc)
    return {
        "status": status,
        "ms": round((time.perf_counter() - start) * 1000, 2),
        "response": payload,
    }


def stats(rows):
    times = sorted(r["ms"] for r in rows if r["status"] == 200)
    return {
        "requests": len(rows),
        "http_200": len(times),
        "http_503": sum(r["status"] == 503 for r in rows),
        "p50_ms": statistics.median(times) if times else None,
        "p95_ms": times[int(0.95 * (len(times) - 1))] if times else None,
        "max_ms": max(times) if times else None,
        "over_1000ms": sum(t > 1000 for t in times),
    }


def run(args):
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    fixtures = [json.loads(l) for l in Path(args.requests).read_text().splitlines()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = json.load(urllib.request.urlopen(args.url + "/v1/info"))
    assert info["revision"] == "e8254243" and info["engine"]["backend"] == "rknn", info
    (out / "info.json").write_text(json.dumps(info, indent=2))
    rows = []
    differences = []
    drift = 0.0
    answer_count = 0
    with (out / "responses.jsonl").open("w") as f:
        for case in fixtures:
            row = post(args.url, case["body"])
            row.update(id=case["id"], set=case["set"])
            if row["status"] == 200:
                assert row["response"]["revision"] == "e8254243"
                for qid, want in case["expected"].items():
                    got = row["response"]["answers"][qid]
                    answer_count += 1
                    if got["choice"] != want:
                        differences.append(
                            {
                                "id": case["id"],
                                "qid": qid,
                                "expected": want,
                                "actual": got["choice"],
                            }
                        )
                    ref = case.get("reference", {}).get(qid)
                    if ref:
                        drift = max(
                            drift,
                            max(
                                abs(v - got["probabilities"][k])
                                for k, v in ref["probabilities"].items()
                            ),
                        )
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if len(rows) % 25 == 0:
                print("completed", len(rows), flush=True)
    summary = {
        "revision": info["revision"],
        "serial": stats(rows),
        "answers": answer_count,
        "choice_differences": differences,
        "max_probability_diff_frozen_acceptance": drift,
        "by_set": {
            name: stats([r for r in rows if r["set"] == name])
            for name in sorted({r["set"] for r in rows})
        },
    }
    # Two independent clients, max_pending=1: busy must fail quickly, never queue unboundedly.
    concurrent_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for index in range(args.pairs):
            barrier = threading.Barrier(2)

            def send():
                barrier.wait()
                return post(args.url, fixtures[index % 20]["body"])

            futures = [pool.submit(send) for _ in range(2)]
            concurrent_rows.extend(f.result() for f in futures)
    summary["concurrent"] = stats(concurrent_rows)
    (out / "concurrent.json").write_text(json.dumps(concurrent_rows, ensure_ascii=False, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:18774")
    p.add_argument("--requests", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", type=int, default=20)
    run(p.parse_args())
