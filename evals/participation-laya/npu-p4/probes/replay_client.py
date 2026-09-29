"""Post each replayed DecisionRequest to the participation service; compare with the expected result."""
import json, statistics, sys, time, urllib.request
url = sys.argv[2]; rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
same = n = 0; lat = {}; diffs = []
for r in rows:
    body = json.dumps(r["request"]).encode()
    t = time.perf_counter()
    got = json.loads(urllib.request.urlopen(urllib.request.Request(url, body, {"content-type": "application/json"}), timeout=15).read())
    ms = (time.perf_counter() - t) * 1000
    want = r["expected"]; mine = {"status": got["status"], "proposal": got["proposal"]}
    n += 1; ok = mine == want; same += ok
    kind = (got["proposal"] or {}).get("action", "abstained"); lat.setdefault(kind, []).append(ms)
    if not ok: diffs.append({"id": r["request"]["decision_id"], "want": want, "got": mine})
    time.sleep(0.2)
allms = [x for v in lat.values() for x in v]
q = lambda xs, p: round(sorted(xs)[int(p * (len(xs) - 1))], 1)
print(json.dumps({"requests": n, "same_as_mac_pytorch": same, "p50_ms": q(allms, .5), "p95_ms": q(allms, .95),
                  "by_kind_p50": {k: (len(v), q(v, .5)) for k, v in lat.items()}, "diffs": diffs[:6]}, ensure_ascii=False))
