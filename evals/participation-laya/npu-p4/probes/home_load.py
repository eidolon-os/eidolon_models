"""Send the smart-home sample requests to the live laya service at a fixed interval; print latency stats."""
import json, statistics, sys, time, urllib.request
reqs = json.load(open(sys.argv[1])); n = int(sys.argv[2]); gap = float(sys.argv[3])
reqs = reqs if isinstance(reqs, list) else reqs.get("requests", list(reqs.values()))
wall, server, errors = [], [], 0
for i in range(n):
    body = reqs[i % len(reqs)]
    body = body.get("body", body) if isinstance(body, dict) else body
    t = time.perf_counter()
    try:
        r = urllib.request.urlopen(urllib.request.Request(sys.argv[4] if len(sys.argv) > 4 else "http://127.0.0.1:8771/v1/systemone", json.dumps(body).encode(),
                                   {"content-type": "application/json"}), timeout=10)
        d = json.loads(r.read()); wall.append((time.perf_counter() - t) * 1000); server.append(d.get("timing_ms", {}).get("total", 0))
    except Exception as e:
        errors += 1; print("err", type(e).__name__, str(e)[:120], file=sys.stderr)
    time.sleep(max(0, gap - (time.perf_counter() - t)))
q = lambda xs, p: round(sorted(xs)[int(p * (len(xs) - 1))], 1) if xs else None
print(json.dumps({"n": n, "ok": len(wall), "errors": errors, "wall_p50": q(wall, .5), "wall_p95": q(wall, .95),
                  "server_p50": q(server, .5), "server_p95": q(server, .95)}))
