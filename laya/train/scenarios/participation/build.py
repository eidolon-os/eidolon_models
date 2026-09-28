"""Put one run's inputs in place: episode files -> records under train/runs/<run>/, with their hashes.

    uv run --extra torch --extra train python train/scenarios/participation/build.py --run p1

Training episodes come from train/data/participation/ (Claude: claude-*.jsonl, GLM: glm-*/episodes.jsonl);
eval episodes from evals/participation-laya/{dev,test}/. Then run the pipeline with the same run id.
Training episodes get the QA corrections in train/data/participation/fixes.jsonl (per line: drop the episode, or
set one event's gold and optionally cut the events after it); eval episodes are never touched.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from episodes import load, to_records, validate  # noqa: E402

from eidolon_laya_train.records import write_jsonl  # noqa: E402

LAYA = HERE.parents[2]
DATA = LAYA / "train" / "data" / "participation"
EVALS = LAYA.parent / "evals" / "participation-laya"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def apply_fixes(eps: list[dict], fixes: list[dict]) -> tuple[list[dict], int]:
    by: dict[str, list[dict]] = {}
    for f in fixes:
        by.setdefault(f["episode"], []).append(f)
    out, n = [], 0
    for ep in eps:
        fs = by.get(ep["episode"], [])
        if any(f.get("drop") for f in fs):
            n += 1
            continue
        for f in sorted((f for f in fs if "event" in f), key=lambda f: -f["event"]):
            ep["events"][f["event"]]["gold"] = f["gold"]
            if f.get("truncate"):
                ep["events"] = ep["events"][: f["event"] + 1]
            n += 1
        out.append(ep)
    return out, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--variants", type=int, default=2)
    a = ap.parse_args()
    run = LAYA / "train" / "runs" / a.run
    run.mkdir(parents=True, exist_ok=True)
    groups = {
        "claude": sorted(DATA.glob("claude-*.jsonl")),
        "glm": sorted(p for p in DATA.glob("glm-s*/episodes.jsonl")),
        "eval-dev": sorted((EVALS / "dev").glob("*.jsonl")),
        "eval-test": sorted((EVALS / "test").glob("*.jsonl")),
    }
    fixes_path = DATA / "fixes.jsonl"
    fixes = [json.loads(x) for x in fixes_path.read_text("utf-8").splitlines() if x.strip()] if fixes_path.exists() else []
    manifest = {}
    for name, files in groups.items():
        eps = load([str(f) for f in files]) if files else []
        fixed = 0
        if not name.startswith("eval"):
            eps, fixed = apply_fixes(eps, fixes)
        bad = [e for ep in eps for e in validate(ep)]
        if bad:
            print("\n".join(bad[:20]), file=sys.stderr)
            raise SystemExit(f"{name}: {len(bad)} invalid episodes")
        recs = to_records(eps, a.variants, name)
        write_jsonl(run / f"{name}.jsonl", recs)
        manifest[name] = {"episodes": len(eps), "records": len(recs), "fixes_applied": fixed,
                          "files": {str(f.relative_to(LAYA.parent)): sha(f) for f in files}}
        print(f"{name}: {len(eps)} episodes -> {len(recs)} records ({fixed} fixes)")
    if fixes_path.exists():
        manifest["fixes"] = {str(fixes_path.relative_to(LAYA.parent)): sha(fixes_path)}
    (run / "inputs.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
