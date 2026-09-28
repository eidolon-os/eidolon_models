"""Promote explicitly reviewed GLM drafts into per-slice, versioned train data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parents[1]
REVIEW = HERE / "review-v2.yaml"
DEST = TRAIN / "data/generated/ip-team-v4/glm-reviewed-v2"


def write_immutable(path: Path, data: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise ValueError(f"refusing to overwrite changed reviewed data: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=Path, default=REVIEW)
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--output-name", default="train.yaml", choices=("train.yaml", "dev.yaml"))
    ap.add_argument("--namespace", default="", help="prefix family and pair IDs when a separate plan reuses call IDs")
    args = ap.parse_args()
    review = yaml.safe_load(args.review.read_text(encoding="utf-8"))
    source = (HERE / review["source"]).resolve()
    selected, rejected = review["selected"], review["rejected"]
    calls = {p.stem: p for p in (source / "calls").glob("*.json")}
    if set(calls) != set(selected) | set(rejected) or set(selected) & set(rejected):
        raise ValueError("every GLM call must have exactly one editorial disposition")
    combined, provenance = [], []
    for call_id in sorted(selected):
        meta = json.loads(calls[call_id].read_text(encoding="utf-8"))
        if meta["status"] != "draft_for_review" or meta["plan_version"] != review["source_plan_version"]:
            raise ValueError(f"{call_id}: source status/version does not qualify")
        raw = (source / "responses" / f"{call_id}.txt").read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta["response_sha256"]:
            raise ValueError(f"{call_id}: source API response changed")
        draft_path = source / "drafts" / f"{call_id}.yaml"
        draft = yaml.safe_load(draft_path.read_text(encoding="utf-8"))
        if not isinstance(draft, list) or len(draft) != meta["episode_count"]:
            raise ValueError(f"{call_id}: draft count changed")
        if any(item["tag"] != meta["slice"] or not item["id"].startswith("glm53_flash_") for item in draft):
            raise ValueError(f"{call_id}: draft identity changed")
        if args.namespace:
            for item in draft:
                item["id"] = args.namespace + item["id"]
                if item.get("pair_id"):
                    item["pair_id"] = args.namespace + item["pair_id"]
        text = yaml.safe_dump(draft, allow_unicode=True, sort_keys=False)
        write_immutable(args.dest / "by-slice" / meta["slice"] / f"{call_id}.yaml", text)
        combined.extend(draft)
        provenance.append({"call_id": call_id, "slice": meta["slice"],
                           "response_sha256": meta["response_sha256"],
                           "review_reason": selected[call_id], "episodes": len(draft)})
    ids = [item["id"] for item in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate reviewed family id")
    write_immutable(args.dest / args.output_name, yaml.safe_dump(combined, allow_unicode=True, sort_keys=False))
    manifest = {"review_version": review["version"],
        "source_plan_version": review["source_plan_version"], "selected_calls": provenance,
        "rejected_calls": rejected, "families": len(combined)}
    if args.namespace:
        manifest["namespace"] = args.namespace
    write_immutable(args.dest / "provenance.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"reviewed {len(combined)} families across {len(provenance)} GLM calls -> {args.dest}")


if __name__ == "__main__":
    main()
