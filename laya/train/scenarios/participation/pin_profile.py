"""Pin the participation task profile into a packaged model, so the service can serve it.

    uv run --extra torch --extra train python train/scenarios/participation/pin_profile.py \
        --model-dir models/laya-participation/<rev> --min-confidence 0.9

Writes <model-dir>/participation.json (schema 2, state format participation-laya-v1: the questions
exactly as scenario.yaml trained them, plus the bounded clarification task per reason) and pins its
sha256 in the model's manifest.json under task_profiles. Whether a clarify may carry these per-reason
tasks (the reply model words the question from the public history) or must abstain is a policy choice
(docs/IP团队/决策模型集成边界审查-20260928.md §4); --no-clarify-tasks gives the abstaining profile. min_confidence is the calibrated action
confidence below which the service abstains and the agent decides (choose it on p-dev, PLAN.md §3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from eidolon_laya_train.scenario import Scenario  # noqa: E402

# Bounded tasks for the speaker who asks; the reply model words the actual question.
CLARIFY_INSTRUCTIONS = {
    "指代不明": "用户说的对象不明确（不确定是哪一位或哪一个）。用一句话请用户说清楚指的是谁或哪一个，不要替用户猜。",
    "要求不明": "不确定用户想让大家做什么。用一句话请用户说清楚具体想要什么。",
    "对象不在场": "用户点到的角色不在场。用一句话告诉用户这位不在，并问用户想让在场的哪一位来回应。",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--min-confidence", type=float, required=True)
    ap.add_argument("--max-candidates", type=int, default=5, help="largest team the release was trained on")
    ap.add_argument("--policy-version", help="default: participation-laya-v1/<revision>/min<conf>")
    ap.add_argument("--no-clarify-tasks", action="store_true",
                    help="write no clarification tasks: every clarify then abstains (as ip-team-v3 does)")
    a = ap.parse_args()
    model_dir = Path(a.model_dir)
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    revision = manifest["source"]["revision"]
    scn = Scenario.load(HERE)
    speaker = scn.questions["speaker"]
    questions = {
        "action": scn.questions["action"].materialize(),
        "speaker": {"type": speaker.type, "instructions": speaker.instructions},
        "clarify_about": scn.questions["clarify_about"].materialize(),
    }
    if set(questions["clarify_about"]["criteria"]) != set(CLARIFY_INSTRUCTIONS):
        raise SystemExit("clarify reasons in scenario.yaml and CLARIFY_INSTRUCTIONS differ")
    profile = {
        "schema_version": 2,
        "task": "ip_team.participation",
        "state_format": "participation-laya-v1",
        "model_revision": revision,
        "policy_version": a.policy_version or f"participation-laya-v1/{revision}/min{a.min_confidence:g}",
        "min_confidence": a.min_confidence,
        "max_candidates": a.max_candidates,
        "questions": questions,
        "clarify_instructions": {} if a.no_clarify_tasks else CLARIFY_INSTRUCTIONS,
    }
    raw = (json.dumps(profile, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    (model_dir / "participation.json").write_bytes(raw)
    manifest["task_profiles"] = {"ip_team.participation": {
        "path": "participation.json", "sha256": hashlib.sha256(raw).hexdigest(),
    }}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"pinned {model_dir / 'participation.json'} ({profile['policy_version']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
