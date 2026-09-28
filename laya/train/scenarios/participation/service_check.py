"""Serve p-dev through the service adapter and compare with the offline eval, decision point by decision point.

    uv run --extra torch --extra train python train/scenarios/participation/service_check.py \
        --model-dir <packaged model with participation.json> --report train/runs/<run>/eval/eval-dev.json

Builds each variant-0 snapshot as an SDK DecisionRequest (candidates in that variant's slot order, so the
service state is the training state byte for byte), runs LayaParticipationPredictor on the real engine and
checks that action / speaker match the offline eval's predictions. Any mismatch means the service does not
see what the model was trained and evaluated on.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))

from eidolon_sdk.biz.participation import Candidate, Context, DecisionRequest, Message  # noqa: E402

from episodes import HISTORY, load, snapshots  # noqa: E402

EVALS = HERE.parents[3] / "evals" / "participation-laya"


def requests_for(ep: dict, seed: int = 7):
    rng = random.Random(f"{seed}:{ep['episode']}")
    snaps = snapshots(ep, rng)  # variant 0: the first draw, as to_records makes it
    slot_order = [c["编号"] for c in snaps[0][0]["候选"]]
    by_slot = {}
    for c in ep["candidates"]:
        for s in snaps[0][0]["候选"]:
            if s["名字"] == c["name"] and s["角色"] == c["role"] and s["编号"] not in by_slot:
                by_slot[s["编号"]] = c
                break
    cands = tuple(Candidate(companion_id=by_slot[s]["id"], display_name=by_slot[s]["name"],
                            description=by_slot[s]["role"]) for s in slot_order)
    msgs, request = [], None
    for step, ev in enumerate(ep["events"]):
        if "user" in ev:
            m = Message(message_id=f"m{step}", author_kind="user", author_id="user", text=ev["user"])
            request = m
        else:
            m = Message(message_id=f"m{step}", author_kind="companion", author_id=ev["say"], text=ev["text"])
        msgs.append(m)
        yield step, DecisionRequest(
            decision_id=f"{ep['episode']}#{step}", context_ref=ep["episode"], context_version=step,
            membership_revision=0, cancellation_epoch=0, user_request=request, trigger=m,
            scene_goal=ep.get("scene_goal", ""), context=Context(recent_messages=tuple(msgs[-HISTORY:])),
            candidates=cands, timeout_ms=5000,
        ), snaps[step][0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    from eidolon_models_laya.config import Settings
    from eidolon_models_laya.engine import load_engine
    from eidolon_models_laya.participation import load_participation_adapter, participation_profile_digest

    settings = Settings(model_dir=Path(a.model_dir), device="cpu")
    engine, manifest = load_engine(settings, log=lambda *_: None)
    adapter = load_participation_adapter(
        Path(a.model_dir) / "participation.json", engine, model_dir=Path(a.model_dir),
        model_revision=manifest.revision, expected_sha256=participation_profile_digest(manifest.raw))
    adapter.min_confidence = 1e-9  # compare raw decisions; the threshold is policy, checked elsewhere
    offline = {}
    for r in json.loads(Path(a.report).read_text("utf-8"))["rows"]:
        t = dict(x.split(":", 1) for x in r["tags"] if ":" in x)
        if t.get("variant") == "0":
            offline.setdefault((t["episode"], int(t["step"])), {})[r["qid"]] = r["pred"]
    eps = load([str(p) for p in sorted((EVALS / "dev").glob("*.jsonl"))])
    n = same_action = same_speaker = n_speaker = state_same = 0
    bad = []
    for ep in eps:
        for step, req, train_state in requests_for(ep):
            state, _, ids = adapter.predictor.state(req)
            state_same += state == train_state
            result = adapter.decide(req)
            got = result.proposal.action if result.proposal else "abstain"
            want = offline.get((ep["episode"], step), {})
            n += 1
            same_action += got == want.get("action")
            if got in ("respond", "clarify") and "speaker" in want:
                slot = f"M{ids.index(result.proposal.participants[0])}"
                n_speaker += 1
                same_speaker += slot == want["speaker"]
            if got != want.get("action"):
                bad.append((ep["episode"], step, got, want.get("action")))
    print(json.dumps({"decision_points": n, "state_identical": state_same, "action_same": same_action,
                      "speaker_same": f"{same_speaker}/{n_speaker}", "mismatches": bad[:10]}, ensure_ascii=False))
    return 0 if same_action == n and state_same == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
