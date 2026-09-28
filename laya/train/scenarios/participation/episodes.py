"""Episodes (WRITING.md format) -> one laya record per decision point, validated.

    uv run --extra torch --extra train python train/scenarios/participation/episodes.py \
        check <episodes.jsonl>...                      # validate, print the decision mix
    uv run --extra torch --extra train python train/scenarios/participation/episodes.py \
        records <episodes.jsonl>... --out <records.jsonl> [--variants 2] [--source <name>]

Every event of an episode becomes a snapshot: the public record so far, the latest event as the
trigger, the round's user request pinned. Candidate ids are mapped to M0..Mn in a random order per
variant (so the model cannot learn positions or ids); the gold follows the mapping. Records are
plain laya records, fed to `assemble` as a source; the gen stage (smart-home adapters) is not used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from eidolon_laya_train.records import Record, record_id, target_vector  # noqa: E402
from eidolon_laya_train.scenario import Scenario  # noqa: E402

ACTIONS = ("respond", "clarify", "wait", "finish")
ABOUT = ("指代不明", "要求不明", "对象不在场")
HISTORY = 16  # public messages kept in the state (the round's user request is always pinned)


def load(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        for n, line in enumerate(Path(p).read_text("utf-8").splitlines(), 1):
            if line.strip():
                ep = json.loads(line)
                ep["_where"] = f"{Path(p).name}:{n}"
                out.append(ep)
    return out


def validate(ep: dict) -> list[str]:
    errs, where = [], ep.get("_where", "?")
    cands = ep.get("candidates") or []
    ids = [c.get("id") for c in cands]
    if ep.get("mode") not in ("team", "companion"):
        errs.append("mode")
    if ep.get("mode") == "companion" and len(cands) != 1:
        errs.append("companion needs exactly one candidate")
    if ep.get("mode") == "team" and not 2 <= len(cands) <= 5:
        errs.append("team needs 2-5 candidates")
    if len(set(ids)) != len(ids) or not all(ids):
        errs.append("candidate ids must be unique and non-empty")
    if not all(c.get("name") and c.get("role") for c in cands):
        errs.append("every candidate needs name and role")
    events = ep.get("events") or []
    if not events or "user" not in events[0]:
        errs.append("first event must be a user message")
    prev = None
    for i, ev in enumerate(events):
        g = ev.get("gold") or {}
        if ("user" in ev) == ("say" in ev):
            errs.append(f"event {i}: exactly one of user / say")
        if "say" in ev and ev["say"] not in ids:
            errs.append(f"event {i}: say by unknown id {ev['say']!r}")
        if "say" in ev and prev is not None and prev.get("action") in ("wait", "finish"):
            errs.append(f"event {i}: a role speaks after {prev['action']} (only the user may follow)")
        if "say" in ev and prev is not None and prev.get("action") in ("respond", "clarify") \
                and ev["say"] not in prev.get("speaker", []):
            errs.append(f"event {i}: {ev['say']} speaks but the previous gold chose {prev.get('speaker')}")
        if not (ev.get("text") if "say" in ev else ev.get("user")):
            errs.append(f"event {i}: empty text")
        a = g.get("action")
        if a not in ACTIONS:
            errs.append(f"event {i}: action {a!r}")
        sp = g.get("speaker")
        if a in ("respond", "clarify"):
            if not sp or not isinstance(sp, list) or not set(sp) <= set(ids):
                errs.append(f"event {i}: speaker must be a non-empty list of candidate ids")
        elif sp:
            errs.append(f"event {i}: {a} takes no speaker")
        if a == "clarify" and g.get("clarify_about") not in ABOUT:
            errs.append(f"event {i}: clarify needs clarify_about in {ABOUT}")
        if a != "clarify" and g.get("clarify_about"):
            errs.append(f"event {i}: clarify_about only with clarify")
        prev = g
    return [f"{where} {ep.get('episode')}: {e}" for e in errs]


def snapshots(ep: dict, rng: random.Random) -> list[tuple[dict, dict]]:
    """[(state, labels)] for one variant: a fresh random id -> M<n> mapping."""
    cands = list(ep["candidates"])
    order = list(range(len(cands)))
    rng.shuffle(order)
    slot = {cands[j]["id"]: f"M{k}" for k, j in enumerate(order)}
    listed = [cands[j] for j in order]
    out, public, request = [], [], None
    for ev in ep["events"]:
        if "user" in ev:
            request = ev["user"]
            public.append({"说话": "用户", "内容": ev["user"]})
        else:
            c = next(c for c in cands if c["id"] == ev["say"])
            public.append({"说话": f"{slot[c['id']]} {c['name']}", "内容": ev["text"]})
        state = {
            "场景目标": ep.get("scene_goal", ""),
            "候选": [{"编号": slot[c["id"]], "名字": c["name"], "角色": c["role"]} for c in listed],
            "本轮用户请求": request,
            "公开记录": public[-HISTORY:],
            "最新一条": f"{public[-1]['说话']}：{public[-1]['内容']}",
        }
        g = ev["gold"]
        labels = {"action": {"gold": g["action"]}}
        if g.get("speaker"):
            labels["speaker"] = {"gold": sorted(slot[s] for s in g["speaker"])}
        if g.get("clarify_about"):
            labels["clarify_about"] = {"gold": g["clarify_about"]}
        out.append((state, labels))
    return out


def to_records(eps: list[dict], variants: int, source: str, seed: int = 7) -> list[Record]:
    scn = Scenario.load(HERE)
    recs = []
    for ep in eps:
        rng = random.Random(f"{seed}:{ep['episode']}")
        for v in range(variants):
            for step, (state, labels) in enumerate(snapshots(ep, rng)):
                cand = {c["编号"]: f"{c['名字']}：{c['角色']}" for c in sorted(state["候选"], key=lambda c: c["编号"])}
                questions = scn.build_questions({"candidates": cand})
                for qid, lab in labels.items():
                    target_vector(questions[qid], lab)  # raises if a gold is not an option
                key = json.dumps(state, ensure_ascii=False, sort_keys=True)
                recs.append(Record(
                    id=record_id(scn.name, source, ep["episode"], f"v{v}", str(step), hashlib.sha1(key.encode()).hexdigest()[:8]),
                    scenario=scn.name, source=f"episodes:{source}", state=state, questions=questions, labels=labels,
                    tags=[f"family:{ep['family']}", f"mode:{ep['mode']}", f"action:{labels['action']['gold']}",
                          f"trigger:{'user' if 'user' in ep['events'][step] else 'role'}",
                          f"source:{ep.get('source', source)}",
                          f"episode:{ep['episode']}", f"step:{step}", f"variant:{v}"],
                    meta={"episode": ep["episode"], "family": ep["family"], "step": step, "variant": v,
                          # every snapshot and variant of an episode is one split group (assemble joins
                          # records by the part of derived_from before any "~")
                          "derived_from": f"episode:{source}:{ep['episode']}"},
                ))
    return recs


def blind(eps: list[dict]) -> list[dict]:
    """The same episodes with every gold removed: what a second annotator labels."""
    out = []
    for ep in eps:
        b = {k: v for k, v in ep.items() if k not in ("_where", "family")}
        b["events"] = [{k: v for k, v in ev.items() if k != "gold"} for ev in ep["events"]]
        out.append(b)
    return out


def blind_snapshots(eps: list[dict]) -> list[dict]:
    """One line per decision point with only the events up to it — what the model sees, no future events
    (whole-episode blind files leak the decision: the next event shows who actually spoke)."""
    out = []
    for ep in eps:
        for i in range(len(ep["events"])):
            out.append({"id": f"{ep['episode']}#{i}", "mode": ep["mode"], "scene_goal": ep.get("scene_goal", ""),
                        "candidates": ep["candidates"],
                        "events": [{k: v for k, v in ev.items() if k != "gold"} for ev in ep["events"][: i + 1]]})
    return out


def snapshot_answers(rows: list[dict]) -> list[dict]:
    """Fold a labelled snapshot file (each line has a top-level "gold") back into episodes for compare()."""
    by: dict[str, dict] = {}
    for r in rows:
        ep_id, i = r["id"].rsplit("#", 1)
        by.setdefault(ep_id, {})[int(i)] = r.get("gold") or {}
    return [{"episode": k, "events": [{"gold": v[i]} for i in sorted(v)]} for k, v in by.items()]


def compare(first: list[dict], second: list[dict]) -> dict:
    """Event-level agreement: action exact; speaker / clarify_about by intersection (lists mean any of)."""
    b = {ep["episode"]: ep for ep in second}
    n = act = spk = spk_n = about = about_n = 0
    dis = []
    for ep in first:
        other = b.get(ep["episode"])
        if other is None:
            continue
        for i, (x, y) in enumerate(zip(ep["events"], other["events"], strict=False)):
            g, h = x["gold"], y.get("gold") or {}
            n += 1
            ok_a = g["action"] == h.get("action")
            act += ok_a
            ok_s = ok_c = True
            if ok_a and g["action"] in ("respond", "clarify"):
                spk_n += 1
                ok_s = bool(set(g.get("speaker", [])) & set(h.get("speaker") or []))
                spk += ok_s
            if ok_a and g["action"] == "clarify":
                about_n += 1
                ok_c = g.get("clarify_about") == h.get("clarify_about")
                about += ok_c
            if not (ok_a and ok_s and ok_c):
                dis.append({"episode": ep["episode"], "event": i, "first": g, "second": h,
                            "text": x.get("user") or f"{x.get('say')}: {x.get('text')}"})
    return {"events": n, "action": round(act / n, 4) if n else None,
            "speaker": round(spk / spk_n, 4) if spk_n else None, "clarify_about": round(about / about_n, 4) if about_n else None,
            "disagreements": dis}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("files", nargs="+")
    r = sub.add_parser("records"); r.add_argument("files", nargs="+"); r.add_argument("--out", required=True)
    r.add_argument("--variants", type=int, default=2); r.add_argument("--source", required=True)
    bl = sub.add_parser("blind"); bl.add_argument("files", nargs="+"); bl.add_argument("--out", required=True)
    bs = sub.add_parser("blind-snapshots"); bs.add_argument("files", nargs="+"); bs.add_argument("--out", required=True)
    cp = sub.add_parser("compare"); cp.add_argument("first"); cp.add_argument("second"); cp.add_argument("--out")
    cp.add_argument("--snapshots", action="store_true", help="second file is a labelled blind-snapshots file")
    a = ap.parse_args()
    if a.cmd == "compare":
        second = load([a.second])
        res = compare(load([a.first]), snapshot_answers(second) if a.snapshots else second)
        if a.out:
            Path(a.out).write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in res["disagreements"]), "utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "disagreements"} | {"n_disagreements": len(res["disagreements"])}, ensure_ascii=False))
        return 0
    eps = load(a.files)
    if a.cmd == "blind-snapshots":
        rows = blind_snapshots(eps)
        Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), "utf-8")
        print(f"{len(rows)} blind snapshots -> {a.out}")
        return 0
    if a.cmd == "blind":
        Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in blind(eps)), "utf-8")
        print(f"{len(eps)} blind episodes -> {a.out}")
        return 0
    errs = [e for ep in eps for e in validate(ep)]
    ids = Counter(ep.get("episode") for ep in eps)
    errs += [f"duplicate episode id {k}" for k, n in ids.items() if n > 1]
    if errs:
        print("\n".join(errs[:50]), file=sys.stderr)
        print(f"{len(errs)} problems in {len(eps)} episodes", file=sys.stderr)
        return 1
    mix = Counter(ev["gold"]["action"] for ep in eps for ev in ep["events"])
    fam = Counter(ep["family"] for ep in eps)
    print(f"{len(eps)} episodes, {sum(mix.values())} decision points; actions {dict(mix)}; "
          f"modes {dict(Counter(ep['mode'] for ep in eps))}; families {len(fam)}")
    if a.cmd == "records":
        from eidolon_laya_train.records import write_jsonl

        recs = to_records(eps, a.variants, a.source)
        write_jsonl(a.out, recs)
        print(f"wrote {len(recs)} records -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
