"""Two members in one conversation, with device templates (design §8.2 A1: the two-person re-test).

Channel assigns each segment to the active track of its own voice by acoustic
evidence alone (``track_continuation_llr``, design §3.2); a segment that joins
the other member's track takes that track's identity. B1 measured another
voice joining about 1 % of the time and estimated that, once device templates
exist, misattribution for ``personal_memory`` could reach 0.6 %. This replays
it on the same held-out folds as ``track_eval.py``:

- each member first holds conversations alone on the held-out far device;
  attributed tracks become device templates there, as in B1;
- then the two talk in the same conversations, their segments interleaved at
  random, tracked as Channel tracks them (scored on a track's latest segments
  within the calibrated length, at most six tracks), and every turn is
  decided on its track with the delivered policy;
- tracks of those mixed conversations that get attributed become templates
  too (as Channel submits them), and a second mixed conversation uses them.

Reported per use: turns named correctly and wrongly, for each mixed
conversation, at the policy's continuation threshold and at stricter ones
(inf: a segment never continues a track). Templates made from mixed tracks
are counted with their purity.

    eidolon_channel/.venv/bin/python -I \\
        eidolon_models/evals/speaker-verification/tracks-20261102/two_person_eval.py \\
        --corpus <3d-speaker> --embeddings <3d-speaker>/embeddings.json \\
        --policy eidolon_models/speaker_verification/campplus_zh_16k_common/624dcb0f/identity-policy.json \\
        --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAX_TRACKS = 6
MAX_SEGMENTS = 8
WARM_SESSIONS = 2
MIXED_CONVERSATIONS = 2
STRICTER = (8.0, 9.0, 10.0, math.inf)


def track_eval():
    spec = importlib.util.spec_from_file_location("track_eval", HERE / "track_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


te = track_eval()
it = te.it


def assign(scorer, tracks, segment, threshold):
    """Channel's rule: continue the best-matching track at or above the threshold, else open one."""
    scored = [(scorer.llr("far_same", te.track_of(t[-max(it.TRACKS["far_same"]):]), [segment]), t)
              for t in tracks]
    matching = [(llr, t) for llr, t in scored if llr >= threshold]
    if matching:
        track = max(matching, key=lambda x: x[0])[1]
    else:
        track = []
        tracks.append(track)
        if len(tracks) > MAX_TRACKS:
            tracks.pop(0)
    track.append(segment)
    del track[:-MAX_SEGMENTS]
    return track


def mixed_conversation(scorer, policy, members, conversation, device, threshold, tally, key, made):
    """One conversation of both members; templates made from its attributed tracks go to ``made``."""
    tracks = []
    final = {}
    for segment in conversation:
        track = assign(scorer, tracks, segment, threshold)
        lead, _, _, named = te.decide(scorer, policy, members, track, device)
        tally.add(key, named, segment["speaker"])
        final[id(track)] = (track, lead, named)
    for track, lead, named in final.values():
        if named["personal_memory"] is not None and te.track_of(track)[1] >= te.ATTRIBUTION_SPEECH_MS:
            purity = float(np.mean([u["speaker"] == lead for u in track]))
            made[lead].append((te.track_of(track), purity))


def interleave(sessions, rng):
    """Both members' segments in one conversation, each in its own order, turns at random."""
    queue = [list(s) for s in sessions]
    order = rng.permutation([i for i, s in enumerate(queue) for _ in s])
    return [queue[i].pop(0) for i in order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "work"
    work.mkdir(exist_ok=True)
    calibration, _ = it.wespeaker()
    policy = json.loads(args.policy.read_text())
    thresholds = (policy["track_continuation_llr"], *STRICTER)
    utts = it.corpus(args.corpus, args.embeddings)
    places = sorted({p for u in utts if (p := it.place(u))})
    near_places = [p for p in places if p.startswith("near:")]
    far_devices = sorted({it.device_of(p) for p in places if p.startswith("far:")})
    speakers = sorted({u["speaker"] for u in utts})
    halves = (set(speakers[0::2]), set(speakers[1::2]))
    tested = te.leak_free(utts, set(places))

    tallies = {t: te.Tally() for t in thresholds}
    templates_made = {t: [] for t in thresholds}
    households = 0
    fold = 0
    for train_spk, test_spk in (halves, halves[::-1]):
        for near_out in near_places:
            for far_out in far_devices:
                fold += 1
                rng = np.random.default_rng(fold)
                seen = [p for p in places if p != near_out
                        and not (p.startswith("far:") and it.device_of(p) == far_out)]
                scorer = te.fold_backend(calibration, utts, train_spk, seen, rng, work, fold)
                by = defaultdict(lambda: defaultdict(list))
                for u in tested:
                    p = it.place(u)
                    if u["speaker"] in test_spk and p and (p == near_out or (
                            p.startswith("far:") and it.device_of(p) == far_out)):
                        by[u["speaker"]][p].append(u)
                regs, warm, mixed = {}, {}, {}
                for s, g in sorted(by.items()):
                    phone = sorted(g.get(near_out, []), key=lambda u: u["segment"])
                    far = [u for p, us in g.items() if p.startswith("far:") for u in us]
                    sessions = te.sessions_of(far, rng)
                    if len(phone) <= it.REGISTRATION_SEGMENTS or len(sessions) < WARM_SESSIONS + MIXED_CONVERSATIONS:
                        continue
                    regs[s] = it.template(phone[:it.REGISTRATION_SEGMENTS])
                    warm[s] = sessions[:WARM_SESSIONS]
                    mixed[s] = sessions[WARM_SESSIONS:WARM_SESSIONS + MIXED_CONVERSATIONS]
                people = sorted(regs)
                for a, b in zip(people[0::2], people[1::2]):
                    households += 1
                    clean = te.household_run(scorer, policy, regs, warm, a, b, far_out,
                                             te.Tally(), defaultdict(int))
                    conversations = [interleave((mixed[a][i], mixed[b][i]), rng)
                                     for i in range(MIXED_CONVERSATIONS)]
                    for threshold in thresholds:
                        templates = {m: list(clean[m]) for m in (a, b)}
                        for i, conversation in enumerate(conversations, 1):
                            members = {m: (regs[m], templates[m][-te.MAX_DEVICE_TEMPLATES:]) for m in (a, b)}
                            made = {a: [], b: []}
                            warmth = "warm" if any(templates.values()) else "cold"
                            mixed_conversation(scorer, policy, members, conversation, far_out, threshold,
                                               tallies[threshold], f"mixed{i}/{warmth}", made)
                            for m in (a, b):
                                templates[m] += [t for t, _ in made[m]]
                                templates_made[threshold] += [purity for _, purity in made[m]]
                print(f"fold {fold}: {near_out} / {far_out}: {len(people)} speakers", flush=True)

    report = {"setup": {"folds": fold, "households": households, "warm_sessions": WARM_SESSIONS,
                        "mixed_conversations": MIXED_CONVERSATIONS, "policy": policy["version"]},
              "by_continuation_threshold": {
                  str(t): {"turns": tallies[t].report(),
                           "templates_from_mixed_tracks": {
                               "n": len(templates_made[t]),
                               "impure": int(sum(p < 1 for p in templates_made[t])),
                               "min_purity": round(min(templates_made[t]), 3) if templates_made[t] else None}}
                  for t in thresholds}}
    (args.out / "two_person.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
