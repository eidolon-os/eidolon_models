"""Tracks and per-device observation templates on live multi-device speech (design §3.2, §8.2 B1).

Registration happens on the phone; speech arrives on a far device. Within a
conversation, a voice's turns form a track scored as one probe (its mean
vector, voiced speech summed: design §3.3 L5). Once a track is attributed
(``personal_memory`` met, at least 8 s of voiced speech: design §3.8 ②), it
becomes that Person's observation template for the device, and later
conversations there compare on the same device (``far_same``).

The 3D-Speaker test set stands in, held out twice as in
``identity-20261101``: each half of the speakers is tested with a back-end
trained on the other, and every fold holds out one close device (the phone)
and one far device (the board), so neither was seen in training.

A household of two registered members talks on the held-out far device,
conversation by conversation (A, B, A, B, ...), each conversation one track of
a few segments. Every turn is decided on its track so far with the delivered
policy's thresholds. Strangers (other held-out speakers on the same device)
probe the household with and without its device templates, and a member is
probed against a household where only the other is registered.

Simultaneous recording: 3D-Speaker records each sentence on up to seven
devices at once. A far segment that is the very sentence of a registration is
not a fair probe (the product never hears the registration again), so such
segments are excluded here, as the back-end's training excludes them.

    eidolon_channel/.venv/bin/python -I \\
        eidolon_models/evals/speaker-verification/tracks-20261102/track_eval.py \\
        --corpus /Users/manson/ai/data/speaker-corpora/3d-speaker \\
        --embeddings /Users/manson/ai/data/speaker-corpora/3d-speaker/embeddings.json \\
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
SESSION_SEGMENTS = 6
MAX_SESSIONS = 4
ATTRIBUTION_SPEECH_MS = 8000
MAX_DEVICE_TEMPLATES = 3
STRANGERS_PER_HOUSEHOLD = 12
USES = ("personal_memory", "addressing", "speaker_label")


def identity_train():
    spec = importlib.util.spec_from_file_location(
        "identity_train", HERE.parent / "identity-20261101" / "identity_train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


it = identity_train()


def registration_segments(groups, near_places):
    """The sentences of a speaker's registrations on these close devices."""
    out = set()
    for p in near_places:
        us = sorted(groups.get(p, []), key=lambda u: u["segment"])
        if len(us) > it.REGISTRATION_SEGMENTS:
            out |= {u["segment"] for u in us[:it.REGISTRATION_SEGMENTS]}
    return out


def leak_free(utts, places):
    """Drop far segments that are a registration's very sentence (recorded at the same moment)."""
    by = defaultdict(lambda: defaultdict(list))
    for u in utts:
        if (p := it.place(u)) in places:
            by[u["speaker"]][p].append(u)
    near = [p for p in places if p.startswith("near:")]
    banned = {s: registration_segments(g, near) for s, g in by.items()}
    return [u for u in utts if not ((p := it.place(u)) and p.startswith("far:")
                                    and u["segment"] in banned.get(u["speaker"], ()))]


class Scorer:
    """The runtime's arithmetic for one fold's back-end: a track is scored on its latest
    segments within the longest track the condition was calibrated on."""

    def __init__(self, space, calibrations):
        self.space, self.calibrations = space, calibrations  # condition -> (w, b)

    def llr(self, condition, template, segments):
        (tv, tms), (pv, pms) = template, track_of(segments[-max(it.TRACKS[condition]):])
        enroll, track = self.space.side(tv, tms, "template"), self.space.side(pv, pms, "track")
        w, b = self.calibrations[condition]
        return float(it.trial_features(enroll, track, condition, it.MAX_SPEECH_MS) @ w + b)


def posteriors(llrs, prior_unknown):
    """Open-set posteriors: each enrolled member equally likely a priori."""
    names = list(llrs)
    log_each = math.log((1 - prior_unknown) / len(names))
    terms = np.array([math.log(prior_unknown)] + [log_each + llrs[n] for n in names])
    top = terms.max()
    w = np.exp(terms - top)
    w /= w.sum()
    return dict(zip(names, w[1:]))


def decide(scorer, policy, members, segments, device):
    """The leading member, the condition that decided, and each use's named member (or None).

    The track is scored on its latest segments within each condition's calibrated length.
    """
    best = {}
    for name, (registration, device_templates) in members.items():
        options = [("far_cross", registration)] + [("far_same", t) for t in device_templates]
        best[name] = max(((scorer.llr(c, t, segments), c) for c, t in options), key=lambda x: x[0])
    post = posteriors({n: v[0] for n, v in best.items()}, policy["prior_unknown"])
    lead = max(post, key=post.get)
    condition = best[lead][1]
    thresholds = policy["conditions"][condition]
    named = {u: (lead if thresholds[u] is not None and post[lead] >= thresholds[u] else None) for u in USES}
    return lead, post[lead], condition, named


def track_of(segments):
    return it.template(segments)


def sessions_of(us, rng):
    us = [us[i] for i in rng.permutation(len(us))]
    out = [us[i:i + SESSION_SEGMENTS] for i in range(0, len(us) - SESSION_SEGMENTS + 1, SESSION_SEGMENTS)]
    return out[:MAX_SESSIONS]


class Tally:
    def __init__(self):
        self.c = defaultdict(lambda: defaultdict(float))

    def add(self, key, named, truth):
        c = self.c[key]
        c["n"] += 1
        for u in USES:
            if named[u] is None:
                continue
            c[f"{u}:{'correct' if named[u] == truth else 'wrong'}"] += 1

    def report(self):
        return {k: {"n": int(c["n"]), **{m: round(v / c["n"], 5) for m, v in sorted(c.items()) if m != "n"}}
                for k, c in sorted(self.c.items())}


def bucket(j):
    return f"k{j}" if j < 3 else "k3+"


def household_run(scorer, policy, regs, sess, a, b, device, own, events):
    """Members A and B alternate conversations; templates are added as tracks get attributed."""
    templates = {a: [], b: []}
    order = [(m, i) for i in range(MAX_SESSIONS) for m in (a, b) if i < len(sess[m])]
    for speaker, i in order:
        segments = sess[speaker][i]
        warm = bool(templates[speaker])
        for j in range(1, len(segments) + 1):
            track = track_of(segments[:j])
            members = {m: (regs[m], templates[m]) for m in (a, b)}
            lead, p, condition, named = decide(scorer, policy, members, segments[:j], device)
            own.add(f"{'warm' if warm else 'cold'}/{bucket(j)}", named, speaker)
            own.add(f"session{i + 1}/{bucket(j)}", named, speaker)
        if named["personal_memory"] is not None and track[1] >= ATTRIBUTION_SPEECH_MS:
            events["attributed" if lead == speaker else "misattributed"] += 1
            templates[lead] = (templates[lead] + [track])[-MAX_DEVICE_TEMPLATES:]
        else:
            events["not_attributed"] += 1
    return templates


def probe_household(scorer, policy, members, sessions, device, tally, prefix, truth):
    for segments in sessions:
        for j in range(1, len(segments) + 1):
            _, _, _, named = decide(scorer, policy, members, segments[:j], device)
            tally.add(f"{prefix}/{bucket(j)}", named, truth)


CURVE_BUCKETS = ((1, 1), (2, 2), (3, 5), (6, 11), (12, 17), (18, 99))


def curve_bucket(k):
    return next(f"k{lo}-{hi}" if hi < 99 else f"k{lo}+" for lo, hi in CURVE_BUCKETS if lo <= k <= hi)


def accumulation_rows(scorer, policy, regs, sess, a, b, strangers, rows):
    """Cold start only: a voice's segments accumulated across all its conversations on the device.

    rows[kind] collects (k, posterior of the leader, leader is the speaker) for the
    members ('own'), for strangers against the household ('stranger'), and for a
    member against a household where only the other is registered ('member').
    """
    def walk(members, speaker, kind):
        segments = [u for s in sess[speaker] for u in s]
        for j in range(1, len(segments) + 1):
            lead, p, _, _ = decide(scorer, policy, members, segments[:j], None)
            rows[kind].append((j, p, lead == speaker))
    household = {m: (regs[m], []) for m in (a, b)}
    for m in (a, b):
        walk(household, m, "own")
    for s in strangers:
        walk(household, s, "stranger")
    walk({a: (regs[a], [])}, b, "member")
    walk({b: (regs[b], [])}, a, "member")


def operating_points(rows, policy):
    """Per accumulated length: coverage at the policy's far_cross thresholds, and the best any
    threshold could do there while strangers and members stay within each use's limit."""
    limits = {"personal_memory": 0.005, "addressing": 0.02}
    out = {}
    for lo, hi in CURVE_BUCKETS:
        pick = {kind: [(p, ok) for k, p, ok in rows[kind] if lo <= k <= hi] for kind in rows}
        if not pick["own"]:
            continue
        own = np.array(pick["own"], dtype=float)
        fa = {kind: np.array([p for p, _ in pick[kind]]) for kind in ("stranger", "member")}
        entry = {"own_n": len(own), "stranger_n": len(fa["stranger"]), "member_n": len(fa["member"])}
        for use, limit in limits.items():
            t_policy = policy["conditions"]["far_cross"][use]
            entry[f"{use}:policy_coverage"] = round(float(np.mean((own[:, 0] >= t_policy) & (own[:, 1] == 1))), 4)
            candidates = np.unique(np.concatenate([own[:, 0], *fa.values()]))
            ok = np.ones(len(candidates), bool)
            for values in fa.values():
                if len(values):
                    above = 1 - np.searchsorted(np.sort(values), candidates, side="left") / len(values)
                    ok &= above <= limit
            best = candidates[np.argmax(ok)] if ok.any() else None
            entry[f"{use}:best_coverage"] = (None if best is None else
                                            round(float(np.mean((own[:, 0] >= best) & (own[:, 1] == 1))), 4))
        out[curve_bucket(lo)] = entry
    return out


def continuation_rows(scorer, sess, a, b, rows):
    """A segment against the track so far of its own voice ('same') and of the other member ('other'),
    on the same device: the evidence for continuing a track (design §3.2)."""
    for x, y in ((a, b), (b, a)):
        for i, segments in enumerate(sess[x]):
            other = sess[y][i] if i < len(sess[y]) else sess[y][-1]
            for j in range(1, len(segments)):
                probe = segments[j:j + 1]
                rows["same"].append((j, scorer.llr("far_same", track_of(segments[:j]), probe)))
                rows["other"].append((j, scorer.llr("far_same", track_of(other[:j]), probe)))


def continuation_points(rows):
    """Continuing one's own track, at thresholds where a different voice joins a track this rarely."""
    same = np.array([v for _, v in rows["same"]])
    other = np.sort([v for _, v in rows["other"]])
    out = {"same_n": len(same), "other_n": len(other)}
    for rate in (0.01, 0.005, 0.001):
        t = float(other[min(len(other) - 1, int(math.ceil((1 - rate) * len(other))))])
        out[f"other_joins_{rate}"] = {"llr": round(t, 3), "own_continues": round(float(np.mean(same > t)), 4)}
    return out


def fold_backend(calibration, utts, train_spk, seen, rng, work, fold):
    space = it.fit_space([u for u in utts if u["speaker"] in train_spk and it.place(u) in seen])
    train = it.structures(utts, train_spk, set(seen), rng)
    calibrations = {}
    for c in ("far_cross", "far_same"):
        rows = it.feature_rows(space, c, *train[c], rng)
        w, b, _, _ = it.train_qmf(calibration, rows, work, f"fold{fold}-{c}")
        calibrations[c] = (w, b)
    return Scorer(space, calibrations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True, help="the delivered identity-policy.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "work"
    work.mkdir(exist_ok=True)
    calibration, _ = it.wespeaker()
    policy = json.loads(args.policy.read_text())
    utts = it.corpus(args.corpus, args.embeddings)
    places = sorted({p for u in utts if (p := it.place(u))})
    near_places = [p for p in places if p.startswith("near:")]
    far_devices = sorted({it.device_of(p) for p in places if p.startswith("far:")})
    speakers = sorted({u["speaker"] for u in utts})
    halves = (set(speakers[0::2]), set(speakers[1::2]))
    tested = leak_free(utts, set(places))

    own, stranger, member = Tally(), Tally(), Tally()
    curve = {"own": [], "stranger": [], "member": []}
    joins = {"same": [], "other": []}
    events = defaultdict(int)
    fold = 0
    for train_spk, test_spk in (halves, halves[::-1]):
        for near_out in near_places:
            for far_out in far_devices:
                fold += 1
                rng = np.random.default_rng(fold)
                seen = [p for p in places if p != near_out
                        and not (p.startswith("far:") and it.device_of(p) == far_out)]
                scorer = fold_backend(calibration, utts, train_spk, seen, rng, work, fold)
                by = defaultdict(lambda: defaultdict(list))
                for u in tested:
                    p = it.place(u)
                    if u["speaker"] in test_spk and p and (p == near_out or (
                            p.startswith("far:") and it.device_of(p) == far_out)):
                        by[u["speaker"]][p].append(u)
                regs, sess = {}, {}
                for s, g in sorted(by.items()):
                    phone = sorted(g.get(near_out, []), key=lambda u: u["segment"])
                    far = [u for p, us in g.items() if p.startswith("far:") for u in us]
                    if len(phone) <= it.REGISTRATION_SEGMENTS or len(far) < SESSION_SEGMENTS:
                        continue
                    regs[s] = it.template(phone[:it.REGISTRATION_SEGMENTS])
                    sess[s] = sessions_of(far, rng)
                people = sorted(regs)
                for a, b in zip(people[0::2], people[1::2]):
                    templates = household_run(scorer, policy, regs, sess, a, b, far_out, own, events)
                    others = [s for s in people if s not in (a, b)]
                    strangers = list(rng.permutation(others)[:STRANGERS_PER_HOUSEHOLD])
                    accumulation_rows(scorer, policy, regs, sess, a, b, strangers, curve)
                    continuation_rows(scorer, sess, a, b, joins)
                    for s in strangers:
                        cold = {m: (regs[m], []) for m in (a, b)}
                        warm = {m: (regs[m], templates[m]) for m in (a, b)}
                        probe_household(scorer, policy, cold, sess[s], far_out, stranger, "cold", s)
                        probe_household(scorer, policy, warm, sess[s], far_out, stranger, "warm", s)
                    for x, y in ((a, b), (b, a)):
                        probe_household(scorer, policy, {x: (regs[x], templates[x])}, sess[y],
                                        far_out, member, "warm", y)
                        probe_household(scorer, policy, {x: (regs[x], [])}, sess[y],
                                        far_out, member, "cold", y)
                print(f"fold {fold}: {near_out} / {far_out}: {len(people)} speakers", flush=True)

    report = {"setup": {"folds": fold, "session_segments": SESSION_SEGMENTS, "max_sessions": MAX_SESSIONS,
                        "attribution_speech_ms": ATTRIBUTION_SPEECH_MS, "policy": policy["version"]},
              "events": dict(events), "own": own.report(), "stranger": stranger.report(),
              "member_as_stranger": member.report(),
              "cold_start_accumulation": operating_points(curve, policy),
              "track_continuation": continuation_points(joins)}
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
