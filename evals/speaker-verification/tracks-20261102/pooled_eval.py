"""Device templates pooled from attributed turns, under Channel's real track continuation (A1).

``track_eval.py`` took each conversation as one track. Channel continues a
track only on acoustic evidence, and on a far device a voice continues its
own track about a fifth of the time (B1), so tracks there rarely reach the
8 s a template needs, and in ``two_person_eval.py`` no mixed conversation
made one. The alternative tested here: every turn that meets
``personal_memory`` is attributed on its own evidence, and its segment joins
that Person's pool on the device; consecutive pooled segments form a
template once they hold 8 s of voiced speech (at most three, the latest), and
a conversation never compares against what it pooled itself.

On the held-out far device of each fold, for speakers with enough segments
there: the first two thirds of a member's segments are their use (single-
speaker conversations of six, A and B alternating), the last third tests
each turn on its track, with the pooled templates and with the phone
template alone. Strangers (other held-out speakers there) probe the household
with its pooled templates; each member is probed against a household where
only the other is registered. Pool purity counts other voices pooled.

    eidolon_channel/.venv/bin/python -I \\
        eidolon_models/evals/speaker-verification/tracks-20261102/pooled_eval.py \\
        --corpus <3d-speaker> --embeddings <3d-speaker>/embeddings.json \\
        --policy eidolon_models/speaker_verification/campplus_zh_16k_common/624dcb0f/identity-policy.json \\
        --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MIN_SEGMENTS = 24
CONVERSATION = 6
STRANGERS_PER_HOUSEHOLD = 12


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


tp = module("two_person_eval")
te, it = tp.te, tp.it


def templates_of(pool):
    """Consecutive pooled segments, each group reaching the template's voiced speech; the latest three."""
    groups, current = [], []
    for u in pool:
        current.append(u)
        if sum(x["speech_ms"] for x in current) >= te.ATTRIBUTION_SPEECH_MS:
            groups.append(te.track_of(current))
            current = []
    return groups[-te.MAX_DEVICE_TEMPLATES:]


def converse(scorer, policy, members, segments, device, threshold, tally, key, truth):
    """One conversation tracked as Channel tracks it; returns the segments each member was attributed."""
    tracks, pooled = [], defaultdict(list)
    for segment in segments:
        track = tp.assign(scorer, tracks, segment, threshold)
        lead, _, _, named = te.decide(scorer, policy, members, track, device)
        if tally is not None:
            tally.add(key, named, truth)
        if named["personal_memory"] is not None:
            pooled[named["personal_memory"]].append(segment)
    return pooled


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
    threshold = policy["track_continuation_llr"]
    utts = it.corpus(args.corpus, args.embeddings)
    places = sorted({p for u in utts if (p := it.place(u))})
    near_places = [p for p in places if p.startswith("near:")]
    far_devices = sorted({it.device_of(p) for p in places if p.startswith("far:")})
    speakers = sorted({u["speaker"] for u in utts})
    halves = (set(speakers[0::2]), set(speakers[1::2]))
    tested = te.leak_free(utts, set(places))

    own, stranger, member = te.Tally(), te.Tally(), te.Tally()
    pools = {"segments": 0, "other_voice": 0, "members": 0, "with_templates": 0, "templates": 0}
    fold = 0
    for train_spk, test_spk in (halves, halves[::-1]):
        for near_out in near_places:
            for far_out in far_devices:
                fold += 1
                rng = np.random.default_rng(fold)
                seen = [p for p in places if p != near_out
                        and not (p.startswith("far:") and it.device_of(p) == far_out)]
                by = defaultdict(lambda: defaultdict(list))
                for u in tested:
                    p = it.place(u)
                    if u["speaker"] in test_spk and p and (p == near_out or (
                            p.startswith("far:") and it.device_of(p) == far_out)):
                        by[u["speaker"]][p].append(u)
                regs, use, test, probes = {}, {}, {}, {}
                for s, g in sorted(by.items()):
                    phone = sorted(g.get(near_out, []), key=lambda u: u["segment"])
                    far = [u for p, us in g.items() if p.startswith("far:") for u in us]
                    if len(phone) <= it.REGISTRATION_SEGMENTS or len(far) < CONVERSATION:
                        continue
                    regs[s] = it.template(phone[:it.REGISTRATION_SEGMENTS])
                    far = [far[i] for i in rng.permutation(len(far))]
                    probes[s] = far[:CONVERSATION]
                    if len(far) >= MIN_SEGMENTS:
                        cut = len(far) * 2 // 3
                        use[s], test[s] = far[:cut], far[cut:]
                people = sorted(use)
                if len(people) < 2:
                    continue
                scorer = te.fold_backend(calibration, utts, train_spk, seen, rng, work, fold)
                for a, b in zip(people[0::2], people[1::2]):
                    pool = {a: [], b: []}
                    chunks = {m: [use[m][i:i + CONVERSATION] for i in range(0, len(use[m]), CONVERSATION)]
                              for m in (a, b)}
                    for i in range(max(len(chunks[a]), len(chunks[b]))):
                        for m in (a, b):
                            if i >= len(chunks[m]):
                                continue
                            members = {x: (regs[x], templates_of(pool[x])) for x in (a, b)}
                            pooled = converse(scorer, policy, members, chunks[m][i], far_out, threshold,
                                              None, None, m)
                            for x, segments in pooled.items():
                                pool[x] += segments
                    warm = {x: (regs[x], templates_of(pool[x])) for x in (a, b)}
                    cold = {x: (regs[x], []) for x in (a, b)}
                    for x in (a, b):
                        pools["members"] += 1
                        pools["segments"] += len(pool[x])
                        pools["other_voice"] += sum(u["speaker"] != x for u in pool[x])
                        pools["with_templates"] += bool(warm[x][1])
                        pools["templates"] += len(warm[x][1])
                        for i in range(0, len(test[x]), CONVERSATION):
                            segments = test[x][i:i + CONVERSATION]
                            key = "warm" if warm[x][1] else "no_template_yet"
                            converse(scorer, policy, warm, segments, far_out, threshold, own, key, x)
                            converse(scorer, policy, cold, segments, far_out, threshold, own,
                                     "cold/" + key, x)
                    others = [s for s in probes if s not in (a, b)]
                    for s in list(rng.permutation(others)[:STRANGERS_PER_HOUSEHOLD]):
                        converse(scorer, policy, warm, probes[s], far_out, threshold, stranger, "warm", s)
                    for x, y in ((a, b), (b, a)):
                        alone = {x: warm[x]}
                        converse(scorer, policy, alone, test[y][:CONVERSATION], far_out, threshold,
                                 member, "warm", y)
                print(f"fold {fold}: {near_out} / {far_out}: {len(people)} speakers with enough use",
                      flush=True)

    report = {"setup": {"folds": fold, "min_segments": MIN_SEGMENTS, "conversation": CONVERSATION,
                        "continuation_llr": threshold, "policy": policy["version"]},
              "pools": pools, "own": own.report(), "stranger": stranger.report(),
              "member_as_stranger": member.report()}
    (args.out / "pooled.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
