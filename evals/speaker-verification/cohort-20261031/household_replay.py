"""The identity back-end on a household's own phone channel (design §8.2, 1c-a).

Public speakers were replayed next to the household's phone and captured
through its app ("声纹环境校准", ``replay_corpus.py``): in-domain impostors
the public-corpus calibration never saw. This script

1. assigns each captured segment to the one replayed speaker it overlaps
   (Admin stamps the time a segment arrives, the Mac logs every speaker's
   block; a segment touching two blocks is dropped) and checks the
   assignment against the speakers' own embeddings from ``build_cohort.py``;
2. measures the delivered back-end on the household: replayed speakers as
   strangers, each member's registration as genuine (one recording left out
   of the template in turn, plus the trial recording);
3. adapts the back-end with the replayed cohort speakers, using WeSpeaker's
   own scripts through ``identity_eval.py``: in the AS-norm cohort, in the
   QMF's training trials, or both. Each is compared on replayed held-out
   speakers, on the household and on the public-corpus held-out trials.

Two checks say how far the replay can be trusted. ``channel`` compares
similarities across speakers: replayed through a loudspeaker, members
speaking live, and the public corpus. ``members_as_strangers`` treats each
member as a stranger to a household where only the other is enrolled: the
one pair of live, in-room impostors there is.

The household's registrations are read from Data's SQLite, read-only; the
captured segments from their export (``<batch>.samples.json``, a list of
``{"captured_at", "sample"}`` taken from Data's ``channel_calibration_samples``
before migration 0030 dropped it). Writes aggregates only: no names, no
vectors.

    eidolon_channel/.venv/bin/python -I \\
        eidolon_models/evals/speaker-verification/cohort-20261031/household_replay.py \\
        --db .eidolon/mac-product/state/eidolon-system.sqlite3 \\
        --log .eidolon/speaker-cohort-20261031/replay/<batch>.jsonl \\
        --samples .eidolon/speaker-cohort-20261031/replay/<batch>.samples.json \\
        --embeddings .eidolon/speaker-cohort-20261031/build/embeddings.json \\
        --delivered eidolon_models/speaker_verification/campplus_zh_16k_common/624dcb0f \\
        --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHANNEL = "mobile"
# Between the end of a recording and Admin's stamp: stopping, encoding, upload.
ARRIVAL_SLACK_S = 1.5
POLICY_V2 = {"personal_memory": 0.99, "addressing": 0.90}
# The variant with the replayed channel in both the cohort and the QMF is the
# factory back-end v3 (design §8.2 1c).
FACTORY = "cohort+qmf"
FACTORY_VERSION = "campplus-identity-evidence-v3"
DISCOVERY = 0.99  # on the posterior of someone not enrolled
MIN_UNKNOWN_SPEECH_MS = 1500


REPLAY_CHANNEL = {
    "capture": "AISHELL speakers replayed through a laptop speaker, recorded by the phone app",
    "segments_s": 8,
}


def _module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ie = _module("identity_eval")
unit, posteriors, load_backend = ie.unit, ie.posteriors, ie.load_backend


def connect(db: Path):
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def captured(paths: list[Path]):
    rows = [row for path in paths for row in json.loads(path.read_text())]
    return sorted(((row["captured_at"], row["sample"]) for row in rows), key=lambda r: r[0])


def household(db: Path):
    """Each registered member: template, the three registration samples, the trial sample."""
    with connect(db) as con:
        rows = con.execute(
            "select p.template_json, e.samples_json from speaker_profiles p "
            "join voice_enrollments e on e.enrollment_id = p.enrollment_id "
            "where p.origin = 'enrollment' order by p.profile_id").fetchall()
    members = []
    for template, samples in rows:
        template, samples = json.loads(template), json.loads(samples)
        registration = [samples["registration"][k] for k in sorted(samples["registration"])]
        mean = unit(np.mean([r["vector"] for r in registration], axis=0))[0]
        assert float(mean @ unit(template["vector"])[0]) > 0.999, "template is not its registration's mean"
        members.append({"template": (np.asarray(template["vector"]), template["speech_ms"]),
                        "registration": registration, "trial": samples["evaluation"]["trial"]})
    return members


def assign(samples, blocks):
    """(speaker, sample) for segments inside exactly one speaker's block."""
    kept, dropped = [], defaultdict(int)
    for at, sample in samples:
        start = at - sample["quality"]["audio_ms"] / 1000 - ARRIVAL_SLACK_S
        hit = [b["speaker"] for b in blocks if b["start"] < at and b["end"] > start]
        if len(hit) == 1:
            kept.append((hit[0], sample))
        else:
            dropped["silence_only" if not hit else "two_speakers"] += 1
    return kept, dict(dropped)


def agreement(kept, data):
    """How often a segment's nearest public speaker is the one it was assigned to."""
    ids = sorted(data["speakers"])
    refs = np.stack([unit(np.mean([r["sample"]["vector"] for r in data["speakers"][s]
                                   if r["channel"] == "phone"], axis=0))[0] for s in ids])
    hits = sum(ids[int(np.argmax(refs @ unit(sample["vector"])[0]))] == speaker for speaker, sample in kept)
    return round(hits / max(1, len(kept)), 4)


def windows(sample):
    """The whole recording and its registration-style windows long enough to compare."""
    return [(np.asarray(w["vector"]), w["speech_ms"]) for w in [sample] + sample.get("crops", [])
            if w["speech_ms"] >= ie.MIN_SPEECH_MS]


def decide(backend, templates, probe):
    """Who the product would name, its posterior, the LLRs and P(not enrolled); like the runtime."""
    side = backend.side(probe[0], probe[1], CHANNEL)
    llrs = {i: backend.llr(t, side) for i, t in templates.items()}
    priors = {i: (1 - ie.PRIOR_UNKNOWN) / len(templates) for i in templates}
    post, unknown = posteriors(llrs, priors, ie.PRIOR_UNKNOWN)
    best = max(post, key=post.get)
    return best, post[best], llrs, unknown


def floors_v4(templates_raw, probe):
    """The cosine floors running before 1b, for comparison: who they would name."""
    vec, ms = probe
    if ms < 600:
        return None
    cos = {i: float(unit(t)[0] @ unit(vec)[0]) for i, t in templates_raw.items()}
    best = max(cos, key=cos.get)
    second = max([v for k, v in cos.items() if k != best], default=-1)
    floor = max(f for start, f in ie.BANDS if ms >= start)
    return best if cos[best] >= floor and cos[best] - second >= 0.06 else None


def band(ms):
    return "lt_1s" if ms < 1000 else ("1_2.5s" if ms < 2500 else "ge_2.5s")


def channel_check(replay, members, data):
    """Median and tail cosine across speakers in each source, and between sources."""
    rng = np.random.default_rng(0)

    def cross(groups, n=20000):
        keys = [k for k, v in groups.items() if v]
        out = []
        for _ in range(n):
            a, b = rng.choice(len(keys), 2, replace=False)
            va, vb = groups[keys[a]], groups[keys[b]]
            out.append(float(va[rng.integers(len(va))] @ vb[rng.integers(len(vb))]))
        return np.asarray(out)

    def q(x):
        return {f"p{p}": round(float(np.quantile(x, p / 100)), 3) for p in (50, 90, 99)}

    rep = {s: [unit(r["sample"]["vector"])[0] for r in rows] for s, rows in replay.items()}
    mem = {i: [unit(r["vector"])[0] for r in m["registration"] + [m["trial"]]] for i, m in enumerate(members)}
    sim = {s: [unit(r["sample"]["vector"])[0] for r in data["speakers"][s] if r["channel"] == "phone"]
           for s in rep if s in data["speakers"]}
    report = {"replay_across_speakers": q(cross(rep)), "public_phone_across_speakers": q(cross(sim))}
    if len(mem) >= 2:
        report["members_across"] = q(cross(mem))
        report["replay_vs_members"] = q(np.asarray([float(a @ b) for v in rep.values() for a in v
                                                    for w in mem.values() for b in w]))
    return report


def members_as_strangers(backend, members):
    """Each member against a household where only the other is enrolled."""
    llrs = []
    for i, m in enumerate(members):
        for j, other in enumerate(members):
            if i == j:
                continue
            enrolled = backend.side(other["template"][0], other["template"][1], CHANNEL)
            for probe in [w for r in m["registration"] for w in windows(r)] + windows(m["trial"]):
                llrs.append(backend.llr(enrolled, backend.side(probe[0], probe[1], CHANNEL)))
    if not llrs:
        return {}
    llr = np.asarray(llrs)
    # One enrolled Person: P = (1 - π0) e^LLR / (π0 + (1 - π0) e^LLR).
    post = (1 - ie.PRIOR_UNKNOWN) * np.exp(llr) / (ie.PRIOR_UNKNOWN + (1 - ie.PRIOR_UNKNOWN) * np.exp(llr))
    return {"n": len(llr), "llr_p50": round(float(np.median(llr)), 2), "llr_max": round(float(llr.max()), 2),
            **{f"named_{use}": round(float(np.mean(post >= threshold)), 4) for use, threshold in POLICY_V2.items()}}


def household_eval(backend, members, strangers):
    """Per use: how often a stranger is named (wrong) and a member named correctly."""
    full = {i: backend.side(m["template"][0], m["template"][1], CHANNEL) for i, m in enumerate(members)}
    raw = {i: m["template"][0] for i, m in enumerate(members)}
    tally = defaultdict(lambda: defaultdict(int))
    stranger_llr = []

    def count(key, truth, best, post, v4, unknown, ms):
        tally[key]["n"] += 1
        tally[key][f"discovery:{'yes' if unknown >= DISCOVERY and ms >= MIN_UNKNOWN_SPEECH_MS else 'no'}"] += 1
        for use, threshold in POLICY_V2.items():
            named = post >= threshold
            tally[key][f"{use}:{'correct' if named and best == truth else ('wrong' if named else 'none')}"] += 1
        tally[key][f"v4_floors:{'none' if v4 is None else ('correct' if v4 == truth else 'wrong')}"] += 1

    for probe in strangers:
        best, post, llrs, unknown = decide(backend, full, probe)
        stranger_llr.append(max(llrs.values()))
        for key in (("stranger", band(probe[1])), ("stranger", "all")):
            count(key, None, best, post, floors_v4(raw, probe), unknown, probe[1])
    for i, m in enumerate(members):
        reg = m["registration"]
        for k in range(len(reg)):
            rest = [r for j, r in enumerate(reg) if j != k]
            own = (unit(np.mean([r["vector"] for r in rest], axis=0))[0], sum(r["speech_ms"] for r in rest))
            templates = {**full, i: backend.side(own[0], own[1], CHANNEL)}
            for probe in windows(reg[k]):
                best, post, _, unknown = decide(backend, templates, probe)
                for key in (("own", band(probe[1])), ("own", "all")):
                    count(key, i, best, post, floors_v4({**raw, i: own[0]}, probe), unknown, probe[1])
        trial = (np.asarray(m["trial"]["vector"]), m["trial"]["speech_ms"])
        best, post, _, unknown = decide(backend, full, trial)
        count(("own_trial", "all"), i, best, post, floors_v4(raw, trial), unknown, trial[1])
    rates = {f"{kind}/{b}": {"n": c["n"], **{k: round(v / c["n"], 4) for k, v in sorted(c.items()) if k != "n"}}
             for (kind, b), c in sorted(tally.items())}
    llr = np.asarray(stranger_llr)
    rates["stranger_max_llr"] = {q: round(float(np.quantile(llr, p)), 2)
                                 for q, p in (("p50", 0.5), ("p99", 0.99), ("max", 1.0))} if len(llr) else {}
    return rates


def verification(backend, rows):
    """EER, Cllr and DCFs overall and by condition (probe channel, short probes)."""
    if not rows:
        return {"targets": 0, "nontargets": 0}
    llr = np.array([backend.llr(backend.side(r["template"], r["e_ms"], r["enroll_channel"]),
                                backend.side(r["probe"], r["p_ms"], r["channel"])) for r in rows])
    y = np.array([r["label"] for r in rows])
    channel = np.array([r["channel"] for r in rows])
    short = np.array([r["p_ms"] < 1000 for r in rows])
    out = {}
    for label, mask in {"all": np.ones(len(y), bool), **{f"{c}_probe": channel == c for c in sorted(set(channel))},
                        "lt_1s": short}.items():
        if mask.sum() == 0 or y[mask].all() or not y[mask].any():
            continue
        act, mindcf = ie.dcf(llr[mask], y[mask])
        out[label] = {"targets": int(y[mask].sum()), "nontargets": int((~y[mask]).sum()),
                      "eer": round(ie.eer(llr[mask], y[mask]), 4), "cllr": round(ie.cllr(llr[mask], y[mask]), 4),
                      "act_dcf_0.01": round(act, 4), "min_dcf_0.01": round(mindcf, 4)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--log", type=Path, nargs="+", required=True, help="replay logs")
    parser.add_argument("--samples", type=Path, nargs="+", required=True, help="captured segments")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--delivered", type=Path, required=True, help="the model package directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    work = args.out / "work"
    work.mkdir(parents=True, exist_ok=True)
    (score_norm, calibration), revision = ie.wespeaker()
    data = json.loads(args.embeddings.read_text())
    cohort_ids, held = data["cohort"], data["held_out"]

    blocks = [json.loads(line) for log in args.log for line in log.read_text().splitlines() if line.strip()]
    samples = captured(args.samples)
    kept, dropped = assign(samples, blocks)
    replay = defaultdict(list)
    for n, (speaker, sample) in enumerate(kept):
        replay[speaker].append({"file": f"replay-{n:04d}", "channel": CHANNEL, "sample": sample})
    replay_data = {"speakers": dict(replay)}
    members = household(args.db)
    report = {"wespeaker": revision, "capture": {
        "segments": len(samples), "kept": len(kept), "dropped": dropped,
        "speakers": len(replay), "per_speaker_median": float(np.median([len(v) for v in replay.values()])),
        "nearest_public_speaker_agrees": agreement(kept, data),
        "speech_ms_median": float(np.median([s["speech_ms"] for _, s in kept])),
    }, "household_members": len(members), "channel": channel_check(replay, members, data)}

    def probes(ids):
        return [w for s in ids for r in replay.get(s, []) for w in windows(r["sample"])]

    rng = np.random.default_rng(0)
    pool, owners = ie.cohort_pool(data, cohort_ids)
    mean = pool.mean(axis=0)
    replay_cohort = [s for s in cohort_ids if s in replay]
    extra = np.asarray([r["sample"]["vector"] for s in replay_cohort for r in replay[s]], dtype=np.float64)
    extra_owners = np.asarray([cohort_ids.index(s) for s in replay_cohort for _ in replay[s]])
    adapted_pool, adapted_owners = np.vstack([pool, extra]), np.concatenate([owners, extra_owners])

    def rows(source, ids, use_pool, use_owners, *, train, enroll_channel):
        return ie.trial_rows(score_norm, source, ids, mean, use_pool, use_owners, cohort_ids,
                             leave_out=train, rng=rng, per=800 if train else 2000,
                             enroll_channel=enroll_channel)

    variants = {}
    for name, (use_pool, use_owners, in_domain) in {
        "qmf": (pool, owners, True),
        "cohort": (adapted_pool, adapted_owners, False),
        "cohort+qmf": (adapted_pool, adapted_owners, True),
    }.items():
        train = rows(data, cohort_ids, use_pool, use_owners, train=True, enroll_channel=ie.ENROLL_CHANNEL)
        if in_domain:
            train += rows(replay_data, replay_cohort, use_pool, use_owners, train=True, enroll_channel=CHANNEL)
        weights, bias, _ = ie.train_qmf(calibration, train, work, name.replace("+", "_"))
        ie.save_backend(args.out / name, data["identity"], mean, use_pool, weights, bias,
                        trained_with={"wespeaker": revision, "trials": len(train),
                                      "targets": sum(r["label"] for r in train)},
                        cohort={"speakers": len(cohort_ids), "vectors": int(len(use_pool)),
                                "dataset": data["dataset"], "dataset_revision": data["dataset_revision"],
                                "channels": {"phone": data["phone"], "device": data["device"],
                                             **({"replay": {**REPLAY_CHANNEL, "speakers": len(replay_cohort),
                                                           "vectors": len(extra)}}
                                                if use_pool is adapted_pool else {})}},
                        version=FACTORY_VERSION if name == FACTORY else ie.VERSION)
        variants[name] = (load_backend(args.out / name), use_pool, use_owners)
    variants = {"delivered": (load_backend(args.delivered), pool, owners), **variants}

    held_replay = [s for s in held if s in replay]
    report["variants"] = {}
    for name, (backend, use_pool, use_owners) in variants.items():
        report["variants"][name] = {
            "in_domain_held_out": verification(backend, rows(replay_data, held_replay, use_pool, use_owners,
                                                             train=False, enroll_channel=CHANNEL)),
            "public_held_out": verification(backend, rows(data, held, use_pool, use_owners,
                                                          train=False, enroll_channel=ie.ENROLL_CHANNEL)),
            "household_vs_held_out_strangers": household_eval(backend, members, probes(held_replay)),
            "members_as_strangers": members_as_strangers(backend, members),
        }
        print(name, json.dumps(report["variants"][name]["in_domain_held_out"].get("all")), flush=True)
    # The delivered back-end never saw any replayed speaker: all of them are strangers to it.
    report["delivered_household_vs_all_strangers"] = household_eval(variants["delivered"][0], members,
                                                                     probes(sorted(replay)))
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report["capture"], ensure_ascii=False))


if __name__ == "__main__":
    main()
