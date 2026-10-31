"""Train the identity back-end and evaluate it the way the product decides.

Design §3.3 / §8.2-1a. The back-end is WeSpeaker's (Apache-2.0): subtract a
domain mean, AS-norm over each side's top-N cohort scores
(``wespeaker/bin/score_norm.py``), and an affine QMF calibration trained with
a Cllr objective and L-BFGS (``wespeaker/bin/score_calibration.py``). The
runtime arithmetic is Channel's ``evidence.py``; this script only fits the
weights and measures.

Trials mirror the product. An enrollment is a registration: a speaker's first
three recordings on the phone channel, averaged, with their voiced speech
summed. Probes are the other recordings, on the phone or the device channel,
whole or as registration-style windows. Training uses cohort speakers, each
trial's two speakers removed from the cohort its statistics come from;
evaluation uses held-out speakers only, against the full cohort.

    eidolon_channel/.venv/bin/python -I \
        eidolon_models/evals/speaker-verification/cohort-20261031/identity_eval.py \
        --embeddings .eidolon/speaker-cohort-20261031/build/embeddings.json \
        --out <dir for identity-backend.json, identity-cohort.npz and the report>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eidolon.livekit.plugins.speaker_verification.evidence import (
    FEATURES,
    Backend,
    Side,
    posteriors,
    trial_features,
    unit,
)

TOP_N = 100
MAX_SPEECH_MS = 20000
MIN_SPEECH_MS = 600
ENROLL_FILES = 3
USES = {"personal_memory": 0.99, "addressing": 0.90, "speaker_label": 0.90}
# Guests, television and unenrolled children are not rare; no informative prior.
PRIOR_UNKNOWN = 0.5
ENROLL_CHANNEL = "phone"
BANDS = ((600, 0.55), (1200, 0.58), (2500, 0.58))  # v4 floors, for comparison


def registrations(rows):
    """(template vector, speech_ms) and probes [(vector, speech_ms, channel)]."""
    files = sorted({r["file"] for r in rows})
    enroll = [r for r in rows if r["file"] in files[:ENROLL_FILES] and r["channel"] == "phone"]
    if len(enroll) < ENROLL_FILES:
        return None
    template = unit(np.mean([r["sample"]["vector"] for r in enroll], axis=0))[0]
    speech = sum(r["sample"]["speech_ms"] for r in enroll)
    probes = []
    for r in rows:
        if r["file"] in files[:ENROLL_FILES]:
            continue
        for w in [r["sample"]] + r["sample"]["crops"]:
            if w["speech_ms"] >= MIN_SPEECH_MS:
                probes.append((np.asarray(w["vector"]), w["speech_ms"], r["channel"]))
    return (template, speech), probes


def cohort_pool(data, ids):
    vectors, owners = [], []
    for i, s in enumerate(ids):
        for r in data["speakers"][s]:
            vectors.append(r["sample"]["vector"])
            owners.append(i)
    return np.asarray(vectors, dtype=np.float64), np.asarray(owners)


def side_stats(prepared, cohort, cohort_owners, exclude_rows):
    """Top-N mean/std of each row's cohort scores, ignoring excluded speakers."""
    scores = prepared @ cohort.T
    for row, owners in enumerate(exclude_rows):
        if owners:
            scores[row, np.isin(cohort_owners, list(owners))] = -np.inf
    top = -np.sort(-scores, axis=1)[:, :TOP_N]
    return top.mean(axis=1), np.maximum(top.std(axis=1), 1e-6)


def features_matrix(e_vec, e_ms, e_mean, e_std, p_vec, p_ms, p_mean, p_std, cross):
    raw = np.sum(e_vec * p_vec, axis=1)
    asnorm = 0.5 * ((raw - e_mean) / e_std + (raw - p_mean) / p_std)
    d = np.sort(np.stack([np.minimum(e_ms, MAX_SPEECH_MS), np.minimum(p_ms, MAX_SPEECH_MS)], 1) / 1000, 1)
    m = np.sort(np.stack([e_mean, p_mean], 1), 1)
    return np.stack([asnorm, d[:, 0], d[:, 1], d[:, 1] - d[:, 0], d[:, 1] / np.maximum(d[:, 0], 1e-3),
                     m[:, 0], m[:, 1], m[:, 1] - m[:, 0], cross, cross * asnorm], 1)


def trials(data, ids, mean, cohort, cohort_owners, cohort_ids, *, leave_out, rng, nontarget_per_speaker):
    """Feature rows for target and non-target trials among ``ids``."""
    regs = {s: registrations(data["speakers"][s]) for s in ids}
    regs = {s: r for s, r in regs.items() if r}
    index = {s: cohort_ids.index(s) for s in ids if s in cohort_ids}
    rows, labels, meta = [], [], []
    speakers = sorted(regs)
    for a in speakers:
        (template, e_ms), _ = regs[a]
        e_vec = unit(template - mean)[0]
        candidates = [(b, p) for b in speakers for p in regs[b][1]]
        targets = [(b, p) for b, p in candidates if b == a]
        others = [(b, p) for b, p in candidates if b != a]
        pick = rng.choice(len(others), size=min(nontarget_per_speaker, len(others)), replace=False)
        chosen = targets + [others[i] for i in pick]
        p_vec = unit(np.stack([p[0] for _, p in chosen]) - mean)
        p_ms = np.array([p[1] for _, p in chosen])
        excl = [{index.get(a), index.get(b)} - {None} if leave_out else set() for b, _ in chosen]
        p_mean, p_std = side_stats(p_vec, cohort, cohort_owners, excl)
        e_mean, e_std = side_stats(np.repeat(e_vec[None], len(chosen), 0), cohort, cohort_owners, excl)
        cross = np.array([float(p[2] != ENROLL_CHANNEL) for _, p in chosen])
        rows.append(features_matrix(np.repeat(e_vec[None], len(chosen), 0), np.full(len(chosen), e_ms),
                                    e_mean, e_std, p_vec, p_ms, p_mean, p_std, cross))
        labels += [b == a for b, _ in chosen]
        meta += [(a, b, p[1], p[2]) for b, p in chosen]
    return np.concatenate(rows), np.asarray(labels), meta


def fit(x, y):
    """WeSpeaker score_calibration: affine layer, Cllr loss, L-BFGS."""
    xt, yt = torch.tensor(x, dtype=torch.float64), torch.tensor(y)
    w = torch.full((x.shape[1],), 1.0 / x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.01)

    def cllr():
        llr = xt @ w + b
        return 0.5 * (torch.nn.functional.softplus(-llr[yt]).mean()
                      + torch.nn.functional.softplus(llr[~yt]).mean()) / math.log(2)

    def closure():
        opt.zero_grad()
        loss = cllr()
        loss.backward()
        return loss

    best = float("inf")
    for _ in range(50):
        loss = float(opt.step(closure).detach())
        if best - loss < 1e-4:
            break
        best = min(best, loss)
    return w.detach().numpy(), float(b.detach()), float(cllr())


def cllr(llr, y):
    t, n = llr[y], llr[~y]
    return float(0.5 * (np.logaddexp(0, -t).mean() + np.logaddexp(0, n).mean()) / math.log(2))


def eer(score, y):
    order = np.argsort(-score)
    s, yy = score[order], y[order]
    tp = np.cumsum(yy) / yy.sum()
    fp = np.cumsum(~yy) / (~yy).sum()
    i = int(np.argmin(np.abs((1 - tp) - fp)))
    return float(((1 - tp[i]) + fp[i]) / 2)


def dcf(llr, y, p_target=0.01):
    """Normalized actual DCF at the Bayes threshold, and minimum DCF over thresholds."""
    threshold = -math.log(p_target / (1 - p_target))
    act = p_target * np.mean(llr[y] < threshold) + (1 - p_target) * np.mean(llr[~y] >= threshold)
    norm = min(p_target, 1 - p_target)
    order = np.sort(np.unique(llr))
    best = min(p_target * np.mean(llr[y] < t) + (1 - p_target) * np.mean(llr[~y] >= t) for t in order[:: max(1, len(order) // 2000)])
    return float(act / norm), float(best / norm)


def household_eval(data, held, backend, rng):
    """Two-person households; decide every probe per use, like the product."""
    regs = {s: registrations(data["speakers"][s]) for s in held}
    regs = {s: r for s, r in regs.items() if r}
    speakers = sorted(regs)
    groups = [speakers[i:i + 2] for i in range(0, len(speakers) - 1, 2)]
    tally = defaultdict(lambda: defaultdict(int))
    for group in groups:
        members = {s: backend.side(regs[s][0][0], regs[s][0][1], ENROLL_CHANNEL) for s in group}
        priors = {s: (1 - PRIOR_UNKNOWN) / len(group) for s in group}
        for truth in speakers:
            kind = "genuine" if truth in group else "stranger"
            for vec, ms, channel in regs[truth][1]:
                probe = backend.side(vec, ms, channel)
                llrs = {s: backend.llr(members[s], probe) for s in group}
                post, _ = posteriors(llrs, priors, PRIOR_UNKNOWN)
                cos ={s: float(unit(regs[s][0][0])[0] @ unit(vec)[0]) for s in group}
                band = "lt_1s" if ms < 1000 else ("1_2.5s" if ms < 2500 else "ge_2.5s")
                key = (kind, channel, band)
                for use, threshold in USES.items():
                    best = max(post, key=post.get)
                    hit = post[best] >= threshold
                    tally[key][f"{use}:{'correct' if hit and best == truth else ('wrong' if hit else 'none')}"] += 1
                best = max(cos, key=cos.get)
                second = max([v for k, v in cos.items() if k != best], default=-1)
                v2 = cos[best] >= 0.55 and cos[best] - second >= 0.1
                floor = max(f for start, f in BANDS if ms >= start) if ms >= 600 else 1
                v4 = cos[best] >= floor and cos[best] - second >= 0.06
                for name, hit in (("v2_fixed", v2), ("v4_floors", v4)):
                    tally[key][f"{name}:{'correct' if hit and best == truth else ('wrong' if hit else 'none')}"] += 1
                tally[key]["n"] += 1
    return tally


def summarize(tally):
    out = {}
    merged = defaultdict(lambda: defaultdict(int))
    for (kind, channel, band), counts in tally.items():
        for k in ((kind, channel, band), (kind, channel, "all"), (kind, "all", "all")):
            for name, n in counts.items():
                merged[k][name] += n
    for (kind, channel, band), counts in sorted(merged.items()):
        n = counts["n"]
        out[f"{kind}/{channel}/{band}"] = {"n": n, **{k: round(v / n, 4) for k, v in sorted(counts.items()) if k != "n"}}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.embeddings.read_text())
    cohort_ids, held = data["cohort"], data["held_out"]
    rng = np.random.default_rng(0)

    vectors, owners = cohort_pool(data, cohort_ids)
    mean = vectors.mean(axis=0)
    cohort = unit(vectors - mean)

    x, y, _ = trials(data, cohort_ids, mean, cohort, owners, cohort_ids, leave_out=True, rng=rng,
                     nontarget_per_speaker=800)
    weights, bias, train_cllr = fit(x, y)

    npz = args.out / "identity-cohort.npz"
    with open(npz, "wb") as f:
        np.savez(f, mean=mean.astype(np.float32), cohort=vectors.astype(np.float32))
    spec = {
        "version": "campplus-identity-evidence-v1",
        "identity": data["identity"],
        "top_n": TOP_N, "max_speech_ms": MAX_SPEECH_MS,
        "features": list(FEATURES), "weights": [round(float(w), 6) for w in weights], "bias": round(bias, 6),
        "cohort": {"path": npz.name, "sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                   "speakers": len(cohort_ids), "vectors": int(len(vectors)),
                   "dataset": data["dataset"], "dataset_revision": data["dataset_revision"],
                   "channels": {"phone": data["phone"], "device": data["device"]}},
        "trained_on": {"trials": int(len(y)), "targets": int(y.sum()), "cllr": round(train_cllr, 4)},
    }
    (args.out / "identity-backend.json").write_text(json.dumps(spec, indent=1) + "\n")

    from eidolon.livekit.plugins.speaker_verification.evidence import load_backend
    backend = load_backend(args.out)

    # The vectorized features here must be the runtime's, exactly.
    regs = registrations(data["speakers"][held[0]])
    e = backend.side(regs[0][0], regs[0][1], ENROLL_CHANNEL)
    for vec, ms, channel in regs[1][:40]:
        p = backend.side(vec, ms, channel)
        hm, hs = side_stats(unit(vec - mean), cohort, owners, [set()])
        assert np.allclose(features_matrix(
            e.vector[None], np.array([e.speech_ms]), np.array([e.cohort_mean]), np.array([e.cohort_std]),
            p.vector[None], np.array([p.speech_ms]), hm, hs, np.array([float(channel != ENROLL_CHANNEL)]))[0],
            trial_features(e, p, MAX_SPEECH_MS))

    hx, hy, hmeta = trials(data, held, mean, cohort, owners, cohort_ids, leave_out=False, rng=rng,
                           nontarget_per_speaker=2000)
    llr = hx @ backend.weights + backend.bias
    report = {"backend": spec, "verification": {}}
    for label, mask in {
        "all": np.ones(len(hy), bool),
        "phone_probe": np.array([m[3] == "phone" for m in hmeta]),
        "device_probe": np.array([m[3] == "device" for m in hmeta]),
        "lt_1s": np.array([m[2] < 1000 for m in hmeta]),
        "ge_2.5s": np.array([m[2] >= 2500 for m in hmeta]),
    }.items():
        act, mindcf = dcf(llr[mask], hy[mask])
        report["verification"][label] = {
            "targets": int(hy[mask].sum()), "nontargets": int((~hy[mask]).sum()),
            "eer": round(eer(llr[mask], hy[mask]), 4), "cllr": round(cllr(llr[mask], hy[mask]), 4),
            "act_dcf_0.01": round(act, 4), "min_dcf_0.01": round(mindcf, 4),
        }
    report["household"] = summarize(household_eval(data, held, backend, rng))
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({"trained": spec["trained_on"], "weights": spec["weights"], "bias": spec["bias"],
                      "verification": report["verification"]}, indent=1))


if __name__ == "__main__":
    main()
