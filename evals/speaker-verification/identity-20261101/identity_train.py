"""Train and accept the identity back-end on live multi-device speech (design §3.3, §8.2 1c-2).

Registration happens on the phone; speech arrives on every device. The
3D-Speaker test set (240 speakers recorded live and simultaneously on phones,
a tablet, a PC, a recording pen, a directional microphone and arrays, 0.1-4 m;
CC BY-SA 4.0) stands in for the product's three comparison conditions:

- ``near_same``: a registration (ten segments, about as much voiced speech as
  the phone's three sentences) on a close device (directional microphone at
  0.1 m or PC at 0.2 m) against other segments on that same device;
- ``far_same``: an observation template (three segments) from one device at
  one distance against that device's other segments;
- ``far_cross``: a close-device registration against a far device (>= 1 m).

Probes are single segments or tracks of k segments (their mean vector, voiced
speech summed), as a conversation track accumulates them.

The back-end (Channel's ``evidence.py``): subtract the mean of live speech;
far_cross is scored after an LDA projection; each condition has an affine QMF
over the score and WeSpeaker's duration factors, trained with WeSpeaker's own
``train_calibration_model`` (pinned in ``wespeaker_ref.py``).

Acceptance is held out twice. Speakers are split in halves, each half tested
with the back-end trained on the other; and every fold holds out one close
device and one far device, so the tested devices were never seen in training,
as the product's phone and boards were never seen. Two-person households then
decide every probe per use. Each use's threshold in each condition is the
design's (§3.3 ②) when the held-out false acceptance meets its limit, raised
along a grid when it does not, and off when nothing on the grid does.

The delivered back-end is trained on all speakers and devices; Channel's
runtime is checked against WeSpeaker's ``infer_calibration`` on it.

    eidolon_channel/.venv/bin/python -I \\
        eidolon_models/evals/speaker-verification/identity-20261101/identity_train.py \\
        --corpus /Users/manson/ai/data/speaker-corpora/3d-speaker \\
        --embeddings /Users/manson/ai/data/speaker-corpora/3d-speaker/embeddings.json \\
        --model eidolon_models/speaker_verification/campplus_zh_16k_common/624dcb0f \\
        --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eidolon.livekit.plugins.speaker_verification.evidence import (
    COMPENSATED, FEATURES, load_backend, trial_features,
)
from eidolon_sdk.biz.speaker_identity import IDENTITY_CONDITIONS

HERE = Path(__file__).resolve().parent
VERSION = "campplus-identity-evidence-v4"
POLICY_VERSION = "campplus-identity-policy-v5"
MAX_SPEECH_MS = 20000
LDA_DIM = 128
REGISTRATION_SEGMENTS = 10
OBSERVATION_SEGMENTS = 3
TRACKS = {"near_same": (1, 2), "far_same": (1, 2, 3, 5), "far_cross": (1, 2, 3, 5)}
NEAR = {"DIRECTIONAL": 0.1, "PC": 0.2}
FAR_FROM_M = 1.0
NONTARGETS_PER_TEMPLATE = 60
PRIOR_UNKNOWN = 0.5
# Design §3.3 ②: threshold, false-acceptance limit.
USES = {"personal_memory": (0.99, 0.005), "addressing": (0.90, 0.02), "speaker_label": (0.90, 0.02)}
# Candidate thresholds: 0.1 apart in log-odds, from the design's value up to 0.99999.
GRID_STEP, GRID_TOP = 0.1, 0.99999
HOUSEHOLD_MARGIN_LLR = 1.0
MIN_SPEECH_MS = 600
MIN_UNKNOWN_SPEECH_MS = 1500


def wespeaker():
    spec = importlib.util.spec_from_file_location("wespeaker_ref", HERE / "wespeaker_ref.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load()[1], module.REVISION


def corpus(root: Path, embeddings: Path):
    files = root / "files-x" / "files"
    distances = dict(line.split()[::-1] if len(line.split()) == 2 else (line.strip(), "")
                     for line in (files / "distance_file").read_text().splitlines() if line.strip())
    emb = json.loads(embeddings.read_text())
    utts = []
    with open(files / "test_utt2info.csv", newline="") as f:
        for row in csv.DictReader(f):
            u = row["Utt-id"]
            if u not in emb:
                continue
            d = distances.get(u.split("_")[-2], "")
            utts.append(dict(id=u, speaker=row["Spk-id"], segment=row["Seg-id"], device=row["Device"],
                             distance=float(d) if d else None, vector=np.asarray(emb[u]["vector"]),
                             speech_ms=int(emb[u]["speech_ms"])))
    return utts


def place(u) -> str | None:
    """'near:<device>' for a close device, 'far:<device>@<m>' at 1 m or more, else None."""
    if u["distance"] is not None and NEAR.get(u["device"]) == u["distance"]:
        return "near:" + u["device"]
    if u["distance"] is not None and u["distance"] >= FAR_FROM_M:
        return f"far:{u['device']}@{u['distance']}"
    return None


def device_of(p: str) -> str:
    return p.split(":", 1)[1].split("@")[0]


# ---------------------------------------------------------------- the back-end

def fit_lda(x: np.ndarray, labels: list[str], dim: int) -> np.ndarray:
    """Columns maximising between-speaker over within-speaker scatter (classic channel compensation)."""
    groups = defaultdict(list)
    for v, s in zip(x, labels):
        groups[s].append(v)
    mu = x.mean(0)
    sw = np.zeros((x.shape[1], x.shape[1]))
    sb = np.zeros_like(sw)
    for vs in groups.values():
        vs = np.asarray(vs)
        m = vs.mean(0)
        sw += (vs - m).T @ (vs - m)
        sb += len(vs) * np.outer(m - mu, m - mu)
    sw += 1e-3 * np.trace(sw) / len(sw) * np.eye(len(sw))
    whiten = np.linalg.cholesky(np.linalg.inv(sw))
    values, vectors = np.linalg.eigh(whiten.T @ sb @ whiten)
    return whiten @ vectors[:, ::-1][:, :dim]


class Fitted:
    """The runtime's arithmetic with fitted arrays, before a file exists."""

    def __init__(self, mean, projection):
        self.mean, self.projection = mean, projection

    def side(self, vector, speech_ms, channel):
        from eidolon.livekit.plugins.speaker_verification.evidence import Side, unit
        centred = np.asarray(vector, dtype=np.float64) - self.mean
        return Side(unit(centred)[0], unit(centred @ self.projection)[0], int(speech_ms), channel)


def fit_space(utts) -> Fitted:
    x = np.stack([u["vector"] for u in utts])
    mean = x.mean(0)
    return Fitted(mean, fit_lda(x - mean, [u["speaker"] for u in utts], LDA_DIM))


# ---------------------------------------------------------------- trials

def template(us):
    """A template is the plain mean of its samples' unit vectors, voiced speech summed (as Channel builds it)."""
    return np.mean([u["vector"] for u in us], axis=0), sum(u["speech_ms"] for u in us)


def tracks(us, k, rng):
    us = [us[i] for i in rng.permutation(len(us))]
    return [template(us[j:j + k]) for j in range(0, len(us) - k + 1, k)]


def structures(utts, speakers, places, rng):
    """Per condition: templates {speaker: [(vector, ms, place)]} and probes [(speaker, k, vector, ms, place)]."""
    by = defaultdict(lambda: defaultdict(list))
    for u in utts:
        p = place(u)
        if u["speaker"] in speakers and p in places:
            by[u["speaker"]][p].append(u)
    out = {c: ({}, []) for c in IDENTITY_CONDITIONS}
    for s, groups in by.items():
        near = {p: sorted(us, key=lambda u: u["segment"]) for p, us in groups.items() if p.startswith("near:")}
        far = {p: us for p, us in groups.items() if p.startswith("far:")}
        for p, us in near.items():
            if len(us) <= REGISTRATION_SEGMENTS:
                continue
            reg, rest = us[:REGISTRATION_SEGMENTS], us[REGISTRATION_SEGMENTS:]
            for c in ("near_same", "far_cross"):
                out[c][0].setdefault(s, []).append((*template(reg), p))
            for k in TRACKS["near_same"]:
                out["near_same"][1].extend([(s, k, v, ms, p) for v, ms in tracks(rest, k, rng)])
        for p, us in far.items():
            for k in TRACKS["far_cross"]:
                out["far_cross"][1].extend([(s, k, v, ms, p) for v, ms in tracks(us, k, rng)])
            if len(us) > OBSERVATION_SEGMENTS:
                us = [us[i] for i in rng.permutation(len(us))]
                obs, rest = us[:OBSERVATION_SEGMENTS], us[OBSERVATION_SEGMENTS:]
                out["far_same"][0].setdefault(s, []).append((*template(obs), p))
                for k in TRACKS["far_same"]:
                    out["far_same"][1].extend([(s, k, v, ms, p) for v, ms in tracks(rest, k, rng)])
    return out


def pairs(condition, t_place, p_place):
    """Whether a template and a probe make a trial of this condition."""
    if condition == "far_cross":
        return t_place.startswith("near:") and p_place.startswith("far:")
    return t_place == p_place


def llr_matrix(space, weights, bias, condition, templates, probes):
    """Every probe against every template: the runtime's trial_features and QMF, vectorised.

    templates: [(vector, ms, place)], probes: [(speaker, k, vector, ms, place)]. Unpaired cells are NaN.
    """
    def prepared(vectors):
        centred = np.stack(vectors) - space.mean
        if condition in COMPENSATED:
            centred = centred @ space.projection
        return centred / np.linalg.norm(centred, axis=1, keepdims=True)

    t = prepared([v for v, _, _ in templates])
    p = prepared([v for _, _, v, _, _ in probes])
    score = p @ t.T
    tms = np.minimum([ms for _, ms, _ in templates], MAX_SPEECH_MS)[None, :] / 1000
    pms = np.minimum([ms for _, _, _, ms, _ in probes], MAX_SPEECH_MS)[:, None] / 1000
    low, high = np.minimum(tms, pms), np.maximum(tms, pms)
    llr = (weights[0] * score + weights[1] * low + weights[2] * high + weights[3] * (high - low)
           + weights[4] * (high / low) + bias)
    paired = np.array([[pairs(condition, tp, pp) for _, _, tp in templates] for *_, pp in probes])
    return np.where(paired, llr, np.nan)


def feature_rows(space, condition, templates, probes, rng, *, nontargets=NONTARGETS_PER_TEMPLATE):
    """Training rows: every target trial, and sampled nontargets per template."""
    rows = []
    speakers = np.array([s for s, *_ in probes])
    places = np.array([p for *_, p in probes])
    for s, ts in templates.items():
        for tv, tms, tp in ts:
            enroll = space.side(tv, tms, "enroll")
            paired = np.array([pairs(condition, tp, pp) for pp in places]) if len(places) else np.zeros(0, bool)
            own = np.flatnonzero(paired & (speakers == s))
            others = np.flatnonzero(paired & (speakers != s))
            chosen = list(own) + list(rng.choice(others, size=min(nontargets, len(others)), replace=False))
            for i in chosen:
                ps, k, pv, pms, pp = probes[i]
                probe = space.side(pv, pms, "probe")
                rows.append((ps == s, trial_features(enroll, probe, condition, MAX_SPEECH_MS)))
    return rows


def train_qmf(calibration, rows, work: Path, name: str):
    """WeSpeaker's train_calibration_model on these rows: (weights, bias, factor file, model file).

    Its L-BFGS stops as soon as a step gains less than 1e-4, so factors on
    very different scales (a duration ratio of 30 beside a cosine) leave it
    barely trained. The factors are standardised for training and the weights
    mapped back, so the runtime applies them to the raw factors unchanged.
    """
    x = np.stack([f for _, f in rows])
    centre, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    factors = work / f"{name}.factors"
    with open(factors, "w") as f:
        for i, (label, row) in enumerate(rows):
            z = (row - centre) / scale
            f.write(f"e{i} t{i} {'target' if label else 'nontarget'} " + " ".join(f"{v:.8f}" for v in z) + "\n")
    model = work / f"{name}.pt"
    calibration.train_calibration_model(str(factors), str(model))
    state = torch.load(model)
    w_std = np.asarray(state["linear.weight"].reshape(-1).tolist())
    b_std = float(state["linear.bias"].reshape(-1)[0])
    weights = w_std / scale
    return weights, b_std - float(weights @ centre), factors, model


# ---------------------------------------------------------------- measures

def cllr(llr, y):
    t, n = llr[y], llr[~y]
    return float(0.5 * (np.logaddexp(0, -t).mean() + np.logaddexp(0, n).mean()) / math.log(2))


def eer(score, y):
    order = np.argsort(-score)
    yy = y[order]
    tp = np.cumsum(yy) / yy.sum()
    fp = np.cumsum(~yy) / (~yy).sum()
    i = int(np.argmin(np.abs((1 - tp) - fp)))
    return float(((1 - tp[i]) + fp[i]) / 2)


def dcf(llr, y, p_target=0.01):
    threshold = -math.log(p_target / (1 - p_target))
    act = p_target * np.mean(llr[y] < threshold) + (1 - p_target) * np.mean(llr[~y] >= threshold)
    norm = min(p_target, 1 - p_target)
    grid = np.sort(np.unique(llr))[:: max(1, len(llr) // 2000)]
    best = min(p_target * np.mean(llr[y] < t) + (1 - p_target) * np.mean(llr[~y] >= t) for t in grid)
    return float(act / norm), float(best / norm)


def verification(llr, y):
    act, mindcf = dcf(llr, y)
    return {"targets": int(y.sum()), "nontargets": int((~y).sum()), "eer": round(eer(llr, y), 4),
            "cllr": round(cllr(llr, y), 4), "act_dcf_0.01": round(act, 4), "min_dcf_0.01": round(mindcf, 4)}


def per_speaker(llr, templates_owner, speakers):
    """Each speaker's best template per probe (NaN where none pairs)."""
    out = np.full((llr.shape[0], len(speakers)), np.nan)
    for j, s in enumerate(speakers):
        cols = [i for i, o in enumerate(templates_owner) if o == s]
        if cols:
            with np.errstate(all="ignore"):
                block = llr[:, cols]
                out[:, j] = np.where(np.all(np.isnan(block), axis=1), np.nan, np.nanmax(np.nan_to_num(block, nan=-np.inf), axis=1))
    return out


def decide(fold_rows, thresholds):
    """Two-person households (consecutive held-out speakers); every probe decided per use.

    fold_rows: [(speakers, probe_speaker, k, L)] with L[probe, speaker] the best LLR.
    Rates per kind ('own' / 'stranger') and track length, summed over folds.
    """
    log_half = math.log((1 - PRIOR_UNKNOWN) / 2)
    totals = defaultdict(lambda: defaultdict(float))
    for speakers, probe_speaker, ks, L in fold_rows:
        index = {s: j for j, s in enumerate(speakers)}
        for a, b in zip(speakers[0::2], speakers[1::2]):
            la, lb = L[:, index[a]], L[:, index[b]]
            ok = ~np.isnan(la) & ~np.isnan(lb)
            terms = np.stack([np.full(la.shape, math.log(PRIOR_UNKNOWN)), log_half + la, log_half + lb])
            top = np.nanmax(terms, axis=0)
            weights = np.exp(terms - top)
            post = weights / weights.sum(axis=0)
            lead = np.where(post[1] >= post[2], 0, 1)
            p_lead = np.maximum(post[1], post[2])
            who = np.where(lead == 0, a, b)
            own = (probe_speaker == a) | (probe_speaker == b)
            other_llr = np.where(probe_speaker == a, lb, la)
            alone = 1 / (1 + PRIOR_UNKNOWN / ((1 - PRIOR_UNKNOWN) * np.exp(np.minimum(other_llr, 700))))
            for kind, mask in (("own", own & ok), ("stranger", ~own & ok)):
                for kk in np.unique(ks):
                    m = mask & (ks == kk)
                    for key in ((kind, "all"), (kind, f"k{kk}")):
                        c = totals[key]
                        c["n"] += m.sum()
                        for use, t in thresholds.items():
                            if t is None:
                                continue
                            named = m & (p_lead >= t)
                            if kind == "own":
                                c[f"{use}:correct"] += (named & (who == probe_speaker)).sum()
                                c[f"{use}:wrong"] += (named & (who != probe_speaker)).sum()
                                c[f"{use}:member_as_stranger"] += (m & (alone >= t)).sum()
                            else:
                                c[f"{use}:wrong"] += named.sum()
    return {f"{kind}/{k}": {"n": int(c["n"]), **{m: round(v / c["n"], 5) for m, v in sorted(c.items()) if m != "n"}}
            for (kind, k), c in sorted(totals.items()) if c["n"]}


def grid(default):
    logit = lambda p: math.log(p / (1 - p))
    xs = np.arange(logit(default), logit(GRID_TOP) + 1e-9, GRID_STEP)
    return [round(float(1 / (1 + math.exp(-x))), 6) for x in xs]


def worst_rates(rates, use):
    """The highest false acceptance over track lengths: stranger named, member named
    as the other member, or member named when only the other is enrolled."""
    rows = [(f"{key}:{m}", r.get(f"{use}:{m}", 0)) for key, r in rates.items() if key != "own/all"
            and key != "stranger/all" for m in ("wrong", "member_as_stranger")]
    return max(rows, key=lambda x: x[1])


def choose(fold_rows):
    """Each use's threshold: the smallest on the grid from the design's value whose
    held-out false acceptance meets the use's limit everywhere; off if none does."""
    chosen, binding = {}, {}
    for use, (default, limit) in USES.items():
        chosen[use] = None
        for t in grid(default):
            where, worst = worst_rates(decide(fold_rows, {use: t}), use)
            if worst <= limit:
                chosen[use] = t
                break
            binding[use] = (t, where, worst)
    return chosen, binding


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="CAM++ package whose identity this trains for")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "work"
    work.mkdir(exist_ok=True)
    calibration, revision = wespeaker()
    utts = corpus(args.corpus, args.embeddings)
    places = sorted({p for u in utts if (p := place(u))})
    near_places = [p for p in places if p.startswith("near:")]
    far_devices = sorted({device_of(p) for p in places if p.startswith("far:")})
    speakers = sorted({u["speaker"] for u in utts})
    halves = (set(speakers[0::2]), set(speakers[1::2]))

    # Held out twice: speaker halves, and one close and one far device per fold.
    held = {c: {"llr": [], "y": [], "k": []} for c in IDENTITY_CONDITIONS}
    household_rows = {c: [] for c in IDENTITY_CONDITIONS}
    fold = 0
    for train_spk, test_spk in (halves, halves[::-1]):
        for near_out in near_places:
            for far_out in far_devices:
                fold += 1
                rng = np.random.default_rng(fold)
                seen = [p for p in places if p != near_out and not (p.startswith("far:") and device_of(p) == far_out)]
                unseen = [near_out] + [p for p in places if p.startswith("far:") and device_of(p) == far_out]
                space = fit_space([u for u in utts if u["speaker"] in train_spk and place(u) in seen])
                train = structures(utts, train_spk, set(seen), rng)
                test = structures(utts, test_spk, set(unseen), rng)
                for c in IDENTITY_CONDITIONS:
                    rows = feature_rows(space, c, *train[c], rng)
                    w, b, _, _ = train_qmf(calibration, rows, work, f"fold{fold}-{c}")
                    templates, probes = test[c]
                    flat = [(tv, tms, tp) for s in sorted(templates) for tv, tms, tp in templates[s]]
                    owner = [s for s in sorted(templates) for _ in templates[s]]
                    if not flat or not probes:
                        continue
                    llr = llr_matrix(space, w, b, c, flat, probes)
                    if fold == 1 and c == "far_cross":  # the vectorised arithmetic is the runtime's
                        e = space.side(flat[0][0], flat[0][1], "e")
                        pr = space.side(probes[0][2], probes[0][3], "p")
                        expected = trial_features(e, pr, c, MAX_SPEECH_MS) @ w + b
                        assert abs(expected - llr[0, 0]) < 1e-6 * max(1, abs(expected)), (expected, llr[0, 0])
                    spk = sorted(templates)
                    L = per_speaker(llr, owner, spk)
                    ps = np.array([p[0] for p in probes])
                    ks = np.array([p[1] for p in probes])
                    household_rows[c].append((spk, ps, ks, L))
                    for j, s in enumerate(spk):
                        col = L[:, j]
                        keep = ~np.isnan(col)
                        held[c]["llr"] += col[keep].tolist()
                        held[c]["y"] += (ps[keep] == s).tolist()
                        held[c]["k"] += ks[keep].tolist()
                print(f"fold {fold}: held out {near_out}, {far_out}", flush=True)

    report = {"corpus": {"utterances": len(utts), "speakers": len(speakers), "places": places,
                         "folds": fold, "lda_dim": LDA_DIM},
              "held_out": {}, "households": {}, "policy": {}}
    policy_conditions = {}
    for c in IDENTITY_CONDITIONS:
        llr, y, k = (np.asarray(held[c][x]) for x in ("llr", "y", "k"))
        y = y.astype(bool)
        report["held_out"][c] = {f"k{kk}": verification(llr[k == kk], y[k == kk]) for kk in TRACKS[c]}
        chosen, binding = choose(household_rows[c])
        report.setdefault("binding", {})[c] = {u: {"rejected_threshold": v[0], "by": v[1], "rate": round(v[2], 5)}
                                              for u, v in binding.items()}
        policy_conditions[c] = {**chosen, "candidate_evidence": None, "discovery": None}
        report["households"][c] = decide(household_rows[c], chosen)
        report["policy"][c] = policy_conditions[c]
        print(c, json.dumps(report["held_out"][c]), json.dumps(chosen), flush=True)

    # The delivered back-end: every speaker, every device.
    rng = np.random.default_rng(0)
    space = fit_space([u for u in utts if place(u)])
    full = structures(utts, set(speakers), set(places), rng)
    conditions, checks = {}, []
    for c in IDENTITY_CONDITIONS:
        rows = feature_rows(space, c, *full[c], rng)
        w, b, factors, model = train_qmf(calibration, rows, work, f"final-{c}")
        conditions[c] = {"weights": [round(float(x), 6) for x in w], "bias": round(b, 6),
                         "trials": len(rows), "targets": int(sum(r[0] for r in rows))}
        checks.append((c, factors, model, rows))
    identity_spec = json.loads((args.model / "identity-policy.json").read_text())
    identity = [identity_spec["model_id"], identity_spec["weights_hash"], identity_spec["preprocessing_version"]]
    arrays = args.out / "identity-backend.npz"
    with open(arrays, "wb") as f:
        np.savez(f, mean=space.mean.astype(np.float32), projection=space.projection.astype(np.float32))
    spec = {
        "version": VERSION, "identity": identity, "max_speech_ms": MAX_SPEECH_MS,
        "features": list(FEATURES), "conditions": conditions,
        "arrays": {"path": arrays.name, "sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
                   "lda_dim": LDA_DIM},
        "trained_with": {"wespeaker": revision, "dataset": "3D-Speaker test set (CC BY-SA 4.0)",
                         "speakers": len(speakers), "places": places},
    }
    (args.out / "identity-backend.json").write_text(json.dumps(spec, indent=1) + "\n")
    policy = {
        "version": POLICY_VERSION, "model_id": identity[0], "weights_hash": identity[1],
        "preprocessing_version": identity[2], "backend_version": VERSION,
        "min_speech_ms": MIN_SPEECH_MS, "min_unknown_speech_ms": MIN_UNKNOWN_SPEECH_MS,
        "prior_unknown": PRIOR_UNKNOWN, "household_margin_llr": HOUSEHOLD_MARGIN_LLR,
        "conditions": policy_conditions,
    }
    (args.out / "identity-policy.json").write_text(json.dumps(policy, indent=2) + "\n")

    # Channel's runtime must give WeSpeaker's LLRs (factors written at six decimals).
    backend = load_backend(args.out)
    drift = 0.0
    for c, factors, model, rows in checks:
        out_llr = work / f"final-{c}.llr"
        calibration.infer_calibration(str(factors), str(model), str(out_llr))
        theirs = np.array([float(line.split()[2]) for line in out_llr.read_text().splitlines()])
        weights, bias = backend.calibrations[c]
        ours = np.array([x @ weights + bias for _, x in rows])
        drift = max(drift, float(np.max(np.abs(ours - theirs))))
    assert drift < 0.01, f"runtime LLRs drift {drift} from WeSpeaker's"
    report["delivered"] = {"backend": spec, "runtime_vs_wespeaker_max_llr_diff": round(drift, 6)}
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({"drift": drift, "policy": policy_conditions}, indent=1))


if __name__ == "__main__":
    main()
