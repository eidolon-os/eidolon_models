"""Train the identity back-end with WeSpeaker's own scripts; evaluate it per use.

Design §3.3 / §8.2. Every step of the back-end is WeSpeaker's (Apache-2.0,
pinned in ``wespeaker_ref.py``): cohort statistics from
``score_norm.get_mean_std`` and its AS-norm formula, quality factors from
``score_calibration.gather_calibration_factors``, weights from
``score_calibration.train_calibration_model``, LLRs from
``infer_calibration``. Two columns are appended to WeSpeaker's factors: the
channel condition and its product with the score. Channel's runtime
(``evidence.py``) must reproduce WeSpeaker's LLRs; this script checks it.

Trials mirror the product. An enrollment is a registration: a speaker's
first three recordings on the phone channel, averaged, voiced speech summed.
Probes are the other recordings on the phone or the device channel, whole or
as registration-style windows. Training uses cohort speakers, each trial's
two speakers left out of the cohort its statistics come from; evaluation
uses held-out speakers against the full cohort.

    eidolon_channel/.venv/bin/python -I \\
        eidolon_models/evals/speaker-verification/cohort-20261031/identity_eval.py \\
        --embeddings .eidolon/speaker-cohort-20261031/build/embeddings.json \\
        --out <dir for identity-backend.json, identity-cohort.npz and the report>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eidolon.livekit.plugins.speaker_verification.evidence import (
    FEATURES,
    load_backend,
    posteriors,
    unit,
)

HERE = Path(__file__).resolve().parent
TOP_N = 100
MAX_SPEECH_MS = 20000
MIN_SPEECH_MS = 600
ENROLL_FILES = 3
USES = {"personal_memory": 0.99, "addressing": 0.90, "speaker_label": 0.90}
# Guests, television and unenrolled children are not rare; no informative prior.
PRIOR_UNKNOWN = 0.5
ENROLL_CHANNEL = "phone"
BANDS = ((600, 0.55), (1200, 0.58), (2500, 0.58))  # v4 floors, for comparison
VERSION = "campplus-identity-evidence-v2"


def wespeaker():
    spec = importlib.util.spec_from_file_location("wespeaker_ref", HERE / "wespeaker_ref.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load(), module.REVISION


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


def trial_rows(score_norm, data, ids, mean, pool, owners, cohort_ids, *, leave_out, rng, per):
    """Score-norm rows built with WeSpeaker's get_mean_std and AS-norm formula."""
    regs = {s: registrations(data["speakers"][s]) for s in ids}
    regs = {s: r for s, r in regs.items() if r}
    index = {s: cohort_ids.index(s) for s in ids if s in cohort_ids}
    centred_pool = pool - mean
    rows = []
    speakers = sorted(regs)
    for a in speakers:
        (template, e_ms), _ = regs[a]
        e_raw = template - mean
        candidates = [(b, p) for b in speakers for p in regs[b][1]]
        others = [i for i, (b, _) in enumerate(candidates) if b != a]
        keep = set(rng.choice(others, size=min(per, len(others)), replace=False).tolist())
        keep |= {i for i, (b, _) in enumerate(candidates) if b == a}
        by_speaker = defaultdict(list)
        for i in sorted(keep):
            by_speaker[candidates[i][0]].append(candidates[i][1])
        for b, probes in by_speaker.items():
            excluded = {index.get(a), index.get(b)} - {None} if leave_out else set()
            cohort = centred_pool[~np.isin(owners, list(excluded))] if excluded else centred_pool
            e_mean, e_std = score_norm.get_mean_std(e_raw[None], cohort, TOP_N)
            p_raw = np.stack([p[0] for p in probes]) - mean
            p_mean, p_std = score_norm.get_mean_std(p_raw, cohort, TOP_N)
            for j, (vec, ms, channel) in enumerate(probes):
                score = float(unit(e_raw)[0] @ unit(p_raw[j])[0])
                # score_norm.main's normalisation, verbatim.
                normed = 0.5 * ((score - e_mean[0]) / e_std[0] + (score - p_mean[j]) / p_std[j])
                rows.append(dict(
                    label=a == b, normed=normed, e_ms=e_ms, p_ms=ms,
                    e_mag=float(np.linalg.norm(e_raw)), p_mag=float(np.linalg.norm(p_raw[j])),
                    e_mean=float(e_mean[0]), p_mean=float(p_mean[j]),
                    cross=float(channel != ENROLL_CHANNEL), template=template, probe=vec, channel=channel,
                ))
    return rows


def factor_file(calibration, rows, work: Path, name: str) -> Path:
    """WeSpeaker's gather_calibration_factors, plus the two condition columns."""
    norm, durations, gathered = work / f"{name}.norm", work / f"{name}.dur", work / f"{name}.gathered"
    with open(norm, "w") as f, open(durations, "w") as d:
        for i, r in enumerate(rows):
            label = "target" if r["label"] else "nontarget"
            f.write(f"e{i} t{i} {r['normed']:.5f} {label} {r['e_mag']:.4f} {r['p_mag']:.4f} "
                    f"{r['e_mean']:.4f} {r['p_mean']:.4f}\n")
            d.write(f"e{i} {r['e_ms'] / 1000:.3f}\nt{i} {r['p_ms'] / 1000:.3f}\n")
    calibration.gather_calibration_factors(str(durations), MAX_SPEECH_MS / 1000, str(norm), str(gathered))
    out = work / f"{name}.factors"
    with open(gathered) as fin, open(out, "w") as fout:
        for line, r in zip(fin, rows):
            score = float(line.split()[3])
            fout.write(f"{line.rstrip()} {r['cross']:.4f} {r['cross'] * score:.5f}\n")
    return out


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
    """Normalized actual DCF at the Bayes threshold, and minimum DCF over thresholds."""
    threshold = -math.log(p_target / (1 - p_target))
    act = p_target * np.mean(llr[y] < threshold) + (1 - p_target) * np.mean(llr[~y] >= threshold)
    norm = min(p_target, 1 - p_target)
    order = np.sort(np.unique(llr))
    best = min(p_target * np.mean(llr[y] < t) + (1 - p_target) * np.mean(llr[~y] >= t)
               for t in order[:: max(1, len(order) // 2000)])
    return float(act / norm), float(best / norm)


def household_eval(data, held, backend):
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
                post, _ = posteriors({s: backend.llr(members[s], probe) for s in group}, priors, PRIOR_UNKNOWN)
                cos = {s: float(unit(regs[s][0][0])[0] @ unit(vec)[0]) for s in group}
                band = "lt_1s" if ms < 1000 else ("1_2.5s" if ms < 2500 else "ge_2.5s")
                key = (kind, channel, band)
                best = max(post, key=post.get)
                for use, threshold in USES.items():
                    hit = post[best] >= threshold
                    tally[key][f"{use}:{'correct' if hit and best == truth else ('wrong' if hit else 'none')}"] += 1
                best = max(cos, key=cos.get)
                second = max([v for k, v in cos.items() if k != best], default=-1)
                floor = max(f for start, f in BANDS if ms >= start) if ms >= 600 else 1
                for name, hit in (("v2_fixed", cos[best] >= 0.55 and cos[best] - second >= 0.1),
                                  ("v4_floors", cos[best] >= floor and cos[best] - second >= 0.06)):
                    tally[key][f"{name}:{'correct' if hit and best == truth else ('wrong' if hit else 'none')}"] += 1
                tally[key]["n"] += 1
    merged = defaultdict(lambda: defaultdict(int))
    for (kind, channel, band), counts in tally.items():
        for k in ((kind, channel, band), (kind, channel, "all"), (kind, "all", "all")):
            for name, n in counts.items():
                merged[k][name] += n
    return {f"{kind}/{channel}/{band}": {"n": c["n"], **{k: round(v / c["n"], 4)
                                                          for k, v in sorted(c.items()) if k != "n"}}
            for (kind, channel, band), c in sorted(merged.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "work"
    work.mkdir(exist_ok=True)
    (score_norm, calibration), revision = wespeaker()
    data = json.loads(args.embeddings.read_text())
    cohort_ids, held = data["cohort"], data["held_out"]
    rng = np.random.default_rng(0)
    pool, owners = cohort_pool(data, cohort_ids)
    mean = pool.mean(axis=0)

    train = trial_rows(score_norm, data, cohort_ids, mean, pool, owners, cohort_ids,
                       leave_out=True, rng=rng, per=800)
    model = work / "qmf.pt"
    calibration.train_calibration_model(str(factor_file(calibration, train, work, "train")), str(model))
    state = torch.load(model)
    weights = state["linear.weight"].reshape(-1).tolist()
    bias = float(state["linear.bias"].reshape(-1)[0])

    npz = args.out / "identity-cohort.npz"
    with open(npz, "wb") as f:
        np.savez(f, mean=mean.astype(np.float32), cohort=pool.astype(np.float32))
    spec = {
        "version": VERSION,
        "identity": data["identity"],
        "top_n": TOP_N, "max_speech_ms": MAX_SPEECH_MS,
        "features": list(FEATURES), "weights": [round(w, 6) for w in weights], "bias": round(bias, 6),
        "trained_with": {"wespeaker": revision, "trials": len(train),
                         "targets": sum(r["label"] for r in train)},
        "cohort": {"path": npz.name, "sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                   "speakers": len(cohort_ids), "vectors": int(len(pool)),
                   "dataset": data["dataset"], "dataset_revision": data["dataset_revision"],
                   "channels": {"phone": data["phone"], "device": data["device"]}},
    }
    (args.out / "identity-backend.json").write_text(json.dumps(spec, indent=1) + "\n")
    backend = load_backend(args.out)

    test = trial_rows(score_norm, data, held, mean, pool, owners, cohort_ids,
                      leave_out=False, rng=rng, per=2000)
    test_factors = factor_file(calibration, test, work, "test")
    calibration.infer_calibration(str(test_factors), str(model), str(work / "test.llr"))
    llr = np.array([float(line.split()[2]) for line in (work / "test.llr").read_text().splitlines()])
    y = np.array([r["label"] for r in test])

    # Channel's runtime must give WeSpeaker's LLRs (factors are written at 4-5 decimals).
    runtime = np.array([backend.llr(backend.side(r["template"], r["e_ms"], ENROLL_CHANNEL),
                                    backend.side(r["probe"], r["p_ms"], r["channel"]))
                        for r in test[:2000]])
    drift = float(np.max(np.abs(runtime - llr[:2000])))
    assert drift < 0.05, f"runtime LLRs drift {drift} from WeSpeaker's"

    report = {"backend": spec, "runtime_vs_wespeaker_max_llr_diff": round(drift, 5), "verification": {}}
    cross = np.array([r["cross"] for r in test]).astype(bool)
    duration = np.array([r["p_ms"] for r in test])
    for label, mask in {"all": np.ones(len(y), bool), "phone_probe": ~cross, "device_probe": cross,
                        "lt_1s": duration < 1000, "ge_2.5s": duration >= 2500}.items():
        act, mindcf = dcf(llr[mask], y[mask])
        report["verification"][label] = {
            "targets": int(y[mask].sum()), "nontargets": int((~y[mask]).sum()),
            "eer": round(eer(llr[mask], y[mask]), 4), "cllr": round(cllr(llr[mask], y[mask]), 4),
            "act_dcf_0.01": round(act, 4), "min_dcf_0.01": round(mindcf, 4),
        }
    report["household"] = household_eval(data, held, backend)
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({"trained_with": spec["trained_with"], "drift": report["runtime_vs_wespeaker_max_llr_diff"],
                      "verification": report["verification"]}, indent=1))


if __name__ == "__main__":
    main()
