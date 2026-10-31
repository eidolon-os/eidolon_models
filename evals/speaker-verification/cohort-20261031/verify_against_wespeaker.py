"""Check our back-end against WeSpeaker's own scripts on the same trials.

Loads ``wespeaker/bin/score_norm.py`` and ``score_calibration.py`` from a
WeSpeaker checkout by file (its package ``__init__`` pulls the whole
inference CLI, which these scripts do not need) and compares, on the
held-out trials of ``identity_eval.py``:

1. AS-norm: WeSpeaker's ``get_mean_std`` and normalisation versus Channel's
   ``evidence.trial_features`` — must agree to rounding.
2. QMF calibration on WeSpeaker's exact features (score; duration, magnitude
   and cohort mean as min / max / diff / ratio): WeSpeaker's
   ``train_calibration_model`` / ``infer_calibration`` versus our ``fit``.
3. What the deployed feature set and the channel condition are worth.

    eidolon_channel/.venv/bin/python -I \\
        eidolon_models/evals/speaker-verification/cohort-20261031/verify_against_wespeaker.py \\
        --wespeaker <checkout> --deps <dir with fire, kaldiio, termcolor> \\
        --embeddings .eidolon/speaker-cohort-20261031/build/embeddings.json --work <scratch dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wespeaker(checkout: Path):
    file_utils = load("wespeaker.utils.file_utils", checkout / "wespeaker/utils/file_utils.py")
    package, utils = types.ModuleType("wespeaker"), types.ModuleType("wespeaker.utils")
    utils.file_utils = file_utils
    package.utils = utils
    sys.modules.update({"wespeaker": package, "wespeaker.utils": utils,
                        "wespeaker.utils.file_utils": file_utils})
    return (load("ws_score_norm", checkout / "wespeaker/bin/score_norm.py"),
            load("ws_score_calibration", checkout / "wespeaker/bin/score_calibration.py"))


def components(ev, data, ids, cohort_ids, mean, cohort, owners, *, leave_out, rng, per):
    """Per-trial raw quantities, with each trial's speakers left out of its cohort statistics."""
    regs = {s: ev.registrations(data["speakers"][s]) for s in ids}
    regs = {s: r for s, r in regs.items() if r}
    index = {s: cohort_ids.index(s) for s in ids if s in cohort_ids}
    out = {k: [] for k in ("score", "e_ms", "p_ms", "e_mag", "p_mag", "e_mean", "e_std",
                           "p_mean", "p_std", "cross", "label")}
    speakers = sorted(regs)
    for a in speakers:
        (template, e_ms), _ = regs[a]
        e_raw = template - mean
        e_vec = e_raw / np.linalg.norm(e_raw)
        cands = [(b, p) for b in speakers for p in regs[b][1]]
        targets = [(b, p) for b, p in cands if b == a]
        others = [(b, p) for b, p in cands if b != a]
        pick = rng.choice(len(others), size=min(per, len(others)), replace=False)
        chosen = targets + [others[i] for i in pick]
        p_raw = np.stack([p[0] for _, p in chosen]) - mean
        p_vec = p_raw / np.linalg.norm(p_raw, axis=1, keepdims=True)
        excl = [{index.get(a), index.get(b)} - {None} if leave_out else set() for b, _ in chosen]
        pm, ps = ev.side_stats(p_vec, cohort, owners, excl)
        em, es = ev.side_stats(np.repeat(e_vec[None], len(chosen), 0), cohort, owners, excl)
        out["score"] += list(p_vec @ e_vec)
        out["e_ms"] += [e_ms] * len(chosen)
        out["p_ms"] += [p[1] for _, p in chosen]
        out["e_mag"] += [float(np.linalg.norm(e_raw))] * len(chosen)
        out["p_mag"] += list(np.linalg.norm(p_raw, axis=1))
        out["e_mean"] += list(em); out["e_std"] += list(es)
        out["p_mean"] += list(pm); out["p_std"] += list(ps)
        out["cross"] += [float(p[2] != ev.ENROLL_CHANNEL) for _, p in chosen]
        out["label"] += [b == a for b, _ in chosen]
    return {k: np.asarray(v) for k, v in out.items()}


def asnorm(c):
    return 0.5 * ((c["score"] - c["e_mean"]) / c["e_std"] + (c["score"] - c["p_mean"]) / c["p_std"])


def wespeaker_factors(calibration, c, work: Path, name: str):
    """Run WeSpeaker's own gather_calibration_factors on these trials."""
    norm, durs = work / f"{name}.norm", work / f"{name}.dur"
    with open(norm, "w") as f, open(durs, "w") as d:
        for i in range(len(c["label"])):
            label = "target" if c["label"][i] else "nontarget"
            f.write(f"e{i} t{i} {asnorm_values[name][i]:.5f} {label} {c['e_mag'][i]:.4f} "
                    f"{c['p_mag'][i]:.4f} {c['e_mean'][i]:.4f} {c['p_mean'][i]:.4f}\n")
            d.write(f"e{i} {c['e_ms'][i] / 1000:.3f}\nt{i} {c['p_ms'][i] / 1000:.3f}\n")
    factors = work / f"{name}.factors"
    calibration.gather_calibration_factors(str(durs), 20.0, str(norm), str(factors))
    rows = [line.split() for line in factors.read_text().splitlines()]
    return factors, np.array([[float(v) for v in r[3:]] for r in rows])


asnorm_values: dict[str, np.ndarray] = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wespeaker", type=Path, required=True)
    parser.add_argument("--deps", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.deps))
    args.work.mkdir(parents=True, exist_ok=True)
    norm_mod, calibration = wespeaker(args.wespeaker)
    ev = load("identity_eval", HERE / "identity_eval.py")
    from eidolon.livekit.plugins.speaker_verification.evidence import Side, trial_features

    data = json.loads(args.embeddings.read_text())
    cohort_ids, held = data["cohort"], data["held_out"]
    vectors, owners = ev.cohort_pool(data, cohort_ids)
    mean = vectors.mean(axis=0)
    cohort = ev.unit(vectors - mean)
    rng = np.random.default_rng(0)
    train = components(ev, data, cohort_ids, cohort_ids, mean, cohort, owners, leave_out=True, rng=rng, per=800)
    test = components(ev, data, held, cohort_ids, mean, cohort, owners, leave_out=False, rng=rng, per=2000)
    report = {}

    # 1. AS-norm, held-out trials (no leave-out): WeSpeaker versus Channel's runtime arithmetic.
    regs = {s: ev.registrations(data["speakers"][s]) for s in held}
    s = sorted(r for r in regs if regs[r])[0]
    (template, e_ms), probes = regs[s]
    e_m, e_s = norm_mod.get_mean_std((template - mean)[None], vectors - mean, ev.TOP_N)
    p_m, p_s = norm_mod.get_mean_std(np.stack([p[0] for p in probes]) - mean, vectors - mean, ev.TOP_N)
    diffs = []
    for i, (vec, ms, ch) in enumerate(probes):
        raw = float(ev.unit(template - mean)[0] @ ev.unit(vec - mean)[0])
        theirs = 0.5 * ((raw - e_m[0]) / e_s[0] + (raw - p_m[i]) / p_s[i])
        enroll = Side(ev.unit(template - mean)[0], e_ms, float(e_m[0]), float(e_s[0]), "phone")
        probe = Side(ev.unit(vec - mean)[0], ms, float(p_m[i]), float(p_s[i]), ch)
        ours_side = (ev.side_stats(ev.unit(vec - mean), cohort, owners, [set()]))
        diffs.append(abs(trial_features(enroll, probe, ev.MAX_SPEECH_MS)[0] - theirs))
        diffs.append(abs(float(ours_side[0][0]) - float(p_m[i])) + abs(float(ours_side[1][0]) - float(p_s[i])))
    report["asnorm_max_abs_diff"] = float(max(diffs))

    # 2. Calibration on WeSpeaker's exact features: their trainer versus ours.
    asnorm_values["train"], asnorm_values["test"] = asnorm(train), asnorm(test)
    train_file, x_train = wespeaker_factors(calibration, train, args.work, "train")
    test_file, x_test = wespeaker_factors(calibration, test, args.work, "test")
    model = args.work / "wespeaker-qmf.pt"
    calibration.train_calibration_model(str(train_file), str(model))
    calibration.infer_calibration(str(test_file), str(model), str(args.work / "test.llr"))
    theirs = np.array([float(line.split()[2]) for line in (args.work / "test.llr").read_text().splitlines()])
    y_train, y_test = train["label"], test["label"]
    w, b, _ = ev.fit(x_train, y_train)
    ours = x_test @ w + b
    cross = test["cross"].astype(bool)

    def scores(llr):
        out = {}
        for name, mask in (("same_channel", ~cross), ("other_channel", cross), ("all", np.ones_like(cross))):
            act, mn = ev.dcf(llr[mask], y_test[mask])
            out[name] = {"eer": round(ev.eer(llr[mask], y_test[mask]), 4),
                         "cllr": round(ev.cllr(llr[mask], y_test[mask]), 4),
                         "act_dcf": round(act, 4), "min_dcf": round(mn, 4)}
        return out

    report["wespeaker_qmf_13_features"] = scores(theirs)
    report["our_fit_same_13_features"] = scores(ours)
    report["llr_max_abs_diff_ours_vs_wespeaker"] = float(np.max(np.abs(ours - theirs)))

    # 3. The channel condition, on top of WeSpeaker's features and on the deployed ones.
    def with_condition(x, c):
        return np.column_stack([x, c["cross"], c["cross"] * x[:, 0]])
    w2, b2, _ = ev.fit(with_condition(x_train, train), y_train)
    report["wespeaker_13_plus_condition"] = scores(with_condition(x_test, test) @ w2 + b2)

    def deployed(c):
        return ev.features_matrix(np.zeros((len(c["label"]), 1)), c["e_ms"], c["e_mean"], c["e_std"],
                                  np.zeros((len(c["label"]), 1)), c["p_ms"], c["p_mean"], c["p_std"],
                                  c["cross"])
    xd_train, xd_test = deployed(train), deployed(test)
    xd_train[:, 0], xd_test[:, 0] = asnorm(train), asnorm(test)
    w3, b3, _ = ev.fit(xd_train, y_train)
    report["deployed_10_features"] = scores(xd_test @ w3 + b3)
    (args.work / "verification.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()


def household(ev, data, mean, cohort, owners, build_features, weights, bias, prior_unknown=0.5):
    """Two-person households from held-out speakers, decided per use like the product."""
    import math
    from collections import defaultdict
    held = data["held_out"]
    regs = {s: ev.registrations(data["speakers"][s]) for s in held}
    speakers = sorted(s for s in regs if regs[s])
    groups = [speakers[i:i + 2] for i in range(0, len(speakers) - 1, 2)]
    tally = defaultdict(lambda: [0, 0, 0])
    for group in groups:
        for truth in speakers:
            for vec, ms, ch in regs[truth][1]:
                llrs = {}
                for s in group:
                    (template, e_ms), _ = regs[s]
                    e_raw, p_raw = template - mean, vec - mean
                    e_vec, p_vec = e_raw / np.linalg.norm(e_raw), p_raw / np.linalg.norm(p_raw)
                    em, es = ev.side_stats(e_vec[None], cohort, owners, [set()])
                    pm, ps = ev.side_stats(p_vec[None], cohort, owners, [set()])
                    c = {"score": np.array([e_vec @ p_vec]), "e_ms": np.array([e_ms]), "p_ms": np.array([ms]),
                         "e_mag": np.array([np.linalg.norm(e_raw)]), "p_mag": np.array([np.linalg.norm(p_raw)]),
                         "e_mean": em, "e_std": es, "p_mean": pm, "p_std": ps,
                         "cross": np.array([float(ch != ev.ENROLL_CHANNEL)]), "label": np.array([False])}
                    llrs[s] = float(build_features(c)[0] @ weights + bias)
                terms = {k: math.log((1 - prior_unknown) / len(group)) + v for k, v in llrs.items()}
                terms[None] = math.log(prior_unknown)
                top = max(terms.values())
                post = {k: math.exp(v - top) for k, v in terms.items()}
                total = sum(post.values())
                best = max(llrs, key=llrs.get)
                for use, threshold in (("personal_memory", 0.99), ("addressing", 0.9)):
                    hit = post[best] / total >= threshold
                    row = tally[(use, truth in group, ch)]
                    row[0] += 1
                    row[1] += hit and best == truth
                    row[2] += hit and best != truth
    return {f"{use}/{'own' if own else 'stranger'}/{ch}": round((c if own else w) / n, 4)
            for (use, own, ch), (n, c, w) in sorted(tally.items())}
