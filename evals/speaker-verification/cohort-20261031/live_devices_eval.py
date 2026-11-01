"""The identity back-end on live speech across devices and distances (design §8.2 1c).

3D-Speaker (Alibaba DAMO Academy, https://3dspeaker.github.io; CC BY-SA 4.0)
records its 240 test speakers live and simultaneously on phones, a tablet, a
PC, a recording pen, a directional microphone and microphone arrays, 0.1–4 m
away. Two evaluations:

1. the official cross-device and cross-distance trial lists (one utterance
   enrolled), for comparison with published results;
2. the product's own use: each speaker registers with three segments
   (averaged, voiced speech summed, channel ``mobile`` as the phone app
   records it); probes are the speaker's other segments on every device and
   distance; two-person households decide each probe per use, every other
   speaker being a stranger. Each member is also scored as a stranger to a
   household where only the other is enrolled. The test set has no handheld
   phone: phones stand 1–4 m away, and only a directional microphone (0.1 m)
   and a PC (0.2 m) are close. So registration is taken twice: from the
   nearest phone (harder than the product), and from the nearest close device
   standing in for the handheld phone.

Extraction is Channel's own (``ReferenceSpeakerExtractor.extract``), each
recording first brought to the level the phone app delivers (a stand-in for a
device's gain control). Embeddings are cached beside the corpus; results hold
aggregates only.

    eidolon_channel/.venv/bin/python -I \\
        eidolon_models/evals/speaker-verification/cohort-20261031/live_devices_eval.py \\
        --corpus /Users/manson/ai/data/speaker-corpora/3d-speaker \\
        --model-dir eidolon_models/speaker_verification/campplus_zh_16k_common/624dcb0f \\
        --backend v3=eidolon_models/speaker_verification/campplus_zh_16k_common/624dcb0f \\
        --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PHONES = {"ANDROID", "IPHONE", "PHONE"}
LEVEL_RMS = 0.06
MIN_SPEECH_MS = 600
ENROLL_SEGMENTS = 3
STRANGERS_PER_HOUSEHOLD = 40
USES = {"personal_memory": 0.99, "addressing": 0.90}
DISCOVERY = 0.99
MIN_UNKNOWN_SPEECH_MS = 1500
PRIOR_UNKNOWN = 0.5


def _module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ie = _module("identity_eval")
bc = _module("build_cohort")


def channel_of(device: str) -> str:
    """The product's channel class: every phone is one channel."""
    return "mobile" if device in PHONES else device.lower()


def corpus(root: Path):
    files = root / "files-x" / "files"
    distances = dict(line.split()[::-1] if len(line.split()) == 2 else (line.strip(), "")
                     for line in (files / "distance_file").read_text().splitlines() if line.strip())
    wav = dict(line.split(maxsplit=1) for line in (files / "test_wav.scp").read_text().splitlines())
    utts = {}
    with open(files / "test_utt2info.csv", newline="") as f:
        for row in csv.DictReader(f):
            code = row["Utt-id"].split("_")[-2]
            distance = distances.get(code, "")
            utts[row["Utt-id"]] = {
                "speaker": row["Spk-id"], "segment": row["Seg-id"], "device": row["Device"],
                "distance": float(distance) if distance else None,
                "path": root / "test-x" / wav[row["Utt-id"]].strip(),
            }
    return utts


_EXTRACTOR = None


def _start(model_dir):
    global _EXTRACTOR
    _EXTRACTOR = bc.ReferenceSpeakerExtractor(model_dir)
    _EXTRACTOR.warm_up()


def _extract(job):
    out = {}
    for utt, path in job:
        raw = bc.read_pcm(Path(path))
        try:
            sample = _EXTRACTOR.extract(bc.level(raw, LEVEL_RMS).tobytes(), MIN_SPEECH_MS)
        except bc.InvalidSpeakerAudio:
            continue
        out[utt] = {"vector": [round(float(v), 6) for v in sample.vector], "speech_ms": sample.speech_ms}
    return out


def embed(utts, model_dir: Path, cache: Path, jobs: int):
    done = json.loads(cache.read_text()) if cache.exists() else {}
    todo = sorted(u for u in utts if u not in done)
    chunks = [[(u, str(utts[u]["path"])) for u in todo[i:i + 200]] for i in range(0, len(todo), 200)]
    if chunks:
        with ProcessPoolExecutor(jobs, initializer=_start, initargs=(model_dir,)) as pool:
            for n, part in enumerate(pool.map(_extract, chunks), 1):
                done.update(part)
                print(f"embedded {n}/{len(chunks)}", flush=True)
        cache.write_text(json.dumps(done))
    return done


def band(ms):
    return "lt_1s" if ms < 1000 else ("1_2.5s" if ms < 2500 else "ge_2.5s")


def distance_band(d):
    return "unknown" if d is None else ("near" if d < 0.8 else "far")


def metrics(llr, y):
    llr, y = np.asarray(llr), np.asarray(y, bool)
    if y.all() or not y.any():
        return {"n": int(len(y))}
    act, mindcf = ie.dcf(llr, y)
    return {"targets": int(y.sum()), "nontargets": int((~y).sum()), "eer": round(ie.eer(llr, y), 4),
            "cllr": round(ie.cllr(llr, y), 4), "act_dcf_0.01": round(act, 4), "min_dcf_0.01": round(mindcf, 4)}


def official(backend, utts, emb, trials: Path):
    """Single-utterance enrollment, each side in its own device's channel."""
    sides = {}

    def side(u):
        if u not in sides:
            sides[u] = backend.side(np.asarray(emb[u]["vector"]), emb[u]["speech_ms"], channel_of(utts[u]["device"]))
        return sides[u]

    groups = defaultdict(lambda: ([], []))
    for line in trials.read_text().splitlines():
        e, t, label = line.split()
        if e not in emb or t not in emb:
            continue
        llr = backend.llr(side(e), side(t))
        same = channel_of(utts[e]["device"]) == channel_of(utts[t]["device"])
        for key in ("all", "same_channel" if same else "cross_channel"):
            groups[key][0].append(llr)
            groups[key][1].append(label == "target")
    return {k: metrics(*v) for k, v in sorted(groups.items())}


NEAR_M = 0.8
REGISTRATION = {
    "phone": lambda info: info["device"] in PHONES,
    "near": lambda info: info["distance"] is not None and info["distance"] < NEAR_M,
}


def registrations(utts, emb, source):
    """Per speaker: a registration (three segments, nearest eligible recording) and the other segments' probes."""
    eligible = REGISTRATION[source]
    by_speaker = defaultdict(lambda: defaultdict(list))
    for u, info in utts.items():
        if u in emb:
            by_speaker[info["speaker"]][info["segment"]].append(u)
    regs = {}
    for speaker, segments in sorted(by_speaker.items()):
        enroll = []
        for segment in sorted(segments):
            candidates = [u for u in segments[segment] if eligible(utts[u])]
            if candidates:
                enroll.append(min(candidates, key=lambda u: (utts[u]["distance"] is None, utts[u]["distance"] or 0)))
            if len(enroll) == ENROLL_SEGMENTS:
                break
        if len(enroll) < ENROLL_SEGMENTS:
            continue
        used = {utts[u]["segment"] for u in enroll}
        template = ie.unit(np.mean([emb[u]["vector"] for u in enroll], axis=0))[0]
        probes = [u for seg, us in segments.items() if seg not in used for u in us]
        regs[speaker] = ((template, sum(emb[u]["speech_ms"] for u in enroll)), probes)
    return regs


def households(backend, utts, emb, regs, rng):
    """Two-person households deciding each probe per use, like the runtime."""
    speakers = sorted(regs)
    sides = {}

    def side(u):
        if u not in sides:
            sides[u] = backend.side(np.asarray(emb[u]["vector"]), emb[u]["speech_ms"], channel_of(utts[u]["device"]))
        return sides[u]

    templates = {s: backend.side(regs[s][0][0], regs[s][0][1], "mobile") for s in speakers}
    tally = defaultdict(lambda: defaultdict(int))
    alone = []

    def count(kind, u, truth, best, post, unknown):
        info, ms = utts[u], emb[u]["speech_ms"]
        keys = [(kind, "all"), (kind, "channel:" + ("mobile" if info["device"] in PHONES else "other")),
                (kind, "distance:" + distance_band(info["distance"])), (kind, "speech:" + band(ms))]
        for key in keys:
            tally[key]["n"] += 1
            for use, threshold in USES.items():
                named = post >= threshold
                tally[key][f"{use}:{'correct' if named and best == truth else ('wrong' if named else 'none')}"] += 1
            tally[key][f"discovery:{'yes' if unknown >= DISCOVERY and ms >= MIN_UNKNOWN_SPEECH_MS else 'no'}"] += 1

    priors = {0: (1 - PRIOR_UNKNOWN) / 2, 1: (1 - PRIOR_UNKNOWN) / 2}
    for a, b in zip(speakers[0::2], speakers[1::2]):
        members = {0: templates[a], 1: templates[b]}
        others = [s for s in speakers if s not in (a, b)]
        strangers = rng.choice(others, size=min(STRANGERS_PER_HOUSEHOLD, len(others)), replace=False)
        for truth, speaker in ((0, a), (1, b)):
            for u in regs[speaker][1]:
                llrs = {i: backend.llr(t, side(u)) for i, t in members.items()}
                post, unknown = ie.posteriors(llrs, priors, PRIOR_UNKNOWN)
                best = max(post, key=post.get)
                count("own", u, truth, best, post[best], unknown)
                alone.append(llrs[1 - truth])
        for speaker in strangers:
            for u in regs[speaker][1]:
                llrs = {i: backend.llr(t, side(u)) for i, t in members.items()}
                post, unknown = ie.posteriors(llrs, priors, PRIOR_UNKNOWN)
                best = max(post, key=post.get)
                count("stranger", u, None, best, post[best], unknown)
    rates = {f"{kind}/{key}": {"n": c["n"], **{k: round(v / c["n"], 5) for k, v in sorted(c.items()) if k != "n"}}
             for (kind, key), c in sorted(tally.items())}
    llr = np.asarray(alone)
    # One enrolled Person: P = (1 - π0) e^LLR / (π0 + (1 - π0) e^LLR).
    post = (1 - PRIOR_UNKNOWN) * np.exp(np.minimum(llr, 700)) / (
        PRIOR_UNKNOWN + (1 - PRIOR_UNKNOWN) * np.exp(np.minimum(llr, 700)))
    rates["members_as_strangers"] = {
        "n": int(len(llr)), "llr_p50": round(float(np.median(llr)), 2), "llr_p99": round(float(np.quantile(llr, 0.99)), 2),
        "llr_max": round(float(llr.max()), 2),
        **{f"named_{use}": round(float(np.mean(post >= t)), 5) for use, t in USES.items()}}
    return rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="CAM++ package, for extraction")
    parser.add_argument("--backend", action="append", required=True, help="name=model package directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    utts = corpus(args.corpus)
    emb = embed(utts, args.model_dir, args.corpus / "embeddings.json", args.jobs)
    regs = {source: registrations(utts, emb, source) for source in REGISTRATION}
    files = args.corpus / "files-x" / "files"
    report = {"corpus": {
        "utterances": len(utts), "embedded": len(emb), "speakers": len({u["speaker"] for u in utts.values()}),
        "registered": {s: len(r) for s, r in regs.items()}, "devices": sorted({u["device"] for u in utts.values()}),
        "speech_ms_median": float(np.median([e["speech_ms"] for e in emb.values()])),
    }, "backends": {}}
    for spec in args.backend:
        name, path = spec.split("=", 1)
        backend = ie.load_backend(Path(path))
        report["backends"][name] = {
            "version": backend.version,
            "official_cross_device": official(backend, utts, emb, files / "trials_cross_device"),
            "official_cross_distance": official(backend, utts, emb, files / "trials_cross_distance"),
            **{f"household_{source}_registration": households(backend, utts, emb, r, np.random.default_rng(0))
               for source, r in regs.items()},
        }
        print(name, json.dumps(report["backends"][name]["official_cross_device"]["all"]), flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report["corpus"], ensure_ascii=False))


if __name__ == "__main__":
    main()
