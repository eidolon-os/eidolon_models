"""Paired numerical and frozen-policy comparison. Public corpus only, no services."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
from modelscope.models.audio.sv.DTDNN import CAMPPlus

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "anonymous-20261028"))
from analyze import DURATIONS, calibrate, discover, open_set, split_samples, unit, vectors_for


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_features(audio):
    f = torchaudio.compliance.kaldi.fbank(torch.from_numpy(audio)[None], num_mel_bins=80)
    return (f - f.mean(0, keepdim=True)).numpy()


def portable_features(audio):
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = 80
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, audio)
    fbank.input_finished()
    f = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
    return f - f.mean(0, keepdims=True)


def metrics(a, b):
    ua, ub = unit(a), unit(b)
    return dict(cosine=float(np.clip(ua @ ub, -1, 1)),
                unit_abs_max=float(np.max(np.abs(ua - ub))),
                raw_abs_max=float(np.max(np.abs(a - b))))


def aggregate(rows):
    return dict(n=len(rows), cosine_min=min(x["cosine"] for x in rows),
                cosine_mean=float(np.mean([x["cosine"] for x in rows])),
                unit_abs_max=max(x["unit_abs_max"] for x in rows),
                raw_abs_max=max(x["raw_abs_max"] for x in rows))


def decisions(enrollment, probes, policy):
    known = dict(list(enrollment.items())[:2])
    labels = []
    for row in probes:
        scores = sorted(((float(row["vector"] @ v), k) for k, v in known.items()), reverse=True)
        accepted = scores[0][0] >= policy["known"] and scores[0][0] - scores[1][0] >= policy["margin"]
        labels.append(scores[0][1] if accepted else None)
    return labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpus", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("native_weights", type=Path)
    args = p.parse_args()
    output = args.output / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    manifest = json.loads((args.corpus / "assets.json").read_text())
    for row in manifest["samples"]:
        assert sha(args.corpus / row["path"]) == row["sha256"]
    paths = dict(public=args.corpus / "weights/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
                 legacy=args.output / "weights/campplus-legacy.onnx",
                 exported=args.output / "weights/campplus-semantic.onnx")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    sessions = {k: ort.InferenceSession(str(v), options, providers=["CPUExecutionProvider"])
                for k, v in paths.items()}
    model = CAMPPlus(feat_dim=80, embedding_size=192).eval()
    model.load_state_dict(torch.load(args.native_weights, map_location="cpu", weights_only=True), strict=True)

    def run(name, features):
        s = sessions[name]
        return s.run(None, {s.get_inputs()[0].name: features[None]})[0].ravel()

    names = ["native", "legacy_same", "public_same", "exported_same", "public_pipeline", "exported_pipeline"]
    data = {k: {"rows": []} for k in names}
    errors = {k: [] for k in names[1:]}
    feature_errors = []
    with torch.inference_mode():
        for i, sample in enumerate(manifest["samples"]):
            audio, sr = sf.read(args.corpus / sample["path"], dtype="float32")
            assert sr == 16000 and audio.ndim == 1
            for duration in DURATIONS:
                count = len(audio) if duration == "full" else int(duration * 16000)
                offset = (len(audio) - count) // 2
                clip = audio[offset:offset + count].copy()
                f, other = native_features(clip), portable_features(clip)
                assert 30 <= len(f) <= 3000
                feature_errors.append(float(np.max(np.abs(f - other))))
                outputs = {"native": model(torch.from_numpy(f)[None]).numpy().ravel()}
                outputs.update({k + "_same": run(k, f) for k in paths})
                outputs.update({k + "_pipeline": run(k, other) for k in ["public", "exported"]})
                for name, raw in outputs.items():
                    data[name]["rows"].append(dict(source_id=sample["id"], speaker=sample["speaker"],
                                                  duration=duration, status="ok", vector=unit(raw).tolist()))
                    if name != "native":
                        errors[name].append(dict(source_id=sample["id"], duration=duration,
                                                 **metrics(outputs["native"], raw)))
            if (i + 1) % 12 == 0:
                print("paired corpus", i + 1, "/", len(manifest["samples"]), flush=True)
        # Synthetic FBank tensors isolate graph shape behavior; these are not accuracy samples.
        frames = sorted(set([30, 31, 58, 98, 148, 298, 2998, 2999, 3000] +
                            [n + d for n in [200, 400, 600, 1000, 2000] for d in [-2, -1, 0, 1, 2]]))
        boundary = []
        for length in frames:
            f = torch.randn(1, length, 80)
            f -= f.mean(1, keepdim=True)
            native = model(f).numpy().ravel()
            boundary.append(dict(frames=length, models={k: metrics(native, run(k, f[0].numpy())) for k in paths}))
    (args.output / "vectors.json").write_text(json.dumps(data))
    groups, calibration, test = split_samples(manifest)
    policy_results = []
    for duration in DURATIONS:
        ce, cp, _ = vectors_for(data["native"], groups, calibration, duration)
        policy = calibrate(ce, cp)
        ne, np_, _ = vectors_for(data["native"], groups, test, duration)
        expected = decisions(ne, np_, policy)
        base_discovery = [discover(ne, np_, policy, seed=s, can_create=duration == "full" or duration >= 1.5)
                          for s in range(5)]
        variants = {}
        for name in names:
            e, probes, _ = vectors_for(data[name], groups, test, duration)
            discovery = [discover(e, probes, policy, seed=s, can_create=duration == "full" or duration >= 1.5)
                         for s in range(5)]
            variants[name] = dict(open_set=open_set(e, probes, policy),
                                  decision_changes=sum(a != b for a, b in zip(expected, decisions(e, probes, policy))),
                                  old_native_gallery=open_set(ne, probes, policy),
                                  old_gallery_decision_changes=sum(a != b for a, b in zip(expected, decisions(ne, probes, policy))),
                                  discovery=discovery,
                                  discovery_summary_changed_seeds=sum(a != b for a, b in zip(base_discovery, discovery)))
        policy_results.append(dict(duration=duration, frozen_native_policy=policy, variants=variants))
    native = np.asarray([x["vector"] for x in data["native"]["rows"]])
    native_scores = native @ native.T
    score_errors = {}
    for name in names[1:]:
        other = np.asarray([x["vector"] for x in data[name]["rows"]])
        score_errors[name] = dict(within_backend_max=float(np.max(np.abs(other @ other.T - native_scores))),
                                 cross_native_templates_max=float(np.max(np.abs(native @ other.T - native_scores))))
    result = dict(production_ready=False, source_manifest_sha256=sha(args.corpus / "assets.json"),
                  native_weights_sha256=sha(args.native_weights),
                  models={k: dict(sha256=sha(v), bytes=v.stat().st_size) for k, v in paths.items()},
                  features_abs_max=max(feature_errors),
                  parity={k: dict(overall=aggregate(v), by_duration={str(d): aggregate([x for x in v if x["duration"] == d])
                               for d in DURATIONS}, worst=min(v, key=lambda x: x["cosine"])) for k, v in errors.items()},
                  score_errors=score_errors, boundary=boundary, frozen_policy=policy_results,
                  gate=dict(cosine_min=0.999999, normalized_max_error=0.0001))
    result["export_passed"] = all(x["cosine"] >= .999999 and x["unit_abs_max"] <= .0001
                                   for k in ["exported_same", "exported_pipeline"] for x in errors[k]) and all(
        x["models"]["exported"]["cosine"] >= .999999 and x["models"]["exported"]["unit_abs_max"] <= .0001 for x in boundary)
    output.write_text(json.dumps(result, indent=2))
    assert result["export_passed"], "Export is not numerically equivalent within the declared gate"
    print("comparison saved; export gate passed", flush=True)


if __name__ == "__main__":
    main()
