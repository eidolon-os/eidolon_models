"""Isolated CPU extraction using upstream networks/frontends; no service calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from eidolon_models_speaker_eval.evaluation import _unit

WEIGHTS = {
    "campplus_onnx": "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
    "eres2netv2_onnx": "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx",
    "redimnet2_b3": "b3-vb2+vox2+cnc2_v0-lm.pt",
    "redimnet2_b2": "b2-vox2-lm.pt",
}
DURATIONS = (0.6, 1.0, 1.5, 2.0, 3.0, "full")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root, name, native_weights):
    meta = {"backend": "pytorch-cpu", "threads": 1}
    if name.endswith("onnx"):
        import onnxruntime as ort
        import kaldi_native_fbank as knf

        path = root / "weights" / WEIGHTS[name]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        model = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        contract = model.get_modelmeta().custom_metadata_map
        if int(contract["sample_rate"]) != 16000:
            raise ValueError("unexpected sample rate")
        if contract["feature_normalize_type"] != "global-mean":
            raise ValueError("unsupported normalization")
        meta.update(backend="onnxruntime-cpu", onnxruntime=ort.__version__, contract=contract,
                    fbank_version=knf.__version__)

        def infer(audio):
            opts = knf.FbankOptions()
            opts.frame_opts.dither = 0
            opts.frame_opts.samp_freq = 16000
            opts.frame_opts.snip_edges = True
            opts.mel_opts.num_bins = 80
            fbank = knf.OnlineFbank(opts)
            if not int(contract["normalize_samples"]):
                audio = audio * 32768
            fbank.accept_waveform(16000, audio)
            fbank.input_finished()
            features = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
            features -= features.mean(axis=0, keepdims=True)
            return model.run(None, {model.get_inputs()[0].name: features[None]})[0].reshape(-1)
    elif name == "campplus_native":
        from modelscope.models.audio.sv.DTDNN import CAMPPlus

        path = native_weights
        model = CAMPPlus(feat_dim=80, embedding_size=192)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        meta["params"] = sum(p.numel() for p in model.parameters())
        meta["frontend"] = "ModelScope DTDNN: torchaudio.compliance.kaldi.fbank80 + mean subtraction"

        def infer(audio):
            features = torchaudio.compliance.kaldi.fbank(torch.from_numpy(audio)[None], num_mel_bins=80)
            features -= features.mean(dim=0, keepdim=True)
            return model(features[None]).detach().numpy().reshape(-1)
    else:
        sys.path.insert(0, str(root / "redimnet-source"))
        from redimnet2.redimnet2 import ReDimNet2Wrap

        path = root / "weights" / WEIGHTS[name]
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = ReDimNet2Wrap(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        meta.update(params=sum(p.numel() for p in model.parameters()),
                    model_config=checkpoint["model_config"])

        def infer(audio):
            return model(torch.from_numpy(audio)[None]).detach().numpy().reshape(-1)
    meta.update(weights_sha256=sha(path), bytes=path.stat().st_size)
    return infer, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("model", choices=["campplus_native", *WEIGHTS])
    p.add_argument("--native-weights", type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    os.umask(0o077)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    manifest = json.loads((root / "assets.json").read_text())
    for item in manifest["models"] + manifest["code"] + manifest["samples"]:
        if sha(root / item["path"]) != item["sha256"]:
            raise ValueError("input checksum mismatch: " + item["path"])
    output = root / "embeddings" / (args.model + ".json")
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    infer, model_meta = build(root, args.model, args.native_weights)
    load_ms = (time.perf_counter() - start) * 1000
    warm, sr = sf.read(root / manifest["samples"][0]["path"], dtype="float32")
    if sr != 16000:
        raise ValueError("unexpected sample rate")
    rows = []
    with torch.inference_mode():
        start = time.perf_counter()
        infer(warm[:48000])
        cold_infer_ms = (time.perf_counter() - start) * 1000
        for _ in range(3):
            infer(warm[:48000])
        for index, sample in enumerate(manifest["samples"]):
            audio, sr = sf.read(root / sample["path"], dtype="float32")
            if sr != 16000 or audio.ndim != 1:
                raise ValueError("unsupported audio")
            for duration in DURATIONS:
                count = len(audio) if duration == "full" else int(duration * 16000)
                offset = (len(audio) - count) // 2
                clip = audio[offset:offset + count].copy()
                row = {"source_id": sample["id"], "speaker": sample["speaker"],
                       "duration": duration, "offset_samples": offset, "sample_count": count}
                start = time.perf_counter()
                try:
                    row["vector"] = _unit(infer(clip).tolist())
                    row["status"] = "ok"
                except Exception as exc:
                    row.update(status="unavailable", error=type(exc).__name__ + ": " + str(exc)[:200])
                row["elapsed_ms"] = (time.perf_counter() - start) * 1000
                rows.append(row)
            if (index + 1) % 12 == 0:
                print(args.model, index + 1, "/", len(manifest["samples"]), flush=True)
    result = dict(model=args.model, metadata=model_meta, load_ms=load_ms,
                  first_inference_ms=cold_infer_ms,
                  peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  platform=platform.platform(), python=sys.version, torch=torch.__version__,
                  torchaudio=torchaudio.__version__, rows=rows,
                  assets_sha256=sha(root / "assets.json"), production_ready=False,
                  timing_scope="features+embedding+normalization; excludes VAD, IO, queue, transport",
                  sessions_known=False, training_overlap_excluded=False)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("saved", output, "failures", sum(x["status"] != "ok" for x in rows), flush=True)


if __name__ == "__main__":
    main()
