"""Single backend per fresh process. Feature + encoder + L2, CPU single thread."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpus", type=Path)
    p.add_argument("weights", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--backend", choices=["torch", "ort"], required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.corpus / "assets.json").read_text())
    if args.backend == "torch":
        import torch
        import torchaudio
        from modelscope.models.audio.sv.DTDNN import CAMPPlus
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        start = time.perf_counter()
        model = CAMPPlus(feat_dim=80, embedding_size=192).eval()
        model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=True)

        @torch.inference_mode()
        def infer(audio):
            features = torchaudio.compliance.kaldi.fbank(torch.from_numpy(audio)[None], num_mel_bins=80)
            features -= features.mean(0, keepdim=True)
            raw = model(features[None]).numpy().ravel()
            return raw / np.linalg.norm(raw)
    else:
        import onnxruntime as ort
        import kaldi_native_fbank as knf
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        start = time.perf_counter()
        model = ort.InferenceSession(str(args.weights), options, providers=["CPUExecutionProvider"])
        input_name = model.get_inputs()[0].name

        def infer(audio):
            opts = knf.FbankOptions()
            opts.frame_opts.dither = 0
            opts.frame_opts.samp_freq = 16000
            opts.frame_opts.snip_edges = True
            opts.mel_opts.num_bins = 80
            fbank = knf.OnlineFbank(opts)
            fbank.accept_waveform(16000, audio)
            fbank.input_finished()
            features = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
            features -= features.mean(0, keepdims=True)
            raw = model.run(None, {input_name: features[None]})[0].ravel()
            return raw / np.linalg.norm(raw)
    load_ms = (time.perf_counter() - start) * 1000
    warm, _ = sf.read(args.corpus / manifest["samples"][0]["path"], dtype="float32")
    start = time.perf_counter()
    infer(warm[:48000])
    cold_ms = (time.perf_counter() - start) * 1000
    for _ in range(3):
        infer(warm[:48000])
    rows = []
    for i, sample in enumerate(manifest["samples"]):
        audio, sr = sf.read(args.corpus / sample["path"], dtype="float32")
        assert sr == 16000 and audio.ndim == 1
        for duration in [0.6, 1.0, 1.5, 2.0, 3.0, "full"]:
            count = len(audio) if duration == "full" else int(duration * 16000)
            offset = (len(audio) - count) // 2
            clip = audio[offset:offset + count].copy()
            start = time.perf_counter()
            vector = infer(clip)
            elapsed = (time.perf_counter() - start) * 1000
            assert vector.shape == (192,) and np.isfinite(vector).all()
            rows.append(dict(id=sample["id"], duration=duration, elapsed_ms=elapsed))
        if (i + 1) % 48 == 0:
            print(args.backend, i + 1, "/", len(manifest["samples"]), flush=True)
    result = dict(backend=args.backend, platform=platform.platform(), python=platform.python_version(),
                  weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                  versions={k: importlib.metadata.version(k) for k in ["torch", "torchaudio", "onnxruntime", "kaldi-native-fbank", "numpy", "soundfile"]},
                  load_ms=load_ms, cold_3s_inference_ms=cold_ms,
                  peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  torch_imported=args.backend == "torch", threads=1, timing_scope="frontend + encoder + L2; excludes IO, VAD, queue, transport, Person/Memory",
                  rows=rows, summary={str(d): dict(n=sum(r["duration"] == d for r in rows),
                      p50_ms=float(np.percentile([r["elapsed_ms"] for r in rows if r["duration"] == d], 50)),
                      p95_ms=float(np.percentile([r["elapsed_ms"] for r in rows if r["duration"] == d], 95)))
                      for d in [0.6, 1.0, 1.5, 2.0, 3.0, "full"]})
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
