"""Counterfactual graph experiment in memory only; never exports a replacement model."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
import kaldi_native_fbank as knf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    root = p.parse_args().root
    torch.set_num_threads(1)
    manifest = json.loads((root / "assets.json").read_text())
    paths = {x["id"]: root / x["path"] for x in manifest["samples"]}
    native = json.loads((root / "embeddings/campplus_native.json").read_text())["rows"]
    exported = json.loads((root / "embeddings/campplus_onnx.json").read_text())["rows"]
    lookup = {(x["source_id"], str(x["duration"])): x for x in exported}
    rank = sorted(native, key=lambda x: np.dot(x["vector"], lookup[(x["source_id"], str(x["duration"]))]["vector"]))
    path = root / "weights/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    original = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    graph = onnx.load(path)
    pools = []
    for node in graph.graph.node:
        if node.op_type == "AveragePool":
            pools.append(dict(name=node.name, attributes={
                a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}))
            for attribute in node.attribute:
                if attribute.name == "count_include_pad":
                    attribute.i = 0
    counterfactual = ort.InferenceSession(graph.SerializeToString(), options,
                                         providers=["CPUExecutionProvider"])

    def run(net, features):
        output = net.run(None, {net.get_inputs()[0].name: features[None]})[0].ravel()
        return output / np.linalg.norm(output)

    rows = []
    for row in rank[:5] + rank[-5:]:
        audio, _ = sf.read(paths[row["source_id"]], dtype="float32")
        clip = audio[row["offset_samples"]:row["offset_samples"] + row["sample_count"]]
        features = torchaudio.compliance.kaldi.fbank(torch.from_numpy(clip)[None], num_mel_bins=80).numpy()
        features -= features.mean(0, keepdims=True)
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = 80
        fbank = knf.OnlineFbank(opts)
        fbank.accept_waveform(16000, clip)
        fbank.input_finished()
        other = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
        other -= other.mean(0, keepdims=True)
        target = np.asarray(row["vector"])
        rows.append(dict(id=row["source_id"], duration=row["duration"],
                         feature_abs_max=float(np.max(np.abs(features-other))),
                         original_same_features_cos=float(target @ run(original, features)),
                         counterfactual_same_features_cos=float(target @ run(counterfactual, features))))
    report = dict(changed_in_memory_only=True, replacement_model_exported=False,
                  intervention="AveragePool count_include_pad=0", pools=pools, rows=rows)
    output = root / "onnx-pooling-diagnostic.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2))
    print("pool nodes", len(pools), "first", pools[:1])
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
