"""PyTorch checkpoint → ONNX, then prove the two agree before recording the export.

Needs the ``export`` extra. The graph goes to ``onnx/model.onnx`` with the
weights beside it in ``onnx/model.onnx.data`` (external data): post-processing
then only ever loads the small graph proto, which matters on a 7 GB host.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from .artifacts import EXPORT_RECORD, Manifest, sha256_file
from .sequence import (
    INPUT_NAMES,
    QTYPES,
    ModelConfig,
    Tokenizer,
    build_sequence,
    collate,
    to_internal,
)

PARITY_TOLERANCE = 1e-3  # max |Δlogit|; measured ~2e-5 on fp32 CPU


def _sample_batch(tok: Tokenizer, cfg: ModelConfig, state_tokens: int) -> dict[str, np.ndarray]:
    unit = "师父，前面那座山上好像有妖气，八戒说他饿了，沙僧在看行李。"
    state = {"speaker": "主人", "utterance": "悟空，你去看看。", "history": unit}
    while len(tok.encode(json.dumps(state, ensure_ascii=False))) < state_tokens:
        state["history"] += unit
    questions = [
        {
            "type": "choice",
            "instructions": "`utterance` 是说给谁听的？",
            "criteria": {
                "唐僧": "师父",
                "孙悟空": "悟空、猴哥",
                "猪八戒": None,
                "沙僧": None,
                "所有人": None,
            },
        },
        {"type": "noul", "instructions": "需要有人回应吗？"},
        {"type": "score", "instructions": "有多紧急？", "criteria": ["不急", "尽快", "马上"]},
    ]
    items = []
    for qdef in questions:
        q = to_internal(qdef)
        ids, markers, _ = build_sequence(
            tok, state, q, state_tokens + cfg.head_max_len, cfg.head_max_len
        )
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
    return collate(items, tok.pad_token_id)


def export_onnx(manifest: Manifest, *, force: bool = False, log=print) -> Path:
    import laya
    import onnx
    import onnxruntime as ort
    import torch

    out = manifest.onnx_path
    if out.is_file() and manifest.export_record() and not force:
        log(f"already exported: {out} (use --force to redo)")
        return out
    torch.backends.mha.set_fastpath_enabled(False)  # the fused encoder-layer path does not export
    tok = Tokenizer(manifest.tokenizer_dir)
    cfg = ModelConfig.from_dict(manifest.model_config())
    model = laya.load(str(manifest.torch_dir), device="cpu").model.eval()

    class Forward(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
            return self.m(input_ids, attention_mask, marker_pos, marker_mask, qtype)

    wrapper = Forward(model).eval()
    example = _sample_batch(tok, cfg, 256)
    args = tuple(torch.from_numpy(example[k]) for k in INPUT_NAMES)
    batch = torch.export.Dim("batch", min=1, max=64)
    seq = torch.export.Dim("seq", min=16, max=8192)
    markers = torch.export.Dim("markers", min=1, max=64)
    dynamic = {
        "input_ids": {0: batch, 1: seq},
        "attention_mask": {0: batch, 1: seq},
        "marker_pos": {0: batch, 1: markers},
        "marker_mask": {0: batch, 1: markers},
        "qtype": {0: batch},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    for stale in out.parent.glob("*"):
        stale.unlink()
    t0 = time.perf_counter()
    torch.onnx.export(
        wrapper,
        args,
        str(out),
        input_names=list(INPUT_NAMES),
        output_names=["logits", "act"],
        dynamic_shapes=dynamic,
        opset_version=manifest.onnx_opset,
        dynamo=True,
        external_data=True,
    )
    # The exporter records intermediate shapes from the example; some are wrong for
    # other lengths and trip ONNX shape inference. Drop them (graph proto only).
    graph = onnx.load(str(out), load_external_data=False)
    del graph.graph.value_info[:]
    onnx.save(graph, str(out))
    log(f"exported in {time.perf_counter() - t0:.0f}s -> {out}")

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    parity = {}
    for label, state_tokens in (("state256x3q", 256), ("state1024x3q", 1024)):
        batch_np = _sample_batch(tok, cfg, state_tokens)
        with torch.no_grad():
            ref = model(*(torch.from_numpy(batch_np[k]) for k in INPUT_NAMES))[0].numpy()
        got = session.run(None, {k: batch_np[k] for k in INPUT_NAMES})[0]
        mask = batch_np["marker_mask"]
        diff = float(np.abs(got[mask] - ref[mask]).max())
        parity[label] = {"seq": int(batch_np["input_ids"].shape[1]), "max_abs_logit_diff": diff}
        log(f"parity {label}: seq={parity[label]['seq']} max|Δlogit|={diff:.2e}")
        if diff > PARITY_TOLERANCE:
            raise RuntimeError(f"ONNX disagrees with PyTorch on {label}: {diff:.3e}")
    record = {
        "source_revision": manifest.revision,
        "files": {
            p.name: sha256_file(p) for p in sorted(out.parent.iterdir()) if p.name != EXPORT_RECORD
        },
        "opset": manifest.onnx_opset,
        "dynamic_axes": {"batch": [1, 64], "seq": [16, 8192], "markers": [1, 64]},
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "parity": parity,
        "exported_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    (out.parent / EXPORT_RECORD).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return out
