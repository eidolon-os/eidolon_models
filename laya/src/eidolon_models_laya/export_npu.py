"""PyTorch checkpoint → static-shape ONNX for NPUs (RK3588 first), checked against PyTorch.

NPU compilers want fixed shapes, and a 256k-row token-embedding Gather is a poor fit for them,
so the forward is split three ways:

    CPU  token-embedding lookup (fp16 table, memory-mapped) + the question-type row
    NPU  encoder from ``inputs_embeds`` + head layers  →  hidden states ``h`` [1, L, d]
    CPU  gather ``h`` at the option markers + scorer MLP  →  logits

One ONNX per sequence bucket (inputs are right-padded up to the bucket). Everything lands in
``<model>/npu/``; a platform converter (``deploy/rk3588/laya_npu.py``) turns each ONNX into its
own format next to it. Needs the ``export`` extra.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from .artifacts import EXPORT_RECORD, Manifest, sha256_file

BUCKETS = (128, 256, 384, 512)  # smart-home: intent ≤ 107, action ≤ 147, device ≤ 502 tokens (p50 317 on v2-dev)
OPSET = 17  # what rknn-toolkit2 2.3 converts cleanly (the 2026-09-23 spike)
UNSUPPORTED_OPS = {"IsNaN"}  # rknn runtime 2.3: "Unsupport CPU op: IsNaN"
PARITY_TOLERANCE = 1e-3  # max |Δlogit| between PyTorch and ONNX Runtime + numpy scorer


def npu_dir(manifest: Manifest) -> Path:
    return manifest.root / "npu"


def bucket_for(n: int, buckets=BUCKETS) -> int:
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"sequence of {n} tokens exceeds the largest bucket {buckets[-1]}")


def _erf(x: np.ndarray) -> np.ndarray:  # Abramowitz–Stegun 7.1.26, |error| < 1.5e-7
    s = np.sign(x)
    x = np.abs(x)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def score_hidden(h: np.ndarray, markers: list[int], sc: dict[str, np.ndarray]) -> np.ndarray:
    """``h`` [L, d] → one logit per option marker (the model's ``scorer``, exact-erf GELU)."""
    m = h[np.asarray(markers)]
    mu, var = m.mean(-1, keepdims=True), m.var(-1, keepdims=True)
    x = (m - mu) / np.sqrt(var + 1e-5) * sc["ln_w"] + sc["ln_b"]
    x = x @ sc["w1"].T + sc["b1"]
    x = 0.5 * x * (1 + _erf(x / np.sqrt(2)))
    return (x @ sc["w2"].T + sc["b2"])[:, 0]


def npu_inputs(ids: list[int], qtype: int, L: int, emb: np.ndarray, type_emb: np.ndarray) -> list[np.ndarray]:
    """The three NPU graph inputs for one item, right-padded to ``L`` (pad rows are masked)."""
    n = len(ids)
    x = np.zeros((1, L, emb.shape[1]), np.float32)
    x[0, :n] = emb[np.asarray(ids)]
    mask = np.zeros((1, L), np.int64)
    mask[0, :n] = 1
    return [x, mask, type_emb[qtype][None, None, :].astype(np.float32)]


def _sample_items(tok, cfg, n: int = 6) -> list[dict]:
    """Real-shaped items for parity: a short intent-style question and a long option list."""
    from .vendor.laya.agent import Agent
    from .vendor.laya.common import QTYPES, build_sequence

    def devices(rooms):
        d = {f"{r}{t}": f"{r}·{t}" for r in rooms for t in ("灯", "空调", "窗帘", "净化器", "加湿器")}
        return d | {"多个设备或整屋": None, "没有对应的设备": None}

    qs = [  # one question per bucket: ~100, ~200 and ~450 tokens
        {"type": "choice", "instructions": "意图是什么？", "criteria": {"控制": None, "查询": None, "无关": None}},
        {"type": "choice", "instructions": "要操作哪台设备？", "criteria": devices(("客厅", "主卧", "书房"))},
        {"type": "choice", "instructions": "要操作哪台设备？",
         "criteria": devices(("客厅", "主卧", "次卧", "书房", "厨房", "阳台", "玄关", "卫生间"))},
        {"type": "noul", "instructions": "需要回应吗？"},
    ]
    utts = ["书房热得冒汗", "把客厅灯调暗一点", "邻居家的猫吵死了", "卫生间一股味儿，开下排风", "明天几点下雨", "晚安"]
    items = []
    for u in utts[:n]:
        for qd in qs:
            q = Agent._to_internal(qd)
            ids, markers = build_sequence(tok, {"utterance": u}, q, cfg["max_len"], cfg["head_max_len"])
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
    return items


def export_npu(manifest: Manifest, *, buckets=BUCKETS, force: bool = False, log=print) -> Path:
    import onnxruntime as ort
    import torch

    from .vendor import laya

    out = npu_dir(manifest)
    record_path = out / EXPORT_RECORD
    if record_path.is_file() and not force:
        log(f"already exported: {out} (use --force to redo)")
        return out
    torch.backends.mha.set_fastpath_enabled(False)  # the fused encoder-layer path does not export
    agent = laya.load(str(manifest.torch_dir), device="cpu")
    model, tok = agent.model.eval(), agent.tok
    # SDPA exports as softmax → IsNaN → Where (a guard for fully-masked rows); the RK3588 runtime has no
    # IsNaN kernel. Eager attention is the same math without the guard (no row is fully masked here).
    model.encoder.config._attn_implementation = "eager"
    cfg = {"max_len": manifest.model_config().get("max_len", 1024), "head_max_len": manifest.model_config().get("head_max_len", 512)}

    class Hidden(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, inputs_embeds, attention_mask, type_vec):
            h = self.m.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
            h = h + type_vec
            pad = attention_mask == 0
            for layer in self.m.head.layers:
                h = layer(h, src_key_padding_mask=pad)
            return h

    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*"):
        stale.unlink()
    emb = model.encoder.get_input_embeddings().weight.detach().float().numpy()
    np.save(out / "tok_emb_fp16.npy", emb.astype(np.float16))
    type_emb = model.type_emb.weight.detach().float().numpy()
    np.save(out / "type_emb.npy", type_emb)
    s = model.scorer
    sc = {"ln_w": s[0].weight, "ln_b": s[0].bias, "w1": s[1].weight, "b1": s[1].bias, "w2": s[3].weight, "b2": s[3].bias}
    sc = {k: v.detach().float().numpy() for k, v in sc.items()}
    np.savez(out / "scorer.npz", **sc)
    emb16 = np.load(out / "tok_emb_fp16.npy").astype(np.float32)  # parity uses what the device will use

    wrapper = Hidden(model).eval()
    items = _sample_items(tok, cfg)
    parity = {}
    for L in buckets:
        path = out / f"hidden_l{L}.onnx"
        example = npu_inputs([tok.cls_token_id, tok.sep_token_id], 0, L, emb16, type_emb)
        t0 = time.perf_counter()
        torch.onnx.export(
            wrapper, tuple(torch.from_numpy(a) for a in example), str(path),
            input_names=["inputs_embeds", "attention_mask", "type_vec"], output_names=["h"],
            opset_version=OPSET, dynamo=False,
        )
        log(f"exported L={L} in {time.perf_counter() - t0:.0f}s -> {path.name}")
        import onnx

        ops = {n.op_type for n in onnx.load(str(path), load_external_data=False).graph.node}
        if ops & UNSUPPORTED_OPS:
            raise RuntimeError(f"L={L}: graph has ops the NPU runtime lacks: {sorted(ops & UNSUPPORTED_OPS)}")
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        worst, n = 0.0, 0
        for it in items:
            if len(it["ids"]) > L or (L != buckets[0] and len(it["ids"]) <= buckets[buckets.index(L) - 1]):
                continue  # each bucket is checked on the items it would actually serve
            b = {
                "input_ids": torch.tensor([it["ids"]]),
                "attention_mask": torch.ones(1, len(it["ids"]), dtype=torch.long),
                "marker_pos": torch.tensor([it["markers"]]),
                "marker_mask": torch.ones(1, len(it["markers"]), dtype=torch.bool),
                "qtype": torch.tensor([it["qtype"]]),
            }
            with torch.no_grad():
                ref = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])[0][0].numpy()
            h = sess.run(None, dict(zip(["inputs_embeds", "attention_mask", "type_vec"],
                                        npu_inputs(it["ids"], it["qtype"], L, emb16, type_emb))))[0][0]
            got = score_hidden(h, it["markers"], sc)
            worst, n = max(worst, float(np.abs(got - ref).max())), n + 1
        parity[f"l{L}"] = {"items": n, "max_abs_logit_diff": worst}
        log(f"parity L={L}: {n} items, max|Δlogit|={worst:.2e} (fp16 embeddings, padded)")
        if n and worst > PARITY_TOLERANCE * 10:  # fp16 table + padding: allow 1e-2 here, the device check is stricter on decisions
            raise RuntimeError(f"L={L}: ONNX + scorer disagrees with PyTorch: {worst:.3e}")
    record = {
        "source_revision": manifest.revision,
        "buckets": list(buckets),
        "opset": OPSET,
        "split": "cpu: tok_emb_fp16 + type_emb | npu: hidden_l<L>.onnx | cpu: scorer.npz on h[markers]",
        "files": {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != EXPORT_RECORD},
        "parity": parity,
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    record_path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", "utf-8")
    return out
