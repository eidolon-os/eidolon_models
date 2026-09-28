"""Frozen DEV-only, local four-backbone comparison. Never loads sealed/test files."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "laya/train/runs/backbone-comparison-v1"
DATA = {
    "v7": ROOT / "laya/train/data/generated/ip-team-v7/generalization1/named/val.jsonl",
    "v8": ROOT / "laya/train/data/generated/ip-team-v8/boundaries1/named/val.jsonl",
}
CHECKPOINTS = {
    "laya": ROOT / "laya/models/laya-multilingual/1c5edc17/torch",
    "laya-r6": ROOT / "laya/train/runs/ip-ensemble-v7-named-r6/checkpoint",
    "laya-r7": ROOT / "laya/train/runs/ip-ensemble-v8-boundary-r7/checkpoint",
}


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def records():
    return [(split, json.loads(line)) for split, p in DATA.items()
            for line in p.read_text().splitlines()]


def payload(r):
    q = r["questions"]["move"]
    names = list(q["criteria"])
    return json.dumps(r["state"], ensure_ascii=False), q, names, [
        f"{n}: {q['criteria'][n]}" for n in names]


def paths():
    sys.path[:0] = [str(OUT / "runtime-deps"), str(OUT / "models/openjev"), str(OUT / "models/decider"),
                    str(OUT / "jevk5-source")]


def preflight():
    paths()
    from transformers import AutoTokenizer
    from typed_decisions.encoder import Collator
    from typed_decisions.open_jev import OpenJev
    from decider.prompt import build, chat_template, resolve_layout
    from decider.infer import Example, Q, _NoShuffle
    from jevk5.prompt import messages
    from eidolon_models_laya.sequence import Tokenizer
    from eidolon_models_laya.participation_text import check_capacity
    toks = {a: AutoTokenizer.from_pretrained(OUT / "models" / a)
            for a in ("openjev", "decider", "jevk5")}
    lt = Tokenizer(CHECKPOINTS["laya"] / "tokenizer")
    cfg = json.loads((OUT / "models/decider/decider_config.json").read_text())
    chat = chat_template(toks["decider"]) if resolve_layout(cfg) == "chat" else None
    collator = Collator(toks["openjev"], max_state_tokens=256, max_len=512)
    rows = []
    for split, r in records():
        state, q, names, opts = payload(r)
        row = {"split": split, "id": r["id"], "models": {}}
        for a, tok in toks.items():
            ids = tok.encode(state, add_special_tokens=False)
            row["models"][a] = {"state_tokens": len(ids), "state_unknown_tokens":
                                sum(x == tok.unk_token_id for x in ids), "full_information": True}
        oi = row["models"]["openjev"]
        oi["full_information"] = oi["state_tokens"] <= 256
        try:
            oi["input_tokens"] = len(collator.encode_one(state, [OpenJev._question(
                0, dict(type="choice", instructions=q["instructions"], options=opts))])[0])
            oi["runnable"] = True
        except ValueError as e:
            oi.update(full_information=False, runnable=False, error=str(e))
        it = build(Example(state, [Q(q["instructions"], opts)]), toks["decider"],
                   _NoShuffle(), max_ctx_tokens=8192, chat=chat)
        row["models"]["decider"]["input_tokens"] = len(it["ids"])
        assert len(toks["decider"].encode("Context:\n" + state, add_special_tokens=False)) < 8192
        ids = toks["jevk5"].apply_chat_template(messages(state, q["instructions"], opts),
                    tokenize=True, add_generation_prompt=True, enable_thinking=False)
        row["models"]["jevk5"]["input_tokens"] = len(ids)
        assert len(ids) < 16384
        row["models"]["laya"] = {"full_information": True, "input_tokens":
            check_capacity(lt, r["state"], q, max_len=2048, head_max_len=256)}
        for model_cap in row["models"].values():
            model_cap["untruncated"] = model_cap["full_information"]
            model_cap["full_information"] = model_cap["untruncated"] and model_cap.get("state_unknown_tokens", 0) == 0
        row["common_untruncated"] = all(x["untruncated"] for x in row["models"].values())
        row["common_full_information"] = all(x["full_information"] for x in row["models"].values())
        rows.append(row)
    manifest = {"data_sha256": {s: digest(p) for s, p in DATA.items()}, "rows": rows,
                "script_sha256": digest(__file__)}
    dest = OUT / "preflight-v2.json"
    if dest.exists():
        assert json.loads(dest.read_text())["rows"] == rows
    else:
        dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"rows": len(rows), "common": sum(r["common_full_information"] for r in rows),
        "common_untruncated": sum(r["common_untruncated"] for r in rows),
        "by_split": {s: sum(r["common_full_information"] for r in rows if r["split"] == s) for s in DATA}}))


def run(alias, limit):
    paths()
    import torch
    assert torch.backends.mps.is_available(), "MPS unavailable; run with hardware access"
    torch.manual_seed(71)
    torch.set_num_threads(4)
    pre = json.loads((OUT / "preflight-v2.json").read_text())
    assert pre["data_sha256"] == {s: digest(p) for s, p in DATA.items()}
    caps = {(r["split"], r["id"]): r for r in pre["rows"]}
    dest = OUT / (f"{alias}.smoke.jsonl" if limit else f"{alias}.jsonl")
    assert not dest.exists(), f"refusing to overwrite {dest}"
    start = time.perf_counter()
    backend = "official"
    if alias in CHECKPOINTS:
        from eidolon_laya_train.model import load_checkpoint
        from eidolon_laya_train.records import Record
        from eidolon_laya_train.evaluate import score_records
        m = load_checkpoint(CHECKPOINTS[alias], "mps")
        m.cfg.update(max_len=2048, head_max_len=256, temperature=[1., 1., 1.], temperature_by_options={})
        def predict(r):
            return list(score_records(m, [Record.from_dict(r)], batch_size=1)[0]["probabilities"].values())
        dtype = "float32"
    elif alias == "openjev":
        from typed_decisions.open_jev import OpenJev
        m = OpenJev.from_pretrained(str(OUT / "models/openjev"), device="mps")
        def predict(r):
            state, q, names, opts = payload(r)
            return list(m.decide(state, [dict(type="choice", instructions=q["instructions"], options=opts)])[0]["probabilities"].values())
        dtype = "float32"
    elif alias == "decider":
        from decider.infer import Decider
        m = Decider(str(OUT / "models/decider"), device="mps", dtype=torch.float16, use_graphs=False)
        def predict(r):
            state, q, names, opts = payload(r)
            return m.decide(state, [dict(question=q["instructions"], options=opts)], max_ctx_tokens=8192)[0]["probs_list"]
        dtype = "float16"
        backend = "official Decider MPS patch, eager"
    elif alias == "jevk5":
        from jevk5.runtime import JevK5
        # Direct safetensors -> MPS device_map segfaulted before the first forward.
        # Keep the official scorer; stage loading on CPU, then transfer once.
        m = JevK5(str(OUT / "models/jevk5"), device="cpu", dtype=torch.float16, graphs=False)
        m.model.to("mps")
        m.device = "mps"
        m.slot_weight = m.model.lm_head.weight[m.slots].detach().contiguous()
        def predict(r):
            state, q, names, opts = payload(r)
            return list(m.probabilities(state, q)[0].values())
        dtype = "float16"
        backend = "official JevK5 eager, CPU-staged loading to MPS, Transformers PyTorch attention fallback"
    else:
        raise ValueError(alias)
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - start
    metadata = {"alias": alias, "device": "mps", "dtype": dtype, "backend": backend,
                "torch": torch.__version__, "load_seconds": load_seconds, "script_sha256": digest(__file__)}
    (dest.with_suffix(".meta.json")).write_text(json.dumps(metadata, indent=2))
    rs = records()[:limit] if limit else records()
    with dest.open("x") as f:
        for index, (split, r) in enumerate(rs):
            c = caps[split, r["id"]]
            cap = c["models"]["laya" if alias in CHECKPOINTS else alias]
            row = {"split": split, "id": r["id"], "family": r["meta"]["family"],
                   "slice": r["meta"]["slice"], "gold": r["labels"]["move"]["gold"],
                   "capacity": cap, "common_full_information": c["common_full_information"]}
            if cap.get("runnable", True):
                torch.mps.synchronize()
                t = time.perf_counter()
                with torch.inference_mode():
                    probs = predict(r)
                torch.mps.synchronize()
                ms = (time.perf_counter() - t) * 1000
                names = list(r["questions"]["move"]["criteria"])
                assert len(names) == len(probs) and all(math.isfinite(p) and p >= 0 for p in probs)
                assert abs(sum(probs) - 1) < .002
                pred = names[max(range(len(probs)), key=probs.__getitem__)]
                row.update(pred=pred, probabilities=dict(zip(names, probs)), p_top=max(probs),
                           correct=pred in row["gold"], elapsed_ms=ms, first_call=index == 0)
            else:
                row["error"] = cap["error"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if index % 10 == 0 or index == len(rs) - 1:
                print(json.dumps({"model": alias, "done": index + 1, "of": len(rs),
                                  "latest_ms": row.get("elapsed_ms")}), flush=True)
    print("completed", alias, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["preflight", "laya", "laya-r6", "laya-r7", "openjev", "decider", "jevk5"])
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    preflight() if a.model == "preflight" else run(a.model, a.limit)
