"""Loading, batching and saving a laya ``DecisionModel`` — shared by train, calibrate, eval.

A checkpoint directory is exactly what ``eidolon-laya serve`` (via ``laya.load``) reads:
``model.safetensors`` + ``rl_agent_config.json`` + ``encoder/config.json`` + ``tokenizer/``.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from eidolon_models_laya.vendor.laya.agent import Agent
from eidolon_models_laya.vendor.laya.common import (
    QTYPES,
    DecisionModel,
    build_model,
    build_sequence,
    collate_items,
)

from .records import Record, option_names, target_vector

DEFAULT_CFG = {
    "head_layers": 2,
    "max_len": 1024,
    "head_max_len": 256,
    "max_prefixes": 6,
    "act_costs": {"escalate": 0.5},
    "cost_wrong_act": 3.0,
    "amp_dtype": "bf16",
    "temperature": [1.0, 1.0, 1.0],
    "temperature_by_options": {},
}


@dataclass
class Loaded:
    model: DecisionModel
    tok: Any
    cfg: dict
    device: torch.device


def pick_device(name: str | None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: str | Path, device: str | None = None) -> Loaded:
    """A laya checkpoint directory (ours or upstream)."""
    dev = pick_device(device)
    agent = Agent(str(path), device=str(dev))
    return Loaded(agent.model, agent.tok, dict(agent.cfg), agent.device)


def load_encoder(encoder_id: str, device: str | None = None, head_layers: int = 2) -> Loaded:
    """A fresh decision head on a pretrained encoder (mmBERT, Zhinao-ChineseModernBert, ...)."""
    from transformers import AutoTokenizer

    dev = pick_device(device)
    cfg = dict(
        DEFAULT_CFG, encoder=encoder_id, head_layers=head_layers, model_name=Path(encoder_id).name
    )
    model = build_model(cfg, pretrained=True).to(dev)
    tok = AutoTokenizer.from_pretrained(encoder_id)
    return Loaded(model, tok, cfg, dev)


def save_checkpoint(loaded: Loaded, out_dir: str | Path, cfg_updates: dict | None = None) -> Path:
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in loaded.model.state_dict().items()}
    save_file(state, str(out / "model.safetensors"))
    loaded.model.encoder.config.save_pretrained(out / "encoder")
    tok_dir = out / "tokenizer"
    if tok_dir.exists():
        shutil.rmtree(tok_dir)
    loaded.tok.save_pretrained(tok_dir)
    cfg = dict(loaded.cfg)
    cfg.update(cfg_updates or {})
    (out / "rl_agent_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out


# ------------------------------------------------------------------ records → items


def to_internal(q: dict) -> dict:
    return Agent._to_internal(q)


def record_items(
    r: Record,
    tok,
    cfg: dict,
    *,
    shuffle: random.Random | None = None,
    require_label: bool = True,
    truncate_left: bool = False,
) -> list[dict]:
    """One item per question. With ``shuffle`` the choice options are permuted (and the
    target with them) so the model cannot learn positions."""
    items = []
    for qid, q in r.questions.items():
        lab = r.labels.get(qid)
        if require_label and not lab:
            continue
        names = option_names(q)
        order = list(range(len(names)))
        if shuffle is not None and q["type"] != "score" and len(order) > 1:  # laya trains this way
            shuffle.shuffle(order)
        ids, markers = build_sequence(
            tok,
            r.state,
            to_internal(q),
            cfg.get("max_len", 1024),
            cfg.get("head_max_len", 256),
            option_order=order,
            truncate_left=truncate_left,
        )
        if len(markers) != len(order):
            continue  # options fell off the end of the window: no honest target for them
        item = {
            "ids": ids,
            "markers": markers,
            "qtype": QTYPES[q["type"]],
            "record_id": r.id,
            "qid": qid,
            "order": order,
            "names": [names[i] for i in order],
        }
        if lab:
            tv = target_vector(q, lab)
            item["target"] = [tv[i] for i in order]
            item["label"] = max(range(len(order)), key=lambda j: item["target"][j])
        items.append(item)
    return items


def batches(items: list[dict], batch_size: int, pad_id: int):
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        yield chunk, collate_items([[it] for it in chunk], pad_id)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def score_items(loaded: Loaded, items: list[dict], batch_size: int = 16) -> list[dict]:
    """Raw logits per item (temperature not applied), in item order."""
    loaded.model.eval()
    out = []
    for chunk, batch in batches(items, batch_size, loaded.tok.pad_token_id):
        b = to_device(batch, loaded.device)
        logits, _ = loaded.model(
            b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"]
        )
        logits = logits.float().cpu()
        for j, it in enumerate(chunk):
            k = len(it["markers"])
            out.append(dict(it, logits=logits[j, :k].tolist()))
    return out
