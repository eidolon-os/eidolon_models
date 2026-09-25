"""``train``: fine-tune a laya DecisionModel on ``dataset/train.jsonl``.

Loss per question = soft cross-entropy against the target distribution
+ ``brier_weight`` · Brier (Von's recipe: the Brier term keeps probabilities honest)
+ ``proper_weight`` · (−laya's strictly proper scoring reward: log + spherical + RPS for score)
+ ``pg_weight`` · laya's noisy-logit policy gradient (the official notebook's RLCD term;
  ``pg_weight: 1, brier_weight: 0`` reproduces the notebook, σ annealed pg_sigma_start → pg_sigma_end).
Choice and noul options are shuffled every epoch (score levels keep their order); the last
``unfreeze_layers`` encoder layers and the head train, the rest stays frozen (``unfreeze_layers: -1``
trains the whole encoder, as the notebook does). The act head is never trained (it is unused).

Config (yaml)::

    init: models/laya-multilingual/1c5edc17/torch   # checkpoint dir, or encoder: jhu-clsp/mmBERT-base
    epochs: 3
    batch_size: 16
    lr_encoder: 2.5e-5
    lr_head: 1.0e-4
    unfreeze_layers: 8
    brier_weight: 0.5
    proper_weight: 0.0
    warmup: 0.06
    weight_decay: 0.01
    max_len: 1024
    head_max_len: 256
    seed: 7
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from eidolon_models_laya.vendor.laya.common import proper_reward

from .model import (
    Loaded,
    batches,
    load_checkpoint,
    load_encoder,
    record_items,
    save_checkpoint,
    to_device,
)
from .records import read_jsonl


def question_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    qtype: torch.Tensor,
    *,
    brier_weight: float,
    proper_weight: float,
    pg_weight: float = 0.0,
    pg_group: int = 4,
    pg_sigma: float = 0.4,
    pg_w_sph: float = 0.75,
) -> tuple[torch.Tensor, dict]:
    """soft CE + brier·Brier − proper·(proper score) + pg·(laya's noisy-logit policy gradient).

    The policy-gradient term is the official fine-tuning notebook's RLCD step: sample ``pg_group``
    zero-mean Gaussian perturbations of the logits, reward each with the strictly proper score
    against the target, and push the logits toward the better-rewarded samples (GRPO-style
    group baseline). With ``pg_weight=1`` and ``brier_weight=0`` this is exactly the notebook.
    """
    logits = logits.masked_fill(~mask, -1e4)
    logp = F.log_softmax(logits, -1)
    p = logp.exp()
    ce = -(target * logp).sum(-1)
    brier = (((p - target) ** 2) * mask).sum(-1)
    loss = ce + brier_weight * brier
    parts = {"ce": ce.mean().item(), "brier": brier.mean().item()}
    if proper_weight > 0:
        r = proper_reward(p, target, qtype, mask.float())
        loss = loss - proper_weight * r
        parts["proper"] = r.mean().item()
    if pg_weight > 0:
        k = mask.sum(-1, keepdim=True).float()
        eps = torch.randn((pg_group,) + logits.shape, device=logits.device) * pg_sigma * mask
        eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
        z = logits.detach().unsqueeze(0) + eps
        q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
        with torch.no_grad():
            r = proper_reward(
                q, target.unsqueeze(0), qtype, mask.float(), w_sph=pg_w_sph, w_rps=1.0
            )
            adv = r - r.mean(0, keepdim=True)
            adv = adv / (adv.std() + 1e-6)
        logp_z = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * pg_sigma**2)
        loss_rl = -(adv * logp_z).mean(0)
        loss = loss + pg_weight * loss_rl
        parts["pg_reward"] = r.mean().item()
    return loss.mean(), parts


def set_trainable(model, unfreeze_layers: int) -> tuple[list, list]:
    for p in model.parameters():
        p.requires_grad_(False)
    enc_params = []
    layers = list(model.encoder.layers)
    if unfreeze_layers < 0:
        unfreeze_layers = len(layers)
    for layer in layers[len(layers) - unfreeze_layers :] if unfreeze_layers > 0 else []:
        for p in layer.parameters():
            p.requires_grad_(True)
            enc_params.append(p)
    final_norm = getattr(model.encoder, "final_norm", None)
    if final_norm is not None and unfreeze_layers > 0:
        for p in final_norm.parameters():
            p.requires_grad_(True)
            enc_params.append(p)
    head_params = []
    for mod in (model.head, model.type_emb, model.scorer):
        if mod is None:
            continue
        for p in mod.parameters():
            p.requires_grad_(True)
            head_params.append(p)
    return enc_params, head_params


def evaluate_split(loaded: Loaded, items: list[dict], batch_size: int) -> dict:
    loaded.model.eval()
    nll, n, correct = 0.0, 0, 0
    by_type = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for chunk, batch in batches(items, batch_size, loaded.tok.pad_token_id):
            b = to_device(batch, loaded.device)
            logits, _ = loaded.model(
                b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"]
            )
            logits = logits.float().masked_fill(~b["marker_mask"], -1e4)
            logp = F.log_softmax(logits, -1)
            nll += -(b["target"] * logp).sum(-1).sum().item()
            pred = logits.argmax(-1)
            for j, it in enumerate(chunk):
                ok = it["target"][pred[j].item()] == max(it["target"])
                correct += ok
                by_type[it["qtype"]][0] += ok
                by_type[it["qtype"]][1] += 1
            n += len(chunk)
    return {
        "nll": nll / max(n, 1),
        "acc": correct / max(n, 1),
        "n": n,
        "acc_by_qtype": {str(k): v[0] / v[1] for k, v in by_type.items()},
    }


def train(config: dict, dataset_dir: Path, out_dir: Path, log=print) -> dict:
    seed = int(config.get("seed", 7))
    torch.manual_seed(seed)
    rng = random.Random(seed)
    if config.get("init"):
        loaded = load_checkpoint(config["init"], config.get("device"))
        init_name = str(config["init"])
    elif config.get("encoder"):
        loaded = load_encoder(
            config["encoder"], config.get("device"), int(config.get("head_layers", 2))
        )
        init_name = config["encoder"]
    else:
        raise ValueError("train config needs `init` (checkpoint dir) or `encoder` (HF id)")
    cfg = loaded.cfg
    cfg["max_len"] = int(config.get("max_len", cfg.get("max_len", 1024)))
    cfg["head_max_len"] = int(config.get("head_max_len", cfg.get("head_max_len", 256)))
    model, tok, device = loaded.model, loaded.tok, loaded.device
    log(
        f"init {init_name} on {device}; max_len {cfg['max_len']} head_max_len {cfg['head_max_len']}"
    )

    train_records = list(read_jsonl(dataset_dir / "train.jsonl"))
    val_path = dataset_dir / "val.jsonl"
    val_records = list(read_jsonl(val_path)) if val_path.exists() else []
    val_items = [it for r in val_records for it in record_items(r, tok, cfg)]
    log(f"train records {len(train_records)}, val items {len(val_items)}")

    enc_params, head_params = set_trainable(model, int(config.get("unfreeze_layers", 8)))
    wd = float(config.get("weight_decay", 0.01))
    opt = torch.optim.AdamW(
        [
            {
                "params": enc_params,
                "lr": float(config.get("lr_encoder", 2.5e-5)),
                "weight_decay": wd,
            },
            {"params": head_params, "lr": float(config.get("lr_head", 1e-4)), "weight_decay": wd},
        ]
    )
    epochs = int(config.get("epochs", 3))
    batch_size = int(config.get("batch_size", 16))
    brier_w = float(config.get("brier_weight", 0.5))
    proper_w = float(config.get("proper_weight", 0.0))
    pg_w = float(config.get("pg_weight", 0.0))
    pg_group = int(config.get("pg_group", 4))
    sigma_start, sigma_end = (
        float(config.get("pg_sigma_start", 0.4)),
        float(config.get("pg_sigma_end", 0.1)),
    )
    accum = int(config.get("grad_accum", 1))
    # items per epoch is stable (one per labelled question), so the schedule is known up front
    n_items = sum(len(record_items(r, tok, cfg)) for r in train_records)
    steps_per_epoch = math.ceil(n_items / batch_size / accum)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(float(config.get("warmup", 0.06)) * total_steps)

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    amp = device.type == "cuda"
    history, best, best_epoch = [], None, -1
    t0 = time.time()
    step = 0
    for epoch in range(epochs):
        model.train()
        items = [it for r in train_records for it in record_items(r, tok, cfg, shuffle=rng)]
        rng.shuffle(items)
        items.sort(key=lambda it: len(it["ids"]) // 64)  # coarse length buckets, then batch
        groups = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
        rng.shuffle(groups)
        running, parts_sum, nb = 0.0, defaultdict(float), 0
        sigma = sigma_start + (sigma_end - sigma_start) * (
            epoch / max(1, epochs - 1)
        )  # notebook's schedule
        opt.zero_grad(set_to_none=True)
        for gi, chunk in enumerate(groups):
            _, batch = next(batches(chunk, len(chunk), tok.pad_token_id))
            b = to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                logits, _ = model(
                    b["input_ids"],
                    b["attention_mask"],
                    b["marker_pos"],
                    b["marker_mask"],
                    b["qtype"],
                )
            loss, parts = question_loss(
                logits.float(),
                b["target"],
                b["marker_mask"],
                b["qtype"],
                brier_weight=brier_w,
                proper_weight=proper_w,
                pg_weight=pg_w,
                pg_group=pg_group,
                pg_sigma=sigma,
            )
            (loss / accum).backward()
            running += loss.item()
            for k, v in parts.items():
                parts_sum[k] += v
            nb += 1
            if (gi + 1) % accum == 0 or gi + 1 == len(groups):
                torch.nn.utils.clip_grad_norm_(enc_params + head_params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
        rec = {
            "epoch": epoch + 1,
            "loss": running / max(nb, 1),
            **{k: v / max(nb, 1) for k, v in parts_sum.items()},
        }
        if val_items:
            rec["val"] = evaluate_split(loaded, val_items, batch_size)
            score = rec["val"]["nll"]
        else:
            score = rec["loss"]
        rec["elapsed_s"] = round(time.time() - t0, 1)
        history.append(rec)
        log(json.dumps(rec, ensure_ascii=False))
        if best is None or score < best:
            best, best_epoch = score, epoch + 1
            save_checkpoint(
                loaded,
                out_dir,
                {
                    "temperature": [1.0, 1.0, 1.0],
                    "temperature_by_options": {},
                    "model_name": config.get("model_name", out_dir.name),
                    "fine_tuned": True,
                    "fine_tuned_from": init_name,
                    "training": {
                        "updates": step,
                        "epochs_completed": epoch + 1,
                        "hours": round((time.time() - t0) / 3600, 3),
                        "config": {k: v for k, v in config.items() if k != "device"},
                    },
                },
            )
    summary = {
        "best_epoch": best_epoch,
        "best_score": best,
        "history": history,
        "items_per_epoch": n_items,
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
