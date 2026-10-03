"""Make one fixed 50/50 parameter-mean checkpoint; never modifies its sources.

This is an offline model experiment, not an ensemble or a new runtime policy.
Temperatures are reset for the existing calibrate stage. No data is read here.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    configs = [json.loads((p / "rl_agent_config.json").read_text()) for p in (a.left, a.right)]
    for key in ("encoder", "head_layers", "max_len", "head_max_len", "max_prefixes",
                "act_costs", "cost_wrong_act", "amp_dtype", "model_name"):
        if configs[0].get(key) != configs[1].get(key):
            raise ValueError(f"incompatible configuration: {key}")
    for directory in ("encoder", "tokenizer"):
        trees = [{str(f.relative_to(p)): digest(f) for f in sorted((p / directory).rglob("*"))
                  if f.is_file()} for p in (a.left, a.right)]
        if not trees[0] or trees[0] != trees[1]:
            raise ValueError(f"incompatible {directory} files")
    left = load_file(a.left / "model.safetensors", device="cpu")
    right = load_file(a.right / "model.safetensors", device="cpu")
    if left.keys() != right.keys():
        raise ValueError("tensor keys differ")
    merged = {}
    for name, x in left.items():
        y = right[name]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"incompatible tensor: {name}")
        if x.is_floating_point():
            merged[name] = (x.float() * .5 + y.float() * .5).to(x.dtype).contiguous()
        else:
            if not torch.equal(x, y):
                raise ValueError(f"non-floating tensor differs: {name}")
            merged[name] = x.contiguous()
    provenance = {
        "experiment_type": "fixed_equal_weight_mean", "training_performed": False,
        "sources": [{"checkpoint": str(p.resolve()), "weight": .5,
                     "weights_sha256": digest(p / "model.safetensors"),
                     "config_sha256": digest(p / "rl_agent_config.json")}
                    for p in (a.left, a.right)],
        "tensor_count": len(merged), "torch_version": torch.__version__,
        "selection": "fixed 50/50 before evaluation; no new acceptance used",
    }
    a.out.mkdir(parents=True)
    save_file(merged, a.out / "model.safetensors", metadata={"format": "pt"})
    for directory in ("encoder", "tokenizer"):
        shutil.copytree(a.left / directory, a.out / directory)
    cfg = configs[0]
    cfg.pop("training", None)
    cfg.update(temperature=[1.0, 1.0, 1.0], temperature_by_options={},
               fine_tuned_from="fixed 50/50 mean of c5 and c6", weight_mean=provenance)
    (a.out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    provenance["output_weights_sha256"] = digest(a.out / "model.safetensors")
    # Existing stage guard expects this filename; its content explicitly says no new training.
    (a.out / "train_summary.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
