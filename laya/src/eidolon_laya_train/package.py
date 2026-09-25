"""``package``: turn a training checkpoint into a ``models/<name>/<rev>/`` directory.

That layout (``manifest.json`` + ``torch/``) is what ``eidolon-laya serve``, ``export-onnx`` and
``doctor`` read, so a packaged run is served, exported and evaluated exactly like an upstream
checkpoint. ``rev`` is the sha256 of the weights (first 8 hex), so the same training output
always packages to the same directory; ``source.hub`` is ``local`` and ``fetch`` refuses it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from eidolon_models_laya.artifacts import sha256_file

FILES = ("model.safetensors", "rl_agent_config.json", "encoder/config.json")


def package(
    checkpoint: Path, models_root: Path, name: str, *, run_id: str | None = None, notes: str = ""
) -> Path:
    checkpoint = Path(checkpoint)
    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"{checkpoint} has no model.safetensors")
    rev = sha256_file(weights)[:8]
    out = models_root / name / rev
    torch_dir = out / "torch"
    if torch_dir.exists():
        shutil.rmtree(torch_dir)
    torch_dir.mkdir(parents=True)
    files: dict[str, str] = {}
    for rel in FILES:
        src = checkpoint / rel
        dst = torch_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        files[rel] = sha256_file(dst)
    for p in sorted((checkpoint / "tokenizer").rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(checkpoint))
            dst = torch_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
            files[rel] = sha256_file(dst)
    cfg = json.loads((checkpoint / "rl_agent_config.json").read_text(encoding="utf-8"))
    summary = {}
    if (checkpoint / "train_summary.json").is_file():
        summary = json.loads((checkpoint / "train_summary.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "name": name,
        "source": {
            "hub": "local",
            "repo_id": f"eidolon-laya-train/{name}",
            "subfolder": "",
            "revision": rev,
            "run_id": run_id,
            "checkpoint": str(checkpoint.resolve()),
        },
        "license": {
            "checkpoint": "internal",
            "base": cfg.get("fine_tuned_from", cfg.get("encoder")),
        },
        "runtime": {
            "code": "laya (vendored, see src/eidolon_models_laya/vendor/laya/VENDOR_VERSION)",
            "trained_max_len": cfg.get("max_len"),
            "head_max_len": cfg.get("head_max_len"),
            "notes": notes
            or f"fine-tuned from {cfg.get('fine_tuned_from')}; best epoch {summary.get('best_epoch')}",
        },
        "torch": {"dir": "torch", "files": files},
        "onnx": {
            "dir": "onnx",
            "file": "model.onnx",
            "opset": 18,
            "produced_by": "eidolon-laya export-onnx",
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out
