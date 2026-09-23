"""Model files on disk: the manifest, fetching the pinned PyTorch checkpoint, checksums.

Layout of one model version directory::

    models/laya-multilingual/<rev>/
    ├── manifest.json        committed: source, revision, sha256 of every PyTorch file
    ├── torch/               fetched (gitignored): model.safetensors, tokenizer/, encoder/, ...
    └── onnx/                exported (gitignored): model.onnx + export.json
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

EXPORT_RECORD = "export.json"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Manifest:
    root: Path
    raw: dict

    @classmethod
    def load(cls, model_dir: Path) -> Manifest:
        path = model_dir / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"no manifest.json in {model_dir}")
        return cls(root=model_dir, raw=json.loads(path.read_text(encoding="utf-8")))

    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def repo_id(self) -> str:
        return self.raw["source"]["repo_id"]

    @property
    def subfolder(self) -> str:
        return self.raw["source"]["subfolder"]

    @property
    def revision(self) -> str:
        return self.raw["source"]["revision"]

    @property
    def torch_dir(self) -> Path:
        return self.root / self.raw["torch"]["dir"]

    @property
    def torch_files(self) -> dict[str, str]:
        return dict(self.raw["torch"]["files"])

    @property
    def tokenizer_dir(self) -> Path:
        return self.torch_dir / "tokenizer"

    @property
    def onnx_dir(self) -> Path:
        return self.root / self.raw["onnx"]["dir"]

    @property
    def onnx_path(self) -> Path:
        return self.onnx_dir / self.raw["onnx"]["file"]

    @property
    def onnx_opset(self) -> int:
        return int(self.raw["onnx"]["opset"])

    def model_config(self) -> dict:
        return json.loads((self.torch_dir / "rl_agent_config.json").read_text(encoding="utf-8"))

    def export_record(self) -> dict | None:
        path = self.onnx_dir / EXPORT_RECORD
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


def verify_torch(manifest: Manifest, *, checksums: bool = True) -> list[str]:
    """Problems with the PyTorch files; empty means complete and intact."""
    problems = []
    for rel, expected in manifest.torch_files.items():
        path = manifest.torch_dir / rel
        if not path.is_file():
            problems.append(f"missing {path}")
        elif checksums and sha256_file(path) != expected:
            problems.append(f"sha256 mismatch {path}")
    return problems


def verify_onnx(manifest: Manifest, *, checksums: bool = True) -> list[str]:
    path = manifest.onnx_path
    if not path.is_file():
        return [f"missing {path} (run: eidolon-laya export-onnx)"]
    record = manifest.export_record()
    if record is None:
        return [f"missing {manifest.onnx_dir / EXPORT_RECORD}; re-run export-onnx"]
    problems = []
    if record.get("source_revision") != manifest.revision:
        problems.append(
            f"{path} was exported from revision {record.get('source_revision')}, "
            f"manifest pins {manifest.revision}"
        )
    for name, expected in record.get("files", {}).items():
        file = manifest.onnx_dir / name
        if not file.is_file():
            problems.append(f"missing {file}")
        elif checksums and sha256_file(file) != expected:
            problems.append(f"sha256 mismatch {file}")
    return problems


def fetch_torch(
    manifest: Manifest, *, endpoint: str | None = None, use_xet: bool | None = None, log=print
) -> Path:
    """Download the pinned revision and copy it into ``torch/``, checksum-verified.

    Downloads land in the ordinary Hugging Face cache first, so a machine that
    already has the revision does not download it again.

    Behind a mirror (e.g. hf-mirror.com) Xet storage still dials Hugging Face's
    own CAS server directly and fails with 401, so Xet is off unless the
    endpoint is huggingface.co itself.
    """
    if not os.environ.get("HF_ENDPOINT", "").strip():
        os.environ.pop("HF_ENDPOINT", None)  # huggingface_hub would take "" as the endpoint
    endpoint = endpoint or os.environ.get("HF_ENDPOINT") or None
    if use_xet is None:
        use_xet = endpoint is None or "huggingface.co" in endpoint
    if not use_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    from huggingface_hub import snapshot_download  # after the env is set: read at import

    if not verify_torch(manifest):
        log(f"already present and verified: {manifest.torch_dir}")
        return manifest.torch_dir
    log(
        f"fetching {manifest.repo_id}@{manifest.revision[:8]}/{manifest.subfolder} "
        f"via {endpoint or 'https://huggingface.co'} (xet={'on' if use_xet else 'off'})"
    )
    snapshot = (
        Path(
            snapshot_download(
                manifest.repo_id,
                revision=manifest.revision,
                allow_patterns=[f"{manifest.subfolder}/{rel}" for rel in manifest.torch_files],
            )
        )
        / manifest.subfolder
    )
    for rel, expected in manifest.torch_files.items():
        src, dst = snapshot / rel, manifest.torch_dir / rel
        if dst.is_file() and sha256_file(dst) == expected:
            continue
        actual = sha256_file(src)
        if actual != expected:
            raise RuntimeError(f"{rel}: sha256 {actual} does not match manifest {expected}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        shutil.copyfile(src, part)  # the cache holds symlinks into blobs; copy the bytes
        part.replace(dst)
        log(f"  {rel} ok")
    return manifest.torch_dir
