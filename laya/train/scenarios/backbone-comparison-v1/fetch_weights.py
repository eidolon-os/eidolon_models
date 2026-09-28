"""Download public pinned weights in bounded ranges and verify Hub LFS SHA256."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[4] / "laya/train/runs/backbone-comparison-v1"
CHUNK = 32 * 1024 * 1024


def get_part(task):
    alias, url, stage, start, end = task
    part = stage.parent / f".range-{start}-{end}"
    if part.exists() and part.stat().st_size == end - start + 1:
        return task
    for attempt in range(4):
        try:
            req = urllib.request.Request(url + f"?range_start={start}", headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=90) as response:
                assert response.status == 206
                assert response.headers["Content-Range"].startswith(f"bytes {start}-{end}/")
                with part.open("wb") as f:
                    shutil.copyfileobj(response, f, 1024 * 1024)
            assert part.stat().st_size == end - start + 1
            return task
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2)


def main():
    from huggingface_hub import HfApi
    sources = json.loads((ROOT / "sources.json").read_text())
    tasks, manifests = [], {}
    for alias, d in sources.items():
        meta = HfApi().model_info(d["repo"], revision=d["revision"], files_metadata=True, token=False)
        info = next(x for x in meta.siblings if x.rfilename == "model.safetensors")
        size, expected = info.lfs.size, info.lfs.sha256
        dest = ROOT / "models" / alias / "model.safetensors"
        if dest.exists():
            continue
        stage = dest.with_suffix(".assembling")
        prefixes = list((dest.parent / ".cache/huggingface/download").glob(f"*.{expected}.*.incomplete"))
        prefix = max(prefixes, key=lambda p: p.stat().st_size) if prefixes else None
        offset = prefix.stat().st_size if prefix else 0
        if prefix:
            shutil.copyfile(prefix, stage)
        else:
            stage.touch()
        with stage.open("r+b") as f:
            f.truncate(size)
        manifests[alias] = dict(size=size, sha256=expected, reused_prefix_bytes=offset, stage=str(stage), destination=str(dest))
        url = f"https://huggingface.co/{d['repo']}/resolve/{d['revision']}/model.safetensors"
        tasks.extend((alias, url, stage, start, min(start + CHUNK, size) - 1) for start in range(offset, size, CHUNK))
        print(alias, "reuse", offset, "of", size, flush=True)
    (ROOT / "download-manifest.json").write_text(json.dumps(manifests, indent=2))
    # Smaller model first so its evaluation can start while other weights download.
    tasks.sort(key=lambda t: manifests[t[0]]["size"])
    remaining = {a: sum(t[0] == a for t in tasks) for a in manifests}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(get_part, t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            alias, url, stage, start, end = future.result()
            part = stage.parent / f".range-{start}-{end}"
            with stage.open("r+b") as f, part.open("rb") as p:
                f.seek(start)
                shutil.copyfileobj(p, f, 1024 * 1024)
            part.unlink()
            remaining[alias] -= 1
            if remaining[alias] % 10 == 0:
                print(alias, "chunks remaining", remaining[alias], flush=True)
            if remaining[alias] == 0:
                h = hashlib.sha256()
                with stage.open("rb") as f:
                    for data in iter(lambda: f.read(8 * 1024 * 1024), b""):
                        h.update(data)
                assert h.hexdigest() == manifests[alias]["sha256"], f"hash mismatch {alias}"
                os.replace(stage, manifests[alias]["destination"])
                print("verified complete", alias, flush=True)


if __name__ == "__main__":
    main()
