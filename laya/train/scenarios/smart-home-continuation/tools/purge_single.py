"""只对 train/data/authored/smart-home-c3/ 套用 evals/smart-home-v2/purge_training_overlap.py 的同一规则：
与锁定评测集（v1、v2、v3 开发集、验收集）相同或近似的行删掉，被删的写进 smart-home-c3/purged.jsonl。不碰别的 authored 目录。
"""
import json
import sys
from pathlib import Path

LAYA = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(LAYA / "evals/smart-home-v2"))
from check_leakage import norm  # noqa: E402
from purge_training_overlap import hit, locked_texts  # noqa: E402

locked = locked_texts()
root = LAYA / "train/data/authored/smart-home-c3"
purged = []
for f in sorted(root.glob("b*.jsonl")):
    keep = []
    for line in f.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        h = hit(norm(c["text"]), locked)
        if h:
            purged.append({"file": f.name, "text": c["text"], "locked": h})
        else:
            keep.append(line)
    f.write_text("".join(x + "\n" for x in keep), "utf-8")
(root / "purged.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in purged), "utf-8")
print(f"purged {len(purged)}", purged[:10])
