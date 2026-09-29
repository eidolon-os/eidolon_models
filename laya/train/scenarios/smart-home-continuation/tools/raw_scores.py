"""Score continuation splits with raw logits (T=1) vs the fitted T, per family."""
import json, math, sys
from collections import defaultdict
from pathlib import Path

from eidolon_laya_train.model import load_checkpoint, record_items, score_items
from eidolon_laya_train.records import read_jsonl

RUN = Path("/Users/manson/ai/eidolon/eidolon_models/laya/train/runs/home-continuation-20260929")
LOCKED = Path("/Users/manson/ai/eidolon/eidolon_models/laya/train/scenarios/smart-home-continuation/eval/locked.jsonl")
ckpt = sys.argv[1] if len(sys.argv) > 1 else str(RUN / "checkpoint")
loaded = load_checkpoint(ckpt, "cpu")
T_fit = loaded.cfg.get("temperature", [1.0])[0]


def sm(z, T):
    z = [x / T for x in z]; m = max(z); e = [math.exp(x - m) for x in z]; s = sum(e)
    return [x / s for x in e]


def run(name, path):
    recs = {r.id: r for r in read_jsonl(path)}
    items = [it for r in recs.values() for it in record_items(r, loaded.tok, loaded.cfg)]
    scored = score_items(loaded, items)
    out = {}
    for T in (1.0, T_fit):
        n = ok = 0; nll = 0.0; conf = 0.0; cov9 = cov9_ok = 0
        fam = defaultdict(lambda: [0, 0, 0.0])
        for it in scored:
            p = sm(it["logits"], T); top = max(range(len(p)), key=p.__getitem__)
            good = it["target"][top] > 0
            n += 1; ok += good; conf += p[top]
            nll += -math.log(max(sum(t * q for t, q in zip(it["target"], p)), 1e-12))
            if p[top] >= 0.9:
                cov9 += 1; cov9_ok += good
            r = recs[it["record_id"]]
            f = r.meta.get("derived_from", "") or ",".join(r.tags)
            fam[f][0] += 1; fam[f][1] += good; fam[f][2] += p[top]
        out[T] = dict(n=n, acc=round(ok / n, 3), mean_conf=round(conf / n, 3), nll=round(nll / n, 3),
                      cov_0_9=cov9, acc_0_9=(round(cov9_ok / cov9, 3) if cov9 else None),
                      fam={k: f"{v[1]}/{v[0]} conf={v[2]/v[0]:.2f}" for k, v in sorted(fam.items())})
    print(f"== {name}  (T_fit={T_fit})")
    for T, d in out.items():
        fam = d.pop("fam")
        print(f"  T={T}: {d}")
        if T == 1.0:
            for k, v in fam.items():
                print(f"      {k:40} {v}")


run("calib", RUN / "dataset/calib.jsonl")
run("val", RUN / "dataset/val.jsonl")
run("locked", LOCKED)
