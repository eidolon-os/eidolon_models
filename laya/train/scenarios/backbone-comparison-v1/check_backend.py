"""Two fixed DEV inputs: compare native CPU fp32 against the MPS run."""
import argparse
import json
import time

import compare as c


def main(alias):
    c.paths()
    import torch
    torch.set_num_threads(4)
    if alias == "decider":
        from decider.infer import Decider
        m = Decider(str(c.OUT / "models/decider"), device="cpu", dtype=torch.float32, use_graphs=False)
        def predict(r):
            state, q, _, opts = c.payload(r)
            return m.decide(state, [dict(question=q["instructions"], options=opts)], max_ctx_tokens=8192)[0]["probs_list"]
    else:
        from jevk5.runtime import JevK5
        m = JevK5(str(c.OUT / "models/jevk5"), device="cpu", dtype=torch.float32, graphs=False)
        def predict(r):
            state, q, _, _ = c.payload(r)
            return list(m.probabilities(state, q)[0].values())
    previous = {(r["split"], r["id"]): r for r in [json.loads(line) for line in (c.OUT / f"{alias}.jsonl").read_text().splitlines()]}
    rows = []
    for index in (0, 42):
        split, r = c.records()[index]
        t = time.perf_counter()
        with torch.inference_mode():
            probs = predict(r)
        old = previous[split, r["id"]]
        names = list(r["questions"]["move"]["criteria"])
        pred = names[max(range(len(probs)), key=probs.__getitem__)]
        row = dict(split=split, id=r["id"], cpu_probabilities=dict(zip(names, probs)),
                   mps_probabilities=old["probabilities"], cpu_pred=pred, mps_pred=old["pred"],
                   same_answer=pred == old["pred"], max_probability_difference=max(abs(x-y) for x,y in zip(probs,old["probabilities"].values())),
                   seconds=time.perf_counter()-t)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (c.OUT / f"{alias}.backend-check.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("alias", choices=["decider", "jevk5"])
    main(p.parse_args().alias)
