"""Two frozen policy paraphrases, DEV only; no fitting or new gold labels."""
import argparse
import copy
import importlib.util
import json
import math
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = importlib.util.spec_from_file_location("base_comparison", HERE.parent / "backbone-comparison-v1/compare.py")
c = importlib.util.module_from_spec(s)
s.loader.exec_module(c)
OUT = c.ROOT / "laya/train/runs/direction-check-v1"
POLICIES = {
    "A": (
        "补充决策规则：用户明确暂停或结束时，优先选择wait或finish。"
        "用户要求某个特定角色回应时，必须用称呼、身份特征和公开历史判断指向。"
        "若多个候选都符合该指代，且现有信息不能唯一确定目标，选择abstain；"
        "不得用候选排列顺序、性格偏好或自行猜测替用户确定身份。"
        "同名本身不是弃权理由：若附带特征或上下文已唯一确定角色，就由该角色回应。"
        "如果用户只是开放聊天、没有限定某个特定人，可以选择合适角色，不因候选多而弃权。"
    ),
    "B": (
        "执行以下参与约定。暂停请求用wait，本轮结束用finish，优先处理这两类意图。"
        "在用户指定某个人发言时，先根据名字、人物资料与已有对话确定是谁；"
        "有身份线索足以区分，即使重名也应让被指向的人回应。"
        "如果仍有两人或更多人同样符合这个指代，没有证据能区分，就应选abstain。"
        "不要因谁排在前面、谁更讨喜或凭空推断而替用户挑一个。"
        "对于未指定特定人物的普通开放交谈，允许挑选合适的回应者，无需仅因多人可选而弃权。"
    ),
}


def payload(record, variant):
    r = copy.deepcopy(record)
    r["questions"]["move"]["instructions"] += "\n" + POLICIES[variant]
    return c.payload(r)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"policies": POLICIES, "data_sha256": {v:c.digest(p) for v,p in c.DATA.items()},
                "sources": json.loads((c.OUT / "sources.json").read_text()),
                "script_sha256": c.digest(__file__), "design_sha256": c.digest(HERE / "DESIGN.md")}
    p = OUT / "manifest.json"
    if p.exists():
        assert json.loads(p.read_text()) == manifest, "frozen experiment changed"
    else:
        p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        (HERE / "POLICIES.md").write_text("\n\n".join(f"## {v}\n\n{text}" for v,text in POLICIES.items()) + "\n")
    return manifest


def run(alias):
    manifest = freeze()
    c.paths()
    import torch
    assert torch.backends.mps.is_available()
    torch.manual_seed(71)
    torch.set_num_threads(4)
    destinations = [OUT / f"{alias}-{variant}.jsonl" for variant in POLICIES]
    assert not any(p.exists() for p in destinations), "do not overwrite results"
    t = time.perf_counter()
    if alias == "decider":
        from decider.infer import Decider
        m = Decider(str(c.OUT / "models/decider"), device="mps", dtype=torch.float16, use_graphs=False)
        tok = m.m.tok
        def predict(state, q, opts):
            return m.decide(state, [dict(question=q["instructions"],options=opts)], max_ctx_tokens=8192)[0]["probs_list"]
        def size(state, q, opts):
            assert len(tok.encode("Context:\n" + state, add_special_tokens=False)) < 8192
            _, items = m._decide_items([(state,[dict(question=q["instructions"],options=opts)])],max_ctx_tokens=8192)
            return len(items[0]["ids"])
    else:
        from jevk5.runtime import JevK5
        m = JevK5(str(c.OUT / "models/jevk5"),device="cpu",dtype=torch.float16,graphs=False)
        m.model.to("mps")
        m.device="mps"
        m.slot_weight=m.model.lm_head.weight[m.slots].detach().contiguous()
        tok = m.tok
        def predict(state,q,opts):
            return list(m.probabilities(state,q)[0].values())
        def size(state,q,opts):
            n=len(m.encode(state,q["instructions"],opts));assert n<=16384
            return n
    torch.mps.synchronize()
    (OUT/f"{alias}.meta.json").write_text(json.dumps({"load_seconds":time.perf_counter()-t,"torch":torch.__version__,"device":"mps","dtype":"float16","script_sha256":manifest["script_sha256"]},indent=2))
    rs=c.records()
    # Check all inputs before any model scoring; no truncation, no unknown tokens.
    lengths={}
    for variant in POLICIES:
        for split,r in rs:
            state,q,names,opts=payload(r,variant)
            for text in [state,q["instructions"],*opts]:
                assert tok.unk_token_id not in tok.encode(text,add_special_tokens=False)
            lengths[variant,split,r["id"]]=size(state,q,opts)
    for variant,dest in zip(POLICIES,destinations):
        with dest.open("x") as f:
            for i,(split,r) in enumerate(rs):
                state,q,names,opts=payload(r,variant)
                torch.mps.synchronize();start=time.perf_counter()
                with torch.inference_mode():probs=predict(state,q,opts)
                torch.mps.synchronize();ms=(time.perf_counter()-start)*1000
                assert len(probs)==len(names) and all(math.isfinite(v) and 0<=v<=1 for v in probs)
                assert abs(sum(probs)-1)<.002
                pred=names[max(range(len(probs)),key=probs.__getitem__)]
                gold=r["labels"]["move"]["gold"]
                row=dict(split=split,id=r["id"],family=r["meta"]["family"],slice=r["meta"]["slice"],variant=variant,
                         gold=gold,pred=pred,correct=pred in gold,probabilities=dict(zip(names,probs)),p_top=max(probs),
                         input_tokens=lengths[variant,split,r["id"]],elapsed_ms=ms,first_call=i==0)
                f.write(json.dumps(row,ensure_ascii=False)+"\n");f.flush()
                if i%12==0 or i==len(rs)-1:print(json.dumps({"model":alias,"policy":variant,"done":i+1,"total":len(rs)}),flush=True)
    print("complete",alias,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("model",choices=["freeze","decider","jevk5"]);a=p.parse_args()
    freeze() if a.model=="freeze" else run(a.model)
