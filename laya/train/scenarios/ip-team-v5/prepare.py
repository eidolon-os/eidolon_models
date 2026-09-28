"""Freeze audited GLM snapshots; no inherited data, no locally assigned golds."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
from transformers import AutoTokenizer
from eidolon_models_laya.sequence import Tokenizer, build_sequence, to_internal
from eidolon_sdk.biz.participation import Candidate, Context, Constraints, DecisionRequest, Message
from eidolon_laya_train.records import Record, write_jsonl, target_vector
from eidolon_laya_train.model import record_items
from eidolon_models_laya.participation_text import project_request, check_capacity

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parents[1]
NAMES = {"train": ("阿岚", "小禾", "明川", "若竹", "云舟", "星野", "阿乔", "闻溪"),
         "val": ("初晴", "小满", "秋山", "知夏", "路遥", "阿榕", "云鹤", "雨声")}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand(entry, variant):
    case, eid, split = entry["case"], entry["id"], entry["split"]
    slots = [s for s in "abcd" if f"role_{s}" in case]
    rng = random.Random(f"v5:{eid}:{variant}")
    names = dict(zip(slots, rng.sample(NAMES[split], len(slots))))
    if entry["same_name"]:
        names["b"] = names["a"]
    ids = {s: "companion-" + hashlib.sha256(f"{eid}:{variant}:{s}".encode()).hexdigest()[:16] for s in slots}
    def render(text):
        return text.format_map(names)
    def message(mid, author, text):
        return Message(message_id=mid, author_kind="user" if author == "user" else "companion",
                       author_id="owner" if author == "user" else ids[author], text=render(text))
    user = message("user-request", "user", case["user"])
    earlier = [message("earlier", case["earlier_author"], case["earlier_text"])] if case["earlier_author"] else []
    peer = [message("peer", case["peer_author"], case["peer_text"])] if case["peer_author"] else []
    history = [user, *earlier, *peer] if peer else [*earlier, user]
    order = slots[:]
    rng.shuffle(order)
    req = DecisionRequest(decision_id=f"{eid}-v{variant}", context_ref=eid, context_version=len(history),
                          membership_revision=1, cancellation_epoch=0, timeout_ms=3000,
                          user_request=user, trigger=history[-1], context=Context(recent_messages=tuple(history)),
                          candidates=tuple(Candidate(companion_id=ids[s], display_name=names[s], description=render(case[f"role_{s}"])) for s in order),
                          constraints=Constraints(remaining_replies=8-len(peer)))
    state, question, mapping = project_request(req)
    by_id = {cid: slot for slot, cid in mapping.items()}
    gold = ["respond:" + by_id[ids[a.split(":")[1]]] if a.startswith("respond:") else a for a in case["acceptable"]]
    target_vector(question, {"gold": gold})
    return Record(id=f"{eid}~v{variant}", scenario="ip-team-v5-text", source=entry["source"],
                  state=state, questions={"move": question}, labels={"move": {"gold": gold}},
                  tags=[entry["slice"], f"family:{eid}"], split=split,
                  meta={"family":eid, "slice":entry["slice"], "synthetic":True, "variant":variant,
                        "annotation":"glm_generated_and_label_blind_same_model_review; assistant_audited",
                        "request":req.model_dump(mode="json"), "reason":case["reason"],
                        "review_reason":entry["review_reason"], "generator_sha256":entry["generator_sha256"]})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,nargs="+",default=[TRAIN/"private/ip-team-v5-text-pilot1"])
    ap.add_argument("--audit",type=Path,default=HERE/"audit-decisions.json")
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise ValueError("immutable output already exists")
    decisions=json.loads(args.audit.read_text())
    rows=[]
    source_files=sorted(p for root in args.source for p in (root/"reviewed").glob("*.json"))
    for path in source_files:
        rows += json.loads(path.read_text())["cases"]
    if set(decisions) != {e["id"] for e in rows}:
        raise ValueError("every generated case needs an explicit audit decision")
    tok=AutoTokenizer.from_pretrained(TRAIN/"runs/r14/checkpoint/tokenizer",local_files_only=True)
    runtime_tok=Tokenizer(TRAIN/"runs/r14/checkpoint/tokenizer")
    splits={"train":[],"val":[]}; families={}; lengths=[]; texts={}
    accepted=[]
    for entry in rows:
        verdict=decisions[entry["id"]]
        if not verdict.get("reason") or verdict["decision"] not in {"accept","reject"}:
            raise ValueError("audit needs decision and reason")
        if verdict["decision"]=="reject":
            continue
        if entry["status"]!="agreed_pending_audit":
            raise ValueError("disagreement cannot enter this pilot")
        fingerprint=hashlib.sha256(json.dumps({k:v for k,v in entry["case"].items() if k not in {"acceptable","reason"}},ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        if fingerprint in texts:
            raise ValueError("duplicate semantic snapshot")
        texts[fingerprint]=entry["id"]
        entry=dict(entry, planned_slice=entry["slice"], slice=verdict.get("observed_slice",entry["slice"]))
        accepted.append(entry)
        families[entry["id"]]=entry["split"]
        for variant in range(4):
            record=expand(entry,variant)
            length=check_capacity(runtime_tok,record.state,record.questions["move"],2048,256)
            item=record_items(record,tok,{"max_len":2048,"head_max_len":256})[0]
            runtime_ids, runtime_markers, _ = build_sequence(runtime_tok, record.state, to_internal(record.questions["move"]),2048,256)
            if item["ids"]!=runtime_ids or item["markers"]!=runtime_markers or len(item["ids"])!=length:
                raise ValueError("training/runtime tokenizer mismatch")
            lengths.append(length)
            splits[entry["split"]].append(record)
    if not all(splits.values()):
        raise ValueError("empty train/dev split")
    args.out.mkdir(parents=True)
    provenance={"source":[str(p) for p in args.source],"audit_sha256":digest(args.audit),"source_files":{str(p):digest(p) for p in source_files},
                "old_data_inherited":False,"split_policy":"case index4 dev; other indices train; all variants stay within family; no independent test claim",
                "max_input_tokens":max(lengths),"min_input_tokens":min(lengths),"families":families,
                "projection_sha256":digest(HERE.parents[2]/"src/eidolon_models_laya/participation_text.py")}
    for split,records in splits.items():
        p=args.out/f"{split}.jsonl"; write_jsonl(p,records)
        provenance[split]={"records":len(records),"families":len({r.meta['family'] for r in records}),"sha256":digest(p)}
    (args.out/"reviewed-families.json").write_text(json.dumps(accepted,ensure_ascii=False,indent=2))
    (args.out/"lineage.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print(json.dumps(provenance,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
