"""c 系列：剧本 + 写手话语 → laya 记录、规则对照副本、盲标文件与盲标报告。

在 laya/ 下：
    .venv/bin/python train/scenarios/smart-home-continuation/build.py records --run-dir train/runs/c1
    .venv/bin/python train/scenarios/smart-home-continuation/build.py blind
    .venv/bin/python train/scenarios/smart-home-continuation/build.py blind-report

写手输出在 train/data/continuation/claude/<批>.jsonl（WRITING.md §2），修正在 fixes.jsonl（丢弃或改金标，
原始写手文件不改）。评测集写到 evals/smart-home-continuation/，训练池写到 <run-dir>/continuation.jsonl。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from frames import (  # noqa: E402
    CANCEL,
    KIND_LABEL,
    LAYA,
    REDO,
    SPLIT_HOMES,
    Device,
    family_of,
    load_home,
)

from eidolon_laya_train.records import Record, record_id  # noqa: E402
from eidolon_laya_train.scenario import Scenario  # noqa: E402

DATA = LAYA / "train/data/continuation"
EVALS = LAYA / "evals/smart-home-continuation"
SCENARIO = Scenario.load(HERE / "scenario-c.yaml")
STATE_FORMAT = "continuation-laya-v1"
SELECT_FAMILIES = {"P01", "P04", "P05", "P06", "P07"}  # 按房间 / 名字选中：可做“候选里没有它”的对照
ORDINAL_FAMILY = "P03"


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def frames(split: str) -> list[dict]:
    """frames-<split>.jsonl 加上补充剧本 frames-<split>-*.jsonl（c4 起的 F08 补充）。"""
    out = read(DATA / f"frames-{split}.jsonl")
    for p in sorted(DATA.glob(f"frames-{split}-*.jsonl")):
        out += read(p)
    return out


def writes() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted((DATA / "claude").glob("*.jsonl")):
        for row in read(p):
            row["writer_batch"] = p.stem
            if row["frame_id"] in out:
                raise ValueError(f"{p}: {row['frame_id']} written twice")
            out[row["frame_id"]] = row
    return out


def fixes() -> dict[str, dict]:
    p = DATA / "fixes.jsonl"
    return {r["frame_id"]: r for r in read(p)} if p.exists() else {}


def _bigrams(text: str) -> set[str]:
    return {text[k:k + 2] for k in range(len(text) - 1)}


def visible_hits(f: dict, utterance: str) -> dict[str, int]:
    """每个候选在状态里看得见的文字（选项名 + 房间·类型）有几个二字片段出现在话里。"""
    return {lab: sum(b in utterance for b in _bigrams(lab + f["options"][lab])) for lab in f["labels"]}


def check_visible(f: dict, utterance: str, gold: str, split: str) -> tuple[str | None, str | None]:
    """LABELING C10：返回 (金标, 规则名)；金标为 None 表示丢弃。"""
    if f["question"] != "pick" or gold in (CANCEL, REDO):
        return gold, None
    hits = visible_hits(f, utterance)
    unique = hits[gold] > 0 and all(v < hits[gold] for k, v in hits.items() if k != gold)
    if f["family"] == "P02" and not unique:
        return REDO, "c10-invisible-feature"
    if split == "train" and f["family"] in SELECT_FAMILIES and hits[gold] == 0:
        return None, "c10-no-overlap"
    return gold, None


def gold_class(q: str, gold: str) -> str:
    return "取消" if gold == CANCEL else "重新理解" if gold == REDO else "执行"


def to_record(f: dict, utterance: str, gold: str, *, source: str, extra_tags=(), meta=None) -> Record:
    q = f["question"]
    if q == "pick":
        dynamic = {"candidates": {lab: f["options"][lab] for lab in f["labels"]}}
    else:
        dynamic = {"verbs": {v: d for v, d in f["options"].items() if v != REDO}}
    questions = SCENARIO.build_questions(dynamic, only=[q])
    if gold not in questions[q]["criteria"]:
        raise ValueError(f"{f['frame_id']}: gold {gold!r} is not an option")
    state = {"utterance": utterance, "context": f["context"]}
    return Record(
        id=record_id(SCENARIO.name, state, q), scenario=SCENARIO.name, source=source, state=state,
        questions=questions, labels={q: {"gold": gold}},
        tags=[f"{q}-{gold_class(q, gold)}", q, f["family"], *extra_tags],
        meta={"derived_from": f"continuation-c/{f['context_id']}", "frame_id": f["frame_id"],
              "family": f["family"], "home": f["home"], "state_format": STATE_FORMAT, **(meta or {})},
    )


# ------------------------------------------------------------------ 规则对照副本


def _pick_frame_with(f: dict, cands: list[Device]) -> dict:
    """同一剧本换一组候选（顺序即 Agent 那句的顺序），重算选项、标签和 Agent 那句话。"""
    homonym = f["homonym"]
    names = [f["labels"][0].split("（")[0]] * len(cands) if homonym else [d.name for d in cands]
    labels = [f"{nm}（{d.room}）" if names.count(nm) > 1 else nm for d, nm in zip(cands, names, strict=True)]
    g = dict(f)
    g["devices"] = [d.__dict__ for d in cands]
    g["labels"] = labels
    g["options"] = {lab: f"{d.room}·{KIND_LABEL[d.kind]}" for lab, d in zip(labels, cands, strict=True)} | {
        CANCEL: None, REDO: None}
    g["context"] = dict(f["context"], Agent=f"{'、'.join(names)}，要哪一个？")
    return g


def counterparts(f: dict, utterance: str, gold: str, homes: dict[str, list[Device]],
                 rng: random.Random) -> list[tuple[dict, str, str]]:
    """(剧本, 金标, 对照类型)：只做金标能由规则推出的两种。"""
    if f["question"] != "pick":
        return []
    cands = [Device(**d) for d in f["devices"]]
    out = []
    mentioned = lambda d: d.room in utterance or d.name in utterance  # noqa: E731

    def touches(d: Device) -> bool:  # 房间或名字的任一个二字片段出现在话里（“客厅”对“一楼客厅”）
        return any(t[k:k + 2] in utterance for t in (d.room, d.name) for k in range(len(t) - 1))

    if f["family"] in SELECT_FAMILIES and gold not in (CANCEL, REDO):
        i = f["labels"].index(gold)
        dev = cands[i]
        pool = [d for d in homes[f["home"]]
                if family_of(d) == family_of(dev) and d not in cands and not touches(d)
                and d.room not in {c.room for c in cands}]
        if mentioned(dev) and pool:
            rep = rng.choice(pool)
            g = _pick_frame_with(f, cands[:i] + [rep] + cands[i + 1:])
            out.append((g, REDO, "cp-absent"))  # 用户点名的那台不在候选里
    if f["family"] == ORDINAL_FAMILY and gold not in (CANCEL, REDO) and not any(map(touches, cands)):
        i = f["labels"].index(gold)
        order = list(range(len(cands)))
        for _ in range(10):
            rng.shuffle(order)
            if order[i] != i:
                break
        if order[i] != i:
            g = _pick_frame_with(f, [cands[k] for k in order])
            out.append((g, g["labels"][i], "cp-reorder"))  # 同一个“第几个”，换了顺序
    return out


# ------------------------------------------------------------------ 旧 locked 按新契约重标


def diag44() -> list[Record]:
    old = read(HERE / "eval/locked.jsonl")
    out = []
    for r in old:
        ctx, u, gold = r["state"]["context"], r["state"]["utterance"], r["labels"]["next"]["gold"]
        if ctx["status"] == "no_focus":
            continue  # 无可执行选项：Agent 不调用（LABELING §1）
        if ctx["status"] == "awaiting_target":
            action = ctx["pending_action"]
            names = [t["name"] for t in ctx["targets"]]
            f = {"question": "pick", "family": "diag", "home": "diag", "context_id": "diag-pick",
                 "frame_id": r["id"], "labels": names,
                 "options": {n: f"{n[:-1]}·灯" for n in names} | {CANCEL: None, REDO: None},
                 "context": {"上一句": "把灯打开" if action == "打开" else "把灯关了",
                             "Agent": f"{'、'.join(names)}，要哪一个？", "待执行": action}}
            new = {"cancel": CANCEL, "escalate": REDO}.get(gold) or names[int(gold.split("_")[1])]
        else:
            name = ctx["focus"]["name"]
            f = {"question": "follow", "family": "diag", "home": "diag", "context_id": "diag-follow",
                 "frame_id": r["id"],
                 "options": {"打开或启动": "开启、启动、开始运行", "关闭或停止": "关掉、停止、断电、收起",
                             "调高或增大": "调亮、升温、调大音量或档位", "调低或减小": "调暗、降温、调小、降下",
                             "设为指定的数值或模式": "设到具体的温度、档位、百分比或模式", REDO: None},
                 "context": {"上一句": f"把{name}打开", "Agent": f"已打开{name}", "设备": [name]}}
            # C1：焦点态没有取消，“不要再动它 / 保持不动”交还 LLM
            new = {"open": "打开或启动", "close": "关闭或停止", "cancel": REDO}[gold]
        out.append(to_record(f, u, new, source="diag-44-relabel", extra_tags=["diag-44", *r["tags"]]))
    return out


# ------------------------------------------------------------------ 命令


def cmd_records(args) -> None:
    w, fx = writes(), fixes()
    homes = {h: load_home(h) for hs in SPLIT_HOMES.values() for h in hs}
    stats: dict = {}
    by_split: dict[str, list[Record]] = {}
    for split in ("train", "dev", "test"):
        recs, st = [], Counter()
        for f in frames(split):
            row = w.get(f["frame_id"])
            if row is None:
                st["unwritten"] += 1
                continue
            if "utterance" not in row:
                st["skip"] += 1
                continue
            fix = fx.get(f["frame_id"], {})
            if fix.get("drop"):
                st["dropped_by_fix"] += 1
                continue
            u = " ".join(str(row["utterance"]).split())
            gold, rule = check_visible(f, u, fix.get("gold", f["gold"]), split)
            if rule:
                st[rule] += 1
            if gold is None:
                continue
            meta = {"writer_batch": row["writer_batch"]} | ({"rule": rule} if rule else {})
            recs.append(to_record(f, u, gold, source="claude-writer", meta=meta))
            st["written"] += 1
            # 每个剧本自己的随机数：评测集的对照副本不随训练池批次变化
            for g, g_gold, kind in counterparts(f, u, gold, homes, random.Random(f["frame_id"])):
                recs.append(to_record(g, u, g_gold, source="counterpart-rule", extra_tags=[kind], meta=meta))
                st[kind] += 1
        if split != "train":  # 评测里与训练池原句相同的话单独打标，报告时分开看
            seen = {r.state["utterance"] for r in by_split["train"]}
            for r in recs:
                if r.state["utterance"] in seen:
                    r.tags.append("utt-seen-in-train")
                    st["utt-seen-in-train"] += 1
        by_split[split] = recs
        stats[split] = dict(st) | {"records": len(recs),
                                   "by_gold": dict(Counter(r.tags[0] for r in recs))}
    EVALS.mkdir(parents=True, exist_ok=True)
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    outputs = {run / "continuation.jsonl": by_split["train"], EVALS / "c-dev.jsonl": by_split["dev"],
               EVALS / "c-test.jsonl": by_split["test"], EVALS / "diag-44.jsonl": diag44()}
    for path, recs in outputs.items():
        if path.parent == EVALS and path.exists() and not args.write_evals:
            stats[path.name] = "frozen (exists; --write-evals to rebuild)"  # 评测集冻结后不随训练池变化
            continue
        seen = set()
        with path.open("w", encoding="utf-8") as fh:
            for r in recs:
                if r.id in seen:
                    continue
                seen.add(r.id)
                fh.write(r.to_json() + "\n")
        stats[path.name] = len(seen)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


def _blind_row(f: dict, utterance: str) -> dict:
    if f["question"] == "pick":
        opts = {lab: f["options"][lab] for lab in f["labels"]}
        opts |= SCENARIO.questions["pick"].exits
    else:
        opts = {v: d for v, d in f["options"].items() if v != REDO} | SCENARIO.questions["follow"].exits
    return {"frame_id": f["frame_id"], "题目": f["question"],
            "说明": SCENARIO.questions[f["question"]].instructions,
            "context": f["context"], "选项": opts, "utterance": utterance}


def cmd_blind(args) -> None:
    """dev / test 全量、训练池每 10 个上下文抽 1 个剧本，打乱后按 150 条一批写盲标文件。"""
    w = writes()
    rng = random.Random(7)
    qa = DATA / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        rows = []
        fr = frames(split)
        if split == "train":
            by_ctx: dict[str, list[dict]] = {}
            for f in fr:
                by_ctx.setdefault(f["context_id"], []).append(f)
            ctxs = sorted(by_ctx)
            fr = [rng.choice(by_ctx[c]) for c in rng.sample(ctxs, len(ctxs) // 10)]
        for f in fr:
            row = w.get(f["frame_id"])
            if row and "utterance" in row:
                rows.append(_blind_row(f, row["utterance"]))
        rng.shuffle(rows)
        for b in range(0, len(rows), 150):
            p = qa / f"blind-{split}-{b // 150:02d}.jsonl"
            p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[b:b + 150]), "utf-8")
        print(split, len(rows))


def cmd_blind_report(args) -> None:
    key = {f["frame_id"]: f for s in ("train", "dev", "test") for f in frames(s)}
    fx = fixes()
    w = writes()
    report: dict = {}
    disagreements = []
    for p in sorted((DATA / "qa").glob("labels-*.jsonl")):
        split = p.stem.split("-")[1]
        agree = n = 0
        for row in read(p):
            f = key[row["frame_id"]]
            gold = fx.get(f["frame_id"], {}).get("gold", f["gold"])
            gold, _rule = check_visible(f, w[f["frame_id"]]["utterance"], gold, split)
            n += 1
            if row["label"] == gold:
                agree += 1
            else:
                disagreements.append({"frame_id": f["frame_id"], "family": f["family"], "spec": f["spec"],
                                      "context": f["context"], "utterance": w[f["frame_id"]]["utterance"],
                                      "gold": gold, "blind": row["label"], "note": row.get("note")})
        r = report.setdefault(split, {"n": 0, "agree": 0})
        r["n"] += n
        r["agree"] += agree
    for r in report.values():
        r["rate"] = round(r["agree"] / r["n"], 4) if r["n"] else None
    (DATA / "qa" / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    (DATA / "qa" / "disagreements.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in disagreements), "utf-8")
    fam = Counter((d["family"], d["gold"] if d["gold"] in (CANCEL, REDO) else "执行", d["blind"]
                   if d["blind"] in (CANCEL, REDO) else "执行") for d in disagreements)
    print(json.dumps(report, ensure_ascii=False), len(disagreements))
    for k, v in fam.most_common(25):
        print(" ", k, v)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("records")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--write-evals", action="store_true", help="重建 evals/ 下的评测集（默认冻结，已存在就不写）")
    p.set_defaults(func=cmd_records)
    p = sub.add_parser("blind")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.set_defaults(func=cmd_blind)
    p = sub.add_parser("blind-report")
    p.set_defaults(func=cmd_blind_report)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
