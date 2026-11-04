"""Anonymous pools on simulated households: a minimal experiment before B2 (design §3.8).

The pool judgment as planned for Channel, run over 3D-Speaker on the same doubly
held-out folds as ``track_eval.py``, to see whether G2 can be met before the
pipeline is built. Per fold, households of 2-4 members registered from the
held-out phone, and visitors who are not, hold conversations on the held-out
far device in a random order, one per day:

- each turn is decided against everyone known (members, candidates, confirmed
  voices); ``personal_memory`` on an established Person attributes it, anything
  else becomes an unresolved record;
- after each conversation a pool pass groups unresolved records into voices
  (continuing a group by the same LLR rule as a track), then for each group:
  attribute it to a known Person (``personal_memory``, 8 s), join a candidate
  (``candidate_evidence``), or discover a candidate (``discovery``, 3 records,
  4 s); then for each candidate: attribute it to a known Person, merge it into
  an older candidate, or promote it (2 conversations on different days,
  3 turns, 8 s, ``discovery`` against the established).

The two thresholds the policy leaves empty (``candidate_evidence``,
``discovery``) and the grouping LLR are chosen on one half of the speakers and
reported on the other. G2 (design, top): a visitor confirmed after two
conversations >= 70%; a visitor's records attributed to a member or another
visitor <= 0.5%; a visitor split into two anonymous Persons <= 5%; a member
made anonymous and still not attributed to themselves at their fifth
conversation <= 5%.

    eidolon_channel/.venv/bin/python -I \\
        eidolon_models/evals/speaker-verification/pools-20261104/pool_eval.py \\
        --corpus <3d-speaker> --embeddings <3d-speaker>/embeddings.json \\
        --policy eidolon_models/speaker_verification/campplus_zh_16k_common/624dcb0f/identity-policy.json \\
        --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRACKS = HERE.parent / "tracks-20261102"
TURNS = 4                      # turns per conversation
MEMBER_CONVERSATIONS = 6
VISITOR_CONVERSATIONS = (1, 2, 3)
ATTRIBUTE_MS, DISCOVER_MS, PROMOTE_MS = 8000, 4000, 8000
DISCOVER_RECORDS, PROMOTE_TURNS = 3, 3
GRID = {
    "group_llr": (6.936, 3.0),
    "candidate_evidence": (0.99,),
    "discovery": (0.9, 0.99),
}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


tp = module(TRACKS / "two_person_eval.py", "two_person_eval")
te, it = tp.te, tp.it


def posteriors(llrs, prior_unknown):
    """Open-set posteriors over the named voices, and P(someone not among them)."""
    if not llrs:
        return {}, 1.0
    names = list(llrs)
    log_each = math.log((1 - prior_unknown) / len(names))
    terms = np.array([math.log(prior_unknown)] + [log_each + llrs[n] for n in names])
    w = np.exp(terms - terms.max())
    w /= w.sum()
    return dict(zip(names, w[1:])), float(w[0])


class Person:
    def __init__(self, pid, kind, truth=None, registration=None):
        self.pid, self.kind, self.truth, self.registration = pid, kind, truth, registration
        self.records = []        # records attributed to this Person
        self.retired = False

    @property
    def established(self):
        return self.kind in ("member", "confirmed")

    def sides(self):
        """(condition, template) pairs this Person is compared through."""
        out = []
        if self.registration is not None:
            out.append(("far_cross", self.registration))
        if self.kind != "member" and self.records:
            out.append(("far_same", te.track_of([r["seg"] for r in self.records])))
        return out


class Pools:
    def __init__(self, scorer, policy, params):
        self.scorer, self.policy, self.params = scorer, policy, params
        self.persons, self.unresolved, self.next_id = {}, [], 0
        self.events = []         # (kind, truth of records, person truth or None)

    def add_member(self, name, registration):
        self.persons[name] = Person(name, "member", truth=name, registration=registration)

    def score(self, segments, persons):
        best = {}
        for p in persons:
            sides = p.sides()
            if sides:
                best[p.pid] = max(self.scorer.llr(c, t, segments) for c, t in sides)
        return best

    def turn(self, record):
        """The turn's own decision: only ``personal_memory`` on someone established attributes it."""
        known = [p for p in self.persons.values() if not p.retired]
        llrs = self.score([record["seg"]], known)
        post, _ = posteriors(llrs, self.policy["prior_unknown"])
        if post:
            lead = max(post, key=post.get)
            person = self.persons[lead]
            condition = "far_cross" if person.kind == "member" else "far_same"
            if person.established and post[lead] >= self.policy["conditions"][condition]["personal_memory"]:
                self._attribute([record], person, "turn")
                return
        self.unresolved.append(record)

    def _attribute(self, records, person, kind):
        for r in records:
            r["person"] = person.pid
            person.records.append(r)
            self.events.append((kind, r["truth"], person.truth))

    def _groups(self):
        groups = []
        for r in sorted(self.unresolved, key=lambda r: r["order"]):
            scored = [(self.scorer.llr("far_same", te.track_of([x["seg"] for x in g]), [r["seg"]]), g)
                      for g in groups]
            fits = [(llr, g) for llr, g in scored if llr >= self.params["group_llr"]]
            if fits:
                max(fits, key=lambda x: x[0])[1].append(r)
            else:
                groups.append([r])
        return groups

    def pass_(self, day):
        p_mem = self.policy["conditions"]
        for group in self._groups():
            segs = [r["seg"] for r in group]
            speech = sum(r["seg"]["speech_ms"] for r in group)
            active = [p for p in self.persons.values() if not p.retired]
            llrs = self.score(segs, active)
            post, unknown = posteriors(llrs, self.policy["prior_unknown"])
            lead = max(post, key=post.get) if post else None
            person = self.persons.get(lead)
            if person is not None and person.established:
                cond = "far_cross" if person.kind == "member" else "far_same"
                if post[lead] >= p_mem[cond]["personal_memory"] and speech >= ATTRIBUTE_MS:
                    self._take(group, person, "attribute")
                    continue
            if person is not None and person.kind == "candidate" and post[lead] >= self.params["candidate_evidence"]:
                self._take(group, person, "join")
                continue
            if (len(group) >= DISCOVER_RECORDS and speech >= DISCOVER_MS
                    and unknown >= self.params["discovery"] and self._consistent(group)):
                self.next_id += 1
                c = Person(f"c{self.next_id}", "candidate")
                self.persons[c.pid] = c
                self._take(group, c, "discover")
                c.truth = Counter(r["truth"] for r in c.records).most_common(1)[0][0]
        self._candidates(day)

    def _take(self, group, person, kind):
        for r in group:
            self.unresolved.remove(r)
        self._attribute(group, person, kind)

    def _consistent(self, group):
        """Every pair of records the same voice, by the grouping rule."""
        return all(self.scorer.llr("far_same", te.track_of([a["seg"]]), [b["seg"]]) >= self.params["group_llr"]
                   for a, b in itertools.combinations(group, 2))

    def _candidates(self, day):
        p_mem = self.policy["conditions"]
        for c in sorted((p for p in self.persons.values() if p.kind == "candidate"), key=lambda p: p.pid):
            if c.pid not in self.persons:
                continue
            segs = [r["seg"] for r in c.records]
            speech = sum(s["speech_ms"] for s in segs)
            established = [p for p in self.persons.values() if p.established]
            llrs = self.score(segs, established)
            post, unknown = posteriors(llrs, self.policy["prior_unknown"])
            if post:
                lead = max(post, key=post.get)
                person = self.persons[lead]
                cond = "far_cross" if person.kind == "member" else "far_same"
                if post[lead] >= p_mem[cond]["personal_memory"] and speech >= ATTRIBUTE_MS:
                    self._dissolve(c, person, "attribute")
                    continue
            others = [p for p in self.persons.values() if p.kind == "candidate" and p.pid < c.pid]
            if others:
                opost, _ = posteriors(self.score(segs, others), self.policy["prior_unknown"])
                olead = max(opost, key=opost.get)
                if opost[olead] >= self.params["candidate_evidence"]:
                    self._dissolve(c, self.persons[olead], "merge")
                    continue
            days = {r["day"] for r in c.records}
            if (len(days) >= 2 and len(c.records) >= PROMOTE_TURNS and speech >= PROMOTE_MS
                    and unknown >= self.params["discovery"]):
                c.kind = "confirmed"
                self.events.append(("promote", c.truth, c.truth))

    def _dissolve(self, c, into, kind):
        del self.persons[c.pid]
        records, c.records = c.records, []
        self._attribute(records, into, kind)


def household(scorer, policy, params, regs, far, members, visitors, rng):
    pools = Pools(scorer, policy, params)
    for m in members:
        pools.add_member(m, regs[m])
    plan = [(m, i) for m in members for i in range(MEMBER_CONVERSATIONS)]
    plan += [(v, i) for v, n in visitors for i in range(n)]
    order = [plan[i] for i in rng.permutation(len(plan))]
    seen = Counter()
    progress = []          # (speaker, conversation index, snapshot of where their records are)
    k = 0
    for day, (speaker, _) in enumerate(order):
        seen[speaker] += 1
        conv = far[speaker][(seen[speaker] - 1) * TURNS: seen[speaker] * TURNS]
        for seg in conv:
            k += 1
            pools.turn({"seg": seg, "truth": speaker, "day": day, "order": k, "person": None})
        pools.pass_(day)
        progress.append((speaker, seen[speaker], where(pools, speaker)))
    return pools, progress


def where(pools, speaker):
    """For one speaker: the Persons holding their records, by kind and truth."""
    held = Counter()
    for p in pools.persons.values():
        n = sum(r["truth"] == speaker for r in p.records)
        if n:
            held[(p.pid, p.kind, p.truth)] += n
    return held


def metrics(runs):
    visitor_two = visitor_two_confirmed = 0
    visitor_records = visitor_wrong = 0
    visitors_multi = visitors_split = 0
    members = members_orphaned = 0
    member_records = member_wrong = 0
    for pools, progress, members_, visitors_ in runs:
        for v, n in visitors_:
            if n >= 2:
                visitor_two += 1
                after_two = next(h for s, i, h in progress if s == v and i == 2)
                visitor_two_confirmed += any(kind == "confirmed" and truth == v for (_, kind, truth) in after_two)
            final = where(pools, v)
            for (_, kind, truth), count in final.items():
                visitor_records += count
                visitor_wrong += count if truth != v else 0
            anon = {pid for (pid, kind, truth) in final if kind in ("candidate", "confirmed") and truth == v}
            if n >= 2:
                visitors_multi += 1
                visitors_split += len(anon) >= 2
        for m in members_:
            members += 1
            fifth = next(h for s, i, h in progress if s == m and i == 5)
            members_orphaned += any(kind in ("candidate", "confirmed") for (_, kind, _t) in fifth)
            for (_, kind, truth), count in where(pools, m).items():
                member_records += count
                member_wrong += count if truth != m else 0
    rate = lambda a, b: round(a / b, 4) if b else None
    return {
        "visitor_confirmed_after_two": rate(visitor_two_confirmed, visitor_two), "visitors_with_two": visitor_two,
        "visitor_records_wrong": rate(visitor_wrong, visitor_records), "visitor_records": visitor_records,
        "visitor_split": rate(visitors_split, visitors_multi),
        "member_anonymous_at_fifth": rate(members_orphaned, members), "members": members,
        "member_records_wrong": rate(member_wrong, member_records),
    }


def passes(m):
    return (m["visitor_confirmed_after_two"] is not None and m["visitor_confirmed_after_two"] >= 0.70
            and (m["visitor_records_wrong"] or 0) <= 0.005 and (m["visitor_split"] or 0) <= 0.05
            and (m["member_anonymous_at_fifth"] or 0) <= 0.05 and (m["member_records_wrong"] or 0) <= 0.005)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-folds", type=int, default=0, help="stop after this many folds (timing)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "work"
    work.mkdir(exist_ok=True)
    calibration, _ = it.wespeaker()
    policy = json.loads(args.policy.read_text())
    utts = it.corpus(args.corpus, args.embeddings)
    places = sorted({p for u in utts if (p := it.place(u))})
    near_places = [p for p in places if p.startswith("near:")]
    far_devices = sorted({it.device_of(p) for p in places if p.startswith("far:")})
    speakers = sorted({u["speaker"] for u in utts})
    halves = (set(speakers[0::2]), set(speakers[1::2]))
    tested = te.leak_free(utts, set(places))
    params_grid = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    runs = {(half, i): [] for half in (0, 1) for i in range(len(params_grid))}
    fold = 0
    for half, (train_spk, test_spk) in enumerate((halves, halves[::-1])):
        for near_out in near_places:
            for far_out in far_devices:
                fold += 1
                if args.max_folds and fold > args.max_folds:
                    break
                rng = np.random.default_rng(fold)
                seen = [p for p in places if p != near_out
                        and not (p.startswith("far:") and it.device_of(p) == far_out)]
                by = defaultdict(lambda: defaultdict(list))
                for u in tested:
                    p = it.place(u)
                    if u["speaker"] in test_spk and p and (p == near_out or (
                            p.startswith("far:") and it.device_of(p) == far_out)):
                        by[u["speaker"]][p].append(u)
                regs, far = {}, {}
                for s, g in sorted(by.items()):
                    phone = sorted(g.get(near_out, []), key=lambda u: u["segment"])
                    segs = [u for p, us in g.items() if p.startswith("far:") for u in us]
                    if len(segs) < TURNS:
                        continue
                    far[s] = [segs[i] for i in rng.permutation(len(segs))]
                    if len(phone) > it.REGISTRATION_SEGMENTS:
                        regs[s] = it.template(phone[:it.REGISTRATION_SEGMENTS])
                member_pool = [s for s in far if s in regs and len(far[s]) >= TURNS * MEMBER_CONVERSATIONS]
                visitor_pool = [s for s in far if s not in member_pool]
                if len(member_pool) < 2 or len(visitor_pool) < 2:
                    continue
                scorer = te.fold_backend(calibration, utts, train_spk, seen, rng, work, fold)
                households = []
                members_left = list(rng.permutation(member_pool))
                visitors_left = list(rng.permutation(visitor_pool))
                while len(members_left) >= 2 and visitors_left:
                    size = min(len(members_left), int(rng.integers(2, 5)))
                    ms = [members_left.pop() for _ in range(size)]
                    vs = []
                    for _ in range(min(len(visitors_left), int(rng.integers(2, 5)))):
                        v = visitors_left.pop()
                        n = min(int(rng.choice(VISITOR_CONVERSATIONS)), len(far[v]) // TURNS)
                        if n:
                            vs.append((v, n))
                    households.append((ms, vs, int(rng.integers(1 << 30))))
                for i, params in enumerate(params_grid):
                    for ms, vs, seed in households:
                        pools, progress = household(scorer, policy, params, regs, far, ms, vs,
                                                    np.random.default_rng(seed))
                        runs[(half, i)].append((pools, progress, ms, vs))
                print(f"fold {fold}: {near_out} / {far_out}: {len(households)} households", flush=True)

    report = {"setup": {"folds": fold, "turns": TURNS, "member_conversations": MEMBER_CONVERSATIONS,
                        "grid": GRID, "policy": policy["version"]}, "by_params": []}
    for i, params in enumerate(params_grid):
        report["by_params"].append({"params": params,
                                    "half0": metrics(runs[(0, i)]), "half1": metrics(runs[(1, i)])})
    # Choose on one half, report on the other.
    chosen = {}
    for pick, report_half in ((0, 1), (1, 0)):
        ok = [r for r in report["by_params"] if passes(r[f"half{pick}"])]
        best = max(ok or report["by_params"],
                   key=lambda r: (passes(r[f"half{pick}"]), r[f"half{pick}"]["visitor_confirmed_after_two"] or 0))
        chosen[f"chosen_on_half{pick}"] = {"params": best["params"], "passes_on_choice": passes(best[f"half{pick}"]),
                                            "reported_on_half": report_half, "result": best[f"half{report_half}"],
                                            "passes": passes(best[f"half{report_half}"])}
    report["chosen"] = chosen
    (args.out / "pools.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(chosen, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
