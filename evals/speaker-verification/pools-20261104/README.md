# Anonymous pools on simulated households (B2 minimal experiment, 2026-11-04)

Before building B2 (design §3.8, §8.2), the pool judgment as planned for Channel
was run on 3D-Speaker with the same doubly held-out folds as `tracks-20261102`:
households of 2–4 members registered on the held-out phone and 2–4 visitors who
are not, holding 4-turn conversations on the held-out far device, one per day.
Each turn is decided against everyone known; anything not attributed becomes an
unresolved record; after each conversation unresolved records are grouped by the
track continuation rule and then attributed, joined to a candidate or discovered;
candidates are attributed, merged or promoted (2 days, 3 turns, 8 s). The empty
policy thresholds (`candidate_evidence`, `discovery`) and the grouping LLR were
varied; one half of the speakers would choose, the other report.

Result (`results.json`, 16 folds, 155 members, 40 visitors with two or more
conversations):

| grouping LLR | visitor confirmed after two (≥ 70%) | member made anonymous at 5th (≤ 5%) | records wrong (≤ 0.5%) |
|---|---|---|---|
| 6.936 (track continuation) | 0% / 11% | 2–8% | visitors 0–4.2%, members 0–0.7% |
| 3.0 | 16–21% | 8–27% | members 3.7–3.9% |

No setting meets G2. The cause is B1's: on a far device a voice's single
segments link to each other only weakly at an acceptable error
(`tracks-20261102`: other voices join 1% → own continues 20%; 0.5% → 9.7%;
0.1% → 3.4%). Grouping unresolved records is the same linking, so visitors are
not discovered at a strict rule and members are swept into candidates at a
loose one. B2 is not built on this; see the design's §8.2 status.

    eidolon_channel/.venv/bin/python -I \
        eidolon_models/evals/speaker-verification/pools-20261104/pool_eval.py \
        --corpus <3d-speaker> --embeddings <3d-speaker>/embeddings.json \
        --policy eidolon_models/speaker_verification/campplus_zh_16k_common/624dcb0f/identity-policy.json \
        --out <dir>
