# Identity evidence: cohort, calibration and acceptance set

Design §3.3 and §8.2. Calibrated per-Person evidence with WeSpeaker's own back-end (`/Users/manson/ai/thirdpart/wespeaker` at `9fecd6c`, loaded by `wespeaker_ref.py`): domain mean, AS-norm, QMF trained with its `train_calibration_model`, plus a channel condition.

| Script | What it does |
|---|---|
| `fetch_aishell.py` | Streams the first 8 recordings ≥ 4 s of each AISHELL-1 speaker on the pinned Hugging Face revision (100 speakers there). Reads only the head of each archive; audio stays in `.eidolon/speaker-cohort-20261031/`. |
| `build_cohort.py` | Puts every recording through a `phone` channel (registration path) and a `device` channel (far-field board: band limit, synthetic room response, noise, Opus 16 kbps), then through Channel's own `extract_enrollment`. Speakers are split by hash: 70 % cohort pool, 30 % held out. Output: `embeddings.json`. |
| `identity_eval.py` | Builds trials with WeSpeaker's `get_mean_std` (each training trial's speakers left out of the cohort), gathers factors with `gather_calibration_factors` (plus the two condition columns), trains with `train_calibration_model`, writes `identity-backend.json` and `identity-cohort.npz`, checks Channel's runtime LLRs against WeSpeaker's `infer_calibration`, and evaluates held-out speakers: EER, Cllr, act/min DCF, and two-person households per use against the fixed 0.55 threshold and the v4 band floors. |
| `replay_corpus.py` | Plays the 99 speakers through the Mac's speaker in level-matched blocks of eight recordings and logs each block's wall-clock span. The phone records them through the app's own capture path (11-01, design §8.2 1c-a). |
| `household_replay.py` | Assigns captured segments to replayed speakers by the block log (a segment touching two blocks is dropped; the nearest public speaker confirms it). Trains the back-end with the replayed channel added to the cohort, to the QMF trials, or both, through `identity_eval.py`. Compares them on the replayed channel's held-out speakers, the public held-out trials and the household: own registrations left out in turn, and each member as a stranger to the other. Also compares cross-speaker similarity by source. |
| `wespeaker_ref.py` | Loads WeSpeaker's two scoring scripts by file at the pinned revision; their small dependencies (fire, kaldiio, termcolor) sit in `/Users/manson/ai/thirdpart/wespeaker-deps`. |

`results-identity-v2.json` is the run behind backend v2. Held-out speakers, prior of someone not enrolled 0.5:

| | own, same channel | own, other channel | stranger accepted, same / other channel |
|---|---|---|---|
| `personal_memory` (P ≥ 0.99) | 98.6 % | 78.2 % | 0.00 % / 0.19 % |
| `addressing` (P ≥ 0.90) | 99.6 % | 93.6 % | 0.01 % / 1.30 % |
| fixed 0.55 (v2) | 99.4 % | 25.1 % | 0.70 % / 0.02 % |
| v4 band floors | 99.4 % | 18.4 % | 0.27 % / 0.00 % |

Channel's runtime LLRs match WeSpeaker's `infer_calibration` to 0.005 (its factor files keep 4–5 decimals). Without the channel condition, same-channel strangers are over-trusted. The `device` channel is simulated.

`results-household-replay.json` is the run behind backend v3 (variant `cohort+qmf`; `delivered` there is v2). 395 of 452 captured segments kept, all 99 speakers, every kept segment's nearest public speaker the assigned one.

| | replayed channel, held-out | public held-out, all / device probes | household live pair: LLR max, named for `personal_memory` |
|---|---|---|---|
| v2 (public channels only) | EER 5.36 %, Cllr 0.684, actDCF 7.62 | Cllr 0.059 / 0.120 | 12.2, 13.6 % |
| replayed channel in the QMF only | Cllr 0.149 | Cllr 0.073 / 0.129 | −1.4, 0 %, but own `personal_memory` coverage falls to 38 % |
| replayed channel in the cohort only | Cllr 1.026 | Cllr 0.062 / 0.127 | 6.6, 2.3 % |
| **v3: in both** | EER 1.05 %, Cllr 0.038, actDCF 0.20 | Cllr 0.067 / 0.139 | 0.24, 0 %; own coverage 100 % |

Cross-speaker cosine (median): public phone channel 0.27, replayed 0.51, household members live 0.39, replayed speakers against the members 0.29. Replayed voices share the loudspeaker's colouring and are not representative strangers against live registrations, so `stranger/*` in the results file is optimistic and not used as a false-acceptance estimate. The household shares the phone and room with the replay; how v3 does elsewhere is not shown.

Reproduce (Channel environment, which owns the extractor):

```bash
python3 -I eidolon_models/evals/speaker-verification/cohort-20261031/fetch_aishell.py .eidolon/speaker-cohort-20261031
eidolon_channel/.venv/bin/python -I eidolon_models/evals/speaker-verification/cohort-20261031/build_cohort.py \
  --model-dir .eidolon/mac-product/state/models/<campplus artifact> \
  --corpus .eidolon/speaker-cohort-20261031 --out .eidolon/speaker-cohort-20261031/build
eidolon_channel/.venv/bin/python -I eidolon_models/evals/speaker-verification/cohort-20261031/identity_eval.py \
  --embeddings .eidolon/speaker-cohort-20261031/build/embeddings.json --out <dir>
eidolon_channel/.venv/bin/python -I eidolon_models/evals/speaker-verification/cohort-20261031/household_replay.py \
  --db .eidolon/mac-product/state/eidolon-system.sqlite3 \
  --log .eidolon/speaker-cohort-20261031/replay/replay-1793518777202.jsonl \
  --samples .eidolon/speaker-cohort-20261031/replay/replay-1793518777202.samples.json \
  --embeddings .eidolon/speaker-cohort-20261031/build/embeddings.json \
  --delivered <model package to compare against> --out <dir>
```

The replay segments were exported from Data before migration 0030 dropped their table; they are public speakers' embeddings only. The household's registrations are read from Data read-only and never leave it: the results hold aggregates only.

Every step is deterministic (codec, level and channel noise are seeded per recording; the calibration starts from WeSpeaker's fixed initialisation).
