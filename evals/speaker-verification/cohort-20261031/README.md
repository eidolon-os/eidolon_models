# Identity evidence: cohort, calibration and acceptance set

Design §3.3 and §8.2 (1a). Calibrated per-Person evidence, following WeSpeaker's back-end: domain mean, AS-norm, condition-aware QMF trained with Cllr.

| Script | What it does |
|---|---|
| `fetch_aishell.py` | Streams the first 8 recordings ≥ 4 s of each AISHELL-1 speaker on the pinned Hugging Face revision (100 speakers there). Reads only the head of each archive; audio stays in `.eidolon/speaker-cohort-20261031/`. |
| `build_cohort.py` | Puts every recording through a `phone` channel (registration path) and a `device` channel (far-field board: band limit, synthetic room response, noise, Opus 16 kbps), then through Channel's own `extract_enrollment`. Speakers are split by hash: 70 % cohort pool, 30 % held out. Output: `embeddings.json`. |
| `identity_eval.py` | Fits the calibration on cohort speakers (each trial's speakers left out of the cohort statistics), writes `identity-backend.json` and `identity-cohort.npz`, and evaluates held-out speakers. Verification: EER, Cllr, act/min DCF. Two-person households: decisions per use, compared with the fixed 0.55 threshold and the v4 band floors. |

`results-identity-v1.json` is the run behind backend v1. Summary for held-out speakers, prior of someone not enrolled 0.5:

| | own, same channel | own, other channel | stranger accepted, same / other channel |
|---|---|---|---|
| `personal_memory` (P ≥ 0.99) | 98.3 % | 80.0 % | 0.00 % / 0.25 % |
| `addressing` (P ≥ 0.90) | 99.7 % | 94.1 % | 0.01 % / 1.61 % |
| fixed 0.55 (v2) | 99.4 % | 25.1 % | 0.70 % / 0.02 % |
| v4 band floors | 99.4 % | 18.4 % | 0.27 % / 0.00 % |

Without the channel condition in the calibration, same-channel strangers were over-trusted (personal memory 0.96 %, addressing 9.4 %) and other-channel owners under-trusted. The `device` channel is simulated; real boards are checked on household data before the in-domain adaptation is switched on (1c).

Reproduce (Channel environment, which owns the extractor):

```bash
python3 -I eidolon_models/evals/speaker-verification/cohort-20261031/fetch_aishell.py .eidolon/speaker-cohort-20261031
eidolon_channel/.venv/bin/python -I eidolon_models/evals/speaker-verification/cohort-20261031/build_cohort.py \
  --model-dir .eidolon/mac-product/state/models/<campplus artifact> \
  --corpus .eidolon/speaker-cohort-20261031 --out .eidolon/speaker-cohort-20261031/build
eidolon_channel/.venv/bin/python -I eidolon_models/evals/speaker-verification/cohort-20261031/identity_eval.py \
  --embeddings .eidolon/speaker-cohort-20261031/build/embeddings.json --out <dir>
```

Every step is deterministic (codec, level and channel noise are seeded per recording; the calibration starts from WeSpeaker's fixed initialisation).
