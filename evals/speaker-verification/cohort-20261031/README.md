# Impostor cohort and acceptance set (2026-10-31)

Design §3.3 (no labelling; per-Person, per-duration thresholds). Three scripts:

| Script | What it does |
|---|---|
| `fetch_aishell.py` | Streams the first 8 recordings ≥ 4 s of each AISHELL-1 speaker on the pinned Hugging Face revision (100 speakers there). Reads only the head of each archive; audio stays under `.eidolon/`. |
| `build_cohort.py` | Levels each recording to device-like RMS (0.04/0.06/0.08), Opus round trip at 48 kHz (24/32 kbps), then Channel's own `ReferenceSpeakerExtractor.extract_enrollment`: utterance plus registration windows. Speakers split by hash: 70 % cohort (`impostor-cohort.npz`, shipped in the model artifact), 30 % held out (`held-out.json`, never shipped). |
| `evaluate.py` | Held-out speakers form two-person households, register through Channel's `evaluate_enrollment`, and every other recording is decided by Channel's `household_tables` + `decide` and by the fixed v2 policy. |

`results-v3.json` is the run behind the shipped policy (`max_false_accept` 0.003). Summary and limits are in the model's `recognition-policy.md`.

Reproduce (Channel environment, which owns the extractor):

```bash
python3 -I eidolon_models/evals/speaker-verification/cohort-20261031/fetch_aishell.py .eidolon/speaker-cohort-20261031
eidolon_channel/.venv/bin/python -I eidolon_models/evals/speaker-verification/cohort-20261031/build_cohort.py \
  --model-dir .eidolon/mac-product/state/models/<campplus artifact> \
  --corpus .eidolon/speaker-cohort-20261031 --corpus .eidolon/anonymous-speaker-eval-20261028 \
  --out .eidolon/speaker-cohort-20261031/build
eidolon_channel/.venv/bin/python -I eidolon_models/evals/speaker-verification/cohort-20261031/evaluate.py \
  --held-out .eidolon/speaker-cohort-20261031/build/held-out.json --model-dir <dir with policy + cohort>
```

The build is deterministic (codec and level per recording are fixed by position), so the same corpus gives the same cohort file.
