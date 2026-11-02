# Changing the speaker model: a one-sided test on live multi-device speech (design §8.2 trigger)

B1 left far cold start at 13 % (`personal_memory`) and 27 % (`addressing`) per track against targets of 50 % and 70 %, which triggered "change the speaker model". Three ModelScope candidates (Apache 2.0) go through exactly the pipeline that accepted the delivered CAM++: the 3D-Speaker test set embedded as Channel embeds a turn, `identity-20261101/identity_train.py` (held out twice; each candidate's own back-end and per-use thresholds), then `tracks-20261102/track_eval.py` with that candidate's own policy.

| Name | ModelScope model | Trained on | Dim |
|---|---|---|---|
| `delivered` | `iic/speech_campplus_sv_zh-cn_16k-common` (624dcb0f) | 200k speakers | 192 |
| `campplus_3dspeaker` | `iic/speech_campplus_sv_zh-cn_3dspeaker_16k` | 3D-Speaker train set | 512 |
| `eres2net_large_3dspeaker` | `iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k` | 3D-Speaker train set | 512 |
| `eres2netv2_common` | `iic/speech_eres2netv2_sv_zh-cn_16k-common` | 200k speakers | 192 |

**One-sided.** The 3D-Speaker test speakers are not in its train set, but its devices are: a model trained there has seen the tested phones and boards, which the product's never are. On this data such a model can only be rejected, never accepted. Failing here closes the lever; passing would have needed far-field, text-independent speech from devices it never saw (HI-MIA is the wake word only and cannot stand in).

The criteria were fixed before any candidate ran (`compare.py`): far cold start per track of three or more segments, phone template only, `personal_memory` ≥ 50 % and `addressing` ≥ 70 %; strangers named on the far device ≤ 0.5 % and ≤ 2 %; the phone's single turn no worse than the delivered model's.

The delivered model was re-embedded from a fresh download of the test set (MD5 as published) and reproduced every committed number, its back-end arrays and its policy bit for bit.

```bash
eidolon_channel/.venv/bin/python -I \
    eidolon_models/evals/speaker-verification/models-20261102/embed_candidate.py \
    --corpus <3d-speaker> --reference <3d-speaker>/embeddings.json \
    --model-dir <ModelScope model directory> --out <embeddings> [--jobs 1 --device mps]
# then identity_train.py and track_eval.py on <embeddings>, and:
eidolon_channel/.venv/bin/python -I \
    eidolon_models/evals/speaker-verification/models-20261102/compare.py \
    delivered=<run> campplus_3dspeaker=<run> eres2net_large_3dspeaker=<run> eres2netv2_common=<run> \
    --out results.json
```

## Results (`results.json`)

| | delivered | campplus_3dspeaker | eres2net_large_3dspeaker | eres2netv2_common |
|---|---|---|---|---|
| Far cold start, track ≥ 3 segments (`personal_memory` / `addressing`) | 13 % / 27 % | 8 % / 22 % | 7 % / 21 % | 14 % / 34 % |
| Best any threshold could reach there, 3–17 segments (`personal_memory`) | 62–65 % | 36–43 % | 36–44 % | 70–74 % |
| Strangers named on the far device (max over cold and warm) | 0.06 % / 0.30 % | 0.40 % / 1.1 % | 0.09 % / 0.47 % | 0 % / 0.31 % |
| Phone, single turn | 77 % / 94 % | 80 % / 92 % | 85 % / 95 % | 87 % / 96 % |
| With the member's device template, single turn | 58 % / 82 % (n 57) | 100 % / 100 % (n 35) | 93 % / 100 % (n 42) | 42 % / 76 % (n 55) |
| A voice continues its own far track (another joins 1 %) | 20 % | 88 % | 87 % | 11 % |
| Held-out EER, single segment: `near_same` / `far_same` / `far_cross` | 1.3 / 4.4 / 11.2 % | 1.7 / 2.3 / 8.6 % | 1.5 / 2.0 / 8.2 % | 1.4 / 3.4 / 9.8 % |
| `far_cross` Cllr | 0.87 | 1.04 | 1.20 | 0.66 |

No candidate meets the far cold-start criterion, the models trained on 3D-Speaker not even on the devices they were trained with. The lever is closed for far cold start.

- The 3D-Speaker models separate same-device voices far better (device templates, track continuation), as they should on devices they saw; across devices from a phone registration they are worse where it matters, at low false acceptance, although their equal error rate is lower.
- `eres2netv2_common` is a modest gain everywhere the phone template decides (phone +10 points of `personal_memory`, far `addressing` +7 points, better calibrated), far short of the targets, at about three times the delivered model's inference cost (75–85 ms against 25 ms for 3 s of audio on the Mac's CPU).
- For every model the gap between coverage at the policy's thresholds and the best any threshold reaches is the same order: the posteriors on devices calibration never saw, not the embedding, hold far cold start back.
