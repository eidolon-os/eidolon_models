# Identity evidence on live multi-device speech (back-end v4, policy v6)

Design §3.3 and §8.2 1c. Registration happens on the phone; speech arrives on every device. `identity_train.py` trains the back-end Channel runs (`eidolon/livekit/plugins/speaker_verification/evidence.py`) and chooses each use's threshold in each comparison condition, on the 3D-Speaker test set (Alibaba DAMO Academy, CC BY-SA 4.0; 240 speakers recorded live and simultaneously on phones, a tablet, a PC, a recording pen, a directional microphone and arrays, 0.1–4 m).

| Condition | Template | Probe |
|---|---|---|
| `near_same` | ten segments on a close device (directional microphone 0.1 m, PC 0.2 m), about the phone's three registration sentences | other segments on that device |
| `far_same` | three segments of one device at one distance (an observation template) | that device's other segments |
| `far_cross` | the close-device registration | a device 1 m or more away |

Probes are single segments or tracks of k segments (mean vector, voiced speech summed). Embeddings are Channel's own extractor's, cached beside the corpus (`embeddings.json`); voiced speech was measured at the phone app's level, as the runtime now does for every device.

The back-end: subtract the mean of live speech; `far_cross` is scored after an LDA projection (128 dimensions); one affine QMF per condition over the cosine and WeSpeaker's duration factors, trained with WeSpeaker's own `train_calibration_model` (pinned in `wespeaker_ref.py`). The factors are standardised for its L-BFGS, which otherwise stops after a step or two on factors of such different scales, and the weights are mapped back to the raw factors.

The back-end also records, per condition, the longest track its calibration was trained on (`max_track_segments`: 2 close, 5 far); Channel scores a longer conversation track on its latest segments within it (design §3.2). The policy's `track_continuation_llr` is where another voice joins a track on the same device at most once in a thousand held-out trials, measured as the runtime does it: a segment against a track of one up to that many segments, of its own voice or of another speaker on the same device.

**Acceptance is held out twice** (20 folds): each half of the speakers is tested with a back-end trained on the other half, and every fold holds out one close device and one far device. Two-person households decide every probe per use. A use's threshold is the design's (`personal_memory` 0.99, `addressing` and `speaker_label` 0.90) when held-out false acceptance meets its limit (0.5 %, 2 %) at every track length, for strangers and for a member when only the other is enrolled; otherwise the smallest value 0.1 log-odds above it that does; otherwise off. `results.json` holds the held-out verification measures per condition and track length, the household rates at the chosen thresholds, the constraint that bound each threshold, and the delivered back-end with its check against WeSpeaker's `infer_calibration`.

The delivered back-end is trained on all speakers and devices. Policy v6 adds only the continuation threshold to v5; the calibration and every use's threshold are unchanged.

**Known overstatement.** 3D-Speaker records each sentence on several devices at once, and 41.9 % of the far segments repeat a sentence of the speaker's own close-device registration, recorded at the same moment. They stay in this recipe's `far_cross` training and testing, so the held-out `far_cross` coverage here is overstated: without them it is 13 % for `personal_memory` and 27 % for `addressing` on tracks of three or more segments (`tracks-20261102`). Strangers' acceptance, which set the thresholds, is unaffected.

```bash
eidolon_channel/.venv/bin/python -I \
    eidolon_models/evals/speaker-verification/identity-20261101/identity_train.py \
    --corpus /Users/manson/ai/data/speaker-corpora/3d-speaker \
    --embeddings /Users/manson/ai/data/speaker-corpora/3d-speaker/embeddings.json \
    --model eidolon_models/speaker_verification/campplus_zh_16k_common/624dcb0f \
    --out <dir>
```

Every step is deterministic (fold seeds, WeSpeaker's fixed initialisation). Copy `identity-backend.json`, `identity-backend.npz` and `identity-policy.json` from `<dir>` into the model package and update `ops/component.toml`.
