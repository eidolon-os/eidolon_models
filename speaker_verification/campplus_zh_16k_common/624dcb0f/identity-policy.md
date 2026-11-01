# Identity evidence: calibration and per-use thresholds by condition

Recognition returns advisory identity evidence, never admission, authentication or a memory grant on its own. Registration happens on the phone; speech arrives on every device. How well the embedding separates speakers depends on the comparison, so evidence and thresholds depend on its **condition** (design §3.3):

- `near_same`: a phone registration template against speech captured by the phone app;
- `far_same`: a device's own observation template against that device;
- `far_cross`: any other pairing, such as a phone registration against a speaker on a board.

The condition follows the template's and the probe's channels. A probe's channel is its device kind; every phone app counts as one channel (`mobile`), and registration captures only through the phone app.

Channel reads three files beside the CAM++ weights:

- **`identity-backend.json` + `identity-backend.npz`** (back-end v4): the calibration.
  - The mean of live multi-device speech is subtracted from every embedding.
  - `far_cross` is scored after an LDA projection (128 dimensions) trained on live multi-device speech; it compensates the channel and carries over to devices it never saw. Same-device comparisons skip it, because within one device it only loses.
  - Each condition has its own affine quality-measure function (QMF) over the cosine and WeSpeaker's min / max / difference / ratio of the two sides' voiced speech, trained with WeSpeaker's own `train_calibration_model` at `9fecd6c`. Channel's runtime reproduces its LLRs to 1e-5.
  - Each condition records the longest track it was calibrated on (`max_track_segments`: 2 close, 5 far). A longer conversation track is scored on its latest segments within it; beyond it the duration factors would extrapolate and push accumulated speech down.
  - No AS-norm and no cohort. A cohort recorded elsewhere inflated scores on every real channel it met, and on live speech no cohort helped across devices.
- **`identity-policy.json`** (policy v6): how the evidence becomes decisions.
  - Every compared Person gets an open-set posterior; someone not enrolled has prior 0.5.
  - Each use has its own threshold in each condition; the condition of the leading Person's comparison decides. A use without a threshold is off in that condition.
  - Household guard: when one member's registration windows, scored against another member's phone template, would already meet a use's threshold, that use in `near_same` also needs an LLR above the highest of them plus `household_margin_llr`. It only ever makes a use stricter.
  - `candidate_evidence` and `discovery` stay off until the anonymous lifecycle (design A3, B2).
  - Conversation tracks (design §3.2): a turn's segment continues the track of its own voice on the same device when its LLR against that track reaches `track_continuation_llr`, and the turn is then decided on the track. The threshold is where another speaker's segment joins a track at most once in a thousand held-out trials on either kind of device; there a voice continues its own track 53 % of the time on the phone and 20 % on far devices. A segment that joined the wrong track would take that track's identity, so continuation is strict.

A registration template is the mean of three samples, and its voiced speech is their sum. Voiced speech is measured at the phone app's level; the embedding uses the audio as captured.

## Evidence (`evals/speaker-verification/identity-20261101`, `results.json`)

Live speech: the 3D-Speaker test set, 240 speakers recorded live and simultaneously on phones, a tablet, a PC, a recording pen, a directional microphone and arrays, 0.1–4 m away; median voiced speech per utterance 1.65 s. Registrations are ten close-device segments (about the voiced speech of the phone's three sentences); probes are single segments or tracks of k segments.

Held out twice: each half of the speakers is tested with a back-end trained on the other half, and every fold holds out one close device and one far device, so the tested devices were never seen in training (20 folds).

| Condition | EER, single segment | Cllr | Threshold `personal_memory` / `addressing` | Own speech named for `personal_memory` (k = 1 / 2 / 3) | Strangers named for `personal_memory` |
|---|---|---|---|---|---|
| `near_same` | 1.3 % (k = 2: 0.6 %) | 0.05 | 0.9982 / 0.9524 | 77 % / 84 % / – | 0.01 % |
| `far_same` | 4.4 % (k = 3: 2.4 %) | 0.18 | 0.9989 / 0.9950 | 10 % / 12 % / 9 % | 0.01 % |
| `far_cross` | 11.2 % (k = 3: 8.7 %) | 0.87 | 0.99 / 0.90 | 37 % / 41 % / 41 %, overstated (below) | 0.17 % |

Each threshold is the design's (0.99 / 0.90) when held-out false acceptance meets its limit (0.5 % for `personal_memory`, 2 % for `addressing` and `speaker_label`) for every track length, both for strangers in two-person households and for a member when only the other is enrolled; otherwise the smallest value 0.1 log-odds apart above it that does. In `near_same` and `far_same` the binding case is a single-member household.

For comparison, back-end v2 on the same corpus named strangers for `personal_memory` in 2.8–4.2 % of close same-device probes and 4.0 % across all devices.

The household's own phone registrations (read-only, aggregates only): the two members' windows score each other at most LLR 2.9 (back-end v2: 12.2), which meets no threshold, so the household guard stays off; their own windows are named for `personal_memory` in 75 % below 1.5 s of voiced speech and in all of them above, and for `addressing` in all of them.

3D-Speaker records each sentence on several devices at once, and 41.9 % of the far segments repeat a sentence of the speaker's own registration, recorded at the same moment. They are in this table's `far_cross` trials, which overstates its own-speech coverage: without them, a household on a device the back-end never saw names its members for `personal_memory` in 13 % and for `addressing` in 27 % of turns on tracks of three or more segments, with the phone template alone (`evals/speaker-verification/tracks-20261102`). Strangers' acceptance, which set the thresholds, is unaffected.

Coverage far from the phone is low. The embedding carries more (the best any threshold could reach there is 62–65 % for `personal_memory`), but the calibration does not turn it into posteriors on devices it never saw; calibrating each range of track length separately was worse. With a member's own observation template on the device, a single turn is named for `personal_memory` in 58 % and for `addressing` in 82 % of cases; making that template needs an attributed track first.

## Data and its terms

`identity-backend.npz` and the weights in `identity-backend.json` are derived from the 3D-Speaker dataset (Alibaba DAMO Academy, https://3dspeaker.github.io), licensed CC BY-SA 4.0; these two files are distributed under CC BY-SA 4.0 with that attribution. They hold a mean vector, a projection matrix and calibration weights: no audio, transcripts, names or per-speaker embeddings. Citation: Zheng et al., "3D-Speaker: A Large-Scale Multi-Device, Multi-Distance, and Multi-Dialect Corpus for Speech Representation Disentanglement", 2023.

## Unchanged

- Below `min_speech_ms` the result is insufficient speech.
- A probe shorter than `min_unknown_speech_ms` is never evidence of someone new.
- Neither silence padding nor audio from previous turns counts as new identity evidence.
- Missing or mismatched files, model mismatch, dependency failure and expired evidence never produce a decision.
- Memory separately verifies the signed Person and Data's authorization for an enabled Realm.
- The UI consumes the decision for addressing; unknown and unavailable remain normal conversation states.
