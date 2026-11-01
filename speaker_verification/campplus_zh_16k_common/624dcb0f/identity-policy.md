# Identity evidence: calibration and per-use thresholds

Recognition returns advisory identity evidence, never admission, authentication or a memory grant on its own. Channel reads three files beside the CAM++ weights:

- **`identity-backend.json` + `identity-cohort.npz`** (back-end v2): the calibration. It follows WeSpeaker's back-end:
  - subtract a domain mean;
  - AS-norm over each side's 100 highest cohort scores;
  - an affine quality-measure function (QMF) over WeSpeaker's own factors — the normalised score and min / max / difference / ratio of the two sides' voiced speech, embedding magnitude after mean subtraction and cohort mean — plus the **condition**: whether the probe comes through a different channel from the one the template was built on (an indicator, plus its product with the score).

  Every step is WeSpeaker's own code at `9fecd6c` (`get_mean_std`, `gather_calibration_factors`, `train_calibration_model`); Channel's runtime reproduces its LLRs to 0.005.

  The cohort is 1074 utterance embeddings of 69 public AISHELL-1 speakers, each recording passed through a simulated phone channel and a simulated far-field device channel. It holds embeddings only: no audio, transcripts or names.
- **`identity-policy.json`** (policy v3): how the evidence becomes decisions.
  - Every compared Person gets an open-set posterior; someone not enrolled has prior 0.5.
  - Each use has its own threshold. A use without one is off: nothing is ever decided for it.
  - Consumers read the decision for their own use and never set a threshold of their own.

  Policy v3 turns on `addressing` 0.90 and `speaker_label` 0.90 only. `personal_memory`, `candidate_evidence` and `discovery` write lasting records and stay off: on live speech across devices the calibration is over-confident (below). When switched on, their thresholds start at 0.99 / 0.95 / 0.99.

A registration template is the mean of three samples, and its voiced speech is their sum. Its channel is `mobile`, because registration captures only through the phone app. A probe's channel is its device kind; every phone app counts as one channel. The calibration was trained under exactly these conventions.

## Evidence (`evals/speaker-verification/cohort-20261031`)

**Public corpus, simulated channels** (`results-identity-v2.json`; 30 held-out speakers, two-person households): EER 1.02 %, Cllr 0.059. `personal_memory` admits 98.6 % of own same-channel and 78.2 % of own other-channel speech; strangers 0.00 % / 0.19 %.

**Live speech across devices** (`results-live-devices.json`, 2026-11-01): 3D-Speaker test set, 240 speakers recorded live and simultaneously on phones, a tablet, a PC, a recording pen, a directional microphone and arrays, 0.1–4 m away; median voiced speech 1.65 s.
- The official cross-device trials (one utterance enrolled) give EER 19.5 %, Cllr 1.45. Raw cosine gives EER 19.6 %: on short, distant, cross-device speech the embedding itself separates speakers poorly.
- Two-person households, registered from a close device standing in for the handheld phone: strangers named for `personal_memory` in 4.0 % of probes (0.45 % of phone probes), for `addressing` in 10.8 %; a member scored as a stranger to a household where only the other is enrolled is named for `personal_memory` in 8.5 %.

**Back-end v3, withdrawn.** v3 added a recorded channel (AISHELL replayed through a laptop speaker into the phone app) to the cohort and the QMF. On that channel and on one household sharing its phone and room it removed the over-confidence. But on the live multi-device set it was worse than v2 (Cllr 3.47, strangers named for `personal_memory` in 10.7 %), so one added channel does not generalise. See `results-household-replay.json` and `results-live-devices.json`.

**Not shown.** No calibration here bounds false acceptance at 0.5 % on live speech across devices. That needs live, multi-device training data, and evidence accumulated over a conversation rather than per utterance (design §8.2 1c, stage 3).

## Unchanged

- Below `min_speech_ms` the result is insufficient speech.
- A probe shorter than `min_unknown_speech_ms` is never evidence of someone new.
- Neither silence padding nor audio from previous turns counts as new identity evidence.
- Missing or mismatched files, model mismatch, dependency failure and expired evidence never produce a decision.
- Memory separately verifies the signed Person and Data's authorization for an enabled Realm.
- The UI consumes the decision for addressing; unknown and unavailable remain normal conversation states.
