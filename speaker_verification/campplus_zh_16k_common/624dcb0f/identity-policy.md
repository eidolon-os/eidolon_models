# Identity evidence: calibration and per-use thresholds

Recognition returns advisory identity evidence, never admission, authentication or a memory grant on its own. Channel reads three files beside the CAM++ weights:

- **`identity-backend.json` + `identity-cohort.npz`**: the calibration. It follows WeSpeaker's back-end:
  - subtract a domain mean;
  - AS-norm over each side's 100 highest cohort scores;
  - an affine quality-measure function (QMF) trained with a Cllr objective. Its inputs are the normalised score, the two sides' voiced speech (min / max / difference / ratio), their cohort means (min / max / difference), and the **condition**: whether the probe comes through a different channel from the one the template was built on (an indicator, plus its product with the score).

  The cohort is 1074 utterance embeddings of 69 public AISHELL-1 speakers, each recording passed through a phone channel and a simulated far-field device channel. It holds embeddings only: no audio, transcripts or names.
- **`identity-policy.json`**: how the evidence becomes decisions.
  - Every compared Person gets an open-set posterior; someone not enrolled has prior 0.5.
  - Each use has its own threshold: `personal_memory` 0.99, `addressing` 0.90, `speaker_label` 0.90, `candidate_evidence` 0.95, and `discovery` 0.99 on the posterior of someone not enrolled.
  - Consumers read the decision for their own use and never set a threshold of their own.

A registration template is the mean of three samples, and its voiced speech is their sum. Its channel is `mobile`, because registration captures only through the phone app. A probe's channel is its device kind; every phone app counts as one channel. The calibration was trained under exactly these conventions.

## Evidence (2026-10-31, `evals/speaker-verification/cohort-20261031`, `results-identity-v1.json`)

30 held-out speakers (never in the cohort), two-person households, registration through Channel's own `evaluate_enrollment`:

| | own, same channel | own, other channel | stranger accepted, same / other channel |
|---|---|---|---|
| `personal_memory` | 98.3 % | 80.0 % | 0.00 % / 0.25 % |
| `addressing` | 99.7 % | 94.1 % | 0.01 % / 1.61 % |
| previous fixed cosine 0.55 | 99.4 % | 25.1 % | 0.70 % / 0.02 % |

Verification: EER 1.07 %, Cllr 0.067 overall; same channel actDCF 0.016 (min 0.015), other channel 0.42 (min 0.33).

**Not shown.** The other channel is simulated (band limit, synthetic room response, noise, Opus 16 kbps), not a real board. Household adaptation (design §8.2 1c) checks real devices before it is switched on. A cohort recorded elsewhere does not describe a household's own channel: on the first real phone registrations it scored near zero against them (design §3.3). The calibration's channel condition and the household adaptation exist for that reason.

## Unchanged

- Below `min_speech_ms` the result is insufficient speech.
- A probe shorter than `min_unknown_speech_ms` is never evidence of someone new.
- Neither silence padding nor audio from previous turns counts as new identity evidence.
- Missing or mismatched files, model mismatch, dependency failure and expired evidence never produce a decision.
- Memory separately verifies the signed Person and Data's authorization for an enabled Realm.
- The UI consumes the decision for addressing; unknown and unavailable remain normal conversation states.
