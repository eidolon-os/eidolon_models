# Identity evidence: calibration and per-use thresholds

Recognition returns advisory identity evidence, never admission, authentication or a memory grant on its own. Channel reads three files beside the CAM++ weights:

- **`identity-backend.json` + `identity-cohort.npz`** (back-end v3): the calibration. It follows WeSpeaker's back-end:
  - subtract a domain mean;
  - AS-norm over each side's 100 highest cohort scores;
  - an affine quality-measure function (QMF) over WeSpeaker's own factors — the normalised score and min / max / difference / ratio of the two sides' voiced speech, embedding magnitude after mean subtraction and cohort mean — plus the **condition**: whether the probe comes through a different channel from the one the template was built on (an indicator, plus its product with the score).

  Every step is WeSpeaker's own code at `9fecd6c` (`get_mean_std`, `gather_calibration_factors`, `train_calibration_model`); Channel's runtime reproduces its LLRs to 0.005.

  The cohort and the QMF's training trials come from 69 public AISHELL-1 speakers in three channels:
  - a simulated phone channel and a simulated far-field device channel (1074 utterance embeddings);
  - **a real recorded channel**: the same speakers replayed through a laptop speaker and captured by the phone app's own path (275 segment embeddings). Both sides of these trials are replayed.

  It holds embeddings only: no audio, transcripts or names, and no household member's voice.
- **`identity-policy.json`** (policy v4): how the evidence becomes decisions.
  - Every compared Person gets an open-set posterior; someone not enrolled has prior 0.5.
  - Each use has its own threshold. A use without one is off: nothing is ever decided for it.
  - Consumers read the decision for their own use and never set a threshold of their own.

  Policy v4 turns on `addressing` 0.90 and `speaker_label` 0.90 only. `personal_memory`, `candidate_evidence` and `discovery` write lasting records and stay off until the calibration passes a check with live, in-room impostors across households (design §8.2 1c): so far there is one live impostor pair. When switched on, their thresholds start at 0.99 / 0.95 / 0.99.

A registration template is the mean of three samples, and its voiced speech is their sum. Its channel is `mobile`, because registration captures only through the phone app. A probe's channel is its device kind; every phone app counts as one channel. The calibration was trained under exactly these conventions.

## Evidence (2026-11-01, `evals/speaker-verification/cohort-20261031`, `results-household-replay.json`)

Back-end v2 was trained on the two simulated channels only. On a channel it had never seen — enrollment and probes both from the replay recording, 30 held-out speakers — it was badly over-confident. Adding the replayed channel to the cohort and to the QMF's training trials fixed that. It cost little on the simulated channels.

| | replayed channel, held-out speakers | public corpus, held-out (all / device probes) |
|---|---|---|
| v2 | EER 5.36 %, Cllr 0.684, actDCF 7.62 | Cllr 0.059 / 0.120, EER 1.02 % / 2.06 % |
| **v3** | EER 1.05 %, Cllr 0.038, actDCF 0.20 | Cllr 0.067 / 0.139, EER 1.22 % / 2.39 % |

One household's phone, live speech (2 registered members; registration windows, one recording left out of the template in turn, plus the trial recording):

| | own, `addressing` / `personal_memory` | each member as a stranger to a household where only the other is enrolled |
|---|---|---|
| v2 | 100 % / 100 % (42 probes) | LLR max 12.2; named in 25 % / 13.6 % of probes |
| **v3** | 100 % / 100 % | LLR max 0.24; never named |

**Not shown.**
- The replayed recording and the household share the phone and the room, so the household figures do not show how v3 does in other homes.
- Replayed voices carry the loudspeaker's colouring. Against live registrations they are not representative strangers, so they are not used to estimate a household's false acceptance.
- The far-field device channel is still simulated, not a real board.
- One live impostor pair cannot bound false acceptance at 0.5 %. That needs live impostors across households and devices (design §8.2 1c).

## Unchanged

- Below `min_speech_ms` the result is insufficient speech.
- A probe shorter than `min_unknown_speech_ms` is never evidence of someone new.
- Neither silence padding nor audio from previous turns counts as new identity evidence.
- Missing or mismatched files, model mismatch, dependency failure and expired evidence never produce a decision.
- Memory separately verifies the signed Person and Data's authorization for an enabled Realm.
- The UI consumes the decision for addressing; unknown and unavailable remain normal conversation states.
